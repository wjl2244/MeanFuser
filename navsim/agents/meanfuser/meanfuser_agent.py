# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from typing import Any, Union, Dict, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LRScheduler, OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.meanfuser.meanfuser_features import MFFeatureBuilder, MFTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.meanfuser.meanfuser_model import MeanfuserModel
from navsim.agents.meanfuser.meanfuser_config import MeanfuserConfig


class MeanfuserAgent(AbstractAgent):
    def __init__(
            self,
            config: MeanfuserConfig,
            lr: float,
            max_epochs: int = 150,
            checkpoint_path: str = None,
            **kwargs,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.meanfuser_model = MeanfuserModel(config)
        self.max_epochs = max_epochs

        from navsim.agents.meanfuser.pdm_score import ComputePDMScore
        self.pdm_score_computer = ComputePDMScore(cache_path='navtest_v1_metric_cache')

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=True)
        print(f"Loaded model from {self._checkpoint_path}")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[3],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [MFTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [MFFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        return self.meanfuser_model(features, targets)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ):
        loss_dict = self.meanfuser_model.get_meanfuser_loss(predictions, targets)

        if not self.training:
            if self.pdm_score_computer is not None:
                pdm_scores = self.pdm_score_computer.get_pdm_scores(predictions["trajectory"], targets["token"])
                loss_dict.update(pdm_scores)
        
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        
        if self._config.scheduler_type == 'constant':
            return torch.optim.Adam(self.meanfuser_model.parameters(), lr=self._lr)
        elif self._config.scheduler_type == 'warmup_cos':
            backbone_params_name = '_backbone.image_encoder'
            img_backbone_params = list(
                filter(lambda kv: backbone_params_name in kv[0], self.meanfuser_model.named_parameters()))
            default_params = list(
                filter(lambda kv: backbone_params_name not in kv[0], self.meanfuser_model.named_parameters()))
            params_lr_dict = [
                {'params': [tmp[1] for tmp in default_params]},
                {
                    'params': [tmp[1] for tmp in img_backbone_params],
                    'lr': self._lr * self._config.lr_mult_backbone,
                    'weight_decay': self._config.backbone_wd
                }
            ]
            
            optim = torch.optim.AdamW(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
            scheduler = WarmupCosLR(
                optim,
                min_lr=1e-7,
                epochs=self.max_epochs,
                warmup_epochs=3,
            )
            return{
                "optimizer": optim,
                "lr_scheduler": scheduler
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:

        # val_score = "val_loss"
        # filename = "{epoch:02d}-{val_loss:.4f}"
        # mode='min'
        val_score = "val_score"
        filename = "{epoch:02d}-{val_score:.4f}"
        mode='max'

        ckpt_callback = ModelCheckpoint(
            save_top_k=10,
            monitor=val_score,
            mode=mode,
            dirpath=f"{self._config.output_dir}/",
            filename=filename,
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        return [ckpt_callback, lr_monitor,]


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = optimizer.param_groups[0]["lr"]
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
