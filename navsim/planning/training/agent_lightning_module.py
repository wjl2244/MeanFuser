from typing import Dict, Tuple
import torch
import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        
        return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        loss_dict = self._step(batch, "train")
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
        
        return loss_dict['loss']

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        loss_dict = self._step(batch, "val")
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
        
        return loss_dict["loss"]

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            pred_trajectorys = predictions["trajectory"]

        tokens = targets['token']
        result = {}

        if pred_trajectorys[0].shape[0] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        for idx, (pred_trajectory, token) in enumerate(zip(
                pred_trajectorys.cpu().numpy(),
                tokens,
        )):
            result[token] = {
                "trajectory": Trajectory(pred_trajectory, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                } 
        return result
