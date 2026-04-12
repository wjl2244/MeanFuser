import math
import os
import pickle
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from torch.nn.attention import SDPBackend, sdpa_kernel
# from torch.nn.attention import sdpa_kernel as sdp_kernel

from navsim.agents.meanfuser.meanfuser_config import MeanfuserConfig
from navsim.agents.meanfuser.utils import cumsum_traj, diff_traj
from navsim.agents.meanfuser.utils import HORIZON, ACTION_DIM_DELTA, ACTION_DIM_ORI
from navsim.agents.meanfuser.modules import SimpleDiffusionTransformer, SemanticMapHead
from navsim.agents.meanfuser.meanfuser_backbone import ResNetBackbone
from navsim.agents.meanfuser.arm_model import ARMModel


class MeanFlowHead(nn.Module):
    def __init__(
            self,
            config: MeanfuserConfig,
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-0.4,                 # Mean parameter for logit_normal distribution
            time_sigma=1.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.5,      # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            # CFG related params
            cfg_omega=2.0,                # CFG omega param, default 1.0 means no CFG
            cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
            cfg_min_t=0.0,                # Minium CFG trigger time 
            cfg_max_t=0.8,                # Maximum CFG trigger time
            jvp_api='autograd',
            ):
        super().__init__()
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t
        
        obs_len = 1024 + 30
        self.model = SimpleDiffusionTransformer(
            config.tf_d_model, config.decoder_nhead, config.tf_d_ffn, config.decoder_layers,
            input_dim=ACTION_DIM_DELTA * HORIZON,
            obs_len=obs_len,
        )
        self.config = config

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        if self.config.noise_type == 'multi_gaussian':
            with open(self.config.navtrain_mean_std_path, 'rb') as f:
                mean_std = pickle.load(f)
            self.cluster_trajs = nn.Parameter(mean_std['cluster_trajs'], requires_grad=False)
            self.gaussian_mean = nn.Parameter(mean_std['center_points'], requires_grad=False) * 0.5
            self.gaussian_std = nn.Parameter(mean_std['center_std'], requires_grad=False)
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t 
    
    def forward(self, outputs, trajectories):
        """
        Compute MeanFlow loss for trajectory data (shape: [batch, num_points, features])
        """
        batch_size = trajectories.shape[0]
        trajectories = trajectories.float()
        device = trajectories.device
        dtype = trajectories.dtype

        x_t = diff_traj(trajectories)
        
        if self.config.noise_type == 'gaussian':
            e = torch.randn_like(x_t) * self.config.noise_std
        elif self.config.noise_type == 'multi_gaussian':
            cluster_centers = self.cluster_trajs.unsqueeze(0).to(device)  # [1,num_proposals,8,4]
            min_index = torch.norm(trajectories[:,None,] - cluster_centers, dim=-1).mean(dim=-1).argmin(dim=-1)
            mean = self.gaussian_mean.unsqueeze(0).unsqueeze(2).to(device)  # [1, num_proposals, 1, 4]
            std = self.gaussian_std.unsqueeze(0).unsqueeze(2).to(device)
            size = (batch_size, mean.shape[1], HORIZON, ACTION_DIM_DELTA)
            e = torch.randn(size, device=device) * std + mean  # [bs, num_proposals, 8, 4]
            e = e.gather(1, min_index[:,None,None,None].expand(-1, 1,HORIZON, ACTION_DIM_DELTA)).squeeze(1).to(dtype)
        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")
        
        # conditions
        bev_query = outputs['bev_query'].contiguous()
        context_query = outputs.get('context_query', None)

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)
        r, t = r.view(-1, 1, 1), t.view(-1, 1, 1)

        z_t = (1 - t) * x_t + t * e
        v_t = e - x_t

        drdt = torch.zeros_like(r)
        dtdt = torch.ones_like(t)

        v_hat = v_t

        def fn_current(z, cur_r, cur_t):
            return self.model(z, cur_r.view(-1), cur_t.view(-1), bev_query, context_query)

        primals = (z_t, r, t)
        tangents = (v_hat, drdt, dtdt)
        
        if self.create_graph:
            # Enable the Math or Efficient attention backends
            with sdpa_kernel([SDPBackend.MATH,]):
                u, dudt = self.jvp_fn(fn_current, primals, tangents, create_graph=True)
            # with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False) :
            #     u, dudt = self.jvp_fn(fn_current, primals, tangents, create_graph=True)
        else:
            u, dudt = self.jvp_fn(fn_current, primals, tangents)

        # u_target = v_hat - (t - r) * dudt[0]
        # error = u[0] - u_target.detach()

        v_pred = u[0] + (t - r) * dudt[0].detach()
        error = v_hat - v_pred
        meanflow_loss = F.l1_loss(v_pred, v_hat)
        
        return meanflow_loss
    
    @torch.no_grad()
    def sample(self, encoder_output):
        result = {}

        bev_query = encoder_output['bev_query']
        context_query = encoder_output.get('context_query', None) 
        device = bev_query.device
        dtype = bev_query.dtype

        BS = bev_query.shape[0]
        NUM_PROPOSALS = self.config.num_proposals

        bev_query = bev_query.repeat_interleave(NUM_PROPOSALS, dim=0)
        if context_query is not None:
            context_query = context_query.repeat_interleave(NUM_PROPOSALS, dim=0)

        size=(BS * NUM_PROPOSALS, HORIZON, ACTION_DIM_DELTA)
        
        if self.config.noise_type == 'gaussian':
            e = torch.randn(size, device=device) * self.config.noise_std
        elif self.config.noise_type == 'multi_gaussian':
            assert self.config.num_proposals % self.gaussian_mean.shape[0] == 0, "gaussian_mean must be divisible by num_proposals"
            repeat_num = self.config.num_proposals // self.gaussian_mean.shape[0]
            mean = self.gaussian_mean.unsqueeze(0).unsqueeze(2).to(device).repeat_interleave(repeat_num, dim=1)
            std = self.gaussian_std.unsqueeze(0).unsqueeze(2).to(device).repeat_interleave(repeat_num, dim=1)
            e = torch.randn((BS, NUM_PROPOSALS, HORIZON, ACTION_DIM_DELTA), device=device) * std + mean
            e = e.view(size).to(dtype)

        if self.config.num_sample_steps == 1:  
            r = torch.zeros(BS*NUM_PROPOSALS, device=device)
            t = torch.ones(BS*NUM_PROPOSALS, device=device)
            u = self.model(e, r, t, bev_query, context_query, training=False)
            x0 = e - u
        
        elif self.config.num_sample_steps > 1:
            z = e
        
            time_steps = torch.linspace(1, 0, self.config.num_sample_steps + 1, device=device)
            
            for i in range(self.config.num_sample_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]
                
                t = torch.full((BS * NUM_PROPOSALS,), t_cur, device=device)
                r = torch.full((BS * NUM_PROPOSALS,), t_next, device=device)
                
                u = self.model(z, r, t, bev_query, context_query, training=False)
                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = z - (t_cur - t_next) * u
            x0 = z
        else:
            raise ValueError(f"Unsupported number of sample steps: {self.config.num_sample_steps}")
        
        traj = cumsum_traj(x0)
        result['pred_diff_traj'] = x0.view(BS, NUM_PROPOSALS, HORIZON, -1)
        result['pred_trajectorys'] = traj.view(BS, NUM_PROPOSALS, HORIZON, ACTION_DIM_ORI)

        return result


class MeanfuserModel(nn.Module):
    def __init__(self, config: MeanfuserConfig):
        super().__init__()
        self._config = config

        self._backbone = ResNetBackbone(config)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self.bev_proj = nn.Sequential(
            nn.Linear(config.tf_d_model+64, config.tf_d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.tf_d_model),
        )

        self._query_splits = [1, config.num_bounding_boxes,]
        self._query_embedding = nn.Embedding(config.num_bounding_boxes+1, config.tf_d_model)
        self._keyval_embedding = nn.Embedding(64+1, config.tf_d_model)
  
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._meanflow_head = MeanFlowHead(config)
        self._semantic_map_head = SemanticMapHead(config)
        self._arm_model_head = ARMModel(config)

        self.use_beyonddrive = os.environ.get('USE_BEYONDDRIVE', None)

    def encoder(self, features):
        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        if isinstance(status_feature, list):
            status_feature = status_feature[-1]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]

        batch_size = status_feature.shape[0]
        encoder_output = {}
       
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        
        encoder_output['bev_feature_upscale'] = bev_feature_upscale.clone()
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)

        # status_feature[:,:4] *= 0
        status_encoding = self._status_encoding(status_feature).unsqueeze(1)
        
        keyval = torch.concatenate([bev_feature, status_encoding], dim=1)
        keyval = keyval + self._keyval_embedding.weight[None, ...]
        
        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)  # [bs,64+d_model,64,64]
        cross_bev_feature = F.interpolate(cross_bev_feature, scale_factor=0.5)
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)
        trajectory_query, context_query = query_out.split(self._query_splits, dim=1)
        
        encoder_output['status_query'] = status_encoding.clone()
        encoder_output['bev_query'] = cross_bev_feature
        encoder_output['trajectory_query'] = trajectory_query.clone()
        encoder_output['context_query'] = context_query.clone()
        return encoder_output

    def get_meanfuser_loss(self, predictions, targets):
        results = {}
        encoder_output = predictions.pop('encoder_output')

        meanflow_loss = self._meanflow_head(encoder_output, targets['trajectory'])
        arm_loss = self._arm_model_head.get_arm_loss(predictions, targets)
        bev_semantic_loss, bev_semantic_map = self._semantic_map_head(encoder_output['bev_feature_upscale'], targets)

        if self.use_beyonddrive is not None:
            negative_index = (targets['negative_trajectory'].sum(-1).sum(-1) > 0)
            delta_negative_trajectory = diff_traj(targets['negative_trajectory'])
            negative_loss = ((predictions['diff_trajectory']-delta_negative_trajectory).abs().mean(-1).mean(-1))[negative_index].mean()
            negative_loss = -negative_loss
        else:
            negative_loss = 0.0

        loss = self._config.meanflow_loss_weight * meanflow_loss \
               + self._config.arm_loss_weight * arm_loss \
               + self._config.bev_semantic_loss_weight * bev_semantic_loss
        
        results.update({
            "loss": loss,
            "meanflow_loss": self._config.meanflow_loss_weight * meanflow_loss,
            "arm_loss": self._config.arm_loss_weight * arm_loss,
            "bev_semantic_loss": self._config.bev_semantic_loss_weight * bev_semantic_loss,
        })

        return results

    def forward(self, features) -> Dict[str, torch.Tensor]:
        results = {}
        
        # Context Encoder
        encoder_output = self.encoder(features)
        results['encoder_output'] = encoder_output

        # Multi-modal Sample
        trajectorys = self._meanflow_head.sample(encoder_output)
        results.update(trajectorys)

        arm_model_output = self._arm_model_head(trajectorys, encoder_output)
        results.update(arm_model_output)
        
        return results
