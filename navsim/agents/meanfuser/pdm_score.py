import os
import torch
from pathlib import Path
import lzma
import pickle
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.evaluate.pdm_score import pdm_score, pdm_score_full_v1
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer


class ComputePDMScore():
    def __init__(self, cache_path):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        hydra.initialize(config_path="../../planning/script/config/pdm_scoring")
        FILTER = "default_run_pdm_score"
        NAVSIM_CACHE_ROOT = os.environ.get("NAVSIM_CACHE_ROOT")
        metric_cache_path = f"{NAVSIM_CACHE_ROOT}/{cache_path}"
        overrides = [
            f"metric_cache_path={metric_cache_path}",
        ]
        pdm_score_cfg = hydra.compose(config_name=FILTER, overrides=overrides)
        self.metric_cache_loader = MetricCacheLoader(Path(pdm_score_cfg.metric_cache_path))
        self.simulator: PDMSimulator = instantiate(pdm_score_cfg.simulator)
        self.scorer: PDMScorer = instantiate(pdm_score_cfg.scorer)

    def get_pdm_scores(self, trajectorys: Trajectory, tokens: str):
        device = trajectorys.device
        trajectorys = trajectorys.cpu().numpy()
        pdm_scores = {}
        for trajectory, token in zip(trajectorys, tokens):
            metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            if trajectory.shape[0] == 40:
                interval_length = 0.1
            else:
                interval_length = 0.5
            trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=interval_length)
            trajectory = Trajectory(trajectory, trajectory_sampling)
            pdm_result = pdm_score(
                            metric_cache=metric_cache,
                            model_trajectory=trajectory,
                            future_sampling=self.simulator.proposal_sampling,
                            simulator=self.simulator,
                            scorer=self.scorer,
                        ).__dict__
            for k in pdm_result.keys():
                if k not in pdm_scores:
                    pdm_scores[k] = []
                pdm_scores[k].append(pdm_result[k])
        for k in pdm_scores.keys():
            pdm_scores[k] = (torch.tensor(pdm_scores[k]).mean() * 100).to(device)
        return pdm_scores
    
    def get_pdm_scores_batch(self, trajectorys: Trajectory, tokens: str):
        device, dtype = trajectorys.device, trajectorys.dtype
        trajectorys = trajectorys.cpu().numpy()
        
        pdm_scores = []
        for trajectory, token in zip(trajectorys, tokens):
            metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            score_sample = []
            score_sample = pdm_score_full_v1(
                            metric_cache=metric_cache,
                            vocab_trajectories=trajectory,
                            future_sampling=self.simulator.proposal_sampling,
                            simulator=self.simulator,
                            scorer=self.scorer,
                        )
            pdm_scores.append(score_sample)
        
        pdm_scores = torch.tensor(pdm_scores, device=device, dtype=dtype)
        return pdm_scores
