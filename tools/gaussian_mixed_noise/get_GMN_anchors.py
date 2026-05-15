import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUM_THREADS"] = "128"
os.environ['OPENBLAS_NUM_THREADS'] = '1'

NAVSIM_WORKSPACE = os.environ.get('NAVSIM_WORKSPACE', None)
os.environ['NAVSIM_CACHE_ROOT'] = f"{NAVSIM_WORKSPACE}/cache"
os.environ['OPENSCENE_DATA_ROOT'] = f"{NAVSIM_WORKSPACE}/dataset"
os.environ['NUPLAN_MAPS_ROOT'] = f"{NAVSIM_WORKSPACE}/dataset/maps"

import pickle
import numpy as np
from tqdm import tqdm
import copy
import torch
import hydra
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
from sklearn.cluster import KMeans
from navsim.agents.mean_flow.utils import cumsum_traj, diff_traj
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.planning.training.dataset import CacheOnlyDataset
from torch.utils.data import DataLoader


def kmean_trajs(trajs, target_num=5):
    fit_trajs = trajs.numpy()

    n_clusters = target_num
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=10,
        max_iter=10000,
        tol=1e-8,
        n_init=n_clusters,
        )

    kmeans_result = kmeans.fit(fit_trajs.reshape(fit_trajs.shape[0], -1))
    ciluster_idx = kmeans_result.labels_
    cluster_trajs = kmeans_result.cluster_centers_.reshape(n_clusters, -1, 3)  # n*8*2
    return cluster_trajs, ciluster_idx


def plot_2d_clustering(trajectories, cluster_trajs, ciluster_idx, title=""):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for i in tqdm(range(cluster_trajs.shape[0])):
        traj = trajectories[ciluster_idx == i]
        traj = traj.reshape(-1, traj.shape[-1])
        ax.scatter(traj[:, 0], traj[:, 1], s=1)
        ax.plot(cluster_trajs[i, :, 0], cluster_trajs[i, :, 1])

    ax.set_title(title)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.savefig(f"tools/gaussian_mixed_noise/clustering_result_{title}.png")
    print(f"save clustering result to tools/gaussian_mixed_noise/clustering_result_{title}.png")
    plt.close()


if __name__ == "__main__":

    split='navtrain'  # dataset
    target_num = 8  # number of GMN components

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../navsim/planning/script/config/training", version_base=None)

    CONFIG_NAME = "default_training"
    NAVSIM_CACHE_ROOT = os.environ.get("NAVSIM_CACHE_ROOT")
    
    overrides = [
        f"train_test_split={split}",
        "agent=transfuser_agent",
        f"cache_path={NAVSIM_CACHE_ROOT}/trainval_v1_cache"
    ]
    pdm_score_cfg = hydra.compose(config_name=CONFIG_NAME, overrides=overrides)
    agent = instantiate(pdm_score_cfg.agent)

    val_data = CacheOnlyDataset(
            cache_path=pdm_score_cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=pdm_score_cfg.train_logs,
            is_training=True,
        )
    print(f"Number of training samples: {len(val_data)}")
    pdm_score_cfg.dataloader.params['batch_size'] = 64
    val_dataloader = DataLoader(val_data, **pdm_score_cfg.dataloader.params)

    all_scenes_trajectory = []
    for batch in tqdm(val_dataloader, desc="Loading validation data"):
        inputs, targets = batch
        all_scenes_trajectory.append(batch[1]['trajectory'])
    all_scenes_trajectory = torch.cat(all_scenes_trajectory, dim=0)
    print(f"Number of trajectories: {all_scenes_trajectory.shape}")

    # K-means cluster
    cluster_trajs, cluster_idx = kmean_trajs(all_scenes_trajectory, target_num=target_num)

    # Plot cluster
    plot_2d_clustering(all_scenes_trajectory, cluster_trajs, cluster_idx, title=f"{split}_{target_num}_trajectorys")

    center_points = []
    center_std = []
    
    # Process the trajectories
    norm_trajs = diff_traj(all_scenes_trajectory).numpy()  # bs*8*4
    norm_cluster_trajs = diff_traj(torch.from_numpy(cluster_trajs))  # bs*8*4
    for i in range(norm_cluster_trajs.shape[0]):
        all_trajs = copy.deepcopy(norm_trajs)
        all_trajs[cluster_idx!=i] *= 0
        center_points.append((all_trajs[cluster_idx==i].mean(axis=0)).mean(axis=0))
        center_std.append((all_trajs[cluster_idx==i]).std(axis=0).mean(axis=0))

    # Save cluster mean and std
    center_points = torch.from_numpy(np.array(center_points)).float() * 1
    center_std = torch.from_numpy(np.array(center_std)).float() * 0 + 0.1
    mean_std_dict = {
            'cluster_trajs': torch.from_numpy(cluster_trajs).float(), 
            'center_points': center_points, 
            'center_std': center_std
            }
    with open(f"tools/gaussian_mixed_noise/{split}_{target_num}_mean_std.pkl", 'wb') as f:
        pickle.dump(mean_std_dict, f)
    print(f"save cluster_anchors to tools/gaussian_mixed_noise/{split}_{target_num}_mean_std.pkl")
