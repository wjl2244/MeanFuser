export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"  # Replace with your actual workspace path
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/MeanFuser"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/MeanFuser/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

# Please use LF in linux/mac and CRLF in windows for this script.

experiment_name=experiment_name  # replace with your experiment name
experiment_uid=$(date +%Y.%m.%d.%H.%M.%S)
output_dir=exp/meanfuser_training/${experiment_name}/${experiment_uid}

lr=2e-4
batch_size=4  # 4 for training on 8 GPU
max_epochs=100

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name=default_training \
    trainer.params.num_nodes=1 \
    agent=meanfuser_agent \
    experiment_name=${experiment_name} \
    experiment_uid=${experiment_uid} \
    train_test_split=navtrain \
    dataloader.params.batch_size=${batch_size} \
    trainer.params.max_epochs=${max_epochs} \
    agent.config.ckpt_path=${experiment_name} \
    agent.config.num_proposals=8 \
    agent.config.noise_type=multi_gaussian \
    agent.lr=${lr} \
    cache_path=${NAVSIM_CACHE_ROOT}/trainval_v1_cache \
    use_cache_without_dataset=True  \
    force_cache_computation=False \
    output_dir=$output_dir \