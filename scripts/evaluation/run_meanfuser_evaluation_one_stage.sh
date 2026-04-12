
export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/MeanFuser"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/MeanFuser/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

split=navtest
traffic_agents=non_reactive

checkpoint_path=/high_perf_store4/evad-tech-vla/wangjunli1/code/MeanFuser_github/exp/meanfuser/reproduction_2/2026.03.08.20.41.12/meanfuser_pdms_89.0.ckpt
output_dir=exp/meanfuser_checkpoints/meanfuser_pdms_89.0_eval/

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py \
    agent=meanfuser_agent \
    traffic_agents=${traffic_agents} \
    agent.checkpoint_path=${checkpoint_path} \
    agent.config.num_proposals=8 \
    agent.config.noise_type=multi_gaussian \
    output_dir=${output_dir} \
    metric_cache_path=${NAVSIM_CACHE_ROOT}/${split}_v2_metric_cache \
    train_test_split=${split}
