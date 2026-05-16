
export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/MeanFuser"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/MeanFuser/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles


split=navhard_two_stage
checkpoint_path=exp/meanfuser_checkpoints/meanfuser_pdms_89.0.ckpt
output_dir=exp/meanfuser_checkpoints/meanfuser_pdms_89.0_eval_NAVSIMv2_navhard/


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_v2.py \
train_test_split=${split} \
agent=meanfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=${checkpoint_path} \
experiment_name=transfuser_agent \
metric_cache_path=${NAVSIM_CACHE_ROOT}/${split}_v2_metric_cache \
output_dir=${output_dir} \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
