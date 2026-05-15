export HYDRA_FULL_ERROR=1
# export OPENBLAS_CORETYPE=HASWELL  # If using NVIDIA H20 GPU, uncomment this line

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/MeanFuser"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/MeanFuser/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

# Please use LF in linux/mac and CRLF in windows for this script.

TRAIN_TEST_SPLIT=navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=${NAVSIM_CACHE_ROOT}/${TRAIN_TEST_SPLIT}_v1_metric_cache