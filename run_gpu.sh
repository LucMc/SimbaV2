#!/bin/bash
# SimbaV2 GPU Training Script

# Set up NVIDIA library paths for JAX CUDA
NVIDIA_LIB_DIR="/home/rs/Documents/luc/SimbaV2/.venv/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR/cusparse/lib:$NVIDIA_LIB_DIR/cublas/lib:$NVIDIA_LIB_DIR/cudnn/lib:$NVIDIA_LIB_DIR/cusolver/lib:$NVIDIA_LIB_DIR/cufft/lib:$NVIDIA_LIB_DIR/cuda_runtime/lib:$NVIDIA_LIB_DIR/nvjitlink/lib:$NVIDIA_LIB_DIR/nccl/lib:$LD_LIBRARY_PATH"

# MuJoCo rendering settings
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"

# JAX memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Build overrides array
OVERRIDES=("+wandb_mode=offline" "entity_name=null")
for arg in "$@"; do
    OVERRIDES+=("$arg")
done

# Convert to --overrides format
OVERRIDE_ARGS=""
for override in "${OVERRIDES[@]}"; do
    OVERRIDE_ARGS="$OVERRIDE_ARGS --overrides $override"
done

# Run training
# Examples:
#   ./run_gpu.sh                                        # Default DMC acrobot-swingup
#   ./run_gpu.sh env.env_name=cheetah-run               # DMC cheetah-run
#   ./run_gpu.sh env=mujoco env.env_name=Humanoid-v4    # MuJoCo Humanoid
#   ./run_gpu.sh env=dmc env.env_name=humanoid-walk     # DMC Hard task

python run_online.py $OVERRIDE_ARGS
