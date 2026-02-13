#!/bin/bash
# SimbaV2 DMC Hard Sweep - Offline wandb mode

# Set up NVIDIA library paths for JAX CUDA
NVIDIA_LIB_DIR="/home/rs/Documents/luc/SimbaV2/.venv/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR/cusparse/lib:$NVIDIA_LIB_DIR/cublas/lib:$NVIDIA_LIB_DIR/cudnn/lib:$NVIDIA_LIB_DIR/cusolver/lib:$NVIDIA_LIB_DIR/cufft/lib:$NVIDIA_LIB_DIR/cuda_runtime/lib:$NVIDIA_LIB_DIR/nvjitlink/lib:$NVIDIA_LIB_DIR/nccl/lib:$LD_LIBRARY_PATH"

# MuJoCo rendering settings
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"

# JAX memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Wandb offline mode - saves locally, sync later with: wandb sync wandb/offline-run-*
export WANDB_MODE=offline

# DMC Hard tasks (7 tasks):
#   humanoid-stand, humanoid-walk, humanoid-run
#   dog-stand, dog-walk, dog-run, dog-trot
#
# Scaled settings (8x vectorized envs):
#   num_train_envs: 8          (8x parallel data collection)
#   buffer.max_length: 8M      (8x buffer to maintain staleness)
#   updates_per_interaction_step: 16  (8x updates to maintain UTD=2.0)

python run_parallel.py \
    --env_type dmc_hard \
    --device_ids 0 \
    --num_seeds 3 \
    --num_exp_per_device 7 \
    --group_name "dmc_hard_sweep" \
    --exp_name "simbaV2" \
    --overrides "+wandb_mode=offline" \
    --overrides "env.num_train_envs=8" \
    --overrides "buffer.max_length=8_000_000" \
    --overrides "updates_per_interaction_step=16" \
    --overrides "evaluation_per_interaction_step=6250" \
    --overrides "logging_per_interaction_step=1250" \
    --overrides "metrics_per_interaction_step=6250" \
    "$@"

echo ""
echo "=========================================="
echo "When complete, sync to wandb with:"
echo "  wandb sync wandb/offline-run-*"
echo "=========================================="
