#!/bin/bash
# Single task evaluation script for Motus policy on RoboTwin platform
# impl and exp of tts and opd

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================


#tts add
TASK_NAME="${1:-click_alarmclock}"
if [ $# -gt 0 ]; then
    shift
fi

# ttsv2 add
GPU_ID=0

# Test-time scaling
TTS_ENABLE=False
TTS_NUM_SAMPLES=8

# Selection method:
#   global_medoid:          average-L2/global-medoid selection
#   keystone:               KeyStone-style guard + kmeans + largest-cluster medoid
#   rank_softmax:           rank-based stochastic selection, P(i)=softmax(-rank_i/tau)
#   cluster_rank_softmax:   guard + kmeans largest-cluster + local rank-softmax
TTS_METHOD="global_medoid"

# KeyStone defaults
TTS_NUM_CLUSTERS=2

# tau meaning:
#   keystone:     unimodality guard threshold, default 0.3
#   rank_softmax: rank-softmax temperature, recommended 1.0
TTS_TAU=0.3

# Rank-softmax temperature for cluster_rank_softmax.
# Keep this separate from TTS_TAU, which is used as the unimodality guard.
TTS_RANK_TAU=1.0

TTS_KMEANS_ITERS=10

# Logging
TTS_LOG_ACTIONS=True

# Deprecated/no-op in the new Python implementation.
# 保留这个参数只是为了兼容旧命令；新版本不再保存 .npz。
TTS_SAVE_FULL_ACTIONS=False
# ttsv2 add end

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --tts)
            TTS_ENABLE=True
            shift
            ;;
        --no-tts)
            TTS_ENABLE=False
            shift
            ;;
        --tts-num-samples)
            TTS_NUM_SAMPLES="$2"
            shift 2
            ;;
        --tts-method)
            TTS_METHOD="$2"
            shift 2
            ;;
        --tts-num-clusters)
            TTS_NUM_CLUSTERS="$2"
            shift 2
            ;;
        --tts-tau)
            TTS_TAU="$2"
            shift 2
            ;;
        --tts-rank-tau)
            TTS_RANK_TAU="$2"
            shift 2
            ;;        
        --tts-kmeans-iters)
            TTS_KMEANS_ITERS="$2"
            shift 2
            ;;            
        --tts-log-actions)
            TTS_LOG_ACTIONS="$2"
            shift 2
            ;;
        --tts-save-full-actions)
            TTS_SAVE_FULL_ACTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# GPU_ID=0                       # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting single task evaluation at $(date)"

# Get script directory (policy/Motus/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create paths_config.yml with required paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (improved - remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# 处理conda报错
#######

# # Activate conda
# if ! command -v conda &> /dev/null; then
#     echo "Error: conda not found."
#     exit 1
# fi

# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV"

# Activate conda
# export PATH="/home/sumita-mana/anaconda3/bin:$PATH"
# source /home/sumita-mana/anaconda3/etc/profile.d/conda.sh

export PATH="/home/ubuntu/miniconda3/bin:$PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
fi

conda activate "$CONDA_ENV"

#######

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${POLICY_DIR}/logs_single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Single Task Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
# echo "TTS Enable:        $TTS_ENABLE" #tts add
# echo "TTS Num Samples:   $TTS_NUM_SAMPLES" #tts add
# echo "TTS Log Actions:   $TTS_LOG_ACTIONS" #tts add
# echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS" #tts add
echo "TTS Enable:        $TTS_ENABLE"
echo "TTS Num Samples:   $TTS_NUM_SAMPLES"
echo "TTS Method:        $TTS_METHOD"
echo "TTS Num Clusters:  $TTS_NUM_CLUSTERS"
echo "TTS Tau:           $TTS_TAU"
echo "TTS Rank Tau:      $TTS_RANK_TAU"
echo "TTS KMeans Iters:  $TTS_KMEANS_ITERS"
echo "TTS Log Actions:   $TTS_LOG_ACTIONS"
echo "TTS Save Full:     $TTS_SAVE_FULL_ACTIONS (deprecated; npz disabled)"
echo "================================================================"
echo ""

# Run evaluation with WAN_PATH passed as argument
echo "Starting evaluation..."

# 处理环境进入问题
#############

echo "Python executable: ${CONDA_ENV}/bin/python"
"${CONDA_ENV}/bin/python" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import sapien
print("sapien ok =", sapien.__file__)
import importlib
m = importlib.import_module("sapien.core")
print("sapien.core ok =", m)
PY

PYTHONWARNINGS=ignore::UserWarning \
"${CONDA_ENV}/bin/python" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --tts_enable "${TTS_ENABLE}" \
    --tts_num_samples "${TTS_NUM_SAMPLES}" \
    --tts_method "${TTS_METHOD}" \
    --tts_num_clusters "${TTS_NUM_CLUSTERS}" \
    --tts_tau "${TTS_TAU}" \
    --tts_rank_tau "${TTS_RANK_TAU}" \
    --tts_kmeans_iters "${TTS_KMEANS_ITERS}" \
    --tts_log_actions "${TTS_LOG_ACTIONS}" \
    --tts_save_full_actions "${TTS_SAVE_FULL_ACTIONS}" \
    2>&1 | tee "$log_file"

#############

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✅ Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "❌ Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi