#!/bin/bash

# Create rerank dataset with per-positive random negative sampling
# and optional difficulty evaluation via OpenAI-compatible API.

set -e
set -x

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# Configuration for data inputs
# ------------------------------------------------------------------------------
TRAIN_PICKLE_FILE="dataset/train_rank_v2_qwen_new_hard_neg_max.pkl"
TEST_PICKLE_FILE="dataset/test_rank_v2_qwen_new_hard_neg_max.pkl"
RANK_RESUME_FILE="dataset/data/processed_seed17/rank_resume_filter_train.json"
ALL_LABELS_CSV="dataset/data/intermediate/all_labels.csv"
JD_FILE="dataset/data/processed/all_jd_texts.csv"
RESUME_FILE="dataset/data/processed/all_resume_texts.csv"

# ------------------------------------------------------------------------------
# Output files
# ------------------------------------------------------------------------------
TRAIN_OUTPUT_FILE="dataset/difficulty/rerank_difficulty_train_dataset.parquet"
TEST_OUTPUT_FILE="dataset/difficulty/rerank_difficulty_test_dataset.parquet"

# ------------------------------------------------------------------------------
# Sampling parameters
# ------------------------------------------------------------------------------
SAMPLES_PER_POSITIVE=3
MAX_POSITIVE_COUNT=11
SEED=42

# ------------------------------------------------------------------------------
# Subset controls (optional, leave empty to process all)
# ------------------------------------------------------------------------------
MAX_TRAIN_JOBS=""
MAX_TEST_JOBS=""

# ------------------------------------------------------------------------------
# Difficulty evaluation settings
# Set RUN_DIFFICULTY=true to enable model-based accuracy annotation
# ------------------------------------------------------------------------------
RUN_DIFFICULTY=false

# Optional difficulty filter (inclusive), e.g. keep samples with 0.4 <= acc <= 1.0
MIN_ACC=""
MAX_ACC=""

# Qwen3-8B thinking mode switch
# Thinking mode:     temperature=0.6, top_p=0.95, top_k=20, min_p=0
# Non-thinking mode: temperature=0.7, top_p=0.8,  top_k=20, min_p=0
ENABLE_THINKING=false

API_BASE="http://localhost:30001/v1"
MODEL_NAME="Qwen/Qwen3-8B"
NUM_RUNS=5
MAX_WORKERS=64
TOP_K=20
MIN_P=0
MAX_RETRIES=3
RETRY_SLEEP=1.0
API_KEY="token-abc123"
ORGANIZATION=""
PROJECT=""

# Ensure output directories exist
mkdir -p "$(dirname "$TRAIN_OUTPUT_FILE")"
mkdir -p "$(dirname "$TEST_OUTPUT_FILE")"

echo "=============================================="
echo "Creating Rerank Difficulty Dataset"
echo "=============================================="
echo "Train pickle:          $TRAIN_PICKLE_FILE"
echo "Test pickle:           $TEST_PICKLE_FILE"
echo "Labels CSV:            $ALL_LABELS_CSV"
echo "Samples per positive:  $SAMPLES_PER_POSITIVE"
echo "Max positive count:    $MAX_POSITIVE_COUNT"
echo "Run difficulty:        $RUN_DIFFICULTY"
if [ "$RUN_DIFFICULTY" = true ]; then
  echo "Enable thinking:       $ENABLE_THINKING"
  echo "Model:                 $MODEL_NAME"
  echo "API base:              $API_BASE"
  echo "Num runs:              $NUM_RUNS"
  echo "Max workers:           $MAX_WORKERS"
  if [ -n "$MIN_ACC" ] || [ -n "$MAX_ACC" ]; then
    echo "Acc filter:            [$MIN_ACC, $MAX_ACC]"
  fi
fi
echo "Train output:          $TRAIN_OUTPUT_FILE"
echo "Test output:           $TEST_OUTPUT_FILE"
echo ""

# ------------------------------------------------------------------------------
# Optional args
# ------------------------------------------------------------------------------
subset_args=()
if [ -n "$MAX_TRAIN_JOBS" ]; then
  subset_args+=(--max_train_jobs "$MAX_TRAIN_JOBS")
fi
if [ -n "$MAX_TEST_JOBS" ]; then
  subset_args+=(--max_test_jobs "$MAX_TEST_JOBS")
fi

difficulty_args=()
if [ "$RUN_DIFFICULTY" = true ]; then
  difficulty_args+=(
    --run_difficulty
    --model_name "$MODEL_NAME"
    --num_runs "$NUM_RUNS"
    --max_workers "$MAX_WORKERS"
    --top_k "$TOP_K"
    --min_p "$MIN_P"
    --max_retries "$MAX_RETRIES"
    --retry_sleep "$RETRY_SLEEP"
  )

  if [ -n "$MIN_ACC" ]; then
    difficulty_args+=(--min_acc "$MIN_ACC")
  fi
  if [ -n "$MAX_ACC" ]; then
    difficulty_args+=(--max_acc "$MAX_ACC")
  fi
  if [ "$ENABLE_THINKING" = false ]; then
    difficulty_args+=(--disable_thinking)
  fi
  if [ -n "$API_BASE" ]; then
    difficulty_args+=(--api_base "$API_BASE")
  fi
  if [ -n "$API_KEY" ]; then
    difficulty_args+=(--api_key "$API_KEY")
  fi
  if [ -n "$ORGANIZATION" ]; then
    difficulty_args+=(--organization "$ORGANIZATION")
  fi
  if [ -n "$PROJECT" ]; then
    difficulty_args+=(--project "$PROJECT")
  fi
fi

# ------------------------------------------------------------------------------
# Main command
# ------------------------------------------------------------------------------
python3 -m data_process.create_rerank_difficulty_dataset \
  --train_pickle_file "$TRAIN_PICKLE_FILE" \
  --test_pickle_file "$TEST_PICKLE_FILE" \
  --rank_resume_file "$RANK_RESUME_FILE" \
  --all_labels_csv "$ALL_LABELS_CSV" \
  --jd_file "$JD_FILE" \
  --resume_file "$RESUME_FILE" \
  --train_output_file "$TRAIN_OUTPUT_FILE" \
  --test_output_file "$TEST_OUTPUT_FILE" \
  --samples_per_positive "$SAMPLES_PER_POSITIVE" \
  --max_positive_count "$MAX_POSITIVE_COUNT" \
  --seed "$SEED" \
  "${subset_args[@]}" \
  "${difficulty_args[@]}"

echo ""
echo "=============================================="
echo "Dataset creation completed!"
echo "=============================================="
echo "Train dataset: $TRAIN_OUTPUT_FILE"
echo "Test dataset:  $TEST_OUTPUT_FILE"
