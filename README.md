# ConFit V3 Reranking

This repository contains the data processing, reranking data construction, difficulty calculation, and VERL-based RL training pipeline for ConFit V3 listwise/rerank experiments. The main workflow is:

1. Use an embedding model to produce initial resume rankings for each job.
2. Convert ranked candidates into listwise VERL training data.
3. Build a rerank difficulty dataset with model-estimated `acc`.
4. Optionally filter samples by difficulty, for example `0.4 <= acc <= 1.0`.
5. Train the reranking model with the generated parquet files.

## Environment Prepare

Install dependencies first:

```bash
chmod +x install.sh
./install.sh
```

Before running scripts, make sure the repository root is visible to Python:

```bash
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=true
```

If you use wandb in training scripts, log in before training:

```bash
wandb login
```

## Data Layout

The scripts assume that job text, resume text, label files, and embedding ranking outputs are already prepared. The commonly used inputs are:

- Job text CSV: `dataset/confit_v3_listwise/job_merged_text.csv`
- Resume text CSV: `dataset/confit_v3_listwise/resume_merged_text.csv`
- Rank-label JSON: `dataset/data/processed_seed17/rank_resume_filter_train.json`
- Train ranking pickle: `dataset/confit_v3_listwise/train_rank.pkl`
- Test ranking pickle: `dataset/confit_v3_listwise/test_rank.pkl`

The ranking pickle should be calculated following the following rank extraction section, the format is like `[{job_id1:{resume_id1:rank_1, resume_id2:rank2,...}, job_id2:...}.` Then, use these ranking files to generate training dataset for RL training. 

For rerank RL training, the final parquet files should contain the VERL-style fields:

- `prompt`: chat messages used as model input
- `reward_model.ground_truth`: accepted candidate labels
- `extra_info.valid_labels`: all candidate labels in the sample
- `extra_info.resume_ids`: resume ids in displayed order
- `extra_info.acc`: model-estimated difficulty accuracy when difficulty calculation is enabled

## Embedding-Based Rank Extraction

Run embedding ranking to produce initial candidate rankings for each job. `--model_path` is optional and can be used when you want to evaluate a customized checkpoint rather than the base encoder.

The embedding model and ranking setup can refer to the ConFit v2 implementation: [jasonyux/ConFit-v2](https://github.com/jasonyux/ConFit-v2).

```bash
python embedding_rank_cal.py \
  --data_dir dataset/confit_v3_listwise \
  --id_dir /path/to/ranking_data \
  --output_dir /path/to/ranking_data \
  --pretrained_encoder Qwen/Qwen3-Embedding-0.6B \
  --model_path /path/to/checkpoint.ckpt
```

The ranking outputs are later used by both `listwise_data_prepare.py` and the rerank difficulty data builder.

## VERL Data Preparation Based On Rank

`listwise_data_prepare.py` converts embedding rankings into listwise training samples. For every job, it takes the top ranked resumes, samples negatives around positives, formats prompts, and writes train/test parquet files.

```bash
python listwise_data_prepare.py \
  --data-dir dataset/confit_v3_aliyun_data/ranking \
  --output-dir dataset/confit_v3_aliyun_data/ranking/verl_dataset \
  --splits train test \
  --top-k 20 \
  --num-negatives 3 \
  --seed 42
```

Important arguments:

- `--top-k`: number of top ranked resumes considered per job.
- `--num-negatives`: number of negatives sampled for each positive resume.
- `--splits`: choose `train`, `test`, or both.
- `--analyze-lengths`: optionally analyze prompt token length distribution.

## Rerank Difficulty Calculation

The rerank difficulty builder is implemented in `data_process/create_rerank_difficulty_dataset.py` and can be launched with:

```bash
chmod +x scripts/run_create_rerank_difficulty_dataset.sh
bash scripts/run_create_rerank_difficulty_dataset.sh
```

This script builds train/test parquet files that are directly usable by the rerank RL training script. Each sample contains one positive resume and three sampled negative resumes, with the candidate order shuffled to avoid position bias.

### Output Files

By default, the script writes:

```text
dataset/difficulty/rerank_difficulty_train_dataset.parquet
dataset/difficulty/rerank_difficulty_test_dataset.parquet
```

It also writes metadata JSON files next to the parquet files. When difficulty filtering is enabled, the metadata records the original sample count, filtered sample count, and the selected `acc` range.

### Difficulty Acc

Set `RUN_DIFFICULTY=true` in `scripts/run_create_rerank_difficulty_dataset.sh` to call an OpenAI-compatible endpoint and annotate each sample with `extra_info.acc`.

The current `acc` definition is top-1 positive accuracy:

1. The evaluator model ranks the four displayed resumes.
2. The first predicted label is used as the top-1 candidate.
3. If the top-1 label is one of the accepted labels, that run is correct.
4. `acc = correct_count / NUM_RUNS`.

### Difficulty Filter

You can filter the final output by `acc` range. For example, to keep all samples with `0.4 <= acc <= 1.0`, set:

```bash
RUN_DIFFICULTY=true
MIN_ACC=0.4
MAX_ACC=1.0
```

The interval is inclusive. If `MIN_ACC` and `MAX_ACC` are empty, no difficulty filtering is applied.

### Key Script Parameters

In `scripts/run_create_rerank_difficulty_dataset.sh`, the most important settings are:

- `TRAIN_PICKLE_FILE` / `TEST_PICKLE_FILE`: embedding ranking pickle files.
- `RANK_RESUME_FILE` and `ALL_LABELS_CSV`: files used to load ground-truth positive resumes.
- `SAMPLES_PER_POSITIVE`: number of random negative batches sampled for each positive.
- `MAX_POSITIVE_COUNT`: skip jobs with too many positives in top-20.
- `RUN_DIFFICULTY`: whether to call a model for `acc` calculation.
- `MODEL_NAME`, `API_BASE`, `API_KEY`: OpenAI-compatible evaluator endpoint configuration.
- `NUM_RUNS`: repeated evaluator runs per sample.
- `MAX_WORKERS`: concurrent evaluator requests.
- `ENABLE_THINKING`: Qwen3 thinking-mode switch.
- `MIN_ACC` / `MAX_ACC`: optional inclusive difficulty filter.

## Reranking Model Training

The RL training entrypoint is:

```bash
bash scripts/train_rearank_rl.sh
```

By default, `scripts/train_rearank_rl.sh` points to the generated rerank difficulty dataset:

```bash
train_dset=$proj_dir/dataset/difficulty/rerank_difficulty_train_dataset.parquet
test_dset=$proj_dir/dataset/difficulty/rerank_difficulty_test_dataset.parquet
```

Before training, set:

- `proj_dir`: absolute path to this repository.
- `CUDA_VISIBLE_DEVICES`: GPUs used by training.
- `N_GPUS`: number of GPUs used by VERL.
- `model_name`: base model or checkpoint path.
- `train_bsize`, `micro_train_bsize`, `per_device_micro_batch_size`: batch size settings.
- `experiment_name`: wandb/checkpoint experiment name.

The reward function is configured as:

```bash
reward_fpath=$proj_dir/confit_v3/trainer/reward_fns_listwise_reason.py
reward_fn_name=my_reward_fn_rearank
```

`my_reward_fn_rearank` uses the listwise ranking output, compares it with `reward_model.ground_truth`, and also reads `extra_info.acc` for hard-sample metrics such as `is_hard` and `hard_acc`.

## Reranking Evaluation / Inference

### Local Hosted Model

For a local OpenAI-compatible endpoint, for example vLLM:

```bash
export VLLM_API_KEY=EMPTY

python -m reranking_clean.parallel_reranking_response_length \
  --base-url http://localhost:8004/v1 \
  --model Qwen/Qwen3-8B \
  --jd-csv dataset/confit_v3_listwise/job_merged_test.csv \
  --resume-csv dataset/confit_v3_listwise/resume_merged_test.csv \
  --rank-resume-json dataset/confit_v3_listwise/rank_resume.json \
  --labels-json dataset/confit_v3_listwise/rank_resume.json \
  --init-ranking-pkl dataset/confit_v3_listwise/test_ranking.pkl \
  --window-size 4 \
  --stride 2 \
  --num-passes 1 \
  --num-workers 8 \
  --temperature 0.6 \
  --max-tokens 8192 \
  --timeout 120 \
  --output-dir ./reranking_clean_outputs/qwen_local
```

### GPT / Claude

For hosted models, set the corresponding API key and base URL:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key

python -m reranking_clean.parallel_reranking_universal \
  --base-url https://api.openai.com/v1 \
  --model gpt-5-mini \
  --jd-csv dataset/confit_v3_listwise/job_merged_test.csv \
  --resume-csv dataset/confit_v3_listwise/resume_merged_test.csv \
  --rank-resume-json dataset/confit_v3_listwise/rank_resume.json \
  --labels-json dataset/confit_v3_listwise/rank_resume.json \
  --init-ranking-pkl dataset/confit_v3_listwise/test_ranking.pkl \
  --window-size 4 \
  --stride 2 \
  --num-passes 1 \
  --num-workers 8 \
  --temperature 0.6 \
  --max-tokens 8192 \
  --timeout 120 \
  --output-dir ./reranking_clean_outputs/gpt
```

For Claude-compatible usage, switch `--base-url` and `--model` to the Anthropic-compatible endpoint/model used in your environment.


## Citation
```bash
@misc{yu2026confitv3improvingresumejob,
      title={ConFit v3: Improving Resume-Job Matching with LLM-based Re-Ranking}, 
      author={Xiao Yu and Ruize Xu and Chengyuan Xue and Junyu Chen and Matthew So and Shijun Ma and Bo Liu and Xiangye Liang and Zhou Yu},
      year={2026},
      eprint={2605.09760},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.09760}, 
}
```
