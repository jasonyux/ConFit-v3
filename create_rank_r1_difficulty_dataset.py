#!/usr/bin/env python3
"""
Create Rank-R1 dataset with per-positive random negative sampling.

For each job's top-20 resumes:
  - Identify positives (satisfied==1) and negatives (satisfied==0).
  - Skip jobs with no positives or too many positives (>= max_positive_count).
  - For each positive, randomly sample 3 negatives to form a 4-resume batch.
    Repeat ``samples_per_positive`` times per positive, with deduplication.
  - The positive is randomly shuffled among the 4 positions to avoid position bias.

Optionally (--run_difficulty), query a model to estimate per-sample accuracy.
"""

import argparse
import json
import os
import pickle
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI
from openai import APIError  # type: ignore
from openai import RateLimitError  # type: ignore
from tqdm import tqdm
from transformers import AutoTokenizer


# --------------------------------------------------------------------------------------
# Data loading utilities
# --------------------------------------------------------------------------------------


def load_pickle_data(pickle_file: str) -> Dict[str, Any]:
    """Load a pickle file containing resume rankings per job."""
    print(f"Loading data from: {pickle_file}")
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} jobs")
    return data


def load_job_descriptions(jd_file: str) -> Dict[str, str]:
    """Load job descriptions from CSV file."""
    jd_df = pd.read_csv(jd_file)
    jd_dict = {}
    for _, row in jd_df.iterrows():
        jd_dict[row["jd_no"]] = row["job_text"]
    print(f"Loaded {len(jd_dict)} job descriptions")
    return jd_dict


def load_resume_texts(resume_file: str) -> Dict[str, str]:
    """Load resume texts from CSV file."""
    resume_df = pd.read_csv(resume_file)
    resume_dict = {}
    for _, row in resume_df.iterrows():
        resume_dict[row["user_id"]] = row["resume_text"]
    print(f"Loaded {len(resume_dict)} resume texts")
    return resume_dict


def load_ground_truth_labels(rank_resume_file: str, all_labels_csv: str) -> Dict[str, Dict]:
    """Load ground truth labels."""
    print(f"Loading ground truth labels from: {rank_resume_file} and {all_labels_csv}")
    with open(rank_resume_file, "r") as f:
        json.load(f)  # validate JSON

    labels_df = pd.read_csv(all_labels_csv)
    labels: Dict[str, Dict] = {}
    for _, row in labels_df.iterrows():
        job_id = row["jd_no"]
        user_id = row["user_id"]
        satisfied = row["satisfied"]
        if job_id not in labels:
            labels[job_id] = {"user_ids": [], "satisfied": []}
        labels[job_id]["user_ids"].append(user_id)
        labels[job_id]["satisfied"].append(satisfied)

    print(f"Loaded labels for {len(labels)} jobs")
    return labels


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------


def _truncate_resume_text(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate a single resume text to fit within max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return truncated + "\n... [truncated]"


def create_rank_r1_prompt(
    job_description: str,
    batch_resumes: List[str],
    resume_texts: Dict[str, str],
    *,
    tokenizer=None,
    max_prompt_tokens: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], bool]:
    """Create listwise-style chat prompts for a batch of resumes.

    Returns:
        (chat_messages, was_truncated)
    """
    # Template overhead (instructions + JD placeholder) is roughly ~350 tokens.
    # If max_prompt_tokens is set, budget the rest to JD + 4 resumes.
    resume_text_list = [
        resume_texts.get(rid, f"Resume text for {rid}") for rid in batch_resumes
    ]
    was_truncated = False

    if tokenizer is not None and max_prompt_tokens is not None:
        # Count template + JD tokens
        template_overhead = 400  # conservative estimate for instruction text
        jd_tokens = len(tokenizer.encode(job_description, add_special_tokens=False))
        remaining = max_prompt_tokens - template_overhead - jd_tokens
        if remaining < 0:
            remaining = 2000  # fallback minimum for resumes

        per_resume_budget = remaining // len(batch_resumes)
        new_texts = []
        for text in resume_text_list:
            t_len = len(tokenizer.encode(text, add_special_tokens=False))
            if t_len > per_resume_budget:
                text = _truncate_resume_text(text, tokenizer, per_resume_budget)
                was_truncated = True
            new_texts.append(text)
        resume_text_list = new_texts

    valid_labels = [str(i + 1) for i in range(len(batch_resumes))]
    resumes_text = "\n\n".join(
        [f"[{label}] Resume {label}:\n{text}" for label, text in zip(valid_labels, resume_text_list)]
    )
    answer_format = " > ".join(f"[{label}]" for label in valid_labels)

    system_prompt = (
        "You are an expert technical recruiter that can rank resumes based on their matching degree to the "
        "job description. You first analyze each resume individually, then compare them systematically, and "
        "finally provide the ranking. "
        f"I will provide you with {len(batch_resumes)} resumes, each indicated by a numeric identifier []. "
        f"Rank the {len(batch_resumes)} resumes based on their matching degree to the job description. "
        "The resumes should be listed in descending order using identifiers. The most relevant resumes should "
        "be listed first. "
        f"The output format should be <answer> {answer_format} </answer>."
    )
    user_prompt = (
        f"Resumes:\n{resumes_text}\n\n"
        f"Please rank these resumes according to their matching degree to the JOB DESCRIPTION: [{job_description}]\n"
        "Follow these steps exactly:\n"
        "1. First, think to summarize the job description and analyze EACH resume briefly: evaluate how well it "
        "matches the job description and mandatory criteria.\n"
        "2. Then, think to COMPARE the resumes and determine which candidates are better fits and why.\n"
        "3. Finally, within <answer> tags, provide ONLY the final ranking of the resumes from best to worst fit "
        "using their numerical identifiers.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], was_truncated


# --------------------------------------------------------------------------------------
# New sampling logic
# --------------------------------------------------------------------------------------


def _get_resume_label(resume_id: str, job_labels: Dict) -> int:
    """Return satisfied label for a resume under a job, defaulting to 0."""
    if resume_id in job_labels["user_ids"]:
        idx = job_labels["user_ids"].index(resume_id)
        return job_labels["satisfied"][idx]
    return 0


def process_jobs(
    job_data: Dict[str, Any],
    labels: Dict[str, Dict],
    job_descriptions: Dict[str, str],
    resume_texts: Dict[str, str],
    max_jobs: Optional[int],
    split: str,
    *,
    samples_per_positive: int = 3,
    max_positive_count: int = 11,
    seed: int = 42,
    tokenizer=None,
    max_prompt_tokens: Optional[int] = None,
) -> List[Dict]:
    """
    Build samples using per-positive random negative sampling.

    For each job:
      1. Take the top-20 ranked resumes.
      2. Split into positives and negatives based on ground truth.
      3. Skip if 0 positives or >= max_positive_count positives.
      4. For each positive, randomly pick 3 negatives ``samples_per_positive`` times.
      5. Shuffle the 4-resume order randomly; record ground_truth accordingly.
      6. Deduplicate: no two samples for the same job may share the same 4-resume set.
      7. If max_prompt_tokens is set, truncate individual resumes to fit.
    """
    rng = random.Random(seed)
    samples: List[Dict] = []
    jobs_processed = 0
    skipped_no_positive = 0
    skipped_too_many_positive = 0
    skipped_not_enough_resumes = 0
    dedup_dropped = 0
    truncated_count = 0

    for job_id, job_rankings in job_data.items():
        if max_jobs and jobs_processed >= max_jobs:
            break

        # Filter to resume ranking entries only
        resume_rankings = {k: v for k, v in job_rankings.items() if isinstance(v, int)}
        if len(resume_rankings) < 20:
            skipped_not_enough_resumes += 1
            continue

        sorted_rankings = sorted(resume_rankings.items(), key=lambda x: x[1])
        top_20_resumes = [rid for rid, _rank in sorted_rankings[:20]]

        # Classify into positives / negatives
        job_labels = labels.get(job_id)
        if not job_labels:
            skipped_no_positive += 1
            jobs_processed += 1
            continue

        positives = [r for r in top_20_resumes if _get_resume_label(r, job_labels) == 1]
        negatives = [r for r in top_20_resumes if _get_resume_label(r, job_labels) != 1]

        if len(positives) == 0:
            skipped_no_positive += 1
            jobs_processed += 1
            continue

        if len(positives) >= max_positive_count:
            skipped_too_many_positive += 1
            jobs_processed += 1
            continue

        if len(negatives) < 3:
            # Not enough negatives to form a valid batch
            skipped_not_enough_resumes += 1
            jobs_processed += 1
            continue

        job_description = job_descriptions.get(job_id, f"Job description for {job_id}")

        # Dedup set for this job: frozenset of 4 resume IDs
        seen_batches: Set[frozenset] = set()
        batch_counter = 0

        for pos_resume in positives:
            for _ in range(samples_per_positive):
                # Random sample 3 negatives
                neg_sample = rng.sample(negatives, 3)
                batch_set = frozenset([pos_resume] + neg_sample)

                if batch_set in seen_batches:
                    dedup_dropped += 1
                    continue
                seen_batches.add(batch_set)

                # Shuffle positions randomly
                batch_resumes = [pos_resume] + neg_sample
                rng.shuffle(batch_resumes)

                valid_labels = [str(i + 1) for i in range(len(batch_resumes))]
                accepted_labels = [
                    label for r, label in zip(batch_resumes, valid_labels)
                    if _get_resume_label(r, job_labels) == 1
                ]

                rank_r1_prompt, was_truncated = create_rank_r1_prompt(
                    job_description, batch_resumes, resume_texts,
                    tokenizer=tokenizer, max_prompt_tokens=max_prompt_tokens,
                )
                if was_truncated:
                    truncated_count += 1

                sample = {
                    "job_description": job_description,
                    "job_id": job_id,
                    "data_source": "rank_r1_difficulty",
                    "prompt": rank_r1_prompt,
                    "ability": "ranking",
                    "reward_model": {
                        "ground_truth": accepted_labels,
                        "style": "rule",
                    },
                    "extra_info": {
                        "index": len(samples),
                        "job_id": job_id,
                        "split": split,
                        "batch_idx": batch_counter,
                        "resume_ids": batch_resumes,
                        "valid_labels": valid_labels,
                        "accepted_labels": accepted_labels,
                        "positive_resume": pos_resume,
                    },
                }
                samples.append(sample)
                batch_counter += 1

        jobs_processed += 1
        if jobs_processed % 200 == 0:
            print(f"[{split}] Processed {jobs_processed} jobs, {len(samples)} samples so far...")

    print(f"\n[{split}] Summary:")
    print(f"  Jobs processed:            {jobs_processed}")
    print(f"  Samples created:           {len(samples)}")
    print(f"  Skipped (no positive):     {skipped_no_positive}")
    print(f"  Skipped (>= {max_positive_count} positives): {skipped_too_many_positive}")
    print(f"  Skipped (< 20 resumes):    {skipped_not_enough_resumes}")
    print(f"  Dedup dropped:             {dedup_dropped}")
    print(f"  Prompts truncated:         {truncated_count}")
    return samples


# --------------------------------------------------------------------------------------
# Inference / difficulty evaluation utilities
# --------------------------------------------------------------------------------------


class OpenAIChatClient:
    """Client for interacting with OpenAI (or OpenAI-compatible) chat completion endpoint.

    Supports Qwen3 thinking mode via vLLM's ``extra_body`` parameter.
    When ``enable_thinking=True``, vLLM will let the model produce an internal
    reasoning trace (``<think>…</think>``) before the final answer.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        timeout: int,
        max_retries: int,
        request_sleep: float,
        enable_thinking: bool = True,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "token-abc123"

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_sleep = request_sleep
        self.enable_thinking = enable_thinking

        client_kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        if project:
            client_kwargs["project"] = project

        self.client = OpenAI(**client_kwargs)

    def query(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat completion request.

        For Qwen3 on vLLM, ``enable_thinking`` and ``top_k`` are passed via
        ``extra_body`` so that the model produces a reasoning trace.
        """
        extra_body: Dict[str, Any] = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body,
                    timeout=self.timeout,
                )

                choices = getattr(response, "choices", None)
                if not choices:
                    raise ValueError("Empty choices returned by the model.")
                message = choices[0].message
                if not message:
                    raise ValueError("Empty message returned by the model.")

                # vLLM may return thinking in reasoning_content (separate field)
                # or the model may embed <think> in content. Combine both.
                content = getattr(message, "content", "") or ""
                reasoning = getattr(message, "reasoning_content", "") or ""

                # If reasoning_content exists, prepend it for downstream parsing
                full_output = content
                if reasoning:
                    full_output = f"<think>{reasoning}</think>\n{content}"

                if not full_output.strip():
                    raise ValueError("Empty content returned by the model.")
                return full_output
            except (RateLimitError, APIError, ValueError) as exc:
                print(f"Request failed (attempt {attempt}/{self.max_retries}): {exc}")
                if attempt == self.max_retries:
                    raise
                time.sleep(self.request_sleep * attempt)

        raise RuntimeError("Failed to obtain model response after retries.")


def parse_model_ranking(model_output: str, valid_labels: List[str]) -> Optional[List[str]]:
    """Parse ranking labels from model output in a robust way."""
    answer_pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    answer_match = answer_pattern.search(model_output)
    parse_target = answer_match.group(1) if answer_match else model_output

    found_labels = re.findall(r"\[(\d+)\]", parse_target)
    if not found_labels:
        return None

    valid_set = set(valid_labels)
    ranking: List[str] = []
    seen = set()
    for label in found_labels:
        if label in valid_set and label not in seen:
            ranking.append(label)
            seen.add(label)
    return ranking if ranking else None


def evaluate_single_sample(
    sample: Dict,
    client: OpenAIChatClient,
    *,
    num_runs: int,
) -> Dict:
    """Evaluate a single sample with multiple model runs and record accuracy metrics."""
    prompt_messages = sample["prompt"]
    valid_labels = sample.get("extra_info", {}).get("valid_labels", ["1", "2", "3", "4"])
    accepted_labels = [str(label) for label in sample.get("reward_model", {}).get("ground_truth", [])]

    evaluation_runs: List[Dict[str, Any]] = []
    correct_count = 0

    error_types: List[str] = []

    for run_idx in range(num_runs):
        error_type = None
        try:
            response_text = client.query(prompt_messages)
        except Exception as exc:
            response_text = f"<error>{exc}</error>"
            predicted_ranking = None
            predicted_top1 = None
            is_correct = False
            exc_str = str(exc)
            if "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
                error_type = "timeout"
            elif "maximum context length" in exc_str.lower() or "code: 400" in exc_str.lower():
                error_type = "context_overflow"
            else:
                error_type = "other_error"
            error_types.append(error_type)
        else:
            predicted_ranking = parse_model_ranking(response_text, valid_labels)
            predicted_top1 = predicted_ranking[0] if predicted_ranking else None
            is_correct = predicted_top1 is not None and predicted_top1 in accepted_labels
            if is_correct:
                correct_count += 1

        evaluation_runs.append(
            {
                "run": run_idx,
                "response": response_text,
                "predicted_ranking": predicted_ranking,
                "predicted_top1": predicted_top1,
                "is_correct": is_correct,
                "error_type": error_type,
            }
        )

    miss_count = num_runs - correct_count
    error_count = len(error_types)
    accuracy = correct_count / num_runs if num_runs > 0 else 0.0

    augmented_sample = dict(sample)
    augmented_extra_info = dict(sample.get("extra_info", {}))
    augmented_extra_info.update(
        {
            "acc": accuracy,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "num_runs": num_runs,
            "evaluation_runs": evaluation_runs,
            "miss_count": miss_count,
            "error_count": error_count,
            "error_types": error_types,
        }
    )

    augmented_sample["accuracy"] = accuracy
    augmented_sample["extra_info"] = augmented_extra_info
    augmented_sample["model_correct_count"] = correct_count
    augmented_sample["model_miss_count"] = miss_count
    augmented_sample["error_count"] = error_count

    return augmented_sample


def evaluate_samples_with_concurrency(
    samples: List[Dict],
    client: OpenAIChatClient,
    *,
    num_runs: int,
    max_workers: int,
    split: str,
) -> List[Dict]:
    """Run model evaluation concurrently across samples."""
    print(
        f"Evaluating {len(samples)} {split} samples using {max_workers} workers "
        f"and {num_runs} runs per sample..."
    )

    results: List[Optional[Dict]] = [None] * len(samples)

    def worker(idx: int, sample: Dict) -> Tuple[int, Dict]:
        annotated = evaluate_single_sample(sample, client, num_runs=num_runs)
        return idx, annotated

    # Counters for real-time stats
    error_counter: Counter = Counter()
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(worker, idx, sample): idx
            for idx, sample in enumerate(samples)
        }

        pbar = tqdm(
            total=len(samples),
            desc=f"[{split}] Evaluating",
            unit="sample",
            dynamic_ncols=True,
        )
        for future in as_completed(future_to_idx):
            idx, annotated_sample = future.result()
            results[idx] = annotated_sample

            # Track errors in real-time
            sample_errors = annotated_sample.get("extra_info", {}).get("error_types", [])
            if sample_errors:
                for et in sample_errors:
                    error_counter[et] += 1
            else:
                success_count += 1

            # Show live error stats in progress bar
            err_str = " ".join(f"{k}={v}" for k, v in error_counter.items()) if error_counter else ""
            pbar.set_postfix_str(f"ok={success_count} {err_str}".strip())
            pbar.update(1)
        pbar.close()

    # Print error summary
    total_runs = len(samples) * num_runs
    total_errors = sum(error_counter.values())
    print(f"\n[{split}] Evaluation summary:")
    print(f"  Total samples:        {len(samples)}")
    print(f"  Total runs:           {total_runs}")
    print(f"  Total errors:         {total_errors} ({total_errors/total_runs*100:.1f}% of runs)")
    if error_counter:
        for err_type, count in error_counter.most_common():
            print(f"    {err_type}: {count}")

    assert all(r is not None for r in results), "Some samples failed to evaluate."
    return [r for r in results if r is not None]


def filter_samples_by_acc(
    samples: List[Dict],
    *,
    min_acc: Optional[float],
    max_acc: Optional[float],
    split: str,
) -> Tuple[List[Dict], int]:
    """Filter samples by acc interval [min_acc, max_acc]."""
    if min_acc is None and max_acc is None:
        return samples, len(samples)

    if min_acc is not None and not (0.0 <= min_acc <= 1.0):
        raise ValueError("min_acc must be within [0, 1].")
    if max_acc is not None and not (0.0 <= max_acc <= 1.0):
        raise ValueError("max_acc must be within [0, 1].")
    if min_acc is not None and max_acc is not None and min_acc > max_acc:
        raise ValueError("min_acc cannot be greater than max_acc.")

    lower = min_acc if min_acc is not None else 0.0
    upper = max_acc if max_acc is not None else 1.0
    before_count = len(samples)

    filtered: List[Dict] = []
    missing_acc = 0
    for sample in samples:
        acc = sample.get("extra_info", {}).get("acc")
        if acc is None:
            missing_acc += 1
            continue
        if lower <= float(acc) <= upper:
            filtered.append(sample)

    print(
        f"[{split}] Difficulty filter applied: acc in [{lower:.4f}, {upper:.4f}] | "
        f"before={before_count}, after={len(filtered)}, missing_acc={missing_acc}"
    )
    return filtered, before_count


# --------------------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------------------


def save_dataset(
    samples: List[Dict],
    output_file: str,
    split_name: str,
    *,
    num_runs: Optional[int] = None,
    min_acc: Optional[float] = None,
    max_acc: Optional[float] = None,
    pre_filter_count: Optional[int] = None,
) -> None:
    """Save dataset to parquet file with metadata."""
    df = pd.DataFrame(samples)

    metadata: Dict[str, Any] = {
        "total_samples": len(samples),
        "split": split_name,
        "description": (
            "Rank-R1 dataset with per-positive random negative sampling. "
            "Each sample is 1 positive + 3 negatives, shuffled."
        ),
    }

    if num_runs is not None and num_runs > 0:
        accuracy_counts = {f"{i / num_runs:.4f}": 0 for i in range(num_runs + 1)}
        if not df.empty and "model_correct_count" in df.columns:
            counts = df["model_correct_count"].value_counts()
            for correct_value, count in counts.items():
                try:
                    correct_int = int(correct_value)
                except (TypeError, ValueError):
                    continue
                if 0 <= correct_int <= num_runs:
                    accuracy_key = f"{correct_int / num_runs:.4f}"
                    accuracy_counts[accuracy_key] = int(count)
        metadata["num_runs"] = num_runs
        metadata["accuracy_counts"] = accuracy_counts
    if pre_filter_count is not None:
        metadata["difficulty_filter"] = {
            "min_acc": min_acc,
            "max_acc": max_acc,
            "pre_filter_count": pre_filter_count,
            "post_filter_count": len(samples),
        }

    df.to_parquet(output_file, index=False)

    metadata_file = output_file.replace(".parquet", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{split_name.capitalize()} dataset saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Total {split_name} samples: {len(samples)}")
    if "accuracy_counts" in metadata:
        print(f"Accuracy distribution: {metadata['accuracy_counts']}")


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def create_rank_r1_difficulty_dataset(
    train_pickle_file: str,
    test_pickle_file: str,
    rank_resume_file: str,
    all_labels_csv: str,
    jd_file: str,
    resume_file: str,
    train_output_file: str,
    test_output_file: str,
    *,
    max_train_jobs: Optional[int] = None,
    max_test_jobs: Optional[int] = None,
    samples_per_positive: int = 3,
    max_positive_count: int = 11,
    seed: int = 42,
    # Difficulty evaluation options
    run_difficulty: bool = False,
    model_name: str = "Qwen/Qwen3-8B",
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: int = 20,
    min_p: float = 0.0,
    max_response_tokens: Optional[int] = None,
    request_timeout: Optional[int] = None,
    num_runs: int = 5,
    max_workers: int = 4,
    max_retries: int = 3,
    retry_sleep: float = 1.0,
    enable_thinking: bool = True,
    max_prompt_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    min_acc: Optional[float] = None,
    max_acc: Optional[float] = None,
) -> None:
    """Create Rank-R1 dataset with optional difficulty annotations."""

    # ---- Auto-select Qwen3 best-practice params based on thinking mode ----
    # Thinking mode:     temperature=0.6, top_p=0.95, top_k=20, min_p=0
    # Non-thinking mode: temperature=0.7, top_p=0.8,  top_k=20, min_p=0
    if enable_thinking:
        temperature = temperature if temperature is not None else 0.6
        top_p = top_p if top_p is not None else 0.95
        max_response_tokens = max_response_tokens if max_response_tokens is not None else 8192
        request_timeout = request_timeout if request_timeout is not None else 300
    else:
        temperature = temperature if temperature is not None else 0.7
        top_p = top_p if top_p is not None else 0.8
        max_response_tokens = max_response_tokens if max_response_tokens is not None else 4096
        request_timeout = request_timeout if request_timeout is not None else 120

    # Auto-compute max_prompt_tokens from model context window if not specified
    # Qwen3-8B context window = 40960 tokens
    MODEL_CONTEXT_WINDOW = 40960
    if max_prompt_tokens is None:
        max_prompt_tokens = MODEL_CONTEXT_WINDOW - max_response_tokens
    print(f"\n[Prompt budget] max_prompt_tokens={max_prompt_tokens} "
          f"(context={MODEL_CONTEXT_WINDOW} - completion={max_response_tokens})")

    if run_difficulty:
        mode_str = "THINKING" if enable_thinking else "NON-THINKING"
        print(f"[Qwen3 {mode_str} mode] temperature={temperature}, top_p={top_p}, "
              f"top_k={top_k}, min_p={min_p}, max_tokens={max_response_tokens}, "
              f"timeout={request_timeout}s")
    elif min_acc is not None or max_acc is not None:
        raise ValueError("Difficulty filter (min_acc/max_acc) requires --run_difficulty.")

    # Load shared resources
    labels = load_ground_truth_labels(rank_resume_file, all_labels_csv)
    job_descriptions = load_job_descriptions(jd_file)
    resume_texts = load_resume_texts(resume_file)

    # Load tokenizer for prompt length control
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    sampling_kwargs = dict(
        samples_per_positive=samples_per_positive,
        max_positive_count=max_positive_count,
        seed=seed,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
    )

    # ---- Train split ----
    print("Loading training data...")
    train_data = load_pickle_data(train_pickle_file)
    print(f"Processing {len(train_data)} training jobs...")
    train_samples = process_jobs(
        train_data, labels, job_descriptions, resume_texts,
        max_train_jobs, "train", **sampling_kwargs,
    )

    # ---- Test split ----
    print("\nLoading test data...")
    test_data = load_pickle_data(test_pickle_file)
    print(f"Processing {len(test_data)} test jobs...")
    test_samples = process_jobs(
        test_data, labels, job_descriptions, resume_texts,
        max_test_jobs, "test", **sampling_kwargs,
    )

    # ---- Optional difficulty evaluation ----
    if run_difficulty:
        print("\n--- Running difficulty evaluation ---")
        client = OpenAIChatClient(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_response_tokens,
            timeout=request_timeout,
            max_retries=max_retries,
            request_sleep=retry_sleep,
            enable_thinking=enable_thinking,
            organization=organization,
            project=project,
        )

        if num_runs <= 0:
            raise ValueError("num_runs must be a positive integer.")

        train_samples = evaluate_samples_with_concurrency(
            train_samples, client, num_runs=num_runs, max_workers=max_workers, split="train",
        )
        test_samples = evaluate_samples_with_concurrency(
            test_samples, client, num_runs=num_runs, max_workers=max_workers, split="test",
        )

        train_samples, train_pre_filter_count = filter_samples_by_acc(
            train_samples, min_acc=min_acc, max_acc=max_acc, split="train"
        )
        test_samples, test_pre_filter_count = filter_samples_by_acc(
            test_samples, min_acc=min_acc, max_acc=max_acc, split="test"
        )
    else:
        train_pre_filter_count = None
        test_pre_filter_count = None

    # ---- Save ----
    save_dataset(
        train_samples, train_output_file, "training",
        num_runs=num_runs if run_difficulty else None,
        min_acc=min_acc,
        max_acc=max_acc,
        pre_filter_count=train_pre_filter_count,
    )
    save_dataset(
        test_samples, test_output_file, "test",
        num_runs=num_runs if run_difficulty else None,
        min_acc=min_acc,
        max_acc=max_acc,
        pre_filter_count=test_pre_filter_count,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Rank-R1 dataset with per-positive random negative sampling "
            "and optional difficulty annotations."
        )
    )

    # Data paths
    parser.add_argument("--train_pickle_file", type=str, required=True)
    parser.add_argument("--test_pickle_file", type=str, required=True)
    parser.add_argument("--rank_resume_file", type=str, required=True)
    parser.add_argument("--all_labels_csv", type=str, required=True)
    parser.add_argument("--jd_file", type=str, required=True)
    parser.add_argument("--resume_file", type=str, required=True)
    parser.add_argument("--train_output_file", type=str, required=True)
    parser.add_argument("--test_output_file", type=str, required=True)

    # Sampling parameters
    parser.add_argument(
        "--samples_per_positive", type=int, default=3,
        help="Number of random negative samples per positive resume (default: 3).",
    )
    parser.add_argument(
        "--max_positive_count", type=int, default=11,
        help="Skip jobs with >= this many positives in top-20 (default: 11).",
    )
    parser.add_argument("--max_train_jobs", type=int, default=None)
    parser.add_argument("--max_test_jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # Difficulty evaluation
    parser.add_argument(
        "--run_difficulty", action="store_true", default=False,
        help="If set, run model inference to annotate per-sample accuracy.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-8B",
        help="Model name for difficulty evaluation (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--api_base", type=str, default=None,
        help="Optional base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (auto: thinking=0.6, non-thinking=0.7).")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p sampling (auto: thinking=0.95, non-thinking=0.8).")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling (Qwen3 recommends 20, default: 20).")
    parser.add_argument("--min_p", type=float, default=0.0,
                        help="Min-p sampling (Qwen3 recommends 0, default: 0).")
    parser.add_argument("--max_response_tokens", type=int, default=None,
                        help="Max tokens (auto: thinking=8192, non-thinking=4096).")
    parser.add_argument("--request_timeout", type=int, default=None,
                        help="Timeout in seconds (auto: thinking=300, non-thinking=120).")
    parser.add_argument(
        "--disable_thinking", action="store_true", default=False,
        help="Disable Qwen3 thinking mode. Auto-switches to non-thinking params.",
    )
    parser.add_argument(
        "--max_prompt_tokens", type=int, default=None,
        help="Max prompt tokens (auto: model_context - max_response_tokens). "
             "Resumes are truncated proportionally to fit.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=5,
        help="Number of evaluation runs per sample (default: 5).",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4,
        help="Number of concurrent workers for inference (default: 4).",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=1.0)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--organization", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument(
        "--min_acc", type=float, default=None,
        help="Optional lower bound of acc filter (inclusive), requires --run_difficulty.",
    )
    parser.add_argument(
        "--max_acc", type=float, default=None,
        help="Optional upper bound of acc filter (inclusive), requires --run_difficulty.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.train_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output_file), exist_ok=True)

    create_rank_r1_difficulty_dataset(
        train_pickle_file=args.train_pickle_file,
        test_pickle_file=args.test_pickle_file,
        rank_resume_file=args.rank_resume_file,
        all_labels_csv=args.all_labels_csv,
        jd_file=args.jd_file,
        resume_file=args.resume_file,
        train_output_file=args.train_output_file,
        test_output_file=args.test_output_file,
        max_train_jobs=args.max_train_jobs,
        max_test_jobs=args.max_test_jobs,
        samples_per_positive=args.samples_per_positive,
        max_positive_count=args.max_positive_count,
        seed=args.seed,
        run_difficulty=args.run_difficulty,
        model_name=args.model_name,
        api_base=args.api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_response_tokens=args.max_response_tokens,
        request_timeout=args.request_timeout,
        num_runs=args.num_runs,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        enable_thinking=not args.disable_thinking,
        max_prompt_tokens=args.max_prompt_tokens,
        api_key=args.api_key,
        organization=args.organization,
        project=args.project,
        min_acc=args.min_acc,
        max_acc=args.max_acc,
    )


if __name__ == "__main__":
    main()
