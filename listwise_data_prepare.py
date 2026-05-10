import argparse
import os
import pickle
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from confit_v3.trainer.load_data import (
    load_all_resume_texts,
    load_job_descriptions,
    load_rank_resume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/test VERL ranking datasets from embedding ranking data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/confit_v3_data/ranking",
        help="Directory containing job/resume text files, rank files, and split id files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where parquet files will be saved. Defaults to <data-dir>/verl_dataset.",
    )
    parser.add_argument("--job-text-file", type=str, default="job_text.csv")
    parser.add_argument("--resume-text-file", type=str, default="resume_text.csv")
    parser.add_argument("--labels-file", type=str, default="rank_resume.json")
    parser.add_argument("--train-rank-file", type=str, default="train_rank.pkl")
    parser.add_argument("--test-rank-file", type=str, default="test_rank.pkl")
    parser.add_argument("--train-job-ids-file", type=str, default="train_job_ids.pkl")
    parser.add_argument("--train-resume-ids-file", type=str, default="train_resume_ids.pkl")
    parser.add_argument("--test-job-ids-file", type=str, default="test_job_ids.pkl")
    parser.add_argument("--test-resume-ids-file", type=str, default="test_resume_ids.pkl")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Dataset splits to build.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of ranked resumes to consider per job.")
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=3,
        help="Number of negative resumes sampled for each positive resume.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling and shuffling.")
    parser.add_argument("--data-source", type=str, default="confit_v3_final")
    parser.add_argument("--output-prefix", type=str, default="listwise_data")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle each final dataset before saving.",
    )
    parser.add_argument(
        "--analyze-lengths",
        action="store_true",
        help="Analyze prompt token lengths after each split is built.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Tokenizer used when --analyze-lengths is enabled.",
    )
    return parser.parse_args()


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def to_str_set(values: Iterable[Any]) -> Set[str]:
    return {str(value) for value in values}


def stringify_rank(rank: Mapping[Any, Mapping[Any, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(job_id): {str(resume_id): score for resume_id, score in resume_rank.items()}
        for job_id, resume_rank in rank.items()
    }


def stringify_labels(labels: Mapping[Any, Mapping[str, Sequence[Any]]]) -> Dict[str, Dict[str, List[Any]]]:
    normalized: Dict[str, Dict[str, List[Any]]] = {}
    for job_id, item in labels.items():
        normalized[str(job_id)] = {
            "user_ids": [str(user_id) for user_id in item["user_ids"]],
            "satisfied": list(item["satisfied"]),
        }
    return normalized


def stringify_text_dict(texts: Mapping[Any, str]) -> Dict[str, str]:
    return {str(key): value for key, value in texts.items()}


def load_all_inputs(args: argparse.Namespace) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, List[Any]]], Dict[str, Any]]:
    job_df = load_job_descriptions(os.path.join(args.data_dir, args.job_text_file))
    job_texts = {str(row["jd_no"]): row["job_text"] for _, row in job_df.iterrows()}
    resume_texts = stringify_text_dict(load_all_resume_texts(os.path.join(args.data_dir, args.resume_text_file)))
    labels = stringify_labels(load_rank_resume(os.path.join(args.data_dir, args.labels_file)))

    split_inputs = {
        "train": {
            "job_ids": to_str_set(load_pickle(os.path.join(args.data_dir, args.train_job_ids_file))),
            "resume_ids": to_str_set(load_pickle(os.path.join(args.data_dir, args.train_resume_ids_file))),
            "rank": stringify_rank(load_pickle(os.path.join(args.data_dir, args.train_rank_file))),
        },
        "test": {
            "job_ids": to_str_set(load_pickle(os.path.join(args.data_dir, args.test_job_ids_file))),
            "resume_ids": to_str_set(load_pickle(os.path.join(args.data_dir, args.test_resume_ids_file))),
            "rank": stringify_rank(load_pickle(os.path.join(args.data_dir, args.test_rank_file))),
        },
    }
    return job_texts, resume_texts, labels, split_inputs


def get_accepted_resumes(label_item: Mapping[str, Sequence[Any]]) -> Set[str]:
    accepted_resumes = set()
    for user_id, satisfied in zip(label_item["user_ids"], label_item["satisfied"]):
        if satisfied:
            accepted_resumes.add(str(user_id))
    return accepted_resumes


def build_samples_for_split(
    split: str,
    rank: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Sequence[Any]]],
    job_texts: Mapping[str, str],
    resume_texts: Mapping[str, str],
    feasible_job_ids: Optional[Set[str]],
    feasible_resume_ids: Optional[Set[str]],
    top_k: int,
    num_negatives: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    job_stats = defaultdict(int)
    required_window_size = num_negatives + 1

    for job_id, resume_rank in rank.items():
        if feasible_job_ids is not None and job_id not in feasible_job_ids:
            continue
        if job_id not in labels or job_id not in job_texts:
            continue

        accepted_resumes = get_accepted_resumes(labels[job_id])
        if feasible_resume_ids is not None:
            accepted_resumes = accepted_resumes.intersection(feasible_resume_ids)

        ranked_resumes = [
            (str(resume_id), score)
            for resume_id, score in resume_rank.items()
            if feasible_resume_ids is None or str(resume_id) in feasible_resume_ids
        ]
        top_resumes = sorted(ranked_resumes, key=lambda x: x[1])[:top_k]
        top_resume_ids = [resume_id for resume_id, _ in top_resumes]

        positive_in_top = [resume_id for resume_id in top_resume_ids if resume_id in accepted_resumes]
        if not positive_in_top:
            continue

        negative_in_top = [resume_id for resume_id in top_resume_ids if resume_id not in accepted_resumes]
        if len(negative_in_top) < num_negatives:
            continue

        for target_resume_id in positive_in_top:
            selected_ids = [target_resume_id] + rng.sample(negative_in_top, num_negatives)
            rng.shuffle(selected_ids)

            valid_labels = [str(i + 1) for i in range(required_window_size)]
            resumes = []
            accepted_labels = []
            skip_sample = False

            for resume_id, label in zip(selected_ids, valid_labels):
                resume_text = resume_texts.get(resume_id)
                if resume_text is None:
                    skip_sample = True
                    break
                if resume_id in accepted_resumes:
                    accepted_labels.append(label)
                resumes.append((resume_id, resume_text, label))

            if skip_sample:
                continue

            samples.append(
                {
                    "job_id": job_id,
                    "job_description": job_texts[job_id],
                    "resumes": resumes,
                    "accepted_labels": accepted_labels,
                }
            )
            job_stats[job_id] += 1

    print_split_stats(split, samples, job_stats)
    return samples


def print_split_stats(split: str, samples: Sequence[Mapping[str, Any]], job_stats: Mapping[str, int]) -> None:
    print(f"\n[{split}] Dataset statistics")
    print(f"Total samples: {len(samples)}")
    print(f"Total jobs: {len(job_stats)}")
    if job_stats:
        print(f"Average samples per job: {len(samples) / len(job_stats):.2f}")
        print(f"Jobs with multiple samples: {sum(1 for count in job_stats.values() if count > 1)}")
    else:
        print("Average samples per job: 0.00")
        print("Jobs with multiple samples: 0")


def make_map_fn_reason(split: str, data_source: str):
    def process_fn(example: Mapping[str, Any], idx: int) -> Dict[str, Any]:
        job_description = example["job_description"]
        resumes = example["resumes"]
        valid_labels = [str(i + 1) for i in range(len(resumes))]

        resumes_text = []
        for (resume_id, resume_text, label), valid_label in zip(resumes, valid_labels):
            assert label == valid_label, f"Label mismatch: {label} != {valid_label}"
            resumes_text.append(f"[{valid_label}] Resume {valid_label}:\n{resume_text}")

        resumes_section = "\n\n".join(resumes_text)
        answer_format = " > ".join(f"[{label}]" for label in valid_labels)

        system_prompt = (
            "You are an expert technical recruiter that can rank resumes based on their matching degree to the job description. "
            "You first analyze each resume individually, then compare them systematically, and finally provide the ranking. "
            f"I will provide you with {len(resumes)} resumes, each indicated by a numeric identifier []. "
            f"Rank the {len(resumes)} resumes based on their matching degree to the job description. "
            "The resumes should be listed in descending order using identifiers. The most relevant resumes should be listed first. "
            f"The output format should be <answer> {answer_format} </answer>."
        )

        user_prompt = (
            f"Resumes:\n{resumes_section}\n\n"
            f"Please rank these resumes according to their matching degree to the JOB DESCRIPTION: [{job_description}]\n"
            "Follow these steps exactly:\n"
            "1. First, think to summarize the job description and analyze EACH resume briefly: Evaluate how well it matches the job description and mandatory criteria.\n"
            "2. Then, think to COMPARE the resumes and determine which candidates are better fits and why.\n"
            "3. Finally, within <answer> tags, provide ONLY the final ranking of the resumes from best to worst fit using their numerical identifiers.\n"
        )

        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "ability": "ranking",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["accepted_labels"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "job_id": example["job_id"],
                "resume_ids": [resume[0] for resume in resumes],
                "valid_labels": valid_labels,
            },
        }

    return process_fn


def build_dataset(samples: List[Dict[str, Any]], split: str, data_source: str, seed: int, shuffle: bool) -> Dataset:
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(function=make_map_fn_reason(split, data_source), with_indices=True)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset


def analyze_prompt_lengths(
    dataset: Dataset,
    tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_key: str = "prompt",
) -> np.ndarray:
    """
    Compute and print token length statistics for chat prompts in a dataset.

    Args:
        dataset: Dataset containing a chat prompt field.
        tokenizer_name: Name or path of the tokenizer.
        prompt_key: Dataset field that stores the chat prompt.

    Returns:
        A NumPy array containing token lengths for all prompts.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print("Calculating total prompt token lengths...")
    token_lengths = []

    for idx, row in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"Progress: {idx}/{len(dataset)}")

        tokens = tokenizer.apply_chat_template(
            row[prompt_key],
            tokenize=True,
            add_generation_prompt=False,
        )
        token_lengths.append(len(tokens))

    token_lengths = np.array(token_lengths)

    print("\n" + "=" * 60)
    print("Total Prompt Token Length Statistics")
    print("=" * 60)
    print(f"Samples:  {len(token_lengths)}")
    if len(token_lengths) > 0:
        print(f"Min:      {token_lengths.min()}")
        print(f"Max:      {token_lengths.max()}")
        print(f"Mean:     {token_lengths.mean():.0f}")
        print(f"Median:   {np.median(token_lengths):.0f}")
        print(f"P90:      {np.percentile(token_lengths, 90):.0f}")
        print(f"P95:      {np.percentile(token_lengths, 95):.0f}")
        print(f"P99:      {np.percentile(token_lengths, 99):.0f}")
    print("=" * 60)

    return token_lengths


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.data_dir, "verl_dataset")
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading inputs...")
    job_texts, resume_texts, labels, split_inputs = load_all_inputs(args)

    for split in args.splits:
        split_seed = args.seed + (0 if split == "train" else 10_000)
        rng = random.Random(split_seed)
        split_info = split_inputs[split]

        samples = build_samples_for_split(
            split=split,
            rank=split_info["rank"],
            labels=labels,
            job_texts=job_texts,
            resume_texts=resume_texts,
            feasible_job_ids=split_info["job_ids"],
            feasible_resume_ids=split_info["resume_ids"],
            top_k=args.top_k,
            num_negatives=args.num_negatives,
            rng=rng,
        )

        dataset = build_dataset(
            samples=samples,
            split=split,
            data_source=args.data_source,
            seed=split_seed,
            shuffle=args.shuffle,
        )

        output_path = os.path.join(output_dir, f"{args.output_prefix}_{split}.parquet")
        dataset.to_parquet(output_path)
        print(f"[{split}] Saved parquet to: {output_path}")

        if args.analyze_lengths:
            analyze_prompt_lengths(dataset, tokenizer_name=args.tokenizer_name, prompt_key="prompt")


if __name__ == "__main__":
    main()
