"""
Compute Qwen embeddings for job and resume texts, then save pairwise resume ranks.

By default, this script reads pre-merged text CSV files and does not merge raw
feature columns. The expected default files are:

    job_merged_text.csv       columns: jd_no, job_text
    resume_merged_text.csv    columns: user_id, resume_text

The expected split ID files are:

    train_job_ids.pkl
    train_resume_ids.pkl
    test_job_ids.pkl
    test_resume_ids.pkl

Example:
    python compute_job_resume_ranks.py \
        --data_dir /path/to/ranking_data \
        --id_dir /path/to/ranking_data \
        --output_dir /path/to/ranking_data \
        --pretrained_encoder Qwen/Qwen3-Embedding-0.6B \
        --model_path /path/to/checkpoint.ckpt
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------
# Default job/resume column names.
# ---------------------------------------------------------------------
RESUME_ID_COL = "user_id"
RESUME_TEXT_COL = "resume_text"
JOB_ID_COL = "jd_no"
JOB_TEXT_COL = "job_text"


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train/test job-resume ranks from pre-merged text CSVs."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing job/resume text CSV files.",
    )
    parser.add_argument(
        "--id_dir",
        type=str,
        default=None,
        help="Directory containing split ID pickle files. Defaults to --data_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where rank pickle files will be saved. Defaults to --data_dir.",
    )
    parser.add_argument(
        "--job_csv",
        type=str,
        default="job_merged_text.csv",
        help="Job CSV filename or absolute path. Default expects columns jd_no and job_text.",
    )
    parser.add_argument(
        "--resume_csv",
        type=str,
        default="resume_merged_text.csv",
        help="Resume CSV filename or absolute path. Default expects columns user_id and resume_text.",
    )
    parser.add_argument("--job_id_col", type=str, default=JOB_ID_COL)
    parser.add_argument("--job_text_col", type=str, default=JOB_TEXT_COL)
    parser.add_argument("--resume_id_col", type=str, default=RESUME_ID_COL)
    parser.add_argument("--resume_text_col", type=str, default=RESUME_TEXT_COL)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Splits to process.",
    )
    parser.add_argument(
        "--job_ids_template",
        type=str,
        default="{split}_job_ids.pkl",
        help="Filename template for job ID pickle files.",
    )
    parser.add_argument(
        "--resume_ids_template",
        type=str,
        default="{split}_resume_ids.pkl",
        help="Filename template for resume ID pickle files.",
    )
    parser.add_argument(
        "--output_template",
        type=str,
        default="{split}_rank.pkl",
        help="Filename template for output rank pickle files.",
    )
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Tokenizer/model name or path.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional custom checkpoint path. If omitted, the base model is used.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--sim_chunk_size", type=int, default=1000)
    parser.add_argument(
        "--rank_top_k",
        type=int,
        default=None,
        help="Save only top-k ranked resumes per job. Default saves the full ranking.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_or_name: str) -> Path:
    path = Path(path_or_name)
    return path if path.is_absolute() else base_dir / path


# ---------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------
def normalize_ids(values: Iterable[object]) -> Set[str]:
    return {str(value) for value in values}


def load_id_set(id_dir: Path, filename: str) -> Set[str]:
    with open(id_dir / filename, "rb") as f:
        return normalize_ids(pickle.load(f))


def load_texts_from_csv(
    csv_path: Path,
    id_col: str,
    text_col: str,
    target_ids: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Load IDs and pre-merged text from a CSV file."""
    df = pd.read_csv(csv_path)
    missing_columns = [col for col in (id_col, text_col) if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {csv_path}: {missing_columns}")

    df[id_col] = df[id_col].astype(str)
    if target_ids is not None:
        df = df[df[id_col].isin(target_ids)].reset_index(drop=True)

    texts = df[text_col].fillna("").astype(str).tolist()
    ids = df[id_col].tolist()
    return texts, ids


def load_split_texts(
    job_csv: Path,
    resume_csv: Path,
    job_ids: Set[str],
    resume_ids: Set[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load job and resume texts for one split."""
    job_texts, job_ids_list = load_texts_from_csv(
        csv_path=job_csv,
        id_col=args.job_id_col,
        text_col=args.job_text_col,
        target_ids=job_ids,
    )
    resume_texts, resume_ids_list = load_texts_from_csv(
        csv_path=resume_csv,
        id_col=args.resume_id_col,
        text_col=args.resume_text_col,
        target_ids=resume_ids,
    )
    return job_texts, job_ids_list, resume_texts, resume_ids_list


# ---------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def get_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tensor:
    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu()

        all_embeddings.append(embeddings)
        del batch, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_embeddings:
        return torch.empty(0, 0)
    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------
# Similarity and ranking
# ---------------------------------------------------------------------
def batch_cosine_similarity(
    matrix_a: Tensor,
    matrix_b: Tensor,
    device: torch.device,
    chunk_size: int,
) -> Tensor:
    num_rows = matrix_a.shape[0]
    matrix_b_gpu = matrix_b.to(device)
    sim = torch.zeros(num_rows, matrix_b.shape[0], dtype=torch.float32)

    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        a_chunk = matrix_a[start:end].to(device)
        sim[start:end] = (a_chunk @ matrix_b_gpu.T).cpu()
        del a_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del matrix_b_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sim


def build_rank_dict(
    sim_matrix: Tensor,
    job_ids: Sequence[str],
    resume_ids: Sequence[str],
    top_k: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    if top_k is not None and top_k < sim_matrix.size(1):
        _, indices = torch.topk(sim_matrix, k=top_k, dim=1, largest=True)
    else:
        _, indices = torch.sort(sim_matrix, dim=1, descending=True)

    indices_np = indices.numpy()
    ranks = {}
    for job_idx, job_id in enumerate(job_ids):
        ranks[job_id] = {
            resume_ids[resume_idx]: rank
            for rank, resume_idx in enumerate(indices_np[job_idx])
        }
    return ranks


def compute_and_save_ranks(
    job_embs: Tensor,
    job_ids: Sequence[str],
    resume_embs: Tensor,
    resume_ids: Sequence[str],
    device: torch.device,
    output_path: Path,
    split: str,
    sim_chunk_size: int,
    rank_top_k: Optional[int],
) -> None:
    print("\n" + "=" * 60)
    print(f"Processing {split}: {len(job_ids)} jobs x {len(resume_ids)} resumes")
    print("=" * 60)

    sim = batch_cosine_similarity(job_embs, resume_embs, device, sim_chunk_size)
    ranks = build_rank_dict(sim, job_ids, resume_ids, top_k=rank_top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(ranks, f)
    print(f"Saved {split} ranks to: {output_path}")


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_encoder(args: argparse.Namespace, device: torch.device):
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_encoder,
        padding_side="left",
        cache_dir=args.cache_dir,
    )
    model = AutoModel.from_pretrained(
        args.pretrained_encoder,
        cache_dir=args.cache_dir,
    )

    if args.model_path:
        print(f"Loading custom checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        cleaned_state_dict = {}
        # clean confit v2 embedding model loading
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("bert_resume.", "bert_job.", "model.", "module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned_state_dict[new_key] = value

        model.load_state_dict(cleaned_state_dict, strict=True)
        print("Checkpoint loaded.")

    return tokenizer, model.to(device).eval()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    id_dir = Path(args.id_dir) if args.id_dir else data_dir
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    job_csv = resolve_path(data_dir, args.job_csv)
    resume_csv = resolve_path(data_dir, args.resume_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Job CSV: {job_csv}")
    print(f"Resume CSV: {resume_csv}")
    print("Text mode: read pre-merged text columns")

    print("\nLoading split ID files...")
    split_ids = {}
    for split in args.splits:
        split_ids[split] = {
            "job_ids": load_id_set(id_dir, args.job_ids_template.format(split=split)),
            "resume_ids": load_id_set(id_dir, args.resume_ids_template.format(split=split)),
        }
        print(
            f"  {split}: {len(split_ids[split]['job_ids'])} jobs, "
            f"{len(split_ids[split]['resume_ids'])} resumes"
        )

    tokenizer, model = load_encoder(args, device)

    for split in args.splits:
        print("\n" + "=" * 60)
        print(f"PROCESSING {split.upper()} SET")
        print("=" * 60)

        job_texts, job_ids, resume_texts, resume_ids = load_split_texts(
            job_csv=job_csv,
            resume_csv=resume_csv,
            job_ids=split_ids[split]["job_ids"],
            resume_ids=split_ids[split]["resume_ids"],
            args=args,
        )

        if job_texts:
            print(f"\nSample job text:\n{job_texts[0][:500]}\n---")
        if resume_texts:
            print(f"\nSample resume text:\n{resume_texts[0][:500]}\n---")

        job_embs = get_embeddings(
            job_texts,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        resume_embs = get_embeddings(
            resume_texts,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        compute_and_save_ranks(
            job_embs=job_embs,
            job_ids=job_ids,
            resume_embs=resume_embs,
            resume_ids=resume_ids,
            device=device,
            output_path=output_dir / args.output_template.format(split=split),
            split=split.upper(),
            sim_chunk_size=args.sim_chunk_size,
            rank_top_k=args.rank_top_k,
        )

        del job_embs, resume_embs, job_texts, resume_texts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    for split in args.splits:
        print(f"  {split}: {output_dir / args.output_template.format(split=split)}")


if __name__ == "__main__":
    main()
