from __future__ import annotations

import argparse
import json
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm


_DEFAULT_STRIDE = 2
_DEFAULT_WINDOW_SIZE = 4
_DEFAULT_NUM_PASSES = 1
_DEFAULT_NUM_WORKERS = 8
_DEFAULT_MODEL = "Qwen/Qwen3-8B"
_DEFAULT_OUTPUT_DIR = "./"

_DEFAULT_JD_CSV = "dataset/confit_v3_listwise/job_merged_test.csv"
_DEFAULT_RANK_RESUME_JSON = "dataset/confit_v3_listwise/rank_resume.json"
_DEFAULT_RESUME_CSV = "dataset/confit_v3_listwise/resume_merged_test.csv"
_DEFAULT_LABELS_CSV = None
_DEFAULT_LABELS_JSON = "dataset/confit_v3_listwise/rank_resume.json"
_DEFAULT_INIT_RANKING_PKL = "dataset/confit_v3_listwise/test_ranking.pkl"
_DEFAULT_BM25_K_STR = "5,50"
_DEFAULT_EVAL_K_STR = "10,20,100,250,500,1000"
_DEFAULT_TEMPERATURE = 0.6
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_TIMEOUT = 120


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(part) for part in re.split(r"[,\s]+", text.strip()) if part]


@dataclass
class PipelineConfig:
    jd_csv: str
    resume_csv: str
    rank_resume_json: str
    labels_csv: str
    labels_json: str
    init_ranking_pkl: str
    bm25_k_values: List[int]
    eval_k_values: List[int]
    stride: int
    window_size: int
    num_passes: int
    base_url: Optional[str]
    model: str
    num_workers: int
    output_dir: str
    temperature: float
    max_tokens: int
    timeout: int


@dataclass
class PipelineData:
    jd_df: Any
    all_resume_data: Dict[str, str]
    rank_resume: Dict[str, Any]
    all_labels_df: Any
    labels: Dict[str, Any]
    initial_ranking: Dict[str, Any]
    job_ids: List[str]


def build_parser(default_base_url: Optional[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd-csv", default=_DEFAULT_JD_CSV)
    parser.add_argument("--resume-csv", default=_DEFAULT_RESUME_CSV)
    parser.add_argument("--rank-resume-json", default=_DEFAULT_RANK_RESUME_JSON)
    parser.add_argument("--labels-csv", default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--labels-json", default=_DEFAULT_LABELS_JSON)
    parser.add_argument("--init-ranking-pkl", default=_DEFAULT_INIT_RANKING_PKL)
    parser.add_argument("--bm25-k", default=_DEFAULT_BM25_K_STR)
    parser.add_argument("--eval-k", default=_DEFAULT_EVAL_K_STR)
    parser.add_argument("--stride", type=int, default=_DEFAULT_STRIDE)
    parser.add_argument("--window-size", type=int, default=_DEFAULT_WINDOW_SIZE)
    parser.add_argument("--num-passes", type=int, default=_DEFAULT_NUM_PASSES)
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--num-workers", type=int, default=_DEFAULT_NUM_WORKERS)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        jd_csv=args.jd_csv,
        resume_csv=args.resume_csv,
        rank_resume_json=args.rank_resume_json,
        labels_csv=args.labels_csv,
        labels_json=args.labels_json,
        init_ranking_pkl=args.init_ranking_pkl,
        bm25_k_values=parse_int_list(args.bm25_k),
        eval_k_values=parse_int_list(args.eval_k),
        stride=args.stride,
        window_size=args.window_size,
        num_passes=args.num_passes,
        base_url=args.base_url,
        model=args.model,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


def load_pipeline_data(config: PipelineConfig) -> PipelineData:
    from confit_v3.trainer.load_data import (
        load_all_labels_csv,
        load_all_resume_texts,
        load_job_descriptions,
        load_rank_resume,
    )

    with open(config.labels_json, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    with open(config.init_ranking_pkl, "rb") as handle:
        initial_ranking = pickle.load(handle)
    return PipelineData(
        jd_df=load_job_descriptions(config.jd_csv),
        all_resume_data=load_all_resume_texts(config.resume_csv),
        rank_resume=load_rank_resume(config.rank_resume_json),
        all_labels_df=load_all_labels_csv(config.labels_csv) if config.labels_csv else None,
        labels=labels,
        initial_ranking=initial_ranking,
        job_ids=list(initial_ranking.keys()),
    )


def print_config_summary(config: PipelineConfig, data: PipelineData) -> None:
    print("=== Config Summary ===")
    print(f"JD_CSV:           {config.jd_csv}")
    print(f"RESUME_CSV:       {config.resume_csv}")
    print(f"RANK_RESUME_JSON: {config.rank_resume_json}")
    print(f"LABELS_CSV:       {config.labels_csv}")
    print(f"LABELS_JSON:      {config.labels_json}")
    print(f"INIT_RANKING_PKL: {config.init_ranking_pkl}")
    print(f"BM25_K_VALUES:    {config.bm25_k_values}")
    print(f"EVAL_K_VALUES:    {config.eval_k_values}")
    print(f"#JD IDs: {len(data.labels)} | #Labels rows: {len(data.all_labels_df)} | #Test jobs: {len(data.job_ids)}")
    print(f"First 3 Test JD IDs: {data.job_ids[:3]}")
    print(f"STRIDE: {config.stride}  WINDOW_SIZE: {config.window_size}  NUM_PASSES: {config.num_passes}")
    print(f"MODEL: {config.model}  BASE_URL: {config.base_url}")
    print(f"NUM_WORKERS: {config.num_workers}  OUTPUT_DIR: {config.output_dir}")
    print(f"TEMPERATURE: {config.temperature}  MAX_TOKENS: {config.max_tokens}  TIMEOUT: {config.timeout}")
    print("=" * 22)


def get_resume_getter(data: PipelineData) -> Callable[[str], str]:
    def get_resume_from_id(resume_id: str) -> str:
        return data.all_resume_data[resume_id]

    return get_resume_from_id


def get_job_description_getter(data: PipelineData) -> Callable[[str], str]:
    def get_job_description_from_id(job_id: str) -> str:
        series = data.jd_df.loc[data.jd_df["jd_no"] == job_id, "job_text"].dropna()
        if series.empty:
            raise KeyError(f"job_id not found or has no job_text: {job_id}")
        return str(series.iloc[0]).strip()

    return get_job_description_from_id


def ordered_entries_by_rank(data: Dict[str, Dict[str, int]], top_k: Optional[int] = None) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for job_id, inner in data.items():
        ordered_pairs = sorted(
            [item for item in inner.items() if isinstance(item[1], int)],
            key=lambda pair: (pair[1], pair[0]),
        )
        if top_k is not None:
            ordered_pairs = ordered_pairs[:top_k]
        result[job_id] = [resume_id for resume_id, _ in ordered_pairs]
    return result


def print_length_stats(lengths: List[int], label: str) -> None:
    if not lengths:
        print(f"  [{label}] No data collected.")
        return
    sorted_lengths = sorted(lengths)

    def percentile(percent: float) -> float:
        if len(sorted_lengths) == 1:
            return float(sorted_lengths[0])
        rank = (len(sorted_lengths) - 1) * percent / 100.0
        low = int(rank)
        high = min(low + 1, len(sorted_lengths) - 1)
        weight = rank - low
        return sorted_lengths[low] * (1.0 - weight) + sorted_lengths[high] * weight

    mean_value = sum(sorted_lengths) / len(sorted_lengths)
    print(f"  [{label}] count={len(sorted_lengths)}")
    print(f"    Mean:   {mean_value:.0f}")
    print(f"    P80:    {percentile(80):.0f}")
    print(f"    P90:    {percentile(90):.0f}")
    print(f"    P95:    {percentile(95):.0f}")
    print(f"    P99:    {percentile(99):.0f}")
    print(f"    Max:    {max(sorted_lengths):.0f}")


def strip_thinking(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return stripped if stripped else text[-500:]


class TokenUsageTrackerWrapper:
    def __init__(self, real_reranker):
        self._real = real_reranker
        self.prompt_tokens: List[int] = []
        self.completion_tokens: List[int] = []
        self.total_tokens: List[int] = []
        self._patch_session()

    def _patch_session(self) -> None:
        session = getattr(self._real, "session", None)
        if session is None:
            return

        original_post = session.post
        tracker = self

        def patched_post(*args, **kwargs):
            response = original_post(*args, **kwargs)
            try:
                data = response.json()
                usage = data.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    completion_tokens = usage.get("completion_tokens", 0) or 0
                    total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                    tracker.prompt_tokens.append(prompt_tokens)
                    tracker.completion_tokens.append(completion_tokens)
                    tracker.total_tokens.append(total_tokens)

                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
                    if content:
                        non_thinking = strip_thinking(content)
                        has_answer_tag = bool(re.search(r"<answer>.*?</answer>", content, re.DOTALL | re.IGNORECASE))
                        answer_numbers = []
                        if has_answer_tag:
                            match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL | re.IGNORECASE)
                            if match:
                                answer_numbers = re.findall(r"\[(\d+)\]", match.group(1))
                        invalid_numbers = [number for number in answer_numbers if number not in ("1", "2", "3", "4")]
                        if not has_answer_tag:
                            print(f"\n[RESPONSE DEBUG] No <answer> tag. Non-thinking output:\n  {non_thinking[:400]}")
                        elif invalid_numbers:
                            print(f"\n[RESPONSE DEBUG] Invalid nums {invalid_numbers} in <answer>. Non-thinking output:\n  {non_thinking[:400]}")
            except Exception:
                pass
            return response

        session.post = patched_post

    def rerank(self, *args, **kwargs):
        return self._real.rerank(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def rerank_parallel(
    reranker_factory: Callable[[], Any],
    input_ranking: Dict[str, Any],
    num_workers: int,
    desc: str,
    track_token_usage: bool,
) -> Dict[str, Any]:
    thread_local = threading.local()
    wrappers: List[TokenUsageTrackerWrapper] = []
    wrappers_lock = threading.Lock()

    def get_reranker():
        if not hasattr(thread_local, "reranker"):
            reranker = reranker_factory()
            if track_token_usage:
                wrapped = TokenUsageTrackerWrapper(reranker)
                thread_local.reranker = wrapped
                with wrappers_lock:
                    wrappers.append(wrapped)
            else:
                thread_local.reranker = reranker
        return thread_local.reranker

    def rerank_one(job_id: str):
        reranker = get_reranker()
        output = reranker.rerank({job_id: input_ranking[job_id]})
        return job_id, output[job_id]

    job_ids = list(input_ranking.keys())
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(rerank_one, job_id): job_id for job_id in job_ids}
        with tqdm(total=len(futures), desc=desc, unit="job") as progress:
            for future in as_completed(futures):
                job_id = futures[future]
                try:
                    completed_job_id, ranking = future.result()
                    results[completed_job_id] = ranking
                except Exception as exc:
                    print(f"[ERROR] job_id={job_id} failed: {exc}")
                    results[job_id] = input_ranking[job_id]
                finally:
                    progress.update(1)

    if track_token_usage:
        all_prompt_tokens: List[int] = []
        all_completion_tokens: List[int] = []
        all_total_tokens: List[int] = []
        for wrapper in wrappers:
            all_prompt_tokens.extend(wrapper.prompt_tokens)
            all_completion_tokens.extend(wrapper.completion_tokens)
            all_total_tokens.extend(wrapper.total_tokens)
        if all_prompt_tokens:
            total_prompt = sum(all_prompt_tokens)
            total_completion = sum(all_completion_tokens)
            print(f"\n{'=' * 55}")
            print(f"Token Usage Statistics [{desc}]")
            print(f"  Total API calls: {len(all_prompt_tokens)}")
            print(f"  Total prompt tokens:     {total_prompt:,}")
            print(f"  Total completion tokens: {total_completion:,}")
            print(f"  Total tokens:            {total_prompt + total_completion:,}")
            print(f"{'=' * 55}")
            print_length_stats(all_prompt_tokens, "Prompt tokens")
            print_length_stats(all_completion_tokens, "Completion tokens")
            print_length_stats(all_total_tokens, "Total tokens (prompt+completion)")
            print(f"{'=' * 55}\n")

    return {job_id: results[job_id] for job_id in job_ids}


def evaluate_ranking(
    ranking: Dict[str, Any],
    job_ids: List[str],
    labels: Dict[str, Any],
    bm25_k_values: List[int],
    eval_k_values: List[int],
    tag: str,
    model_provider: Any,
    model_name: Any,
) -> List[dict]:
    from confit_v3.trainer.metric import calculate_metrics

    metrics_rows = []
    for job_id in tqdm(job_ids, desc=f"Evaluating [{tag}]", unit="job"):
        valid_resumes = ranking[job_id]
        search_result = {"hits": {"hits": [{"_id": resume_id} for resume_id in valid_resumes]}}
        metrics = calculate_metrics(search_result, job_id, labels, bm25_k_values, eval_k_values)
        metrics_rows.append(
            {
                "job_id": job_id,
                "prompt_num": -1,
                "model_provider": model_provider,
                "model_name": model_name,
                "hits_count": -1,
                "query_length": -1,
                **metrics,
            }
        )
    return metrics_rows


def make_reranker_factory(
    reranker_cls,
    config: PipelineConfig,
    data: PipelineData,
    num_comparison_elements: int,
    window_stride: int,
    num_passes: int,
) -> Callable[[], Any]:
    get_resume_from_id = get_resume_getter(data)
    get_job_description_from_id = get_job_description_getter(data)

    def factory():
        return reranker_cls(
            base_url=config.base_url,
            model=config.model,
            top_k=20,
            get_resume_from_id=get_resume_from_id,
            get_job_description_from_id=get_job_description_from_id,
            order_entries_callback=ordered_entries_by_rank,
            num_comparison_elements=num_comparison_elements,
            window_stride=window_stride,
            num_passes=num_passes,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    return factory


def run_two_stage_pipeline(
    reranker_cls,
    config: PipelineConfig,
    data: PipelineData,
    *,
    track_token_usage: bool,
) -> None:
    from confit_v3.trainer.metric import save_final_results

    print("\n[Step 0] Evaluating ConFit v2 baseline ...")
    baseline_metrics = evaluate_ranking(
        data.initial_ranking,
        data.job_ids,
        data.labels,
        config.bm25_k_values,
        config.eval_k_values,
        "CONFIT v2 BASELINE",
        "model_provider",
        "model_name",
    )
    save_final_results(
        config.output_dir,
        baseline_metrics,
        "model_provider",
        "model_name",
        config.bm25_k_values,
        config.eval_k_values,
    )
    print("  -> Baseline results saved.")

    print("\n[Step 1] Reranking pass-1 (window=4, stride=2, passes=1) ...")
    reranked_pass1 = rerank_parallel(
        reranker_factory=make_reranker_factory(reranker_cls, config, data, 4, 2, 1),
        input_ranking=data.initial_ranking,
        num_workers=config.num_workers,
        desc="Reranking pass-1",
        track_token_usage=track_token_usage,
    )
    print("[Step 1] Evaluating pass-1 results ...")
    pass1_metrics = evaluate_ranking(
        reranked_pass1,
        data.job_ids,
        data.labels,
        config.bm25_k_values,
        config.eval_k_values,
        "POST RERANKING pass-1",
        config.model,
        "pass1",
    )
    save_final_results(
        config.output_dir,
        pass1_metrics,
        config.model,
        "pass1",
        config.bm25_k_values,
        config.eval_k_values,
    )
    print("  -> Pass-1 results saved.")

    print(
        f"\n[Step 2] Reranking pass-2 (window={config.window_size}, stride={config.stride}, passes={config.num_passes}) ..."
    )
    reranked_pass2 = rerank_parallel(
        reranker_factory=make_reranker_factory(
            reranker_cls,
            config,
            data,
            config.window_size,
            config.stride,
            config.num_passes,
        ),
        input_ranking=reranked_pass1,
        num_workers=config.num_workers,
        desc="Reranking pass-2",
        track_token_usage=track_token_usage,
    )
    print("[Step 2] Evaluating pass-2 results ...")
    pass2_metrics = evaluate_ranking(
        reranked_pass2,
        data.job_ids,
        data.labels,
        config.bm25_k_values,
        config.eval_k_values,
        "POST RERANKING pass-2",
        config.model,
        "pass2",
    )
    save_final_results(
        config.output_dir,
        pass2_metrics,
        config.model,
        "pass2",
        config.bm25_k_values,
        config.eval_k_values,
    )
    print("  -> Pass-2 results saved.")
