"""Reward functions and utilities for listwise ranking tasks.

The expected model output contains a ranking inside optional ``<answer>`` tags,
for example:

    <answer>[1] > [2] > [3] > [4]</answer>

Labels can be numeric (``[1]``) or uppercase alphabetic (``[A]``), depending on
``valid_labels`` provided in ``extra_info``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_NUMERIC_LABELS = ["1", "2", "3", "4"]


# -----------------------------------------------------------------------------
# Parsing utilities
# -----------------------------------------------------------------------------

def _get_valid_labels(extra_info: Optional[Dict[str, Any]], default: Sequence[str]) -> List[str]:
    """Read valid labels from ``extra_info`` with a safe fallback."""
    if extra_info and isinstance(extra_info.get("valid_labels"), list):
        return [str(label) for label in extra_info["valid_labels"]]
    return list(default)


def _get_accepted_labels(ground_truth: Any) -> List[str]:
    """Normalize accepted labels from the reward ground truth."""
    if isinstance(ground_truth, list):
        return [str(label) for label in ground_truth]
    return []


def extract_answer_content(response: str) -> tuple[str, bool]:
    """Extract text inside ``<answer>`` tags, or return the full response."""
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip(), True
    return response, False


def _label_pattern(valid_labels: Sequence[str]) -> str:
    """Choose a bracketed-label regex based on the expected label type."""
    if valid_labels and all(str(label).isdigit() for label in valid_labels):
        return r"\[(\d+)\]"
    return r"\[([A-Z])\]"


def parse_ranking_robust(response: str, valid_labels: List[str]) -> Optional[List[str]]:
    """Parse a ranking from model output.

    The parser first looks inside ``<answer>`` tags. If no tags are found, it
    falls back to the entire response. Duplicate labels are removed while
    preserving the first occurrence. Parsing succeeds only when all valid labels
    appear exactly once after filtering.
    """
    if not response or not valid_labels:
        return None

    answer_content, _ = extract_answer_content(response)
    found_labels = re.findall(_label_pattern(valid_labels), answer_content)

    seen = set()
    filtered_labels = []
    for label in found_labels:
        if label in valid_labels and label not in seen:
            filtered_labels.append(label)
            seen.add(label)

    if len(filtered_labels) != len(valid_labels):
        return None
    if set(filtered_labels) != set(valid_labels):
        return None
    return filtered_labels


def parse_ranking_with_diagnostics(response: str, valid_labels: List[str]) -> Dict[str, Any]:
    """Parse a ranking and return diagnostics useful for debugging."""
    if not response or not valid_labels:
        return {
            "ranking": None,
            "answer_content": "",
            "found_labels": [],
            "filtered_labels": [],
            "is_valid": False,
            "has_answer_tags": False,
            "error": "Empty response or valid_labels",
        }

    answer_content, has_answer_tags = extract_answer_content(response)
    found_labels = re.findall(_label_pattern(valid_labels), answer_content)

    seen = set()
    filtered_labels = []
    for label in found_labels:
        if label in valid_labels and label not in seen:
            filtered_labels.append(label)
            seen.add(label)

    is_valid = len(filtered_labels) == len(valid_labels) and set(filtered_labels) == set(valid_labels)

    error = None
    if not has_answer_tags:
        error = "No <answer> tags found"
    elif not is_valid:
        if len(filtered_labels) != len(valid_labels):
            error = f"Found {len(filtered_labels)} labels, expected {len(valid_labels)}"
        else:
            error = f"Labels mismatch: found {set(filtered_labels)}, expected {set(valid_labels)}"

    return {
        "ranking": filtered_labels if is_valid else None,
        "answer_content": answer_content,
        "found_labels": found_labels,
        "filtered_labels": filtered_labels,
        "is_valid": is_valid,
        "has_answer_tags": has_answer_tags,
        "error": error,
    }


# -----------------------------------------------------------------------------
# Ranking metrics
# -----------------------------------------------------------------------------

def compute_positive_at_top_percentage(
    predicted_ranking: List[str],
    accepted_labels: List[str],
    valid_labels: List[str],
) -> float:
    """Return the percentage of positives ranked before every negative."""
    if not accepted_labels:
        return 0.0

    non_accepted_labels = [label for label in valid_labels if label not in accepted_labels]
    if not non_accepted_labels:
        return 100.0

    non_accepted_positions = [predicted_ranking.index(label) for label in non_accepted_labels]
    first_negative_position = min(non_accepted_positions)

    positives_before_negatives = sum(
        1 for label in accepted_labels if predicted_ranking.index(label) < first_negative_position
    )
    return (positives_before_negatives / len(accepted_labels)) * 100.0


def check_perfect_accuracy(
    predicted_ranking: List[str],
    accepted_labels: List[str],
    valid_labels: List[str],
) -> bool:
    """Check whether all positives are ranked before all negatives."""
    if not accepted_labels:
        return True

    non_accepted_labels = [label for label in valid_labels if label not in accepted_labels]
    if not non_accepted_labels:
        return True

    accepted_positions = [predicted_ranking.index(label) for label in accepted_labels]
    non_accepted_positions = [predicted_ranking.index(label) for label in non_accepted_labels]

    return max(accepted_positions) < min(non_accepted_positions)


def calculate_ndcg_at_k(ranking: List[str], accepted_labels: List[str], k: int = 4) -> float:
    """Calculate NDCG@k for binary relevance labels."""
    if not ranking or k <= 0:
        return 0.0

    ranking_at_k = ranking[: min(k, len(ranking))]

    dcg = 0.0
    for idx, label in enumerate(ranking_at_k):
        relevance = 1.0 if label in accepted_labels else 0.0
        dcg += relevance / math.log2(idx + 2)

    idcg = 0.0
    for idx in range(min(len(accepted_labels), k)):
        idcg += 1.0 / math.log2(idx + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _best_positive_rank(predicted_ranking: List[str], accepted_labels: List[str]) -> int:
    """Return the best positive rank as a 1-based index, or 0 if absent."""
    if not accepted_labels:
        return 0

    positions = [predicted_ranking.index(label) for label in accepted_labels if label in predicted_ranking]
    return min(positions) + 1 if positions else 0


def _parse_failure_result(solution_str: str, valid_labels: List[str], best_positive_rank: Optional[int] = None) -> Dict[str, Any]:
    """Create the common parse-failure reward payload."""
    return {
        "score": -1.0,
        "accuracy": 0,
        "format_crash": 1,
        "answer_length": len(solution_str),
        "positive_at_top_pct": 0.0,
        "best_positive_rank": best_positive_rank if best_positive_rank is not None else len(valid_labels) + 1,
    }


# -----------------------------------------------------------------------------
# Reward functions
# -----------------------------------------------------------------------------

def my_reward_fn_ranking_perfect_accuracy(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Binary reward: 1.0 when all positives are before all negatives."""
    valid_labels = _get_valid_labels(extra_info, DEFAULT_NUMERIC_LABELS)
    accepted_labels = _get_accepted_labels(ground_truth)
    predicted_ranking = parse_ranking_robust(solution_str, valid_labels)

    if predicted_ranking is None:
        return _parse_failure_result(solution_str, valid_labels)

    positive_at_top_pct = compute_positive_at_top_percentage(
        predicted_ranking, accepted_labels, valid_labels
    )
    is_perfect = check_perfect_accuracy(predicted_ranking, accepted_labels, valid_labels)

    return {
        "score": 1.0 if is_perfect else 0.0,
        "accuracy": 1 if is_perfect else 0,
        "format_crash": 0,
        "answer_length": len(solution_str),
        "positive_at_top_pct": positive_at_top_pct,
        "best_positive_rank": _best_positive_rank(predicted_ranking, accepted_labels),
    }


def my_reward_fn_ranking_continuous(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Continuous reward equal to ``positive_at_top_pct / 100``."""
    valid_labels = _get_valid_labels(extra_info, DEFAULT_NUMERIC_LABELS)
    accepted_labels = _get_accepted_labels(ground_truth)
    predicted_ranking = parse_ranking_robust(solution_str, valid_labels)

    if predicted_ranking is None:
        result = _parse_failure_result(solution_str, valid_labels, best_positive_rank=0)
        result.pop("best_positive_rank")
        return result

    positive_at_top_pct = compute_positive_at_top_percentage(
        predicted_ranking, accepted_labels, valid_labels
    )

    return {
        "score": positive_at_top_pct / 100.0,
        "accuracy": 1 if positive_at_top_pct == 100.0 else 0,
        "format_crash": 0,
        "answer_length": len(solution_str),
        "positive_at_top_pct": positive_at_top_pct,
    }


def my_reward_fn_ranking_top2_hit(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reward based on whether the best positive appears at rank 1 or rank 2."""
    valid_labels = _get_valid_labels(extra_info, DEFAULT_NUMERIC_LABELS)
    accepted_labels = _get_accepted_labels(ground_truth)
    predicted_ranking = parse_ranking_robust(solution_str, valid_labels)

    if predicted_ranking is None:
        return _parse_failure_result(solution_str, valid_labels, best_positive_rank=0)

    best_positive_rank = _best_positive_rank(predicted_ranking, accepted_labels)
    if best_positive_rank == 1:
        reward = 1.0
    elif best_positive_rank == 2:
        reward = 0.5
    else:
        reward = 0.0

    positive_at_top_pct = compute_positive_at_top_percentage(
        predicted_ranking, accepted_labels, valid_labels
    )
    is_perfect = check_perfect_accuracy(predicted_ranking, accepted_labels, valid_labels)

    return {
        "score": reward,
        "accuracy": 1 if is_perfect else 0,
        "format_crash": 0,
        "answer_length": len(solution_str),
        "positive_at_top_pct": positive_at_top_pct,
        "best_positive_rank": best_positive_rank,
    }


def my_reward_fn_rearank(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reward based on NDCG@4 improvement over the original label order."""
    valid_labels = _get_valid_labels(extra_info, DEFAULT_NUMERIC_LABELS)
    accepted_labels = _get_accepted_labels(ground_truth)
    difficulty = extra_info.get("acc", 0) if extra_info else 0

    baseline_ranking = valid_labels
    predicted_ranking = parse_ranking_robust(solution_str, valid_labels)

    if predicted_ranking is None:
        return {
            "score": -1.0,
            "accuracy": 0,
            "format_crash": 1,
            "answer_length": len(solution_str),
            "old_ndcg": 0.0,
            "new_ndcg": 0.0,
            "ndcg_improvement": 0.0,
            "best_positive_rank": len(valid_labels) + 1,
            "is_hard": 1 if difficulty == 0 else 0,
            "hard_acc": 0,
        }

    old_ndcg = calculate_ndcg_at_k(baseline_ranking, accepted_labels, k=len(valid_labels))
    new_ndcg = calculate_ndcg_at_k(predicted_ranking, accepted_labels, k=len(valid_labels))
    ndcg_improvement = new_ndcg - old_ndcg

    upperbound = 1.0
    if abs(upperbound - old_ndcg) < 1e-9:
        reward = 1.0 if ndcg_improvement >= 0 else -1.0
    else:
        reward = ndcg_improvement / (upperbound - old_ndcg)

    is_perfect = check_perfect_accuracy(predicted_ranking, accepted_labels, valid_labels)

    return {
        "score": reward,
        "accuracy": 1 if is_perfect else 0,
        "format_crash": 0,
        "answer_length": len(solution_str),
        "old_ndcg": old_ndcg,
        "new_ndcg": new_ndcg,
        "ndcg_improvement": ndcg_improvement,
        "best_positive_rank": _best_positive_rank(predicted_ranking, accepted_labels),
        "is_hard": 1 if difficulty == 0 else 0,
        "hard_acc": 1 if difficulty == 0 and is_perfect else 0,
    }
