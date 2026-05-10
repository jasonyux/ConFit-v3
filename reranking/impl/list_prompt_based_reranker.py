from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from reranking_clean.interface.reranker import Reranker
from reranking_clean.query_store import query_store


class ListPromptBasedReranker(Reranker):
    def __init__(
        self,
        order_entries_callback,
        get_job_description_from_id,
        get_resume_from_id,
        top_k: Optional[int] = None,
        num_passes: int = 1,
        num_comparison_elements: int = 3,
        window_stride: Optional[int] = None,
    ):
        assert num_comparison_elements >= 2, "num_comparison_elements must be >= 2"
        self.top_k = top_k
        self.num_passes = max(1, int(num_passes))
        self.k = int(num_comparison_elements)
        self.stride = int(window_stride) if window_stride is not None else 1
        self.order_entries_by_rank_callback = order_entries_callback
        self.get_job_description_from_id = get_job_description_from_id
        self.get_resume_from_id = get_resume_from_id

    def _rank_block(
        self,
        job_description: str,
        labeled_block: List[Tuple[str, str, str]],
    ) -> List[str]:
        raise NotImplementedError

    def _log_after_window(
        self,
        *,
        job_id: str,
        job_description: str,
        labeled_block: List[Tuple[str, str, str]],
        final_window_order_ids: List[str],
    ) -> None:
        resume_ids_in_prompt_order = [resume_id for resume_id, _, _ in labeled_block]
        try:
            query_store.record_query(
                job_id=job_id,
                job_description=job_description,
                resume_ids_in_prompt_order=resume_ids_in_prompt_order,
                ordered_resume_ids=list(final_window_order_ids),
                model_name=getattr(self, "model", None),
                prompt_style=getattr(self, "prompt_style", None),
                messages=None,
                response_text=None,
            )
        except Exception:
            pass

    def rerank(self, original_ranking_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        lists = self.order_entries_by_rank_callback(original_ranking_dict, self.top_k)
        out_lists: Dict[str, List[str]] = {}

        count = 0
        for job_id, resume_ids in tqdm(lists.items(), desc="Reranking jobs", unit="job"):
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} jobs...")

            if len(resume_ids) < 2:
                out_lists[job_id] = resume_ids[:]
                continue

            job_description = self.get_job_description_from_id(job_id)
            resume_cache: Dict[str, str] = {}

            def get_resume_text(resume_id: str) -> str:
                if resume_id not in resume_cache:
                    resume_cache[resume_id] = self.get_resume_from_id(resume_id)
                return resume_cache[resume_id]

            current_ids = resume_ids[:]
            total = len(current_ids)
            window_size = min(self.k, max(2, total))
            stride = max(1, self.stride)

            for _ in range(self.num_passes):
                start = max(0, total - window_size)
                while start >= 0:
                    end = min(start + window_size, total)
                    if end - start < 2:
                        break

                    window_ids = current_ids[start:end]
                    labels = [chr(ord("A") + idx) for idx in range(len(window_ids))]
                    labeled_block = [
                        (resume_id, get_resume_text(resume_id), label)
                        for resume_id, label in zip(window_ids, labels)
                    ]

                    try:
                        ordered_ids = self._rank_block(job_description, labeled_block)
                        remaining_ids = [resume_id for resume_id in window_ids if resume_id not in ordered_ids]
                        merged_ids = [resume_id for resume_id in ordered_ids if resume_id in window_ids] + remaining_ids
                        if len(merged_ids) == len(window_ids):
                            current_ids[start:end] = merged_ids

                        self._log_after_window(
                            job_id=job_id,
                            job_description=job_description,
                            labeled_block=labeled_block,
                            final_window_order_ids=current_ids[start:end],
                        )
                    except Exception as exc:
                        print(f"Block ranking failed for job {job_id} [{start}:{end}]: {exc}")

                    start -= stride

            out_lists[job_id] = current_ids

        print("Reranking complete.")
        return {
            job_id: {resume_id: rank for rank, resume_id in enumerate(resume_ids)}
            for job_id, resume_ids in out_lists.items()
        }
