from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_STORE: Dict[str, Dict[str, Any]] = {}
_MAX_EXAMPLES_PER_KEY = 3
_NUMBER_INSERTIONS = 0


def _key(job_description: str, resume_ids: List[str]) -> str:
    jd_hash = hashlib.sha256(job_description.encode("utf-8")).hexdigest()[:16]
    ids_part = "|".join(sorted(str(rid) for rid in resume_ids))
    return f"{jd_hash}::set[{ids_part}]"


def record_query(
    *,
    job_id: str,
    job_description: str,
    resume_ids_in_prompt_order: List[str],
    ordered_resume_ids: List[str],
    model_name: Optional[str] = None,
    prompt_style: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    response_text: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    key = _key(job_description, resume_ids_in_prompt_order)
    now = time.time()

    global _NUMBER_INSERTIONS
    _NUMBER_INSERTIONS += 1
    if _NUMBER_INSERTIONS % 200 == 0:
        print(f"[QueryStore] Number of insertions: {_NUMBER_INSERTIONS}")

    example: Dict[str, Any] = {
        "ts": now,
        "job_id": job_id,
        "model": model_name,
        "prompt_style": prompt_style,
        "resume_ids_in_prompt_order": list(resume_ids_in_prompt_order),
        "ordered_resume_ids": list(ordered_resume_ids),
    }
    if messages is not None:
        example["messages"] = messages
    if response_text is not None:
        example["response_text"] = response_text
    if extra:
        example["extra"] = extra

    with _LOCK:
        record = _STORE.get(key)
        if record is None:
            record = {
                "key": key,
                "job_description": job_description,
                "resume_ids_set": sorted(set(str(rid) for rid in resume_ids_in_prompt_order)),
                "num_calls": 0,
                "first_ts": now,
                "last_ts": now,
                "last_ordering": list(ordered_resume_ids),
                "examples": [],
            }

        record["num_calls"] = int(record.get("num_calls", 0)) + 1
        record["last_ts"] = now
        record["last_ordering"] = list(ordered_resume_ids)

        examples = list(record.get("examples", []))
        examples.append(example)
        record["examples"] = examples[-_MAX_EXAMPLES_PER_KEY:]
        _STORE[key] = record


def snapshot() -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        return dict(_STORE)


def dump_json(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"records": list(snapshot().values())}, handle, ensure_ascii=False, indent=2)


def clear() -> None:
    with _LOCK:
        _STORE.clear()
