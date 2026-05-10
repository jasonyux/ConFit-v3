from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import requests

from reranking_clean.impl.list_prompt_based_reranker import ListPromptBasedReranker


class QwenListPromptRerankerNum(ListPromptBasedReranker):
    LABEL_TO_NUMBER = {"A": "1", "B": "2", "C": "3", "D": "4"}
    NUMBER_TO_LABEL = {value: key for key, value in LABEL_TO_NUMBER.items()}

    def __init__(
        self,
        order_entries_callback,
        get_job_description_from_id,
        get_resume_from_id,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: int = 60,
        top_k: Optional[int] = None,
        num_passes: int = 2,
        num_comparison_elements: int = 4,
        window_stride: Optional[int] = None,
    ):
        super().__init__(
            order_entries_callback=order_entries_callback,
            get_job_description_from_id=get_job_description_from_id,
            get_resume_from_id=get_resume_from_id,
            top_k=top_k,
            num_passes=num_passes,
            num_comparison_elements=num_comparison_elements,
            window_stride=window_stride,
        )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'EMPTY')}"}

    @classmethod
    def _build_messages(
        cls,
        job_description: str,
        labeled_block: List[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        resumes = []
        for _, resume_text, label in labeled_block:
            number = cls.LABEL_TO_NUMBER[label]
            resumes.append(f"[{number}] Resume {number}:\n{resume_text}")
        resumes_section = "\n\n".join(resumes)
        valid_labels = [str(index + 1) for index in range(len(labeled_block))]
        answer_format = " > ".join(f"[{label}]" for label in valid_labels)

        system = (
            "You are an expert technical recruiter that can rank resumes based on their matching degree "
            "to the job description. You first analyze each resume individually, then compare them "
            "systematically, and finally provide the ranking. "
            f"I will provide you with {len(labeled_block)} resumes, each indicated by a numeric identifier []. "
            f"Rank the {len(labeled_block)} resumes based on their matching degree to "
            "the job description. The resumes should be listed in descending order using identifiers. "
            "The most relevant resumes should be listed first. "
            f"The output format should be <answer> {answer_format} </answer>."
        )
        user = (
            f"Resumes:\n{resumes_section}\n\n"
            f"Please rank these resumes according to their matching degree to the JOB DESCRIPTION: [{job_description}]\n"
            "Follow these steps exactly:\n"
            "1. First, think to summarize the job description and analyze EACH resume briefly: Evaluate how well it matches the job description and mandatory criteria.\n"
            "2. Then, think to COMPARE the resumes and determine which candidates are better fits and why.\n"
            "3. Finally, within <answer> tags, provide ONLY the final ranking of the resumes from best to worst fit using their numerical identifiers.\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        chunks = [f"{message['role'].upper()}:\n{message['content']}" for message in messages]
        return "\n\n".join(chunks) + "\n\nASSISTANT:\n"

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        response = None
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError:
            if response is None or response.status_code not in (404, 405):
                raise

        payload = {
            "model": self.model,
            "prompt": self._messages_to_prompt(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self.session.post(
            f"{self.base_url}/completions",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    @classmethod
    def parse_ranking_robust(cls, response: str, valid_labels: List[str]) -> Optional[List[str]]:
        if not response or not valid_labels:
            return None

        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
        answer_content = answer_match.group(1).strip() if answer_match else response
        found_numbers = re.findall(r"\[(\d+)\]", answer_content)

        seen = set()
        ordered_labels: List[str] = []
        for number in found_numbers:
            label = cls.NUMBER_TO_LABEL.get(number)
            if label in valid_labels and label not in seen:
                ordered_labels.append(label)
                seen.add(label)

        for label in valid_labels:
            if label not in seen:
                ordered_labels.append(label)

        return ordered_labels

    def _rank_block(
        self,
        job_description: str,
        labeled_block: List[Tuple[str, str, str]],
    ) -> List[str]:
        messages = self._build_messages(job_description, labeled_block)
        content = self._call_llm(messages)
        allowed_labels = [label for _, _, label in labeled_block]
        labels_in_order = self.parse_ranking_robust(content, allowed_labels) or list(allowed_labels)
        label_to_id = {label: resume_id for resume_id, _, label in labeled_block}
        return [label_to_id[label] for label in labels_in_order if label in label_to_id]
