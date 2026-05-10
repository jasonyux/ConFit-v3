from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import requests

from reranking_clean.impl.list_prompt_based_reranker import ListPromptBasedReranker


class UniversalListPromptRerankerNum(ListPromptBasedReranker):
    LABEL_TO_NUMBER = {"A": "1", "B": "2", "C": "3", "D": "4"}
    NUMBER_TO_LABEL = {value: key for key, value in LABEL_TO_NUMBER.items()}

    def __init__(
        self,
        order_entries_callback,
        get_job_description_from_id,
        get_resume_from_id,
        base_url: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: Optional[str] = None,
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
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.session = requests.Session()
        self.provider = self._infer_provider(model=model, base_url=base_url)
        self.base_url = self._resolve_base_url(self.provider, base_url)
        self.api_key = self._resolve_api_key(self.provider, api_key)

    @staticmethod
    def _infer_provider(model: str, base_url: Optional[str]) -> str:
        model_lower = (model or "").strip().lower()
        base_url_lower = (base_url or "").strip().lower()
        if model_lower.startswith("claude-") or "anthropic" in model_lower:
            return "anthropic"
        if (
            model_lower.startswith("gpt-")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or model_lower.startswith("o4")
            or model_lower.startswith("o5")
        ):
            return "openai"
        if "anthropic.com" in base_url_lower:
            return "anthropic"
        if "api.openai.com" in base_url_lower:
            return "openai"
        return "openai_compatible"

    @staticmethod
    def _resolve_base_url(provider: str, base_url: Optional[str]) -> str:
        if base_url:
            return base_url.rstrip("/")
        if provider == "openai":
            return "https://api.openai.com/v1"
        if provider == "anthropic":
            return "https://api.anthropic.com/v1"
        return "http://localhost:8000/v1"

    @staticmethod
    def _resolve_api_key(provider: str, api_key: Optional[str]) -> str:
        if api_key:
            return api_key
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        if provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        return os.getenv("VLLM_API_KEY", "EMPTY")

    @classmethod
    def _build_messages(
        cls,
        job_description: str,
        labeled_block: List[Tuple[str, str, str]],
    ) -> List[Dict[str, str]]:
        resumes = []
        for _, resume_text, label in labeled_block:
            number = cls.LABEL_TO_NUMBER.get(label, label)
            resumes.append(f"[{number}] Resume {number}:\n{resume_text}")
        resumes_section = "\n\n".join(resumes)

        system = (
            "You are an expert technical recruiter that ranks resumes by match quality against a job description.\n"
            "You analyze each resume, compare them systematically, then provide a final ranking.\n"
            "You will receive exactly 4 resumes, each identified by a numeric label in brackets.\n"
            "Rank them from best fit to worst fit.\n"
            "The preferred final format is:\n"
            "<answer> [1] > [2] > [3] > [4] </answer>\n"
            "If you include analysis before the final answer, make sure the final ranking is still clear and unambiguous."
        )
        user = (
            f"Resumes:\n{resumes_section}\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            "Follow these steps:\n"
            "1. Briefly analyze the job requirements.\n"
            "2. Briefly analyze EACH resume.\n"
            "3. Compare the candidates.\n"
            "4. End with a final ranking from best to worst using the numeric identifiers.\n"
            "Preferred final format:\n"
            "<answer> [1] > [2] > [3] > [4] </answer>\n"
            "Do not output multiple conflicting final rankings."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "openai":
            return self._call_openai_responses(messages)
        if self.provider == "anthropic":
            return self._call_anthropic_messages(messages)
        return self._call_openai_compatible(messages)

    def _call_openai_responses(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_text = []
        user_text = []
        for message in messages:
            if message.get("role") == "system":
                system_text.append(message.get("content", ""))
            else:
                user_text.append(f"{message.get('role', '').upper()}:\n{message.get('content', '')}")

        payload = {
            "model": self.model,
            "input": "\n\n".join(user_text).strip(),
            "max_output_tokens": self.max_tokens,
            "reasoning": {"effort": "minimal"},
            "text": {"verbosity": "low"},
        }
        instructions = "\n\n".join(system_text).strip()
        if instructions:
            payload["instructions"] = instructions

        response = self.session.post(
            f"{self.base_url}/responses",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for {response.url}\n{response.text}",
                response=response,
            )

        data = response.json()
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()

        parts = []
        for item in data.get("output", []):
            for block in item.get("content", []):
                if block.get("type") in {"output_text", "text"} and block.get("text"):
                    parts.append(block["text"])
        return "\n".join(parts).strip()

    def _call_anthropic_messages(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        system_text = None
        anthro_messages = []
        for message in messages:
            if message["role"] == "system":
                system_text = message["content"]
            else:
                anthro_messages.append({"role": message["role"], "content": message["content"]})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthro_messages,
            "temperature": self.temperature,
        }
        if system_text:
            payload["system"] = system_text

        response = self.session.post(
            f"{self.base_url}/messages",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        parts = [block["text"] for block in data.get("content", []) if block.get("type") == "text" and block.get("text")]
        return "\n".join(parts).strip() or str(data)

    def _call_openai_compatible(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
                headers=headers,
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
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    @staticmethod
    def _extract_answer_region(response: str) -> str:
        if not response:
            return ""
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        for pattern in [
            r"final\s*answer\s*[:：]\s*(.*)",
            r"ranking\s*[:：]\s*(.*)",
            r"best\s*to\s*worst\s*[:：]\s*(.*)",
        ]:
            match = re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return response.strip()

    @staticmethod
    def _extract_numeric_order(text: str) -> List[str]:
        if not text:
            return []
        normalized = text.strip().replace("\n", " ").replace("→", ">").replace("—", ">").replace("–", ">").replace(">>", ">")
        candidates = re.findall(r"\[(\d+)\]", normalized)
        if not candidates:
            candidates = re.findall(r"(?:resume|candidate)\s*[\[#\(]?\s*(\d+)\s*[\]#\)]?", normalized, flags=re.IGNORECASE)
        if not candidates:
            candidates = re.findall(r"(?<!\d)(\d+)(?!\d)", normalized)

        seen = set()
        ordered = []
        for candidate in candidates:
            if candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
        return ordered

    @classmethod
    def _numbers_to_labels(cls, numbers: List[str], valid_labels: List[str]) -> List[str]:
        seen = set()
        labels = []
        for number in numbers:
            label = cls.NUMBER_TO_LABEL.get(number)
            if label and label in valid_labels and label not in seen:
                labels.append(label)
                seen.add(label)
        for label in valid_labels:
            if label not in seen:
                labels.append(label)
        return labels[:len(valid_labels)]

    def parse_ranking_robust(self, response: str, valid_labels: List[str]) -> Optional[List[str]]:
        if not response or not valid_labels:
            return None
        answer_region = self._extract_answer_region(response)
        numeric_order = self._extract_numeric_order(answer_region) or self._extract_numeric_order(response)
        labels = self._numbers_to_labels(numeric_order, valid_labels)
        return labels or list(valid_labels)

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
