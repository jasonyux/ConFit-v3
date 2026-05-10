from abc import ABC, abstractmethod
from typing import Dict


class Reranker(ABC):
    @abstractmethod
    def rerank(self, original_ranking_dict: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        raise NotImplementedError
