"""
Searchers package for different search implementations.
"""

from enum import Enum
from importlib import import_module

from .base import BaseSearcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = ("bm25", ".bm25_searcher", "BM25Searcher")
    FAISS = ("faiss", ".faiss_searcher", "FaissSearcher")
    FAISS_HF = ("faiss-hf", ".qwen_faiss_searcher", "QwenFaissSearcher")
    REASONIR = ("reasonir", ".faiss_searcher", "ReasonIrSearcher")
    CUSTOM = ("custom", ".custom_searcher", "CustomSearcher")

    def __init__(self, cli_name, module_name, class_name):
        self.cli_name = cli_name
        self.module_name = module_name
        self.class_name = class_name

    @property
    def searcher_class(self):
        # Import only the selected backend: dense retrieval does not need Java,
        # Pyserini, or training frameworks used by other searchers.
        return getattr(import_module(self.module_name, __name__), self.class_name)

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]


def __getattr__(name):
    for backend in SearcherType:
        if name == backend.class_name:
            return backend.searcher_class
    raise AttributeError(name)
