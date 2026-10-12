"""Qwen embeddings with the published FAISS shards, using Transformers/SDPA.

Query prefix, EOS pooling, and normalization follow the upstream Tevatron
searcher. This inference-only backend avoids its training and FlashAttention
dependencies. Corpus vectors are the published vectors, not re-encoded here.
"""

import glob
import logging
import pickle

import faiss
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

from .base import BaseSearcher

logger = logging.getLogger(__name__)
TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def pool_query(hidden_states, attention_mask):
    if bool(attention_mask[:, -1].all()):
        reps = hidden_states[:, -1]
    else:
        lengths = attention_mask.sum(dim=1) - 1
        reps = hidden_states[torch.arange(len(lengths), device=lengths.device), lengths]
    return torch.nn.functional.normalize(reps, p=2, dim=-1)


class QwenFaissSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument("--index-path", help="Published corpus.shard*.pkl glob")
        parser.add_argument("--embedding-model", help="Pinned Qwen embedding snapshot")
        parser.add_argument("--corpus-path", help="Dataset saved by prepare_assets.py")
        parser.add_argument("--embedding-device", default="cuda")
        parser.add_argument(
            "--embedding-dtype",
            choices=["float16", "bfloat16", "float32"],
            default="float16",
        )
        parser.add_argument("--query-max-length", type=int, default=8192)

    def __init__(self, args):
        self.args = args
        for name in ("index_path", "embedding_model", "corpus_path"):
            if not getattr(args, name, None):
                raise ValueError(
                    f"Missing {name}; run scripts_sglang/prepare_assets.py"
                )
        self.index = None
        self.lookup = []
        paths = sorted(glob.glob(args.index_path))
        if not paths:
            raise ValueError(f"No embedding shards matched {args.index_path}")
        for path in paths:
            # Only use the pinned, published BrowseComp-Plus index shards.
            with open(path, "rb") as source:
                vectors, ids = pickle.load(source)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(ids):
                raise ValueError(f"Invalid embedding shard: {path}")
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            if vectors.shape[1] != self.index.d:
                raise ValueError("Embedding shards have inconsistent dimensions")
            self.index.add(vectors)
            self.lookup.extend(str(docid) for docid in ids)
            del vectors
        if len(set(self.lookup)) != len(self.lookup):
            raise ValueError("Embedding index contains duplicate document IDs")
        self.corpus = load_from_disk(args.corpus_path)
        self.docid_to_row = {
            str(docid): row for row, docid in enumerate(self.corpus["docid"])
        }
        if set(self.lookup) - self.docid_to_row.keys():
            raise ValueError("Embedding index and corpus document IDs do not match")
        logger.info(
            "Loaded %d documents, dimension %d", self.index.ntotal, self.index.d
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.embedding_model, padding_side="left", trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                args.embedding_model,
                dtype=getattr(torch, args.embedding_dtype),
                attn_implementation="sdpa",
                trust_remote_code=False,
            )
            .to(args.embedding_device)
            .eval()
        )
        if self.model.config.hidden_size != self.index.d:
            raise ValueError("The embedding model does not match the published index")

    def search(self, query, k=5):
        if not isinstance(query, str) or not query.strip() or k < 1:
            raise ValueError("Search requires a nonempty query and a positive k")
        inputs = self.tokenizer(
            TASK_PREFIX + query,
            padding=True,
            truncation=True,
            max_length=self.args.query_max_length,
            return_tensors="pt",
        ).to(self.args.embedding_device)
        with torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            query_vector = (
                pool_query(hidden, inputs.attention_mask).float().cpu().numpy()
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Embedding model produced non-finite values")
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        return [
            {
                **self.get_document(self.lookup[int(index)]),
                "score": float(score),
            }
            for score, index in zip(scores[0], indices[0], strict=True)
        ]

    def get_document(self, docid):
        row = self.docid_to_row.get(str(docid))
        if row is None:
            return None
        return {"docid": str(docid), "text": self.corpus[row]["text"]}

    @property
    def search_type(self):
        return "Qwen3-Embedding / FAISS-HF"
