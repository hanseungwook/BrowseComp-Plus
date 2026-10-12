"""Verify the actual Qwen-8B encoder, published vectors, and corpus lookup."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from searcher.searchers.qwen_faiss_searcher import QwenFaissSearcher  # noqa: E402
from search_agent.horizon_sglang_client import SearchTools  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url", help="Also verify a forced tool call and continuation"
    )
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    args = parser.parse_args()
    assets = json.loads((ROOT / "scripts_sglang/assets.local.json").read_text())
    searcher = QwenFaissSearcher(
        SimpleNamespace(
            index_path=assets["index_glob"],
            embedding_model=assets["embedding"]["path"],
            corpus_path=assets["corpus_path"],
            embedding_device="cuda",
            embedding_dtype="float16",
            query_max_length=8192,
        )
    )
    hits = searcher.search("Which city is the capital of France?", 5)
    assert len(hits) == 5 and len({hit["docid"] for hit in hits}) == 5
    assert all(hit["text"].strip() for hit in hits)
    result = {
        "passed": True,
        "model": assets["embedding"],
        "documents": searcher.index.ntotal,
        "dimensions": searcher.index.d,
        "hits": [{"docid": hit["docid"], "score": hit["score"]} for hit in hits],
    }
    if args.base_url:
        handler = SearchTools(
            searcher, AutoTokenizer.from_pretrained(assets["snippet_tokenizer"]["path"])
        )
        client = OpenAI(
            base_url=args.base_url, api_key="local-sglang", timeout=600, max_retries=0
        )
        messages = [
            {
                "role": "user",
                "content": "Use search once to find the capital of France, then answer in one short sentence.",
            }
        ]
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=handler.definitions(),
            tool_choice={"type": "function", "function": {"name": "search"}},
            reasoning_effort="low",
            temperature=0,
            max_tokens=2048,
        )
        first = response.choices[0]
        if first.finish_reason != "tool_calls" or not first.message.tool_calls:
            raise RuntimeError(
                f"Forced tool-call smoke did not call a tool: {first.finish_reason}"
            )
        message = first.message
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "reasoning_content": getattr(message, "reasoning_content", "") or "",
                "tool_calls": [call.model_dump() for call in message.tool_calls],
            }
        )
        for call in message.tool_calls:
            retrieved = handler.execute(call.function.name, call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": json.dumps(retrieved),
                }
            )
        continuation = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=handler.definitions(),
            reasoning_effort="low",
            temperature=0,
            max_tokens=2048,
        )
        final = continuation.choices[0]
        if final.finish_reason != "stop" or not (final.message.content or "").strip():
            raise RuntimeError(
                f"Tool continuation did not finish: {final.finish_reason}"
            )
        result["tool_protocol"] = {
            "passed": True,
            "forced_search_calls": len(message.tool_calls),
            "answer": final.message.content,
            "autonomous_tool_selection_tested": False,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Qwen-8B retrieval smoke passed", flush=True)


if __name__ == "__main__":
    main()
