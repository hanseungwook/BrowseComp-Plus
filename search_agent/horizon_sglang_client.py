"""Run pretrained Horizon with local Qwen retrieval and optional session eviction."""

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path

from openai import BadRequestError, OpenAI
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search_agent.prompts import format_query  # noqa: E402
from search_agent.horizon_template import generation_template  # noqa: E402
from search_agent.horizon_context import ContextLimitError, ContextPolicy  # noqa: E402
from searcher.searchers.qwen_faiss_searcher import QwenFaissSearcher  # noqa: E402


class ToolInputError(ValueError):
    """An invalid model-generated tool call, rather than a retrieval failure."""


class SearchTools:
    def __init__(self, searcher, tokenizer, *, k=5, snippet_max_tokens=512):
        self.searcher = searcher
        self.tokenizer = tokenizer
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.lock = threading.Lock()

    def definitions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def execute(self, name, raw_arguments):
        if name != "search":
            raise ToolInputError(f"Unknown tool: {name}")
        try:
            arguments = json.loads(raw_arguments)
        except (ValueError, TypeError) as exc:
            raise ToolInputError("search arguments must be a JSON object") from exc
        if not isinstance(arguments, dict) or set(arguments) != {"query"}:
            raise ToolInputError("search accepts exactly one argument: query")
        if not isinstance(arguments["query"], str) or not arguments["query"].strip():
            raise ToolInputError("query must be a nonempty string")
        # A single encoder is shared across concurrent conversations on the GPU.
        with self.lock:
            return self._search(arguments["query"])

    def _search(self, query):
        candidates = self.searcher.search(query, self.k)
        results = []
        for item in candidates:
            tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
            snippet = item["text"]
            if len(tokens) > self.snippet_max_tokens:
                snippet = self.tokenizer.decode(
                    tokens[: self.snippet_max_tokens], skip_special_tokens=True
                )
            results.append(
                {
                    "docid": str(item["docid"]),
                    "score": item["score"],
                    "snippet": snippet,
                }
            )
        return results


def run_query(client, tool_handler, *, query_id, query, args, metadata):
    session = client.new_conversation() if hasattr(client, "new_conversation") else None
    if session is not None:
        client = session
    policy = (
        ContextPolicy(
            session,
            final_answer_tokens=getattr(args, "final_answer_tokens", 32768),
            context_margin=getattr(args, "context_margin", 32768),
        )
        if session is not None and session.factory.context_length is not None
        else None
    )
    tools = tool_handler.definitions()
    messages = [
        {
            "role": "user",
            "content": format_query(query, "QUERY_TEMPLATE_NO_GET_DOCUMENT"),
        }
    ]
    result = []
    retrieved = set()
    counts = Counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = "max_iterations"
    error = None
    invalid_response = None
    started = time.monotonic()
    try:
        for iteration in count():
            if args.max_iterations and iteration >= args.max_iterations:
                break
            requested_tokens = (
                policy.prepare_turn(messages, tools, args.max_tokens)
                if policy
                else args.max_tokens
            )
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=tools,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=requested_tokens,
                seed=args.seed,
            )
            if response.usage is None:
                raise RuntimeError(
                    "The server omitted token usage; cannot record generation usage"
                )
            for key in usage:
                usage[key] += getattr(response.usage, key, 0) or 0
            choice = response.choices[0]
            message = choice.message
            reasoning = getattr(message, "reasoning_content", None) or ""
            assistant = {
                "role": "assistant",
                "content": message.content or "",
                # Horizon's template requires this field. Preserve prior
                # reasoning instead of silently turning this into a W0 run.
                "reasoning_content": reasoning,
            }
            if message.tool_calls:
                assistant["tool_calls"] = [
                    call.model_dump() for call in message.tool_calls
                ]
            messages.append(assistant)
            if reasoning:
                result.append({"type": "reasoning", "output": reasoning})
            if message.content:
                result.append({"type": "output_text", "output": message.content})
            if choice.finish_reason == "length":
                status = (
                    "turn_token_limit"
                    if requested_tokens == args.max_tokens
                    else "context_limit"
                )
                break
            if choice.finish_reason not in {"stop", "tool_calls"}:
                # A session stopped by endoftext rather than im_end reports
                # an unfinished assistant. Parsed tool calls do not authorize
                # another turn: count the model output as invalid, without
                # executing its calls, repairing its tokens, or retrying it.
                status = "empty_or_invalid_response"
                invalid_response = {"finish_reason": choice.finish_reason}
                if session is not None and session.turns:
                    turn = session.turns[-1]
                    invalid_response.update(
                        reason="unfinished_assistant_message",
                        session_completed=turn["cache"]["completed"],
                        server_finish_reason=turn["finish_reason"],
                    )
                break
            if message.tool_calls:
                if policy and policy.prompted:
                    # The final-answer request is an instruction, not a promise
                    # that the model obeys. Never resume searching or mark a
                    # returned tool call as a successful final answer.
                    status = "final_answer_tool_call"
                    break
                for call in message.tool_calls:
                    name = call.function.name
                    counts[name] += 1
                    result.append(
                        {
                            "type": "tool_call",
                            "tool_name": name,
                            "arguments": call.function.arguments,
                            "tool_call_id": call.id,
                        }
                    )
                    hits = []
                    if policy and policy.reason:
                        tool_message = policy.stop_search_message(call)
                    else:
                        try:
                            hits = tool_handler.execute(name, call.function.arguments)
                            output = json.dumps(hits, ensure_ascii=False, indent=2)
                        except ToolInputError as exc:
                            output = json.dumps({"error": str(exc)})
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": name,
                            "content": output,
                        }
                        if policy and not policy.tool_result_fits(
                            messages,
                            tools,
                            tool_message,
                            args.max_tokens,
                        ):
                            if hits:
                                tool_message, hits = policy.fit_search_results(
                                    messages, tools, call, hits, args.max_tokens
                                )
                            else:
                                tool_message = policy.stop_search_message(call)
                    if (
                        policy
                        and policy.reason
                        and not policy.tool_result_fits(
                            messages,
                            tools,
                            tool_message,
                            args.max_tokens,
                        )
                    ):
                        # A very large batch of tool calls can make even the
                        # small refusal messages too long. The next user turn
                        # cancels pending searches; cached history is untouched.
                        break
                    retrieved.update(hit["docid"] for hit in hits)
                    messages.append(tool_message)
                    result.append(
                        {
                            "type": "tool_result",
                            "tool_name": name,
                            "tool_call_id": call.id,
                            "output": tool_message["content"],
                        }
                    )
                continue
            status = (
                "completed"
                if choice.finish_reason == "stop" and assistant["content"].strip()
                else "empty_or_invalid_response"
            )
            break
    except ContextLimitError as exc:
        status = "context_limit"
        error = str(exc)
    except BadRequestError as exc:
        # Context exhaustion is a per-question budget failure. Other bad
        # requests may indicate a broken template or client and must stop.
        context_error = (
            "model's context length" in str(exc)
            or "model's maximum context length" in str(exc)
            or "original context-position limit" in str(exc)
        )
        status = "context_limit" if context_error else "error"
        error = str(exc)
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                status = "error"
                error = f"{error or ''} Session cleanup failed: {exc}".strip()
    return {
        "metadata": metadata,
        "query_id": query_id,
        "tool_call_counts": dict(counts),
        "usage": usage,
        "status": status,
        "error": error,
        "retrieved_docids": sorted(retrieved),
        "result": result,
        "messages": messages,
        "elapsed_seconds": time.monotonic() - started,
        **({"invalid_response": invalid_response} if invalid_response else {}),
        **({"session_turns": session.turns} if session is not None else {}),
        **({"finalization": policy.summary()} if policy is not None else {}),
    }


def save_result(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resume_metadata_matches(existing, current, compatible_agent_hashes=()):
    """Allow an explicitly reviewed client-only fix; compare all other settings."""
    if existing.get("configuration_sha256") == current["configuration_sha256"]:
        return True
    if existing.get("agent_sha256") not in compatible_agent_hashes:
        return False
    omitted = {"agent_sha256", "configuration_sha256"}
    return {k: v for k, v in existing.items() if k not in omitted} == {
        k: v for k, v in current.items() if k not in omitted
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--context-length", type=int)
    parser.add_argument(
        "--final-answer-tokens",
        type=int,
        default=32768,
        help="Context reserved for the final answer when appending tool results",
    )
    parser.add_argument("--context-margin", type=int, default=32768)
    parser.add_argument(
        "--condition",
        choices=["chat", "normal", "w1", "w8"],
        default="chat",
        help="Comparison conditions share the token-ID session path; chat is the legacy baseline",
    )
    parser.add_argument("--horizon-tokenizer", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent conversations sharing one query encoder",
    )
    parser.add_argument(
        "--server-metadata",
        type=Path,
        help="Server configuration recorded by run_local.py",
    )
    parser.add_argument(
        "--queries", type=Path, default=ROOT / "topics-qrels/queries.tsv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/qwen3-8b/horizon-0.9b-full"
    )
    parser.add_argument(
        "--assets", type=Path, default=ROOT / "scripts_sglang/assets.local.json"
    )
    parser.add_argument("--snippet-tokenizer")
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Generated-token limit per turn, including reasoning (default: 32768)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Optional generation-step limit for smoke tests; 0 means unlimited",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of queries; 0 means all"
    )
    parser.add_argument(
        "--query-id", action="append", help="Restrict to a query ID; repeatable"
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite previous results"
    )
    parser.add_argument(
        "--resume-compatible-agent-sha256",
        action="append",
        default=[],
        help="Explicitly accept a reviewed prior client hash on resume; all other metadata must match",
    )
    QwenFaissSearcher.parse_args(parser)
    preliminary, _ = parser.parse_known_args()
    assets = json.loads(preliminary.assets.read_text())
    parser.set_defaults(
        embedding_model=assets["embedding"]["path"],
        snippet_tokenizer=assets["snippet_tokenizer"]["path"],
        index_path=assets["index_glob"],
        corpus_path=assets["corpus_path"],
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("workers must be between 1 and the server limit of 4")
    if args.condition != "chat" and args.horizon_tokenizer is None:
        parser.error("Session conditions require --horizon-tokenizer")
    if args.condition != "chat":
        native_context = json.loads(
            (args.horizon_tokenizer / "config.json").read_text()
        )["max_position_embeddings"]
        if args.context_length is None:
            args.context_length = native_context
        if not 0 < args.context_length <= native_context:
            parser.error("context-length must be within the model's native context")
        if (
            not 0
            < args.final_answer_tokens
            <= args.context_margin
            < args.context_length
        ):
            parser.error(
                "Require 0 < final-answer-tokens <= context-margin < context-length"
            )
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0, num-shards)")
    if (
        min(args.max_tokens, args.k, args.snippet_max_tokens) < 1
        or args.max_iterations < 0
        or args.limit < 0
    ):
        parser.error(
            "Turn budget, k, and snippet length must be positive; limits must be nonnegative"
        )
    logging.basicConfig(level=logging.INFO)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key
        not in {
            "base_url",
            "limit",
            "query_id",
            "num_shards",
            "shard_id",
            "force",
            "output_dir",
            "assets",
            "queries",
            "server_metadata",
            "resume_compatible_agent_sha256",
        }
    }
    settings["asset_revisions"] = assets["revisions"]
    if args.server_metadata:
        settings["server"] = json.loads(args.server_metadata.read_text())
    settings["queries_sha256"] = hashlib.sha256(args.queries.read_bytes()).hexdigest()
    settings["query_template"] = "QUERY_TEMPLATE_NO_GET_DOCUMENT"
    settings["reasoning_effort"] = "high"
    settings["generation_budget_scope"] = "per_turn"
    settings["total_generation_budget"] = None
    settings["search_result_policy"] = (
        "untrimmed" if args.condition == "chat" else "ranked-prefix-trimming-v1"
    )
    settings["agent_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings["mode"] = (
        "full-history-chat-completions"
        if args.condition == "chat"
        else "chronological-token-id-sessions"
    )
    if args.condition != "chat":
        template = generation_template(args.horizon_tokenizer)
        settings["chat_template_sha256"] = hashlib.sha256(
            template.read_bytes()
        ).hexdigest()
        settings["session_adapter_sha256"] = hashlib.sha256(
            (ROOT / "search_agent/horizon_sessions.py").read_bytes()
        ).hexdigest()
        settings["context_policy_sha256"] = hashlib.sha256(
            (ROOT / "search_agent/horizon_context.py").read_bytes()
        ).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    metadata = {**settings, "configuration_sha256": fingerprint}
    with args.queries.open(newline="") as source:
        queries = [
            (row[0], row[1])
            for row in csv.reader(source, delimiter="\t")
            if len(row) >= 2
        ]
    if args.query_id:
        queries = [(qid, text) for qid, text in queries if qid in set(args.query_id)]
        missing = set(args.query_id) - {qid for qid, _ in queries}
        if missing:
            parser.error(f"Unknown query IDs: {sorted(missing)}")
    queries = [
        row
        for index, row in enumerate(queries)
        if index % args.num_shards == args.shard_id
    ]
    if args.limit:
        queries = queries[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for qid, text in queries:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", qid):
            raise ValueError("Query ID cannot be used as a result filename")
        path = args.output_dir / f"query_{qid}.json"
        if path.exists() and not args.force:
            existing = json.loads(path.read_text())
            if not resume_metadata_matches(
                existing.get("metadata", {}),
                metadata,
                args.resume_compatible_agent_sha256,
            ):
                raise ValueError(
                    "Output directory contains a different configuration; select another directory"
                )
            if existing.get("status") != "error":
                continue
        pending.append((qid, text, path))
    if not pending:
        print("No pending queries")
        return
    searcher = QwenFaissSearcher(args)
    snippet_tokenizer = AutoTokenizer.from_pretrained(
        args.snippet_tokenizer, trust_remote_code=False
    )
    handler = SearchTools(
        searcher,
        snippet_tokenizer,
        k=args.k,
        snippet_max_tokens=args.snippet_max_tokens,
    )
    if args.condition == "chat":
        client = OpenAI(
            base_url=args.base_url, api_key="local-sglang", timeout=1800, max_retries=0
        )
    else:
        from search_agent.horizon_sessions import SessionFactory

        tokenizer = AutoTokenizer.from_pretrained(
            args.horizon_tokenizer, trust_remote_code=False
        )
        tokenizer.chat_template = generation_template(
            args.horizon_tokenizer
        ).read_text()
        window = {"normal": None, "w1": 1, "w8": 8}[args.condition]
        client = SessionFactory(
            args.base_url, tokenizer, window=window, context_length=args.context_length
        )

    def execute(item):
        qid, query, path = item
        record = run_query(
            client, handler, query_id=qid, query=query, args=args, metadata=metadata
        )
        save_result(path, record)
        print(
            json.dumps(
                {
                    "query_id": qid,
                    "status": record["status"],
                    "tools": record["tool_call_counts"],
                    "tokens": record["usage"]["completion_tokens"],
                }
            ),
            flush=True,
        )
        if record["status"] == "error":
            raise RuntimeError(f"Query {qid} failed; see {path}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, item) for item in pending]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


if __name__ == "__main__":
    main()
