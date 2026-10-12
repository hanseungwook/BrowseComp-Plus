"""Exercise real GPU compaction, exact token deltas, and concurrent sessions."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search_agent.horizon_sessions import SessionFactory, parse_output  # noqa: E402
from search_agent.horizon_template import generation_template  # noqa: E402
from search_agent.horizon_context import ContextPolicy, FINAL_ANSWER_PROMPT  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search a document corpus",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    tokenizer.chat_template = generation_template(args.tokenizer).read_text()
    params = {
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "sampling_seed": 0,
        "max_new_tokens": 256,
    }
    factory = SessionFactory(
        args.base_url, tokenizer, context_length=args.context_length
    )
    payload = factory.renderer.build_turn(
        [{"role": "user", "content": "What is 17 plus 25? Answer briefly."}],
        TOOLS,
        first=True,
        priming="I add seventeen and twenty five by combining ten plus twenty and seven plus five. The total is forty two.\n</ifm|think>\n",
    )
    client = factory.new_conversation()
    try:
        native = client.post(
            "/generate",
            {
                "input_ids": payload["input_ids"],
                "sampling_params": {
                    **params,
                    "stop_token_ids": [factory.renderer.end_id],
                    "skip_special_tokens": False,
                },
            },
        )
    finally:
        client.close()
    assert native["output_ids"][-1] == factory.renderer.end_id, (
        "Native control did not finish"
    )

    def check(window):
        factory = SessionFactory(
            args.base_url, tokenizer, window=window, context_length=args.context_length
        )
        session = factory.new_conversation()
        session.session_id = session.post(
            "/reasoning_sessions/open", {"window": window}
        )["session_id"]
        try:
            first = session.post(
                f"/reasoning_sessions/{session.session_id}/generate",
                {**payload, "request_id": "first", "sampling_params": params},
            )
            cache = first["meta_info"]["reasoning_eviction"]
            assert cache["completed"] and cache["window"] == window
            assert cache["next_absolute_position"] == len(payload["input_ids"]) + len(
                first["output_ids"]
            )
            assert (
                cache["retained_tokens"] + cache["dropped_tokens"]
                == cache["next_absolute_position"]
            )
            if window is None:
                assert cache["dropped_tokens"] == 0
            else:
                assert cache["dropped_tokens"] > 0 and cache["freed_pages"] > 0
            # Exercise the adapter on an actual following tool message, with
            # the exact original output IDs still resident on the server.
            session.transcript_ids = payload["input_ids"] + first["output_ids"]
            session.turns = [{"cache": cache}]
            response = session.create(
                model=args.model,
                messages=[
                    {
                        "role": "tool",
                        "content": "The calculation result is 42. Reply with that number only.",
                    }
                ],
                tools=TOOLS,
                temperature=0,
                top_p=1,
                max_tokens=1024,
                seed=0,
            )
            assert response.choices[0].finish_reason == "stop", (
                "Tool continuation did not finish"
            )
            assert response.choices[0].message.content.strip()
            return {
                "window": window,
                "first_ids": first["output_ids"],
                "first_cache": cache,
                "second_cache": session.turns[-1]["cache"],
                "continuation_tokens": response.usage.completion_tokens,
            }
        finally:
            session.close()

    serial = [check(window) for window in (None, 1, 8)]
    assert all(row["first_ids"] == native["output_ids"] for row in serial), (
        "First-turn greedy output changed before pruning"
    )
    assert (
        serial[2]["first_cache"]["retained_tokens"]
        - serial[1]["first_cache"]["retained_tokens"]
        == 7
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        concurrent = list(pool.map(check, (None, 1, 8, 8)[: args.concurrency]))
    finalizations = []
    for window in (None, 1, 8):
        # Use a small *client* context ceiling to exercise the real final-answer
        # path cheaply. The server still runs at the model's native limit.
        factory = SessionFactory(
            args.base_url, tokenizer, window=window, context_length=32768 + 1024
        )
        session = factory.new_conversation()
        messages = []
        policy = ContextPolicy(session)
        try:
            session.session_id = session.post(
                "/reasoning_sessions/open", {"window": window}
            )["session_id"]
            first = session.post(
                f"/reasoning_sessions/{session.session_id}/generate",
                {**payload, "request_id": "first", "sampling_params": params},
            )
            assert first["meta_info"]["reasoning_eviction"]["completed"]
            session.transcript_ids = payload["input_ids"] + first["output_ids"]
            session.turns = [{"cache": first["meta_info"]["reasoning_eviction"]}]
            before = list(session.transcript_ids)
            call = SimpleNamespace(
                id="trimmed-search", function=SimpleNamespace(name="search")
            )
            hits = [
                {
                    "docid": "verified-calculation",
                    "score": 1.0,
                    "snippet": "The verified calculation is 17 + 25 = 42. " * 1000,
                }
            ]
            tool_message, kept = policy.fit_search_results(
                messages, TOOLS, call, hits, 32768
            )
            assert kept and kept[0]["truncated"]
            assert hits[0]["snippet"].startswith(kept[0]["snippet"])
            assert session.transcript_ids == before
            assert policy.tool_result_fits(messages, TOOLS, tool_message, 32768)
            messages.append(tool_message)
            maximum = policy.prepare_turn(messages, TOOLS, 32768)
            assert maximum == 32768
            assert messages[-1]["content"] == FINAL_ANSWER_PROMPT
            response = session.create(
                model=args.model,
                messages=messages,
                tools=TOOLS,
                temperature=0,
                top_p=1,
                max_tokens=maximum,
                seed=0,
            )
            finalizations.append(
                {
                    "window": window,
                    "policy": policy.summary(),
                    "retained_results": kept,
                    "response": response.model_dump(),
                }
            )
            # Keep diagnostics even when the model fails to follow the prompt.
            args.output.with_name("final-answer-checks.json").write_text(
                json.dumps(finalizations, indent=2) + "\n"
            )
            assert response.choices[0].finish_reason == "stop", response.choices[
                0
            ].finish_reason
            assert "42" in response.choices[0].message.content
            assert not response.choices[0].message.tool_calls
        finally:
            session.close()
    # The same native parser used by the adapter must recognize canonical calls.
    call_text = "Done.\n</ifm|think>\n<ifm|tool_calls>\n<ifm|tool_call>search\n<ifm|arg_key>query</ifm|arg_key>\n<ifm|arg_value>capital of France</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>"
    _, _, calls = parse_output(
        tokenizer, tokenizer.encode(call_text, add_special_tokens=False), TOOLS
    )
    assert len(calls) == 1 and calls[0]["function"]["name"] == "search"
    result = {
        "passed": True,
        "model": args.model,
        "context_length": args.context_length,
        "concurrency": args.concurrency,
        "final_answer_checks": finalizations,
        "trimmed_search_final_answer_checks": True,
        "normal_matches_native_before_pruning": True,
        "w8_retains_seven_more_than_w1": True,
        "serial": serial,
        "concurrent": concurrent,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Normal/W1/W8 session and final-answer checks passed; concurrency={args.concurrency}",
        flush=True,
    )


if __name__ == "__main__":
    main()
