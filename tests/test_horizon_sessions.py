import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search_agent.horizon_sessions import HorizonRenderer, SessionFactory, parse_output  # noqa: E402
from search_agent.horizon_sglang_client import SearchTools, run_query  # noqa: E402
from search_agent.horizon_template import generation_template  # noqa: E402
from search_agent.horizon_context import (  # noqa: E402
    ContextLimitError,
    ContextPolicy,
    FINAL_ANSWER_PROMPT,
)


@pytest.fixture(
    scope="module",
    params=[
        "runtime.local.json",
        "runtime-3.7b.local.json",
        "runtime-fa3-3.7b.local.json",
        "runtime-fa3-7b.local.json",
    ],
)
def tokenizer(request):
    path = ROOT / "scripts_sglang" / request.param
    if not path.exists():
        pytest.skip("Requires the prepared local Horizon tokenizer")
    model = Path(json.loads(path.read_text())["horizon"]["path"])
    result = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    result.chat_template = generation_template(model).read_text()
    return result


@pytest.fixture
def handler():
    searcher = SimpleNamespace(
        search_description=lambda k: "Search the corpus",
        search=lambda query, k: [
            {"docid": "42", "text": "Paris is the capital of France.", "score": 0.5}
        ],
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: list(text),
        decode=lambda tokens, **kwargs: "".join(tokens),
    )
    return SearchTools(searcher, tokenizer)


def test_render_matches_native_template_and_uses_trusted_boundaries(tokenizer, handler):
    from sglang.srt.mem_cache.reasoning_eviction import (
        IFMReasoningMarkers,
        MessageBoundary,
        annotate_reasoning,
    )

    renderer = HorizonRenderer(tokenizer)
    messages = [
        {
            "role": "user",
            "content": "Quoted <ifm|think>not assistant reasoning</ifm|think>",
        }
    ]
    payload = renderer.build_turn(messages, handler.definitions(), first=True)
    native = tokenizer.apply_chat_template(
        messages, tools=handler.definitions(), add_generation_prompt=True, tokenize=True
    )
    if hasattr(native, "input_ids"):
        native = native.input_ids
    assert payload["input_ids"] == native
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant"]
    spans = annotate_reasoning(
        payload["input_ids"],
        [MessageBoundary(**m) for m in payload["messages"]],
        IFMReasoningMarkers.from_tokenizer(tokenizer),
    )
    assert spans == () or spans == []
    continuation = renderer.build_turn(
        [{"role": "tool", "content": "result"}], handler.definitions()
    )
    assert [m["role"] for m in continuation["messages"]] == ["tool", "assistant"]
    assert "# Tools" not in tokenizer.decode(continuation["input_ids"])


TOOL_TEXT = "\nShort reasoning.\n</ifm|think>\n<ifm|tool_calls>\n<ifm|tool_call>search\n<ifm|arg_key>query</ifm|arg_key>\n<ifm|arg_value>capital of France</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>"


def test_native_parsers_keep_reasoning_out_of_tool_arguments(tokenizer, handler):
    reasoning, content, calls = parse_output(
        tokenizer,
        tokenizer.encode(TOOL_TEXT, add_special_tokens=False),
        handler.definitions(),
    )
    assert "Short reasoning" in reasoning
    assert len(calls) == 1 and calls[0]["function"]["name"] == "search"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "capital of France"
    }
    assert "Short reasoning" not in content


@pytest.mark.parametrize("window", [None, 1, 8])
def test_session_tool_loop_keeps_token_history_and_closes(
    tokenizer, handler, monkeypatch, window
):
    requests = []
    total = 0
    generated = []
    outputs = [
        TOOL_TEXT + "<|ifm|im_end|>",
        "Done.\n</ifm|think>\nExact Answer: Paris<|ifm|im_end|>",
    ]

    def respond(request):
        nonlocal total
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/open"):
            return httpx.Response(
                200, json={"session_id": "test", "policy": {"window": window}}
            )
        if request.url.path.endswith("/generate"):
            ids = tokenizer.encode(outputs[len(generated)], add_special_tokens=False)
            total += len(payload["input_ids"]) + len(ids)
            generated.append(ids)
            return httpx.Response(
                200,
                json={
                    "output_ids": ids,
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "reasoning_eviction": {
                            "window": window,
                            "completed": True,
                            "next_absolute_position": total,
                        },
                    },
                },
            )
        return httpx.Response(200, json={"closed": True})

    real_client = httpx.Client
    monkeypatch.setattr(
        "search_agent.horizon_sessions.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    client = SessionFactory("http://localhost/v1", tokenizer, window=window)
    args = SimpleNamespace(
        max_iterations=5,
        max_tokens=1024,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    record = run_query(
        client, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "completed", record["error"]
    assert record["tool_call_counts"] == {"search": 1}
    assert record["retrieved_docids"] == ["42"]
    calls = [p for path, p in requests if path.endswith("/generate")]
    assert [m["role"] for m in calls[1]["messages"]] == ["tool", "assistant"]
    assert calls[1]["sampling_params"]["max_new_tokens"] == 1024
    assert record["usage"]["completion_tokens"] == sum(map(len, generated))
    assert len(record["session_turns"]) == 2
    assert requests[-1][0].endswith("/close")


@pytest.mark.parametrize("window", [None, 1, 8])
def test_endoftext_after_tool_calls_is_invalid_and_closes_session(
    tokenizer, handler, monkeypatch, window
):
    requests = []
    output_ids = tokenizer.encode(
        TOOL_TEXT + "\n<|ifm|endoftext|>", add_special_tokens=False
    )

    def respond(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/open"):
            return httpx.Response(
                200, json={"session_id": "test", "policy": {"window": window}}
            )
        if request.url.path.endswith("/generate"):
            return httpx.Response(
                200,
                json={
                    "output_ids": output_ids,
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": output_ids[-1]},
                        "reasoning_eviction": {
                            "window": window,
                            "completed": False,
                            "next_absolute_position": len(payload["input_ids"])
                            + len(output_ids),
                        },
                    },
                },
            )
        return httpx.Response(200, json={"closed": True})

    real_client = httpx.Client
    monkeypatch.setattr(
        "search_agent.horizon_sessions.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )

    def unexpected_search(*args):
        pytest.fail("Must not execute calls from an unfinished assistant message")

    handler.searcher.search = unexpected_search
    factory = SessionFactory("http://localhost/v1", tokenizer, window=window)
    args = SimpleNamespace(
        max_iterations=5,
        max_tokens=1024,
        model="horizon",
        temperature=1.0,
        top_p=0.95,
        seed=0,
    )
    record = run_query(
        factory, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "empty_or_invalid_response"
    assert record["error"] is None
    assert record["tool_call_counts"] == {}
    assert record["messages"][-1]["tool_calls"][0]["function"]["name"] == "search"
    assert record["session_turns"][0]["output_ids"] == output_ids
    assert record["invalid_response"] == {
        "reason": "unfinished_assistant_message",
        "finish_reason": "content_filter",
        "session_completed": False,
        "server_finish_reason": {"type": "stop", "matched": output_ids[-1]},
    }
    assert sum(path.endswith("/generate") for path, _ in requests) == 1
    assert [path.rsplit("/", 1)[-1] for path, _ in requests[-2:]] == ["cancel", "close"]


def test_transport_retry_reuses_request_id(tokenizer, monkeypatch):
    bodies = []

    def respond(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            raise httpx.ReadTimeout("lost response", request=request)
        return httpx.Response(200, json={"ok": True})

    real_client = httpx.Client
    monkeypatch.setattr(
        "search_agent.horizon_sessions.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr("search_agent.horizon_sessions.time.sleep", lambda _: None)
    session = SessionFactory("http://localhost", tokenizer).new_conversation()
    try:
        assert session.post(
            "/generate", {"request_id": "same-transaction"}, retry=True
        ) == {"ok": True}
        assert bodies[0] == bodies[1]
    finally:
        session.close()


def mocked_sessions(tokenizer, monkeypatch, *, window, context_length, outputs):
    requests = []
    total = 0
    generated = iter(outputs)

    def respond(request):
        nonlocal total
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/open"):
            return httpx.Response(
                200, json={"session_id": "test", "policy": {"window": window}}
            )
        if request.url.path.endswith("/generate"):
            ids = tokenizer.encode(next(generated), add_special_tokens=False)
            requested = payload["sampling_params"]["max_new_tokens"]
            assert total + len(payload["input_ids"]) + requested < context_length
            assert len(ids) <= requested
            total += len(payload["input_ids"]) + len(ids)
            return httpx.Response(
                200,
                json={
                    "output_ids": ids,
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "reasoning_eviction": {
                            "window": window,
                            "completed": True,
                            "next_absolute_position": total,
                        },
                    },
                },
            )
        return httpx.Response(200, json={"closed": True})

    real_client = httpx.Client
    monkeypatch.setattr(
        "search_agent.horizon_sessions.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    factory = SessionFactory(
        "http://localhost", tokenizer, window=window, context_length=context_length
    )
    return factory, requests


FINAL_TEXT = "Done.\n</ifm|think>\nExact Answer: Paris<|ifm|im_end|>"


@pytest.mark.parametrize("window", [None, 1, 8])
def test_reply_is_capped_to_actual_context_space(
    tokenizer, handler, monkeypatch, window
):
    messages = [{"role": "user", "content": "Answer briefly."}]
    tools = handler.definitions()
    prompt = HorizonRenderer(tokenizer).build_turn(messages, tools, first=True)
    limit = len(prompt["input_ids"]) + 65
    factory, requests = mocked_sessions(
        tokenizer,
        monkeypatch,
        window=window,
        context_length=limit,
        outputs=[FINAL_TEXT],
    )
    session = factory.new_conversation()
    try:
        session.create(
            model="horizon",
            messages=messages,
            tools=tools,
            temperature=0,
            top_p=1,
            max_tokens=32768,
            seed=0,
        )
        payload = next(p for path, p in requests if path.endswith("/generate"))
        assert payload["sampling_params"]["max_new_tokens"] == 64
    finally:
        session.close()


def test_context_policy_uses_original_positions_after_eviction(tokenizer, handler):
    factory = SessionFactory(
        "http://localhost", tokenizer, window=1, context_length=131072
    )
    session = factory.new_conversation()
    try:
        session.transcript_ids = [0] * 100388
        session.turns = [
            {"cache": {"retained_tokens": 500, "next_absolute_position": 100388}}
        ]
        messages = [{"role": "tool", "content": "new evidence " * 1500}]
        policy = ContextPolicy(session)
        maximum = policy.prepare_turn(messages, handler.definitions(), 29176)
        assert policy.prompted and 8192 < maximum <= 29176
        assert messages[-1]["content"] == FINAL_ANSWER_PROMPT
        assert session.transcript_ids == [0] * 100388
        assert policy.event["prompt_tokens"] > 100388
        assert maximum <= session.remaining_context(messages, handler.definitions())
    finally:
        session.close()


@pytest.mark.parametrize("window", [None, 1, 8])
def test_oversized_new_search_result_preserves_room_for_final_answer(
    tokenizer, handler, monkeypatch, window
):
    handler.snippet_max_tokens = 100000
    handler.searcher.search = lambda *args: [
        {"docid": "42", "text": "large result " * 5000, "score": 1.0}
    ]
    factory, requests = mocked_sessions(
        tokenizer,
        monkeypatch,
        window=window,
        context_length=2048,
        outputs=[TOOL_TEXT + "<|ifm|im_end|>", FINAL_TEXT],
    )
    args = SimpleNamespace(
        max_iterations=5,
        max_tokens=512,
        final_answer_tokens=128,
        context_margin=256,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    record = run_query(
        factory, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "completed", record["error"]
    assert record["finalization"]["omitted_tool_results"] == 0
    assert record["finalization"]["trimmed_tool_results"] == 1
    assert record["finalization"]["requested"]
    assert record["retrieved_docids"] == ["42"]
    assert record["messages"][-2]["content"] == FINAL_ANSWER_PROMPT
    kept = json.loads(record["messages"][2]["content"])
    assert kept[0]["docid"] == "42" and kept[0]["truncated"]
    assert ("large result " * 5000).startswith(kept[0]["snippet"])
    assert kept[0]["snippet"]
    calls = [p for path, p in requests if path.endswith("/generate")]
    assert [m["role"] for m in calls[1]["messages"]] == ["tool", "user", "assistant"]
    assert 128 <= calls[1]["sampling_params"]["max_new_tokens"] <= 512


@pytest.mark.parametrize("window", [None, 1, 8])
def test_trimming_preserves_ranked_hits_ids_unicode_and_cached_history(
    tokenizer, handler, window
):
    factory = SessionFactory(
        "http://localhost", tokenizer, window=window, context_length=8192
    )
    session = factory.new_conversation()
    messages = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "cached answer"},
    ]
    session.next_message = len(messages)
    session.transcript_ids = [0] * 1000
    session.turns = [
        {"cache": {"next_absolute_position": 1000, "retained_tokens": 100}}
    ]
    call = SimpleNamespace(id="search-1", function=SimpleNamespace(name="search"))
    hits = [
        {"docid": "first", "score": 0.9, "snippet": "First complete snippet."},
        {"docid": "second", "score": 0.8, "snippet": "Second complete snippet."},
        {
            "docid": "third",
            "score": 0.7,
            "snippet": 'Evidence 😺 "quoted" \\ path\nΩ漢字 ' * 100,
        },
        {"docid": "fourth", "score": 0.6, "snippet": "Lower-ranked snippet."},
    ]
    original_hits = copy.deepcopy(hits)
    original_messages = copy.deepcopy(messages)
    original_turns = copy.deepcopy(session.turns)
    policy = ContextPolicy(session, final_answer_tokens=512, context_margin=512)
    target = {
        "role": "tool",
        "content": json.dumps(
            hits[:2]
            + [{**hits[2], "snippet": hits[2]["snippet"][:100], "truncated": True}],
            ensure_ascii=False,
            indent=2,
        ),
    }
    delta = factory.renderer.build_turn(
        [target, policy.final_message()], handler.definitions()
    )
    factory.context_length = (
        len(session.transcript_ids) + len(delta["input_ids"]) + 512 + 1
    )
    try:
        message, kept = policy.fit_search_results(
            messages, handler.definitions(), call, hits, 512
        )
        assert json.loads(message["content"]) == kept
        assert [hit["docid"] for hit in kept] == ["first", "second", "third"]
        assert kept[:2] == hits[:2]
        assert kept[2]["truncated"] and kept[2]["snippet"]
        assert hits[2]["snippet"].startswith(kept[2]["snippet"])
        assert len(kept[2]["snippet"]) < len(hits[2]["snippet"])
        assert policy.tool_result_fits(messages, handler.definitions(), message, 512)
        assert policy.summary()["search_result_trimming"][0]["omitted_hits"] == 1
        assert hits == original_hits and messages == original_messages
        assert session.transcript_ids == [0] * 1000 and session.turns == original_turns
        messages.append(message)
        assert policy.prepare_turn(messages, handler.definitions(), 512) == 512
        assert messages[-1]["content"] == FINAL_ANSWER_PROMPT
    finally:
        session.close()


def test_search_results_that_fit_are_unchanged(tokenizer, handler):
    session = SessionFactory(
        "http://localhost", tokenizer, context_length=131072
    ).new_conversation()
    policy = ContextPolicy(session)
    messages = [{"role": "user", "content": "question"}]
    call = SimpleNamespace(id="search-1", function=SimpleNamespace(name="search"))
    hits = [{"docid": "42", "score": 0.5, "snippet": "Complete evidence."}]
    try:
        message, kept = policy.fit_search_results(
            messages, handler.definitions(), call, hits, 32768
        )
        assert kept == hits
        assert message["content"] == json.dumps(hits, ensure_ascii=False, indent=2)
        assert policy.reason is None and policy.trimming == []
    finally:
        session.close()


@pytest.mark.parametrize("window", [None, 1, 8])
def test_final_instruction_only_when_no_result_or_notice_fits(
    tokenizer, handler, monkeypatch, window
):
    from search_agent.prompts import format_query

    messages = [
        {
            "role": "user",
            "content": format_query("capital?", "QUERY_TEMPLATE_NO_GET_DOCUMENT"),
        }
    ]
    prompt = HorizonRenderer(tokenizer).build_turn(
        messages, handler.definitions(), first=True
    )
    padding = tokenizer.decode(
        tokenizer.encode(" reasoning" * 1200, add_special_tokens=False)[:900]
    )
    factory, requests = mocked_sessions(
        tokenizer,
        monkeypatch,
        window=window,
        context_length=len(prompt["input_ids"]) + 1024 + 128,
        outputs=[padding + TOOL_TEXT + "<|ifm|im_end|>", FINAL_TEXT],
    )
    args = SimpleNamespace(
        max_iterations=0,
        max_tokens=1024,
        final_answer_tokens=1024,
        context_margin=1024,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    record = run_query(
        factory, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "completed", record["error"]
    assert record["finalization"]["requested"]
    assert record["finalization"]["omitted_tool_results"] == 1
    assert record["finalization"]["trimmed_tool_results"] == 0
    assert record["retrieved_docids"] == []
    assert record["messages"][-2]["content"] == FINAL_ANSWER_PROMPT
    calls = [p for path, p in requests if path.endswith("/generate")]
    assert len(calls) == 2
    assert [m["role"] for m in calls[1]["messages"]] == ["user", "assistant"]


def test_trimming_stops_remaining_searches_in_batch(tokenizer, handler, monkeypatch):
    second_call = TOOL_TEXT[
        TOOL_TEXT.index("<ifm|tool_call>") : TOOL_TEXT.index("</ifm|tool_calls>")
    ]
    output = TOOL_TEXT.replace("</ifm|tool_calls>", second_call + "</ifm|tool_calls>")
    executions = []

    def search(*args):
        executions.append(args)
        return [{"docid": "42", "score": 1.0, "text": "large result " * 5000}]

    handler.searcher.search = search
    handler.snippet_max_tokens = 100000
    factory, _ = mocked_sessions(
        tokenizer,
        monkeypatch,
        window=8,
        context_length=2048,
        outputs=[output + "<|ifm|im_end|>", FINAL_TEXT],
    )
    args = SimpleNamespace(
        max_iterations=0,
        max_tokens=512,
        final_answer_tokens=512,
        context_margin=512,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    record = run_query(
        factory, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "completed", record["error"]
    assert len(executions) == 1
    assert record["retrieved_docids"] == ["42"]
    assert record["finalization"]["trimmed_tool_results"] == 1
    assert record["finalization"]["requested"]


@pytest.mark.parametrize("window", [None, 1, 8])
def test_native_sessions_allow_more_than_32k_total_output(
    tokenizer, handler, monkeypatch, window
):
    long_tool = " reasoning" * 18000 + TOOL_TEXT + "<|ifm|im_end|>"
    factory, requests = mocked_sessions(
        tokenizer,
        monkeypatch,
        window=window,
        context_length=131072,
        outputs=[long_tool, long_tool, FINAL_TEXT],
    )
    args = SimpleNamespace(
        max_iterations=0,
        max_tokens=32768,
        final_answer_tokens=32768,
        context_margin=32768,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    record = run_query(
        factory, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "completed", record["error"]
    assert record["usage"]["completion_tokens"] > 32768
    assert record["tool_call_counts"] == {"search": 2}
    assert not record["finalization"]["requested"]
    calls = [p for path, p in requests if path.endswith("/generate")]
    assert len(calls) == 3
    assert all(p["sampling_params"]["max_new_tokens"] == 32768 for p in calls)


def test_final_answer_tool_call_is_not_executed_or_marked_success(
    tokenizer, handler, monkeypatch
):
    factory, _ = mocked_sessions(
        tokenizer,
        monkeypatch,
        window=1,
        context_length=4096,
        outputs=[TOOL_TEXT + "<|ifm|im_end|>"],
    )

    def unexpected(*args):
        pytest.fail("Must not execute searches after asking for the final answer")

    handler.execute = unexpected
    args = SimpleNamespace(
        max_iterations=5,
        max_tokens=32768,
        final_answer_tokens=512,
        context_margin=1024,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    record = run_query(
        factory, handler, query_id="toy", query="capital?", args=args, metadata={}
    )
    assert record["status"] == "final_answer_tool_call"
    assert record["finalization"]["requested"]
    assert record["tool_call_counts"] == {}


def test_no_generation_when_context_is_really_full(tokenizer, handler):
    factory = SessionFactory("http://localhost", tokenizer, context_length=1)
    session = factory.new_conversation()
    try:
        with pytest.raises(ContextLimitError):
            session.create(
                model="horizon",
                messages=[{"role": "user", "content": "question"}],
                tools=handler.definitions(),
                temperature=0,
                top_p=1,
                max_tokens=20,
                seed=0,
            )
        assert session.session_id is None
    finally:
        session.close()
