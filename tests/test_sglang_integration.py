"""CPU contract tests; the Slurm smoke separately exercises real models."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import httpx
from openai import BadRequestError
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_agent.horizon_sglang_client import (
    SearchTools,
    ToolInputError,
    resume_metadata_matches,
    run_query,
)
from searcher.searchers import SearcherType
from searcher.searchers.qwen_faiss_searcher import QwenFaissSearcher, pool_query
from scripts_evaluation.evaluate_run import SGLangJudge, create_judge_prompt


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return next(self.responses)


def completion(
    *,
    tool=False,
    finish=None,
    reasoning="plan",
    tokens=10,
    arguments='{"query":"test"}',
):
    message = {
        "role": "assistant",
        "content": None if tool else "Exact Answer: example",
        "reasoning_content": reasoning,
    }
    if tool:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": arguments},
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "created": 0,
            "model": "horizon",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish or ("tool_calls" if tool else "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": tokens,
                "total_tokens": 20 + tokens,
            },
        }
    )


class FakeSearcher:
    def search_description(self, k):
        return "Search the corpus"

    def search(self, query, k):
        return [{"docid": "42", "text": "abcdef", "score": 0.5}]


@pytest.fixture
def handler():
    tokenizer = NS(
        encode=lambda text, **kwargs: list(text),
        decode=lambda tokens, **kwargs: "".join(tokens),
    )
    return SearchTools(FakeSearcher(), tokenizer, snippet_max_tokens=3)


def run(client, handler, **overrides):
    options = dict(
        max_iterations=5,
        max_tokens=100,
        model="horizon",
        temperature=0,
        top_p=1,
        seed=0,
    )
    options.update(overrides)
    return run_query(
        client,
        handler,
        query_id="synthetic",
        query="test",
        args=NS(**options),
        metadata={},
    )


@pytest.mark.parametrize("reasoning", ["preserve this reasoning", ""])
def test_multiturn_preserves_history_and_resets_turn_budget(handler, reasoning):
    client = FakeClient([completion(tool=True, reasoning=reasoning), completion()])
    record = run(client, handler)
    assert record["status"] == "completed"
    history = client.requests[1]["messages"]
    assert history[1]["reasoning_content"] == reasoning
    assert history[1]["tool_calls"][0]["id"] == history[2]["tool_call_id"] == "call-1"
    assert '"snippet": "abc"' in history[2]["content"]
    assert client.requests[1]["max_tokens"] == 100
    assert record["usage"]["completion_tokens"] == 20
    assert record["retrieved_docids"] == ["42"]
    assert record["tool_call_counts"] == {"search": 1}
    assert record["result"][-1]["type"] == "output_text"


@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"finish": "length"}, "turn_token_limit"),
        ({"tool": True}, "max_iterations"),
    ],
)
def test_incomplete_outputs_are_not_completed(handler, kwargs, status):
    record = run(FakeClient([completion(**kwargs)]), handler, max_iterations=1)
    assert record["status"] == status


def test_unlimited_turns_can_exceed_previous_token_and_iteration_limits(handler):
    client = FakeClient(
        [completion(tool=True, tokens=90) for _ in range(101)] + [completion()]
    )
    record = run(client, handler, max_iterations=0)
    assert record["status"] == "completed"
    assert len(client.requests) == 102
    assert all(request["max_tokens"] == 100 for request in client.requests)
    assert record["usage"]["completion_tokens"] == 9100
    assert record["tool_call_counts"] == {"search": 101}


def test_invalid_tool_input_is_returned_to_model(handler):
    client = FakeClient([completion(tool=True, arguments="invalid JSON"), completion()])
    assert run(client, handler)["status"] == "completed"
    assert "error" in client.requests[1]["messages"][-1]["content"]
    with pytest.raises(ToolInputError):
        handler.execute("unknown", "{}")


def test_retrieval_failure_stops_the_run(handler):
    def fail(*args):
        raise ValueError("non-finite embeddings")

    handler.searcher.search = fail
    record = run(FakeClient([completion(tool=True)]), handler)
    assert record["status"] == "error"
    assert "non-finite" in record["error"]


@pytest.mark.parametrize("tool", [False, True])
def test_invalid_finish_does_not_execute_tools_and_next_question_runs(handler, tool):
    def unexpected_search(*args):
        raise AssertionError("A terminal invalid response must not execute searches")

    handler.searcher.search = unexpected_search
    client = FakeClient([completion(tool=tool, finish="content_filter"), completion()])
    record = run(client, handler)
    assert record["status"] == "empty_or_invalid_response"
    assert record["error"] is None
    assert record["tool_call_counts"] == {}
    assert record["invalid_response"]["finish_reason"] == "content_filter"
    assert len(client.requests) == 1
    assert run(client, handler)["status"] == "completed"


def test_resume_requires_explicit_prior_client_and_identical_other_settings():
    old = {
        "configuration_sha256": "old-config",
        "agent_sha256": "old-agent",
        "temperature": 1.0,
        "server": {"config": {"context_length": 524288}},
    }
    new = {**old, "configuration_sha256": "new-config", "agent_sha256": "new-agent"}
    assert resume_metadata_matches(old, old)
    assert not resume_metadata_matches(old, new)
    assert resume_metadata_matches(old, new, ["old-agent"])
    assert not resume_metadata_matches(old, {**new, "temperature": 0.0}, ["old-agent"])
    assert not resume_metadata_matches(old, {**new, "server": {}}, ["old-agent"])
    assert not resume_metadata_matches(old, {**new, "new_setting": True}, ["old-agent"])


def test_missing_usage_is_an_error(handler):
    response = completion()
    response.usage = None
    assert run(FakeClient([response]), handler)["status"] == "error"


@pytest.mark.parametrize(
    "message,status",
    [
        (
            "Requested token count exceeds the model's maximum context length",
            "context_limit",
        ),
        ("Invalid tool schema", "error"),
    ],
)
def test_context_limit_does_not_hide_other_bad_requests(handler, message, status):
    def fail(**kwargs):
        response = httpx.Response(
            400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
        )
        raise BadRequestError(message, response=response, body=None)

    client = NS(chat=NS(completions=NS(create=fail)))
    assert run(client, handler)["status"] == status


@pytest.mark.parametrize(
    "mask,expected",
    [
        ([[0, 1, 1], [1, 1, 1]], [[0.0, 1.0], [1.0, 0.0]]),
        ([[1, 1, 0], [1, 0, 0]], [[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_eos_pooling_ignores_padding(mask, expected):
    hidden = torch.tensor(
        [[[9.0, 9.0], [3.0, 0.0], [0.0, 2.0]], [[0.0, 4.0], [9.0, 9.0], [2.0, 0.0]]]
    )
    torch.testing.assert_close(
        pool_query(hidden, torch.tensor(mask)), torch.tensor(expected)
    )


def test_optional_backends_are_not_imported():
    assert SearcherType.get_searcher_class("faiss-hf") is QwenFaissSearcher
    assert "pyserini" not in sys.modules
    assert "tevatron" not in sys.modules
    assert "vllm" not in sys.modules


def test_judge_uses_official_prompt_and_nonthinking_settings(monkeypatch):
    client = FakeClient([completion(), completion(finish="length")])
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    judge = SGLangJudge(
        NS(
            base_url="http://localhost/v1",
            model="judge",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_output_tokens=4096,
        )
    )
    messages = [{"role": "user", "content": create_judge_prompt("2+2?", "4", "4")}]
    outputs = judge.chat([messages], chat_template_kwargs={"enable_thinking": False})
    assert outputs[0].outputs[0].text == "Exact Answer: example"
    request = client.requests[0]
    assert request["messages"] == messages
    assert request["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert request["temperature"] == 0.7 and request["top_p"] == 0.8
    with pytest.raises(RuntimeError, match="did not finish"):
        judge.chat([messages], chat_template_kwargs={"enable_thinking": False})
