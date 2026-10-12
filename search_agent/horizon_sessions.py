"""Adapt Horizon's token-ID sessions to the benchmark's chat/tool loop.

All comparison conditions use identical chronological token IDs and message
boundaries. Only the server's window changes; no text history is shortened.
"""

import copy
import time
import uuid

import httpx
from openai import BadRequestError
from openai.types.chat import ChatCompletion

from search_agent.horizon_context import ContextLimitError


class HorizonRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.start = "<|ifm|im_start|>"
        self.end = "<|ifm|im_end|>"
        self.end_id = self.encode(self.end)[0]
        if len(self.encode(self.start)) != 1 or len(self.encode(self.end)) != 1:
            raise ValueError("Horizon message delimiters must each be one token")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def build_turn(self, messages, tools, *, first=False, priming=""):
        if not messages or any(
            m["role"] not in {"user", "tool", "system"}
            or not isinstance(m.get("content"), str)
            for m in messages
        ):
            raise ValueError("Append only new user/tool/system text messages")
        chunks = []
        prefix = self.tokenizer.bos_token or "" if first else ""
        if first:
            # Render the actual template's tool definitions. This trusted system
            # block is sliced at its known outer delimiters, never scanned for
            # roles inside model-generated or retrieved text.
            block = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="high",
            )
            header = self.start + "system\n"
            if not block.startswith(prefix + header) or not block.endswith(self.end):
                raise ValueError("Unexpected Horizon tool-system template")
            chunks.append(("system", block[len(prefix) :], True))
        for message in messages:
            role = message["role"]
            chunks.append(
                (role, self.start + role + "\n" + message["content"] + self.end, True)
            )
        chunks.append(
            ("assistant", self.start + "assistant\n<ifm|think>\n" + priming, False)
        )
        ids, boundaries = self.encode(prefix), []
        for role, text, completed in chunks:
            start = len(ids)
            ids.extend(self.encode(text))
            boundaries.append(
                dict(
                    role=role,
                    start=start,
                    body_start=start + 1 + len(self.encode(role)),
                    end=len(ids),
                    completed=completed,
                )
            )
        # Guard against BPE differences from segment encoding or a changed
        # template. Special message delimiters should make these identical.
        if ids != self.encode(prefix + "".join(chunk[1] for chunk in chunks)):
            raise ValueError("Message segmentation changed tokenization")
        if first and not priming:
            expected = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                reasoning_effort="high",
            )
            if hasattr(expected, "input_ids"):
                expected = expected.input_ids
            if ids != expected:
                raise ValueError("Session prompt differs from the native chat template")
        return {"input_ids": ids, "messages": boundaries}


def parse_output(tokenizer, output_ids, tools):
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    reasoning, content = ReasoningParser(
        model_type="k2_horizon",
        stream_reasoning=False,
        force_reasoning=True,
        tokenizer=tokenizer,
    ).parse_non_stream(text)
    content, parsed = FunctionCallParser(
        [Tool.model_validate(tool) for tool in tools],
        "k2_horizon",
        tokenizer=tokenizer,
    ).parse_non_stream(content or "")
    calls = [
        {
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.parameters,
            },
        }
        for call in parsed
    ]
    return reasoning or "", content or "", calls


class SessionFactory:
    def __init__(
        self, base_url, tokenizer, *, window=None, timeout=1800, context_length=None
    ):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.tokenizer = tokenizer
        self.renderer = HorizonRenderer(tokenizer)
        self.window = window
        self.timeout = timeout
        self.context_length = context_length

    def new_conversation(self):
        return SessionConversation(self)


class SessionConversation:
    def __init__(self, factory):
        self.factory = factory
        self.http = httpx.Client(base_url=factory.base_url, timeout=factory.timeout)
        self.session_id = None
        self.next_message = 0
        self.transcript_ids = []
        self.turns = []
        self.chat = self
        self.completions = self

    def post(self, path, payload=None, *, retry=False):
        for attempt in range(3 if retry else 1):
            try:
                response = self.http.post(path, json=payload or {})
                break
            except httpx.TransportError:
                if not retry or attempt == 2:
                    raise
                time.sleep(1)
        if response.status_code == 400:
            raise BadRequestError(response.text, response=response, body=response.text)
        response.raise_for_status()
        return response.json()

    def remaining_context(self, messages, tools):
        if self.factory.context_length is None:
            raise ValueError("Context accounting requires an explicit context length")
        delta = self.factory.renderer.build_turn(
            messages[self.next_message :], tools, first=not self.turns
        )
        # The server uses a strict '< context_length' admission check, including
        # the generated terminal token. Account for chronological positions,
        # never the shorter resident KV length after W1/W8 compaction.
        return (
            self.factory.context_length
            - len(self.transcript_ids)
            - len(delta["input_ids"])
            - 1
        )

    def create(self, *, model, messages, tools, temperature, top_p, max_tokens, seed):
        if self.factory.context_length is not None:
            max_tokens = min(max_tokens, self.remaining_context(messages, tools))
            if max_tokens < 1:
                raise ContextLimitError("No context space remains for generation")
        if self.session_id is None:
            opened = self.post(
                "/reasoning_sessions/open", {"window": self.factory.window}
            )
            self.session_id = opened["session_id"]
            if opened["policy"]["window"] != self.factory.window:
                raise RuntimeError("Server returned a different eviction window")
        delta = self.factory.renderer.build_turn(
            messages[self.next_message :],
            tools,
            first=not self.turns,
        )
        request_id = uuid.uuid4().hex
        before = len(self.transcript_ids) + len(delta["input_ids"])
        response = self.post(
            f"/reasoning_sessions/{self.session_id}/generate",
            dict(
                **delta,
                request_id=request_id,
                sampling_params={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": -1,
                    "sampling_seed": seed,
                    "max_new_tokens": max_tokens,
                },
            ),
            retry=True,
        )
        output = response["output_ids"]
        meta = response["meta_info"]
        cache = meta["reasoning_eviction"]
        if cache["window"] != self.factory.window:
            raise RuntimeError("Server response has the wrong eviction window")
        if cache["completed"] and (
            not output or output[-1] != self.factory.renderer.end_id
        ):
            raise RuntimeError("Completed session did not return its terminal token")
        self.transcript_ids.extend(delta["input_ids"])
        self.transcript_ids.extend(output)
        if cache["next_absolute_position"] != len(self.transcript_ids):
            raise RuntimeError("Session lost its original token positions")
        self.turns.append(
            {
                "request_id": request_id,
                "input_ids": delta["input_ids"],
                "messages": delta["messages"],
                "output_ids": output,
                "cache": copy.deepcopy(cache),
                "finish_reason": meta["finish_reason"],
                "requested_max_new_tokens": max_tokens,
            }
        )
        decoded_ids = output[:-1] if cache["completed"] else output
        reasoning, content, calls = parse_output(
            self.factory.tokenizer, decoded_ids, tools
        )
        finish = (
            ("tool_calls" if calls else "stop")
            if cache["completed"]
            else (
                "length"
                if meta["finish_reason"]["type"] == "length"
                else "content_filter"
            )
        )
        self.next_message = (
            len(messages) + 1
        )  # Skip the generated assistant on the next call.
        completion = len(output)
        return ChatCompletion.model_validate(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning,
                            "tool_calls": calls or None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": before,
                    "completion_tokens": completion,
                    "total_tokens": before + completion,
                },
            }
        )

    def close(self):
        try:
            if self.session_id:
                # Also handles an uncertain transport failure while generation
                # is still running; cancel is a no-op after a successful turn.
                self.post(f"/reasoning_sessions/{self.session_id}/cancel")
                self.post(f"/reasoning_sessions/{self.session_id}/close")
        finally:
            self.http.close()
