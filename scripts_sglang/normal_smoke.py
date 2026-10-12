"""Inspect normal Horizon generations through native chat and full-history sessions."""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search_agent.horizon_sessions import SessionFactory  # noqa: E402
from search_agent.horizon_template import generation_template  # noqa: E402
from scripts_sglang.session_smoke import TOOLS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model = config["horizon"]["id"]
    sampling = config["sampling"]
    client = OpenAI(
        base_url=args.base_url, api_key="local-sglang", timeout=1800, max_retries=0
    )
    report = {"passed": False, "config": config, "checks": []}

    def record(name, response, expected, cache=None):
        item = {"name": name, "response": response.model_dump(), "cache": cache}
        report["checks"].append(item)
        choice = response.choices[0]
        answer = choice.message.content or ""
        item["passed"] = (
            choice.finish_reason == "stop" and expected.lower() in answer.lower()
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "name": name,
                    "passed": item["passed"],
                    "answer": answer,
                    "completion_tokens": response.usage.completion_tokens,
                }
            ),
            flush=True,
        )
        assert item["passed"], item

    def chat(messages):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            seed=sampling["seed"],
            max_tokens=32768,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
        )

    prompts = [
        ("arithmetic", "What is 17 plus 25? Give the final answer as a number.", "42"),
        (
            "reasoning",
            "A bag contains 3 red and 2 blue balls. Two are drawn without replacement. What is the probability both are red? Give the final answer as a decimal.",
            "0.3",
        ),
    ]
    for name, prompt, expected in prompts:
        record(name, chat([{"role": "user", "content": prompt}]), expected)
    messages = [
        {
            "role": "user",
            "content": "Remember the project code AZURE-47. Reply with Saved.",
        }
    ]
    response = chat(messages)
    record("remember", response, "saved")
    message = response.choices[0].message
    messages += [
        {
            "role": "assistant",
            "content": message.content or "",
            "reasoning_content": getattr(message, "reasoning_content", "") or "",
        },
        {
            "role": "user",
            "content": "What project code did I ask you to remember? Reply with only the code.",
        },
    ]
    record("native_multi_turn", chat(messages), "AZURE-47")

    tokenizer = AutoTokenizer.from_pretrained(
        config["horizon"]["path"], trust_remote_code=False
    )
    tokenizer.chat_template = generation_template(config["horizon"]["path"]).read_text()
    factory = SessionFactory(
        args.base_url, tokenizer, window=None, context_length=config["context_length"]
    )
    session = factory.new_conversation()
    messages = []
    try:
        for name, prompt, expected in [
            (
                "normal_session",
                "What is 17 plus 25? Give the final answer as a number.",
                "42",
            ),
            (
                "normal_session_multi_turn",
                "Add 3 to that result. Give the final answer as a number.",
                "45",
            ),
        ]:
            messages.append({"role": "user", "content": prompt})
            response = session.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                max_tokens=32768,
                seed=sampling["seed"],
            )
            cache = session.turns[-1]["cache"]
            assert cache["window"] is None and cache["dropped_tokens"] == 0
            record(name, response, expected, cache)
            message = response.choices[0].message
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "reasoning_content": getattr(message, "reasoning_content", "")
                    or "",
                }
            )
    finally:
        session.close()
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
