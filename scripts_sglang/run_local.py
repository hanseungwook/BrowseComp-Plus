"""Launch local SGLang servers and run BrowseComp-Plus on an allocated GPU."""

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search_agent.horizon_template import generation_template  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Server:
    def __init__(self, kind, config, assets, logs, *, port=0, sessions=False):
        if not port:
            with socket.socket() as socket_:
                socket_.bind(("127.0.0.1", 0))
                port = socket_.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        model = config["horizon"] if kind == "horizon" else assets["judge"]
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model["path"],
            "--served-model-name",
            model["id"],
            "--model-impl",
            "sglang",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            config.get("attention_backend", "triton")
            if kind == "horizon"
            else "triton",
            "--disable-overlap-schedule",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--context-length",
            str(config["context_length"] if kind == "horizon" else 32768),
            "--mem-fraction-static",
            str(config.get("horizon_mem_fraction_static", 0.55))
            if kind == "horizon"
            else "0.80",
            "--max-running-requests",
            str(config.get("max_running_requests", 4)) if kind == "horizon" else "4",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if kind == "horizon":
            command += [
                "--chat-template",
                str(generation_template(model["path"])),
                "--reasoning-parser",
                "k2_horizon",
                "--tool-call-parser",
                "k2_horizon",
            ]
            if sessions:
                command += [
                    "--enable-streaming-session",
                    "--enable-reasoning-kv-eviction",
                    "--enable-deterministic-inference",
                ]
        self.log_path = logs / f"server-{kind}.log"
        self.log = self.log_path.open("w")
        save(logs / f"command-{kind}.json", command)
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def ready(self, timeout=1200):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Server exited; inspect {self.log_path}")
            try:
                with urlopen(
                    self.base_url.removesuffix("/v1") + "/health", timeout=2
                ) as response:
                    if response.status == 200:
                        return
            except (URLError, OSError):
                pass
            time.sleep(1)
        raise TimeoutError(f"Server startup timed out; inspect {self.log_path}")

    def stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.log.close()


def run(command):
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "agent",
            "evaluate",
            "judge",
            "smoke",
            "session-smoke",
            "replay-smoke",
            "normal-smoke",
            "tool-smoke",
            "retrieval-smoke",
            "serve-horizon",
            "serve-judge",
        ],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "scripts_sglang/runtime-fa3-7b.local.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/qwen3-8b/horizon-0.9b-full"
    )
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--condition", choices=["chat", "normal", "w1", "w8"], default="chat"
    )
    parser.add_argument("--expected-queries", type=int)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    config = json.loads(args.config.read_text())
    if args.mode == "replay-smoke":
        # The oracle holds a warm and a cold session concurrently. With only
        # one request row, opening the second necessarily evicts the first.
        config["max_running_requests"] = max(2, config.get("max_running_requests", 4))
    sampling = config.get("sampling", {"temperature": 0, "top_p": 1, "seed": 0})
    assets = json.loads((ROOT / "scripts_sglang/assets.local.json").read_text())
    if Path(sys.prefix).resolve() != Path(config["environment_prefix"]).resolve():
        raise RuntimeError("Activate sglang-kv-evict before running this script")
    os.environ["PYTHONNOUSERSITE"] = "1"
    # Slurm invokes the environment's Python directly, so expose its build
    # helpers (notably ninja) to SGLang's first-use CUDA compilation too.
    os.environ["PATH"] = (
        str(Path(sys.prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(Path(config["sglang_checkout"]) / "python")]
    )
    os.environ["SGLANG_CACHE_DIR"] = config["sglang_cache_dir"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SGLANG_RUST_BUILD_MODE"] = "never"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    logs = (
        args.logs_dir or ROOT / "logs" / f"{args.mode}-{time.strftime('%Y%m%dT%H%M%S')}"
    )
    logs.mkdir(parents=True, exist_ok=True)
    environment = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "sglang",
            "faiss-cpu",
            "datasets",
            "openai",
        )
    }
    environment.update(
        config=config, assets=assets, mode=args.mode, extra_arguments=extra
    )
    save(logs / "environment.json", environment)
    server_metadata = {
        "config": config,
        "session_api": args.condition != "chat"
        or args.mode in ("session-smoke", "replay-smoke", "normal-smoke"),
        "deterministic_inference": args.condition != "chat"
        or args.mode in ("session-smoke", "replay-smoke", "normal-smoke"),
        "sglang_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config["sglang_checkout"], text=True
        ).strip(),
    }
    save(logs / "server-metadata.json", server_metadata)
    active = None
    try:
        if args.mode in (
            "agent",
            "evaluate",
            "smoke",
            "session-smoke",
            "replay-smoke",
            "normal-smoke",
            "tool-smoke",
            "retrieval-smoke",
            "serve-horizon",
        ):
            active = Server(
                "horizon",
                config,
                assets,
                logs,
                port=args.port or (30000 if args.mode == "serve-horizon" else 0),
                sessions=args.condition != "chat"
                or args.mode in ("session-smoke", "replay-smoke", "normal-smoke"),
            )
            active.ready()
            print(f"Horizon ready at {active.base_url}", flush=True)
            if args.mode == "serve-horizon":
                active.process.wait()
                return
            if args.mode == "normal-smoke":
                run(
                    [
                        "scripts_sglang/normal_smoke.py",
                        "--base-url",
                        active.base_url,
                        "--config",
                        str(args.config.resolve()),
                        "--output",
                        str(logs / "normal-smoke.json"),
                    ]
                )
                return
            if args.mode == "replay-smoke":
                run(
                    [
                        str(
                            Path(config["sglang_checkout"])
                            / "examples/usage/reasoning_eviction/qualify_server.py"
                        ),
                        "--url",
                        active.base_url.removesuffix("/v1"),
                        "--tokenizer",
                        config["horizon"]["path"],
                        "--output-dir",
                        str(logs / "replay-smoke"),
                        "--keep-going",
                    ]
                )
                return
            if args.mode == "session-smoke":
                run(
                    [
                        "scripts_sglang/session_smoke.py",
                        "--base-url",
                        active.base_url,
                        "--tokenizer",
                        config["horizon"]["path"],
                        "--model",
                        config["horizon"]["id"],
                        "--context-length",
                        str(config["context_length"]),
                        "--concurrency",
                        str(config.get("max_running_requests", 4)),
                        "--output",
                        str(logs / "session-smoke.json"),
                    ]
                )
            if args.mode in ("smoke", "tool-smoke", "retrieval-smoke"):
                # Check a real retrieval independently of the model's decision
                # to call a tool on the selected benchmark question.
                run(
                    [
                        "scripts_sglang/retrieval_smoke.py",
                        "--output",
                        str(logs / "retrieval-smoke.json"),
                    ]
                    + (
                        []
                        if args.mode == "retrieval-smoke"
                        else [
                            "--base-url",
                            active.base_url,
                            "--model",
                            config["horizon"]["id"],
                        ]
                    )
                )
            command = [
                "search_agent/horizon_sglang_client.py",
                "--base-url",
                active.base_url,
                "--model",
                config["horizon"]["id"],
                "--output-dir",
                str(args.output_dir),
                "--server-metadata",
                str(logs / "server-metadata.json"),
                "--condition",
                args.condition,
                "--horizon-tokenizer",
                config["horizon"]["path"],
                "--context-length",
                str(config["context_length"]),
                "--final-answer-tokens",
                str(config.get("context_policy", {}).get("final_answer_tokens", 32768)),
                "--context-margin",
                str(config.get("context_policy", {}).get("context_margin", 32768)),
                "--temperature",
                str(sampling["temperature"]),
                "--top-p",
                str(sampling["top_p"]),
                "--seed",
                str(sampling["seed"]),
            ]
            if args.mode == "smoke":
                command += [
                    "--limit",
                    "1",
                    "--max-iterations",
                    "8",
                ]
            if args.mode not in ("tool-smoke", "session-smoke", "retrieval-smoke"):
                run(command + extra)
            active.stop()
            active = None
        if args.mode in ("judge", "evaluate", "smoke", "serve-judge"):
            if args.expected_queries is not None:
                records = [
                    json.loads(p.read_text())
                    for p in args.output_dir.glob("query_*.json")
                ]
                if len(records) != args.expected_queries or any(
                    r["status"] == "error" for r in records
                ):
                    raise RuntimeError(
                        "Refusing to score a missing or failed set of benchmark runs"
                    )
            active = Server(
                "judge",
                config,
                assets,
                logs,
                port=args.port or (30001 if args.mode == "serve-judge" else 0),
            )
            active.ready()
            print(f"Judge ready at {active.base_url}", flush=True)
            if args.mode == "serve-judge":
                active.process.wait()
                return
            if args.mode in ("judge", "evaluate", "smoke"):
                from openai import OpenAI

                sys.path.insert(0, str(ROOT))
                from scripts_evaluation.evaluate_run import (
                    create_judge_prompt,
                    parse_judge_response,
                )

                response = OpenAI(
                    base_url=active.base_url, api_key="local-sglang", max_retries=0
                ).chat.completions.create(
                    model=assets["judge"]["id"],
                    messages=[
                        {
                            "role": "user",
                            "content": create_judge_prompt(
                                "What is 2 + 2?",
                                "Exact Answer: 4\nConfidence: 100%",
                                "4",
                            ),
                        }
                    ],
                    temperature=0,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                judged = parse_judge_response(response.choices[0].message.content or "")
                save(logs / "judge-smoke.json", judged)
                if judged["parse_error"] or not judged["correct"]:
                    raise RuntimeError(
                        "The judge failed the synthetic correctness check"
                    )
            command = [
                "scripts_evaluation/evaluate_run.py",
                "--base-url",
                active.base_url,
                "--model",
                assets["judge"]["id"],
                "--input_dir",
                str(args.output_dir),
            ]
            run(command + (extra if args.mode == "judge" else []))
        save(
            logs / "result.json",
            {"passed": True, "mode": args.mode, "output_dir": str(args.output_dir)},
        )
    except BaseException as error:
        save(
            logs / "result.json",
            {"passed": False, "mode": args.mode, "error": repr(error)},
        )
        raise
    finally:
        if active is not None:
            active.stop()


if __name__ == "__main__":
    main()
