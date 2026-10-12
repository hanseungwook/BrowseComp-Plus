"""Submit normal/W1/W8 with independent GPU replicas and dependent grading."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            "%Y%m%d-%H%M%S"
        ),
    )
    parser.add_argument("--time", default="7-00:00:00")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "scripts_sglang/runtime-fa3-7b.local.json",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        choices=range(1, 9),
        default=1,
        help="Independent single-GPU copies per condition (default: 1)",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    model = config["horizon"]
    sampling = config.get("sampling", {"temperature": 0, "top_p": 1, "seed": 0})
    workers = config.get("max_running_requests", 4)
    if type(workers) is not int or not 1 <= workers <= 4:
        parser.error("Config max_running_requests must be between 1 and 4")
    if (
        not args.label
        or Path(args.label).name != args.label
        or args.label in {".", ".."}
    ):
        parser.error("label must be a single directory name")
    questions = sum(
        1
        for row in (ROOT / "topics-qrels/queries.tsv").read_text().splitlines()
        if row.strip()
    )
    if questions != 830:
        raise RuntimeError("Expected the pinned 830-question BrowseComp-Plus dataset")
    manifest_path = ROOT / "logs" / f"comparison-{args.label}.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Submission already recorded at {manifest_path}; choose another label"
        )
    manifest = {
        "label": args.label,
        "questions_per_condition": questions,
        "generation_budget_scope": "per_turn",
        "max_tokens_per_turn": 32768,
        "total_generation_budget": None,
        "max_iterations": 0,
        "search_result_policy": "ranked-prefix-trimming-v1",
        "workers_per_replica": workers,
        "context_length": config.get("context_length"),
        "context_policy": config.get("context_policy"),
        "replicas_per_condition": args.replicas,
        "model": model["id"],
        "model_revision": model["revision"],
        "attention_backend": config.get("attention_backend", "triton"),
        "sampling": sampling,
        "reasoning_effort": "high",
        "runtime_config": str(config_path),
        "partition": "main",
        "jobs": [],
    }
    outputs = {
        condition: ROOT / "runs" / "qwen3-8b" / args.label / condition
        for condition in ("normal", "w1", "w8")
    }
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(command, condition, role, output):
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=True
        )
        job_id = completed.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {completed.stdout}")
        manifest["jobs"].append(
            {
                "condition": condition,
                "role": role,
                "job_id": job_id,
                "output_dir": str(output),
                "command": command,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["jobs"][-1]), flush=True)
        return job_id

    for condition, output in outputs.items():
        name = f"bcplus-{model['id'].rsplit('-', 1)[-1].lower()}-{condition}"
        scheduler = [
            "sbatch",
            "--parsable",
            "--partition=main",
            f"--time={args.time}",
            "--requeue",
        ]
        common = [
            "--config",
            str(config_path),
            "--condition",
            condition,
            "--output-dir",
            str(output),
        ]
        generation = [
            "--",
            "--limit",
            "0",
            "--max-tokens",
            "32768",
            "--max-iterations",
            "0",
            "--workers",
            str(workers),
            "--temperature",
            str(sampling["temperature"]),
            "--top-p",
            str(sampling["top_p"]),
            "--seed",
            str(sampling["seed"]),
        ]
        if args.replicas == 1:
            submit(
                scheduler
                + [f"--job-name={name}-32k", "scripts_sglang/run.sbatch", "evaluate"]
                + common
                + ["--expected-queries", str(questions)]
                + generation,
                condition,
                "evaluate",
                output,
            )
            continue
        generation_id = submit(
            scheduler
            + [
                f"--job-name={name}-32k",
                f"--array=0-{args.replicas - 1}",
                "scripts_sglang/run.sbatch",
                "agent",
            ]
            + common
            + generation
            + ["--num-shards", str(args.replicas)],
            condition,
            "generation_array",
            output,
        )
        # afterok on the array ID waits for every replica. A failed shard must
        # not produce a score over an incomplete set of benchmark questions.
        submit(
            scheduler
            + [
                f"--job-name={name}-judge",
                f"--dependency=afterok:{generation_id}",
                "--kill-on-invalid-dep=yes",
                "scripts_sglang/run.sbatch",
                "judge",
            ]
            + common
            + ["--expected-queries", str(questions)],
            condition,
            "judge",
            output,
        )
    print(f"Submission manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
