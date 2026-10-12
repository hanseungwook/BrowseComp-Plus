"""Check replica allocation and the all-shards-before-grading contract."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts_sglang import submit_comparison  # noqa: E402


@pytest.mark.parametrize("replicas", [1, 8])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
def test_submission_allocates_replicas_and_gates_grading(
    tmp_path, monkeypatch, replicas, workers, temperature
):
    (tmp_path / "topics-qrels").mkdir()
    (tmp_path / "topics-qrels/queries.tsv").write_text(
        "".join(f"{i}\tquestion {i}\n" for i in range(830))
    )
    config = tmp_path / "runtime.local.json"
    config.write_text(
        json.dumps(
            {
                "horizon": {"id": "IFM/K2-Horizon-3.7B", "revision": "pinned"},
                "max_running_requests": workers,
                "attention_backend": "fa3",
                "sampling": {"temperature": temperature, "top_p": 0.95, "seed": 0},
            }
        )
    )
    calls = []

    def submit(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        calls.append(command)
        return SimpleNamespace(stdout=f"{1000 + len(calls)}\n")

    monkeypatch.setattr(submit_comparison, "ROOT", tmp_path)
    monkeypatch.setattr(submit_comparison.subprocess, "run", submit)
    submit_comparison.main(
        ["--config", str(config), "--label", "test", "--replicas", str(replicas)]
    )
    manifest = json.loads((tmp_path / "logs/comparison-test.json").read_text())
    assert manifest["model"] == "IFM/K2-Horizon-3.7B"
    assert manifest["replicas_per_condition"] == replicas
    assert manifest["generation_budget_scope"] == "per_turn"
    assert manifest["max_tokens_per_turn"] == 32768
    assert manifest["total_generation_budget"] is None
    assert manifest["max_iterations"] == 0
    assert manifest["attention_backend"] == "fa3"
    assert manifest["reasoning_effort"] == "high"
    assert manifest["sampling"] == {
        "temperature": temperature,
        "top_p": 0.95,
        "seed": 0,
    }
    assert len(calls) == (3 if replicas == 1 else 6)
    for condition in ("normal", "w1", "w8"):
        jobs = [job for job in manifest["jobs"] if job["condition"] == condition]
        generation = jobs[0]["command"]
        assert generation[generation.index("--max-tokens") + 1] == "32768"
        assert generation[generation.index("--max-iterations") + 1] == "0"
        assert generation[generation.index("--workers") + 1] == str(workers)
        assert generation[generation.index("--limit") + 1] == "0"
        assert generation[generation.index("--temperature") + 1] == str(temperature)
        assert generation[generation.index("--top-p") + 1] == "0.95"
        assert "--partition=main" in generation
        if replicas == 1:
            assert "evaluate" in generation
            assert generation[generation.index("--expected-queries") + 1] == "830"
        else:
            assert "--array=0-7" in generation and "agent" in generation
            assert generation[generation.index("--num-shards") + 1] == "8"
            judge = jobs[1]["command"]
            assert f"--dependency=afterok:{jobs[0]['job_id']}" in judge
            assert "--kill-on-invalid-dep=yes" in judge
            assert "--partition=main" in judge and "judge" in judge
            assert judge[judge.index("--expected-queries") + 1] == "830"
            assert jobs[0]["output_dir"] == jobs[1]["output_dir"]
    # Refuse a duplicate label without submitting any more jobs.
    before = len(calls)
    with pytest.raises(FileExistsError):
        submit_comparison.main(["--config", str(config), "--label", "test"])
    assert len(calls) == before


def test_array_task_passes_its_index_without_changing_shard_count(tmp_path):
    env_dir = tmp_path / "env/bin"
    env_dir.mkdir(parents=True)
    fake_python = env_dir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "BROWSECOMP_ROOT": str(tmp_path),
        "SGLANG_ENV_PREFIX": str(tmp_path / "env"),
        "SLURM_JOB_ID": "1004",
        "SLURM_ARRAY_JOB_ID": "1000",
        "SLURM_ARRAY_TASK_ID": "4",
        # A retried subset may have a different task count; never use it to
        # repartition questions that other replicas have already generated.
        "SLURM_ARRAY_TASK_COUNT": "1",
    }
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts_sglang/run.sbatch"),
            "agent",
            "--condition",
            "w8",
            "--",
            "--limit",
            "0",
            "--num-shards",
            "8",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(result.stdout)
    assert args[args.index("--num-shards") + 1] == "8"
    assert args[args.index("--shard-id") + 1] == "4"
    assert args[args.index("--logs-dir") + 1].endswith("/logs/job-1000_4")
    assert args.index("--shard-id") > args.index("--")


@pytest.mark.parametrize(
    "task,condition,shard",
    [(0, "normal", 0), (1, "w8", 0), (46, "normal", 23),
     (47, "w8", 23), (62, "normal", 31), (63, "w8", 31)],
)
def test_shared_array_dispatches_conditions_and_keeps_unique_logs(
    tmp_path, task, condition, shard
):
    env_dir = tmp_path / "env/bin"
    env_dir.mkdir(parents=True)
    fake_python = env_dir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    script = ROOT / "scripts_sglang/run_7b_pair.sbatch"
    assert "#SBATCH --array=0-63%64" in script.read_text()
    result = subprocess.run(
        ["bash", str(script), "config.json", str(tmp_path / "saved"), "old-sha"],
        env={
            **os.environ,
            "BROWSECOMP_ROOT": str(tmp_path),
            "SGLANG_ENV_PREFIX": str(tmp_path / "env"),
            "SLURM_ARRAY_JOB_ID": "1000",
            "SLURM_ARRAY_TASK_ID": str(task),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(result.stdout)
    assert args[args.index("--condition") + 1] == condition
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "saved" / condition)
    assert args[args.index("--shard-id") + 1] == str(shard)
    assert args[args.index("--num-shards") + 1] == "32"
    assert args[args.index("--logs-dir") + 1].endswith(f"/logs/job-1000_{task}")
    assert args[args.index("--resume-compatible-agent-sha256") + 1] == "old-sha"
    assert args.index("--shard-id") > args.index("--")
