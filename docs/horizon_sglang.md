# K2 Horizon with SGLang and Qwen3-Embedding-8B

This setup reuses the `sglang-kv-evict` environment and its editable SGLang
checkout. The default generator is `IFM/K2-Horizon-7B`, validated on normal
generations, a small BrowseComp-Plus pilot, and W1/W8 session smoke checks. A separate local config
selects `IFM/K2-Horizon-3.7B`, the dense 4B-class release. Both use the verified
anti-doom-loop weights. The 0.9B configs and past results remain available, but
its evaluation jobs were canceled in favor of 7B. Retrieval uses
`Qwen/Qwen3-Embedding-8B` with the published BrowseComp-Plus vectors. The official
`Qwen/Qwen3-32B` judge scores the answers in a separate serving phase.

## What is configured

- 830 benchmark questions and a fixed corpus of 100,195 documents.
- Four published Qwen3-Embedding-8B index shards, searched with CPU FAISS inner
  product; only the query encoder runs on GPU. Dataset snapshots and models are
  pinned in `scripts_sglang/prepare_assets.py`.
- The upstream retrieval instruction prefix, last-token pooling, normalized
  embeddings, top-5 hits, and 512-token snippets. Snippet lengths use the
  `Qwen/Qwen3-0.6B` tokenizer, as in the existing clients; it is not another
  inference model.
- The upstream search-only question prompt and answer format. Horizon's native
  generation template, reasoning parser, and tool parser are used.
  Full assistant reasoning and tool messages are retained between calls. The
  0.9B generation template is selected explicitly because its default
  `chat_template.jinja` ignores `reasoning_content` history. The pinned 3.7B
  checkpoint provides its generation template as `chat_template.jinja`; it
  already preserves reasoning history. Template selection and hashing use the
  same file in the server and client.
- Each checkpoint's native context: 131,072 tokens for 0.9B and 524,288 for
  3.7B/7B. Model-card sampling: temperature 0.6 for 0.9B, 1.0 for 3.7B/7B,
  top-p 0.95 and high reasoning effort for both. Each request has a
  32,768-token output limit **per turn**, including reasoning and tool calls.
  There is no cumulative generated-token budget and no generation-step limit
  in benchmark runs; the chronological native context is the trajectory limit. These are configurable run
  settings, not a claim to reproduce another model's published score.
- Native BF16 SGLang inference using FA3 attention on Hopper. Eager execution
  and disabled overlap are still required by the eviction runtime. One H200 hosts Horizon
  and the 8B query encoder together. The judge runs after they exit.

The current configs are `runtime-fa3-7b.local.json` (default) and
`runtime-fa3-3.7b.local.json`; `runtime-fa3-0.9b.local.json` preserves the earlier setup. They pin the model snapshots and use
the same final-answer policy across normal, W1, and W8:

- Before each completed-turn continuation, count the exact chronological token
  IDs, including new tool results and the assistant prefix. Removed reasoning
  still counts toward original positions. Each request gets a fresh allowance
  of `min(32768, remaining_context_tokens)`, with one token of admission slack.
  Cumulative token usage is recorded for reporting, never subtracted from the
  next turn's allowance.
- Ask for a final answer when at most 32,768 context tokens remain, or when a
  full generation allowance plus the final instruction would not fit.
- Append an explicit instruction to stop searching, give the best answer from
  available evidence, and use Explanation / Exact Answer / Confidence. The
  final turn receives the same 32,768-token maximum, capped only by remaining
  context. There is no separate 8K final-answer output cap.
- Before adding new search results, retain space for that instruction and up
  to 32,768 final-answer tokens. If a response would consume the reserve, keep
  complete hits in retrieval rank order and fit a prefix of the next snippet
  into the remaining space. Preserve document IDs and scores, mark a shortened
  snippet with `truncated: true`, and omit the remaining lower-ranked hits.
  Fit checks use Horizon's tokenizer on the complete serialized JSON and prompt,
  including the final instruction and message delimiters. A response that fits
  is passed through unchanged. Previously cached history is never edited.
  After trimming, stop remaining searches in that batch and ask for the final
  answer. If no evidence fits, use a short search-stopped notice when possible;
  if even that cannot fit, append only the final-answer instruction. Prepare
  that instruction before context accounting when there are no new tool messages.
  A long individual generation can still consume this space before the next
  turn boundary; the final request then uses only the context that remains.
- Save the finalization reason, wholly omitted tool-response count, trimmed
  response count, and per-call retained/omitted hit counts and truncated-snippet
  lengths. Only document IDs actually supplied to the model enter the retrieved
  ID set. Metadata records `ranked-prefix-trimming-v1`, and results from the
  earlier whole-response omission policy belong in separate output directories.
  A model that keeps
  calling tools after the final instruction receives `final_answer_tool_call`.
  An unfinished response that uses the entire per-turn output allowance receives
  `turn_token_limit`; one stopped by the smaller remaining context receives
  `context_limit`. Neither is counted as a successful answer. The adapter does
  not extend an unfinished assistant message beyond the per-turn limit or
  inject a user instruction into it.
- If generation stops while the session still reports an unfinished assistant
  message (for example, `endoftext` instead of `im_end` after tool calls), record
  `empty_or_invalid_response` and the raw ending in `invalid_response`. Preserve
  the generated tokens, execute no tools from that response, close the session,
  and continue with the next question. Do not repair or retry the model output.
  These records are scored incorrect by the existing evaluator. Genuine API,
  retrieval, and cleanup failures still stop the worker as infrastructure errors.
- `--max-iterations 0` means unlimited turns and is the benchmark default.
  A positive value remains available for bounded smoke tests only.

The 7B native config reserves 72% of GPU memory and the 3.7B config reserves 68%
for Horizon weights and KV,
leaving room for the query encoder, and processes one conversation at a time
per replica so a full 512K history can fit. The historical 0.9B config uses 55% and four
concurrent conversations. All conditions of a given model share these settings.
Earlier 128K and cumulative-32K runs retain their separate configs and output
labels. Never mix their results with the per-turn policy. Run metadata explicitly
records the budget scope, unlimited cumulative budget, and client source hash.
After a reviewed client-only result-classification fix, a resumed worker may use
`--resume-compatible-agent-sha256 OLD_SHA256` to reuse results from that exact
prior client. Every other metadata field must match. New records retain the
actual new client hash; this option never relaxes model, sampling, context,
retrieval, server, or template checks.

The query encoder uses Transformers SDPA instead of Tevatron/FlashAttention.
Pooling and normalization match that backend, but kernel differences can affect
scores and ranking. This setup has not established exact retrieval parity with
the original backend. See [the original Qwen instructions](qwen.md) and
[judge instructions](llm_as_judge.md).

The benchmark comparison uses the same pretrained checkpoint for
three conditions: normal full history, W1 (reasoning markers only), and W8
(reasoning markers plus the last seven interior tokens). W0 is omitted because
it is equivalent to W1 in this implementation. Keep questions, retrieval,
decoding settings, and budgets identical across conditions.

All three conditions use `/reasoning_sessions`, with pruning disabled for
normal. `search_agent/horizon_sessions.py` renders the native template's exact
initial prompt and then appends only new tool/user messages. It preserves
generated token IDs and original positions without re-rendering or shortening
assistant history. The server performs KV compaction after completed assistant
messages. The legacy `--condition chat` path remains available separately;
ordinary `/v1/chat/completions` requests do not enable eviction.

## Submit the three full evaluations

The 7B comparison uses label `horizon-7b-fa3-20261007` and the frozen config
`runtime-fa3-7b-comparison-20261007.local.json`. It began with eight replicas
per condition. On 2026-10-08 the 3.7B and 7B W1 jobs were stopped, with their
saved trajectories retained, and 7B full history and W8 were repartitioned
into 32 shards each. `run_7b_pair.sbatch` runs a shared 64-task array on `main`
with a 64-GPU cap: even task IDs select full history, odd IDs select W8,
and integer division by two gives each condition's shard index. Each shard
skips existing non-infrastructure-error results, including model-invalid
responses. In-flight questions without a saved result start again.
Each condition's judge waits for all 32 of its own array tasks. The current
job IDs, original job history, resume counts, and commands are in
`logs/comparison-horizon-7b-fa3-20261007.json`.
Each condition covers all 830 questions and has a dependent Qwen3-32B judging
job. Before launch, the 7B session smoke verified actual page reclamation,
W8 retaining seven more tokens than W1, tool continuation, and final answers.

```bash
cd ~/BrowseComp-Plus
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sglang-kv-evict
python scripts_sglang/submit_comparison.py --replicas 8
```

The 7B config pins revision `3b6ae394c1b863c27a306d168b25555837955192`.
Its 36 weight shards match `anti_doom_loop_verified`, and the downloaded SHA256
hashes were verified. Normal-generation validation runs with:

```bash
sbatch scripts_sglang/run.sbatch normal-smoke --config scripts_sglang/runtime-fa3-7b.local.json
```

This records arithmetic, reasoning, native multi-turn recall, and unpruned
session continuations with the model-card sampling and full per-turn budget.
On 2026-10-07 all six checks passed. A normal-only pilot on the first three
dataset rows (769/770/771) also finished naturally and received three correct
final-answer judgments from Qwen3-32B. The trajectories used 4/46/40 searches
and 3,486/7,824/12,367 generated tokens, with no pruning, replay, turn-budget
exhaustion, or context finalization. The larger search counts warrant monitoring.
This validates the pipeline on three examples, not full benchmark accuracy or
7B W1/W8 correctness. Pilot outputs live under
`runs/validation/horizon-7b-fa3-20261007/normal/` with matching `evals/` results.

For the cached 3.7B checkpoint, select its separate config and output label:

```bash
python scripts_sglang/submit_comparison.py \
  --config scripts_sglang/runtime-fa3-3.7b.local.json \
  --label horizon-3.7b-comparison --replicas 8
```

The 3.7B FA3 config pins revision `d6e5432c59637093c0e125e791d89ea4a0b53559`,
the latest main revision checked on 2026-10-07. All 36 weight-shard SHA256 hashes
match the `anti_doom_loop_verified` release and the current SGLang cookbook's
`fe504ef19c7efa3120a9352bcfa180aa133946de` revision. The previous
`943ce4e230b461c3c60830a87217992615885797` weights match the pre-update
`sft_2_2500` checkpoint; those evaluations remain separate. The 0.9B weights,
tokenizer, and configuration are already identical to the current cookbook
checkpoint, so its existing pin remains in use. With the default `--replicas 1`, each invocation
submits three jobs on `main`: normal, W1, and W8. Each requests one H200,
eight CPUs, 128 GB of host memory, and a seven-day time limit. Each job processes
all 830 questions with a fresh 32,768-token allowance per turn, unlimited steps,
the config's model-card sampling, and its concurrency (one for 3.7B/7B; four for the historical 0.9B), sharing one Qwen query
encoder. When generation completes, the job shuts down Horizon and the encoder,
then starts Qwen3-32B to grade its results. Scoring requires all 830 result files;
budget-limited answers count as incomplete, while infrastructure errors halt the
run. The job is requeueable and resumes saved per-question results.

With `--replicas 8`, each condition instead gets one generation array with eight
single-GPU tasks and one dependent grading job. Each task runs an independent
TP=1, DP=1 SGLang server and Qwen query encoder, using the config's conversation
concurrency. The tasks handle disjoint question groups (103 or 104 questions
each), writing to the condition's shared result directory. A conversation never
moves between servers. This uses up to 24 generation GPUs across three conditions;
tasks may be placed on different nodes. Every task requests eight CPUs and 128 GB
of host memory. Grading waits for the entire condition's array to succeed and
checks for all 830 results before starting. No eight-GPU tensor-parallel server
or SGLang `--dp 8` router is involved. Array task server logs are in
`logs/job-<array-id>_<task-id>/`; grading has its own job and logs.

The printed `logs/comparison-<label>.json` manifest lists job IDs, exact commands,
and output directories. Results are in `runs/qwen3-8b/<label>/<condition>/` and
scores in the corresponding `evals/qwen3-8b/<label>/<condition>/`. Scheduler logs
are in `logs/slurm-<job>.log`; server commands and logs are in `logs/job-<job>/`.
Session results include token deltas and per-turn retained/dropped-token,
freed-page, replay, and absolute-position statistics. Reported prompt-token usage
counts the chronological prompt length; it is not a measurement of prefill work.

To repeat the GPU session checks without running the dataset:

```bash
sbatch scripts_sglang/run.sbatch session-smoke
# Use --config scripts_sglang/runtime-fa3-3.7b.local.json for the larger model.
```

This checks native/full-session first-turn greedy agreement, actual W1/W8 page
reclamation, the seven-token retention difference, tool-message continuation,
and the configured concurrency. It also supplies an oversized synthetic search
response, verifies that useful evidence is retained with an intact document ID,
and checks that each model can answer after the final instruction in normal/W1/W8
modes. A small client context ceiling exercises the boundary cheaply. This is
not a full-length 512K quality or performance test. Both sizes passed this
check with FA3 on H200. Use `replay-smoke` to run the separate full/W0/W1/W8/
oversized-window warm-versus-cold replay qualification; it saves per-turn
responses and applies the existing 0.05 sampled-logprob tolerance. The eviction implementation remains
experimental; the replay check temporarily allows at least two request rows to
keep its paired warm/cold sessions resident. These checks do not establish full
masked-reference logit parity.

The updated 3.7B FA3 checkpoint passed all 15 warm/cold comparisons with exact
token IDs and sampled logprobs. The 0.9B FA3 checkpoint can change reasoning
wording after cache rebuild even without pruning. An ordinary FA3 server with
the eviction feature disabled reproduced the same warm/cold token difference;
the final arithmetic answer remained unchanged. Strict 0.9B replay parity is
not established. Per-turn replay metadata remains available in every trace.

The eviction runtime currently requires TP=1, DP=1, and PP=1 and rejects
multi-GPU tensor parallelism. Allocating eight GPUs alone does not make these
jobs use them. Use `--replicas 8` above to distribute benchmark work across
independent single-GPU servers. This preserves each conversation's cache on one
server. Eight-GPU tensor-parallel eviction still
needs implementation and distributed correctness validation.

## Existing local installation

```bash
cd ~/BrowseComp-Plus
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sglang-kv-evict
```

Only `faiss-cpu==1.15.1` was added to the existing inference environment. Torch,
Transformers, and SGLang were preserved. Do not run the repository's `uv sync`
inside this environment: that installs the separate upstream vLLM/training
stack. The small additional requirements are in
`scripts_sglang/requirements.txt`.

`scripts_sglang/runtime-fa3-*.local.json` records the environment, SGLang source,
Horizon snapshot, and context length. `scripts_sglang/assets.local.json` records
the downloaded models and dataset paths. These machine-specific files are
ignored by Git. To prepare the pinned retrieval and judge assets again:

```bash
python scripts_sglang/prepare_assets.py
```

The preparer respects an existing `HF_HOME` and otherwise uses a local cache.
Benchmark plaintext, model outputs, and downloaded assets remain local and
ignored by Git. The agent reads only question IDs and questions; reference
answers are loaded separately by the scoring script.

## Smoke test on one H200

Submit from the repository root so Slurm can open `logs/slurm-<job>.log`:

```bash
mkdir -p logs
sbatch scripts_sglang/run.sbatch smoke \
  --output-dir runs/smoke/horizon-7b-qwen8b
```

This checks a real Qwen retrieval and a forced search call with a tool-result
continuation, then runs one benchmark question with a
32,768-token per-turn limit and at most eight generation steps, checks the judge with a
synthetic arithmetic answer, and evaluates the benchmark output. Logs and
server settings are saved under `logs/job-<job>/`. The job owns and shuts down
its servers. A successful pipeline check is not evidence of benchmark accuracy;
inspect the query's `status`, tool counts, and the corresponding evaluation too.
The forced call tests the interface, not autonomous tool selection. To repeat
only the retrieval/tool interface check, use `sbatch scripts_sglang/run.sbatch tool-smoke`.

To check that the larger cache and Qwen encoder fit together, use
`sbatch scripts_sglang/run.sbatch retrieval-smoke --condition normal --config scripts_sglang/runtime-fa3-3.7b.local.json`.
This loads Horizon at 512K and runs actual Qwen retrieval on the same GPU.
The legacy `tool-smoke` also tests `/v1/chat/completions`; the previous 3.7B
checkpoint returned an empty continuation in that separate path. The benchmark
uses the token-session adapter, validated with `session-smoke`, for all conditions.

## Generate and score benchmark runs

Generation defaults to five questions. Use `--limit 0` for all 830; allocate
time appropriate to the chosen checkpoint and budget. For example:

```bash
sbatch --time=7-00:00:00 scripts_sglang/run.sbatch agent --condition normal \
  --output-dir runs/qwen3-8b/horizon-7b-full \
  -- --limit 0 --max-tokens 32768 --max-iterations 0
```

Each question is saved atomically as `query_<id>.json` with its full messages,
trajectory, token usage, retrieved IDs, status, and configuration fingerprint.
Re-running skips existing non-error results with matching settings and retries
infrastructure errors. `--force` overwrites
selected results. Choose another output directory when changing settings.
Infrastructure failures stop the process after saving the failing record;
token, context, or iteration limits remain incomplete results and are counted as such by
the evaluator.

For a small selection, append `--query-id <id>` (repeatable). Parallel jobs may
use `--num-shards 8 --shard-id 0` through `7`; each handles disjoint questions.
Keep their model and run settings identical when sharing an output directory.

After generation finishes:

```bash
sbatch scripts_sglang/run.sbatch judge \
  --output-dir runs/qwen3-8b/horizon-7b-full
```

The existing evaluator accepts `--base-url` for SGLang and retains the upstream
judge prompt, sampling defaults, and metrics. Judge thinking is disabled.
Results appear under `evals/qwen3-8b/horizon-7b-full/`, including
`evaluation_summary.json` and `detailed_judge_results.csv`. API failures abort
scoring rather than dropping those questions from the denominator. Existing
scores are reused; pass `-- --force` to the judge launcher when intentionally
re-evaluating overwritten generations. Calibration metrics on tiny subsets are
not meaningful.

## Interactive use on an allocated GPU

```bash
python scripts_sglang/run_local.py serve-horizon --port 30000
# In a second shell on that same GPU node, with the environment active:
python search_agent/horizon_sglang_client.py \
  --base-url http://127.0.0.1:30000/v1 --model IFM/K2-Horizon-7B --limit 5 \
  --output-dir runs/manual/horizon-7b-full
```

The server binds to loopback. Stop Horizon and the query encoder before starting
`serve-judge --port 30001`, then use:

```bash
python scripts_evaluation/evaluate_run.py \
  --base-url http://127.0.0.1:30001/v1 --model Qwen/Qwen3-32B \
  --input_dir runs/manual/horizon-7b-full
```

CPU contract tests (no model weights are loaded):

```bash
python -m pytest -q tests/test_sglang_integration.py tests/test_horizon_sessions.py tests/test_slurm_comparison.py
```
