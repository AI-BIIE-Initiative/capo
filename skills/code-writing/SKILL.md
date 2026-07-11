---
name: code-writing
description: >
  Authoritative spec for the CAPO fine-tuning scripts the orchestrator generates
  at Attempt 0 (train.py, probe.py, src/eval/evaluate.py, src/eval/plot_eval.py)
  and the contracts the code-repair-critic must preserve when patching them. Use
  when writing these scripts fresh, self-repairing them in the 3-step gate's
  repair ladder, or emitting a patch diff at Attempt 3.
license: MIT
user-invokable: false
compatibility: >
  Loaded by the main orchestrator (Attempts 0-2, in-process) and the
  code-repair-critic subagent (Attempt 3, fresh context). Both must keep every
  contract marked non-negotiable intact.
---

# Code-writing spec for the 3-step gate

Single source of truth for train.py, probe.py, src/eval/evaluate.py, and
src/eval/plot_eval.py. The orchestrator's prompt template references this file
by path — keep both in sync. The scripts are generated once at Attempt 0. If the
gate's schema check or feasibility probe fails with failure_category=script_bug,
the repair ladder runs (Attempt 1 → 2 → 3), and every repair must preserve the
contracts below. Breaking one is a Phase 6 finalizer failure that blocks the run.

Everything below is non-negotiable unless stated otherwise.

## Canonical layout
train.py and probe.py are thin entry-point launchers at the run root that import
from a flat src/ package: src/{data,models,train,eval,utils}/. No deeper nesting.
Configs under configs/, operational state under outputs/, scientific output under
results/.

Run root:
- train.py — launcher → src.train.main()
- probe.py — launcher → src.train.probe()
- requirements.txt — pinned deps, installed on remote in Step 4
- infra.json — written by the infra agent
- manifest.json, state.json, task.md — session bookkeeping (orchestrator-managed)
- RUN_REPORT.md — written by the finalizer (do NOT write here)

src/ (each subpkg has __init__.py):
- data/ — dataset loading, preprocessing, tokenization, collators
- models/ — model loading, PLM wrappers, LoRA/PEFT setup, heads
- train/ — training loop, loss, optimizer/scheduler, callbacks, step.py, canary.py
- eval/ — evaluate.py, plot_eval.py, metrics
- utils/ — logging, seeding, config helpers, file I/O

- configs/: experiment.yaml (run_id, task, model, method, seed), training.yaml (epochs, batch_size, learning_rate, ...), evaluation.yaml (checkpoint_path, metrics, output_dir).
- outputs/ (every operational/log artifact): train.log (nohup stdout), train_err.log (nohup stderr), train.pid (launcher, not train.py), status.json, metrics.jsonl.
- results/ (every scientific output): metrics.json (final, by evaluate.py), eval_metrics.csv, train_metrics.csv, eval_per_class.csv (classification, n_classes ≤ 50), plots/ (overwritten in place), predictions/.
- checkpoints/: best/ (best by val metric, pushed to HF Hub), last/ (most recent, overwritten each save, retained locally, NOT pushed).
- reports/: evaluation_report.md (by evaluate.py), plot_manifest.json, final_summary.json (finalizer — do NOT write here), health/history.jsonl (monitor — do NOT write here).
- scripts/launch_command.sh (only file under scripts/, written by Step 10); probe/ (probe_result.json, probe.log, probe_batch_recipe.json); pricing/ (lambda-<gpu>.json, cost_report.json); profile/ (profile.json, plots/); compaction/.

FORBIDDEN at run root: any subdir named fine-tuning/, finetuning/, training/,
ft/, logs/, probes/, data/, archive/, repairs/, environment/, figures/; any
Python file other than train.py and probe.py; any *.log/*.csv/*.json other than
infra.json, manifest.json, state.json; README.md. The postrun validator
(python -m capo.utils.checks --stage postrun --repair) physically moves
misplaced files and deletes forbidden subdirs after migrating their contents.

## Dataset caching
The HuggingFace datasets cache stays at its default location
(~/.cache/huggingface/datasets/), OUTSIDE the run dir. Never write a local data/
subdir under the run root — this keeps the run dir small and reusable.

## Dataset load + schema validation
Both train.py and probe.py load the same dataset via datasets.load_dataset. Two
failure modes have killed runs deep in multiprocess.pool after 30+ min of
preprocessing: (1) silent cache fallback — if the Hub is unreachable, datasets
warns and returns a STALE cached snapshot whose columns may not match; (2) late
schema mismatch — a renamed column surfaces as KeyError inside _tokenize_batch
after workers fan out. Catch both at the gate.

`dataset_ref` is NOT always a Hub id. It is an HF `owner/name` id OR a relative
local path `inputs/<file>` (a dataset staged into the run dir and rsynced to the
instance — see the orchestrator's `reports/dataset_source.json`). Detect a local
ref (it contains a data-file extension `.csv/.tsv/.parquet/.json/.jsonl/.fasta`,
or resolves to an existing file relative to CWD which is the run dir at runtime)
and route to the builder form instead of a Hub lookup:
  - csv     → `load_dataset('csv',     data_files=dataset_ref)`
  - tsv     → `load_dataset('csv',     data_files=dataset_ref, sep='\t')`
  - parquet → `load_dataset('parquet', data_files=dataset_ref)`
  - json/jsonl → `load_dataset('json', data_files=dataset_ref)`
  - fasta   → parse with Biopython (`from Bio import SeqIO`) into
              `datasets.Dataset.from_dict({...})`; there is no `fasta` builder.
For a local ref the Hub-fallback checks (steps 3, 5 below) do not apply
(`hub_lookup_failed=False`); the missing-dependency and schema-mismatch checks
(steps 2, 4) apply unchanged.

src/data/dataset.py MUST implement:

load_and_validate_dataset(dataset_ref, split, required_columns, logger, *,
deep_check=False, label_columns=None, min_rows=8, allow_zero_positive_classes=False), which:

1. Captures the datasets logger output via a logging.Handler BEFORE load_dataset.
2. Calls load_dataset(dataset_ref, split=split) once; for split=="all", loads and
   validates every split (train, val/validation, test). An import/engine error here
   (ModuleNotFoundError, ImportError, "No module named ...", "Missing optional
   dependency ...", pandas' "Unable to find a usable engine ... pyarrow or
   fastparquet") is a MISSING DEPENDENCY, not a schema problem: write
   dataset_load_error.json with failure_category="missing_dependency",
   missing_packages parsed from the message, observed_columns=[], error_message,
   then sys.exit(5). Do NOT fall through to the schema path — zero columns means
   nothing loaded, not a missing column; the fix is a one-line pip install.
3. After load, scans captured records for "couldn't be found on the Hugging Face
   Hub" or "Using the latest cached version of the dataset". If hit, sets
   hub_lookup_failed=True and logs one ERROR: "HF Hub lookup failed for
   '{dataset_ref}' — falling back to cached snapshot at {cache_path} (last
   modified {iso})".
4. Validates set(required_columns) <= set(ds.column_names). On mismatch, write
   outputs/dataset_load_error.json with failure_category="data_schema_mismatch"
   plus dataset_ref, split, hub_lookup_failed, cache_path, cache_mtime_iso,
   required_columns, observed_columns, missing_columns, error_message; then
   sys.exit(5). Do NOT let it die inside Dataset.map.
5. On Hub fallback alone (columns present, hub_lookup_failed=True), still write
   outputs/dataset_load_warning.json with the same fields and
   failure_category="hub_fallback_stale_cache", and continue. The monitor
   surfaces this as warn severity without killing the run.

Split synthesis (source-agnostic — gate on "val/test missing", NOT on local vs
Hub). A lone local file (and some single-split Hub datasets) load as a DatasetDict
with only `train` — no val/test. When `split=="all"` and the loaded object lacks
a val/validation AND/OR test split, SYNTHESIZE them deterministically instead of
letting downstream train-on-test or KeyError:
  1. Read `seed` and `split_fractions` (default [0.8, 0.1, 0.1] = train/val/test)
     from configs/training.yaml. probe.py, train.py, and eval MUST read the SAME
     values so the probe stays a valid gate and eval never sees training rows.
  2. Split with `Dataset.train_test_split` twice (train→train+temp, temp→val+test),
     seeded. For single-label classification, pass `stratify_by_column=<label>`;
     for regression / multi-label, unstratified shuffle.
  3. Homology safety (protein sequences): a random split leaks homologs across
     train/test and inflates metrics. PREFER a cluster-aware split (see the
     `clustering` skill — mmseqs2/CD-HIT identity-clustered assignment) for
     sequence modality when feasible; fall back to stratified-random otherwise.
  4. Record `synthesized_splits=true` and `split_strategy` ∈
     {"cluster_aware","stratified_random","random"} in outputs/data_validation.json,
     and log one WARNING when a random split is used on sequence data.
Datasets that already ship train+val+test are untouched (synthesis never fires).

required_columns = the sequence column ("sequence" default, --sequence-column
overrides) ∪ every label column in training.yaml (label_columns or
target_column) ∪ any filter column from --species / --filter-column.

Both train.py AND probe.py MUST call this before any tokenize/filter/map step —
the probe failing here IS the gate doing its job.

### Bespoke dataset construction from raw multi-file sources
When train.py BUILDS its dataset from raw external dumps (e.g. ClinVar
variant_summary.txt + hgvs4variation.txt, multi-table joins, comment-prefaced
TSVs) instead of loading a ready-made HF dataset, `load_and_validate_dataset`
only sees the ASSEMBLED result — the bugs that actually kill these runs happen
earlier, in your parsing/join code. Two failure modes have each killed a run:

1. **KeyError on a column that "should" be there** (e.g. `df["VariationID"]`
   raising KeyError while streaming hgvs4variation.txt). Root cause: the header
   was not where the reader assumed — comment-prefaced files (`#`-preamble with
   one description line per column) make `pd.read_csv` treat a comment as the
   header, so the real column never exists. Rules:
   - Detect the header explicitly: scan for the first line whose tab-split cells
     (stripped of a leading `#`) contain your key column, and pass that index as
     `skiprows`. Do not hard-code a line number.
   - **Assert immediately after EVERY raw read, before you index any column:**
     `assert "VariationID" in df.columns, f"hgvs4variation: header not found — got {list(df.columns)[:12]}"`.
     Never write `df[col]` / `chunk[col]` for an externally-sourced column
     without a preceding membership assert. A `usecols` callable that silently
     drops an unmatched name does NOT protect you — it hides the mismatch until
     the later `[col]` access explodes deep in a stream.
   - Log the resolved header (`header at line N, cols=K`) so a failure is
     diagnosable from the log alone.

2. **`data_schema_mismatch: missing required columns ['gene']`** when the run
   built the dataset itself. Root cause: the columns your construction code
   WRITES drifted from the `required_columns` your loader expects (construction
   wrote `GeneSymbol`; the loader asked for `gene`). Rules:
   - Define the column contract ONCE as a module constant and reference it from
     both the writer (the parquet/csv you emit) and `required_columns`. They must
     name the SAME columns — if you emit `GeneSymbol`, require `GeneSymbol`.
   - After construction, still run `load_and_validate_dataset` on the assembled
     dataset. The raw-read asserts (mode 1) and the final gate are complementary.

probe.py / data_smoke MUST schema-check EVERY raw file the construction reads
(not just the first / the label table): a probe that confirms VariationID in
variant_summary.txt but never checks hgvs4variation.txt gives false confidence
and lets the run die 30+ min later. Assert the key column of each source file.

Deep validation (deep_check=True) — called by train.py once on the instance
immediately before the training loop (NOT by probe.py's smoke). For each loaded
split it additionally checks: (1) non-empty, len(ds) >= min_rows; (2) no
null/empty sequences; (3) no duplicate row_id within a split (cross-split dups
logged as warning, not fatal); (4) sequence↔label alignment — every row has
sequence and every label_columns entry populated, and if a label_mask column
exists, len(row["label_mask"])==len(label_columns) and label_mask.shape==labels.shape
after collation; (5) label domain (e.g. {-1,0,1} masked multi-label, {0,1}
binary, finite categorical, finite floats regression). Then, classification
only: (6) pos_weight sanity — for each class c, pos_train[c]=rows where c is
OBSERVED and ==1, neg_train[c]=rows where c is OBSERVED and ==0 (NOT the -1
masked-out values). Require pos_train[c] >= 1 AND neg_train[c] >= 1 and
0 < neg_train[c]/pos_train[c] < 1e6. A class with pos or neg count 0 fails
unless allow_zero_positive_classes is True (then surfaced in warnings, not
failed_checks). Check 6 catches run binding-esm2-20260617-1614-68f0, where
compute_pos_weights counted neg only on mask==True AND label==1.0 rows, giving
pos_weight=0 for all 10 species — the contract for neg_train[c] is OBSERVED 0
labels, not 1 labels.

On any deep-check failure, write outputs/dataset_load_error.json with
failure_category="data_validation_failed", plus failed_checks (e.g.
["train.pos_weight['human_new']=0.0", "val.split_non_empty"]), per_split_stats
(n_rows/null_sequences/duplicate_row_ids per split), per_class_pos_weight,
warnings, error_message; then sys.exit(5). On pass, write
outputs/data_validation.json with the same stats plus failure_category=null and
ok=true — the finalizer records its summary in RUN_REPORT.md under Data integrity.

Exit codes from this helper: always 5 (missing_dependency, data_schema_mismatch,
or data_validation_failed — the JSON identifies which).

## CPU parallelism
Detect cores at runtime, cap at 32 for I/O safety:
N_CPU = min(multiprocessing.cpu_count(), 32).
Use datasets.map(..., num_proc=N_CPU); DataLoader(num_workers=min(N_CPU, 8),
pin_memory=True, persistent_workers=True). Log at startup:
INFO Hardware: {N_CPU} CPU cores, {VRAM_GB} GB VRAM, num_proc={N_CPU}.

## Progress logging (every stage > 30 s)
Python logging at INFO to stdout. tqdm alone is insufficient; with HF Trainer,
add a TrainerCallback that emits these to the standard logger. Required lines
cover: stage start/done with counts and elapsed for load, filter, tokenize
(with periodic "Stage 3/5 progress: {done}/{total} ({pct}%)"); training start;
per-step "Step {step}/{total}: loss={loss} lr={lr}"; per-epoch "Epoch
{epoch}/{epochs}: train_loss={loss} val_mcc={mcc}"; "Checkpoint saved: {path}";
eval start; "Eval done: macro_mcc={mcc} ({elapsed}s)".

## Third-party logger suppression
Inside setup_logging(), immediately after configuring the root logger and file
handler — and BEFORE any import that triggers them (before import trackio and
trackio.init()) — clamp these to WARNING: httpcore, httpx, urllib3, requests,
huggingface_hub, huggingface_hub.file_download, filelock, fsspec, asyncio.
Without this, trackio.init() floods outputs/train.log with DEBUG noise that
breaks the Haiku monitor's log parsing.

## CSV schemas
results/eval_metrics.csv — append-only, header once, one row per eval call.
Columns in order: run_id, model_id, fine_tune_strategy, dataset_ref, git_sha,
seed, timestamp_iso, epoch, step, global_step, split, n_samples, batch_size,
learning_rate, train_loss_running_mean — plus task metrics:
- classification: val_loss, accuracy, macro_f1, mcc, auroc, precision_macro, recall_macro
- regression: val_loss, mse, rmse, mae, r2, spearman, pearson
- language model: val_loss, perplexity
Missing metrics are EMPTY cells, never 0 or NaN-stringified. git_sha defaults to
"unknown" outside a git repo.

results/train_metrics.csv — append-only, one row per --log-every step: run_id,
timestamp_iso, epoch, step, global_step, train_loss, learning_rate, grad_norm,
throughput_samples_per_s.

results/eval_per_class.csv — classification only, n_classes ≤ 50, per eval call:
timestamp_iso, epoch, step, global_step, split, class_id, class_name, support,
precision, recall, f1.

## status.json contract
outputs/status.json is consumed by the Phase B Haiku monitor. Written by train.py
at: training start ({"state":"running","epoch":0,"step":0,"pid":<pid>,"updated_at":...}),
each epoch end (state running plus epoch, step, loss, val_mcc, ...), and on exit
(state completed or failed). updated_at is ISO 8601 UTC.

Crash safety: wrap main() so ANY uncaught exception writes
{"state":"failed","returncode":1,"updated_at":...} BEFORE re-raising. A run that
dies without flipping state (e.g. a crashing subprocess like boltz predict)
leaves the file frozen at "running" and blinds the monitor — the failure that
billed an A100 idle for 2.5h.

Heartbeat: refresh updated_at at least every 60 s even during long SILENT stages
(background heartbeat thread, or per-unit-of-work in embed/inference loops). The
monitor treats a stale updated_at as a dead-process signal independent of the
PID.

## Streaming metrics — outputs/metrics.jsonl
One JSON line per eval step, e.g. {"epoch":3,"step":480,"global_step":480,
"val_loss":0.41,"val_mcc":0.72,"timestamp_iso":...}. Append in real time after
each eval; never rewrite from scratch on resume (appending keeps one continuous
stream across resumes). Final summary metrics go in results/metrics.json instead
(written once by evaluate.py).

## Plot cadence, required plots, atomic write
Cadence: after every epoch and every --eval-every steps, append the CSV row THEN
regenerate every PNG in results/plots/ from the latest CSV state. Plots are
ALWAYS regenerated from CSV, never accumulated in memory — a partial run must
still produce correct plots.

Cadence plots (from results/eval_metrics.csv after every eval):
- classification: loss_curve.png, mcc_curve.png, macro_f1_curve.png,
  auroc_curve.png (binary only), confusion_matrix.png, per_class_f1.png
- regression: loss_curve.png, rmse_curve.png, pred_vs_true_scatter.png
- language model: loss_curve.png, perplexity_curve.png

**Curve correctness — group by seed, sort by x (this is a common, run-ruining
bug).** eval_metrics.csv is append-only and carries a `seed` column; a multi-seed
run (e.g. 3 LoRA seeds) writes several rows at the SAME `global_step`. If you draw
one line through every val row over `global_step`, the line jumps back to a low
step each time the next seed's rows begin — the curve becomes a zig-zag scribble
that looks "completely off" even though the numbers are fine. Rules for EVERY
cadence curve:
  - Filter to the split you are plotting (`split == "val"`), then **GROUP BY
    `seed`** and draw ONE line per seed (distinct palette colours + a legend), OR
    aggregate across seeds into a mean line with a ±1 std band. Never connect rows
    from different seeds into a single line.
  - **Sort each line by the x column** (`global_step`) before plotting — never rely
    on row order.
  - A single-seed run is just the one-group case; the same code path must handle
    1 and N seeds without special-casing.
  - Use `train_metrics.csv` for the train-loss curve — do NOT accept a `train_csv`
    argument and then ignore it, and do NOT fake per-class F1 from the eval CSV's
    macro_f1 (read eval_per_class.csv). A plotting helper that declares a
    parameter must use it or not declare it.

Test-set plots (classification, produced ONCE at end of training and re-produced
by --eval-only recovery, from results/predictions/test_predictions.csv):
- confusion_matrix_test.png — per-class on the held-out test split (one panel per
  class for multi-label)
- mcc_per_class_test.png — bar chart of test MCC per class (with macro line)
- prediction_distribution_test.png — histogram of predicted scores per class,
  split by true label (positives vs negatives)
- label_vs_pred_test.png — predicted vs true intensities scatter/heatmap
  (faceted grid for multi-label)
A missing test plot at end of training is a Phase 6 finalizer failure that
triggers --eval-only recovery.

Atomic write — matplotlib infers format from the EXTENSION, so the temp file must
keep the real extension. A <name>.png.tmp suffix makes savefig see format "tmp"
and raise ValueError, which can kill the run. Insert the temp marker BEFORE the
extension, pass format= explicitly, then os.replace() (atomic on the same
filesystem):

  root, ext = os.path.splitext(final_path)   # ".../loss_curve", ".png"
  tmp = f"{root}.tmp{ext}"                    # ".../loss_curve.tmp.png"
  fig.savefig(tmp, dpi=150, bbox_inches="tight", format=ext.lstrip(".") or "png")
  os.replace(tmp, final_path)
  plt.close(fig)

The dashboard never serves a half-written file and the format is never inferred
from a temp suffix.

Plotting is non-fatal — it MUST NEVER crash training. Wrap the per-eval
generate_plots(...) in try/except: on any exception log a WARNING with traceback,
record it in reports/plot_manifest.json ({"last_error": "<repr>", "at_step":
<global_step>}), and CONTINUE. The CSVs are the source of truth and are written
BEFORE plotting; plot_eval.py and the finalizer regenerate every PNG from the
CSVs alone, so a cosmetic plotting bug must never lose a run with a valid
checkpoint. Order inside do_eval: write CSVs → save checkpoints → then (guarded)
plot. train.py MUST honor --no-inline-plots and CAPO_DISABLE_INLINE_PLOTS=1 to
skip inline plotting while still writing CSVs and checkpoints — the lever the
recovery loop pulls past a persistent plotting bug.

Reproducibility — train.py MUST emit a standalone src/eval/plot_eval.py on first
eval (idempotent overwrite). Single CLI:
python src/eval/plot_eval.py --csv results/eval_metrics.csv
--train-csv results/train_metrics.csv --per-class-csv results/eval_per_class.csv
--out results/plots/. It imports only pandas + matplotlib + numpy, reproduces
every PNG from the CSVs alone (no model, no checkpoint), and is the SAME function
train.py calls after each eval (factor plotting into a shared src/eval/ module —
never duplicate plotting logic). reports/plot_manifest.json is written at first
eval and refreshed each eval. A missing eval_metrics.csv, plot_eval.py, or
plot_manifest.json at end of training is a Phase 6 finalizer failure; the
finalizer re-runs plotting locally if the CSVs exist but plots do not, so
plot_eval.py's standalone reproducibility is what makes self-healing possible.

## Plot color palette (every PNG)
Black for axes, labels, spines, ticks. For data:

| Role | Hex |
|------|-----|
| Primary | #1E5994 (BLUE_0) |
| Accent | #E6905B (ORANGE_50) |
| Third | #713D8F (PURPLE_0) |
| Fourth | #0E625C (GREEN_0) |
| Reference line 1 | #9B3208 (ORANGE_0) |
| Reference line 2 | #713D8F (PURPLE_0) |
| Noise / missing | #AAAAAA |

Colormaps — NEVER use "tab20", "coolwarm", "YlOrRd", or "Blues". Build from
matplotlib.colors.LinearSegmentedColormap.from_list:
- CMAP_SEQ = from_list("capo_seq", ["#C8DFD9", "#78B5B0", "#0E625C"])
- CMAP_DIV = from_list("capo_div", ["#1E5994", "#FFFFFF", "#9B3208"])
- CMAP_BLUE = from_list("capo_blue", ["#BDD9F5", "#8DB8E2", "#1E5994"])

Define palette constants inline at the top of generate_plots — do NOT rely on a
capo.viz.palette import (generated scripts run on the remote instance without the
capo package).

## Checkpoint saving + HF Hub push
train.py MUST save two checkpoints: checkpoints/best/ (best by val metric,
overwritten on new best) and checkpoints/last/ (most recent at any save point,
ALSO written unconditionally at end of training). Both must contain config.json
plus either model.safetensors (single shard) or model.safetensors.index.json +
numbered shards. Always save with safetensors + sharding:
model.save_pretrained(out_dir, max_shard_size="2GB", safe_serialization=True).

The finalizer pushes ONLY checkpoints/best/ to a private HF Hub repo (namespace
from hf whoami, template capo-{run_id}-best). checkpoints/last/ is retained
locally for debugging/resume but NOT pushed; a missing last/ is a soft failure.
HF_TOKEN is pre-installed at ~/.cache/huggingface/token (mode 600) and exported as
HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the tmux session, so train.py needs no
explicit token; to push during training, HfApi().whoami() and
api.upload_folder(..., private=True) work directly.

## Checkpoint resumption
train.py MUST accept --resume-from-checkpoint <path>: load model weights +
optimizer + scheduler + epoch/global_step and continue. Metrics and status.json
keep appending (not overwriting), so a resumed run produces one continuous
metrics.jsonl. For HF Trainer, forward to
trainer.train(resume_from_checkpoint=<path>).

## Idempotent expensive precompute
Any EXPENSIVE per-item precompute stage before training (e.g. boltz predict
writing per-complex embeddings, a feature pass writing per-shard tensors — any
stage whose unit of work produces a durable file) MUST be idempotent and
resumable: (1) derive the EXPECTED unit set from inputs on disk and the DONE set
from valid outputs present; (2) run only missing = expected − done, and skip+log
if none are missing; (3) NEVER force a full recompute of a stage whose valid
outputs exist (no --override on the main call) and never delete/overwrite a valid
expensive artifact — a force-recompute flag is allowed ONLY in the isolated
probe; (4) cheaply validate an output before trusting it (existence + size floor +
readable container) so a truncated file is recomputed while a complete one is
reused. This lets a crashed run resume in minutes; the resume path
(capo.persistence.run_inventory) plans on exactly these on-disk outputs.

## Eval-only mode
train.py MUST accept --eval-only --checkpoint <path> --out <dir>: (1) load model
+ tokenizer from <path> (typically checkpoints/best/); (2) run
src.eval.evaluate.run_final_evaluation(checkpoint=<path>, out=<dir>) on the test
split; (3) write <dir>/eval_metrics.csv, <dir>/metrics.json, <dir>/plots/, and
reports/evaluation_report.md + reports/plot_manifest.json; (4) skip
trackio.init() (no streaming run, no Space write); (5) exit 0 on success,
non-zero on failure (error to stderr). The finalizer relies on this to recover
missing eval artifacts after a run that crashed late or was preempted.

## Trackio integration
train.py MUST accept --trackio-project, --trackio-run, --trackio-space-id
(optional), and define a TrackioLogger helper that: (1) silences httpcore, httpx,
urllib3, requests, huggingface_hub, huggingface_hub.file_download to ERROR before
trackio.init(); (2) pre-creates the bucket with exist_ok=True —
HfApi().create_repo(repo_id=space_id + "-bucket", repo_type="dataset",
exist_ok=True, private=True) — because trackio.init() calls create_repo WITHOUT
exist_ok, so on a second run the existing bucket returns 409 Conflict and no run
is created; pre-creating absorbs the 409; (3) calls trackio.init(project=...,
name=..., space_id=..., config=...) and sets _ok=True only on success; (4)
provides .log(metrics) and .finish() that no-op silently when _ok is False.
Instantiate once at startup after setup_logging(), pass config=vars(args). When
--trackio-space-id is empty, all tracker calls are silent no-ops. The HF_TOKEN is
read from ~/.cache/huggingface/token automatically by huggingface_hub.

## Shared train-step + precision contract
The autocast / mixed-precision call lives in EXACTLY ONE file: src/train/step.py.
Run binding-esm2-20260617-1614-68f0 died because train.py used the modern
signature autocast(device_type="cuda", ...) with the legacy import
from torch.cuda.amp import autocast — a TypeError at the first backward pass — and
the probe missed it because it re-implemented autocast separately. src/train/step.py
MUST expose:

make_train_step(model, optimizer, criterion, cfg) -> Callable[[batch], dict]

returning step(batch) -> {"loss": float, "grad_norm": float, ...} that performs
ONE forward + backward + optimizer step under cfg.precision in {"fp32","bf16","fp16"}
with cfg.amp_enabled. Import surface: from torch.amp import autocast, GradScaler
(modern, torch 2.1+, supports device_type). NEVER from torch.cuda.amp import
autocast. The function body owns the ENTIRE precision call — building the autocast
context, scaling the loss if fp16, calling optimizer.step(). No other file
(train.py, probe.py, any Trainer callback) may re-instantiate autocast or
GradScaler; a precision tweak edits only this file. Both train.py (training loop)
and probe.py (forward+backward smoke) MUST import and invoke make_train_step from
here. probe_result.json records precision, autocast_mode, and
train_step_signature_ok to make this gate-checkable.

## Canary block
train.py MUST run a short canary INSIDE the training process — after
load_and_validate_dataset(deep_check=True) succeeds, after the DataLoader is
built, before the main epoch loop. The canary is the in-training analogue to the
probe gate: the probe proves the backward pass runs once; the canary proves the
model actually learns. Logic lives in src/train/canary.py:

run_canary(train_step, train_loader, n_steps=200, out_dir=Path("outputs")) -> dict

It runs n_steps forward+backward+optimizer steps via train_step (the SAME
callable and DataLoader train.py uses), records per-step loss, grad_norm, lr,
step_latency_s, and writes outputs/canary_summary.json with: passed, n_steps,
mean_loss, first_loss, last_loss, loss_slope (least-squares slope over the last
n_steps//2 steps), max_grad_norm, n_nonfinite_loss, n_nonfinite_grad, precision,
completed_at_iso.

Pass criteria — ALL must hold:
- n_nonfinite_loss == 0 AND n_nonfinite_grad == 0
- last_loss < first_loss * 1.5 (loss not exploding)
- loss_slope < 0 over the back half, OR last_loss < first_loss if the regression
  is too noisy to fit
- max_grad_norm < 1e4 (gradients not exploding)

On failure, write outputs/canary_failure.json with the same fields plus
failure_category="canary_failed", failed_checks (e.g. ["loss_explosion",
"nan_grad"]), error_message, and remediation_hint (e.g. "lr 10x too high | broken
loss masking | <other>"); then sys.exit(6). The launcher's exit-code path writes
status.json state="failed" and post-launch diagnostics classifies via
failure_category="canary_failed" (see src/capo/orchestration/post_launch_diagnostics.py).

Keep it tight — the canary is one focused gate, not a mini-eval. It does NOT run
validation, save checkpoints, or push to trackio (it completes BEFORE
trackio.init for the main run). Adding scope defeats the purpose. train.py MUST
accept --skip-canary for repair-loop reruns where the canary is known good;
default is canary-enabled.

## Probe script contract — probe.py
CLI: --model-id --seq-length --batch-size --out-json --dataset <ref>. probe.py is
a thin launcher; the real logic lives in src/train/ (or src/utils/) and is SHARED
with the training path, so probe failures correlate with training failures by
construction. It MUST NOT import or initialize trackio. Behavior:

1. Dataset binding check — call load_and_validate_dataset(dataset_ref,
   split="train", required_columns=..., deep_check=False) (smoke, not full
   audit), then run the real tokenize_dataset(...) on the first 64 rows of the
   loaded split. This exercises the exact load+filter+tokenize path train.py
   hits, catching schema drift at the gate. Record dataset_load_ok,
   tokenize_smoke_ok, hub_lookup_failed, cache_path, cache_mtime_iso,
   observed_columns.
2. Build a single batch at the requested batch size and seq length.
3. One forward pass — record forward_ok, forward_latency_s, first_loss,
   peak_memory_gb.
4. One forward + backward via make_train_step imported from src/train/step.py
   (do NOT re-implement autocast in probe.py). Record backward_ok,
   backward_latency_s, peak_memory_gb_after_backward, precision, autocast_mode,
   train_step_signature_ok.
5. Write <out_json> (default probe/probe_result.json) with all of the above plus
   success and, on failure, failure_category ∈ {oom, nan_inf, script_bug,
   data_schema_mismatch, data_validation_failed, resource_mismatch,
   hub_fallback_stale_cache}, error_message, traceback. Always write the JSON
   before exit.
6. Exit codes: 0 success, 2 OOM, 3 import error, 4 generic exception, 5
   data_schema_mismatch or data_validation_failed (from load_and_validate_dataset).

External-inference strategies — if the model's real GPU cost is an EXTERNAL
subprocess (e.g. boltz predict) rather than in-process make_train_step, the probe
MUST additionally run that subprocess on ONE minimal input (1 complex, minimal
sampling/recycling) and record subprocess_smoke_ok, subprocess_cmd,
subprocess_returncode. This catches a missing GPU kernel package (boltz lazily
imports cuequivariance_torch at first forward), a failed weight load, or an
invalid ligand SMILES — at the gate for cents, not after a multi-hour launch. A
synthetic-tensor-only probe for such a strategy is a GATE FAILURE: it tests the
cheap ~4% and skips the expensive, failure-prone ~96%. The probe's GPU work must
import and execute the same modules training will; synthetic tensors are
acceptable only when the real model is genuinely in-process and cheap.

## Evaluate contract — src/eval/evaluate.py
Loads a checkpoint from configs/evaluation.yaml, runs inference on the test
split, computes all task-appropriate metrics, writes results/metrics.json and
reports/evaluation_report.md. Called by train.py at end of training via
run_final_evaluation() and again by the finalizer in --eval-only mode when
results are incomplete. The module MUST expose run_final_evaluation(checkpoint,
out) as a callable so both train.py --eval-only and any future re-entry point
reuse it.

Per-split predictions — at end of training and on every --eval-only invocation,
run_final_evaluation MUST write results/predictions/val_predictions.csv and
results/predictions/test_predictions.csv.
- single-label classification / regression: idx, row_id, true_label, pred_label,
  pred_score
- multi-label (label_columns = [c1, c2, ...]): idx, row_id, then
  true_label_{ci}, pred_label_{ci}, pred_score_{ci} for each class

row_id is the dataset's stable per-row identifier when present, else the integer
index in the eval split. pred_label is the thresholded prediction (per-class for
multi-label); pred_score is the raw sigmoid/softmax probability or regression
value. Append-only is NOT required — each eval call overwrites both files
atomically (.tmp + os.replace). In results/eval_metrics.csv, every eval call
emits BOTH a split="val" row AND a split="test" row when the test split is in
that eval. The finalizer's eval-recovery gate (see fine_tuning_finalizer.md Step
3) treats any missing prediction CSV or test plot as a recoverable failure that
re-invokes train.py --eval-only --checkpoint checkpoints/best/.

## Output for the code-repair-critic (Attempt 3 only)
Output one fenced diff block (unified diff that git apply can consume) followed by
one short paragraph explaining why the prior two repair attempts missed the root
cause. No other prose, no analysis sections, no headers. The orchestrator
validates the diff structurally before applying it.
