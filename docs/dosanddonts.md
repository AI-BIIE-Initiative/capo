## CAPO PLM Fine-Tuning Playbook: Do’s, Don’ts, and tips (Data → Ship)

### Map the space: what happens end-to-end (sequential)

* **1) Define the task**

  * Decide: multi-label classification vs regression vs token-level vs generation
  * Lock **primary metric(s)** that reflect deployment (e.g., **PR-AUC / macro-F1 / balanced accuracy** for imbalanced multi-label)

* **2) Pick base PLM + tuning method**

  * Full fine-tune vs **PEFT (LoRA/adapters)** vs freezing most layers

* **3) Lock the data contract + version raw data**

  * Schema, label columns, “unknown” conventions (e.g., `-1`)
  * Save raw snapshots (immutable)

* **4) Data cleaning + preprocessing**

  * Sequence cleaning, ID normalization rules, label cleaning, outlier checks
  * Log counts at every step (before/after + reasons)

* **5) Deduplicate + cluster sequences**

  * Compute similarity / identity clusters; store **cluster IDs**

* **6) Split train/val/test without leakage**

  * Prefer clustered split by identity; align with deployment scenario
  * Freeze a test set

* **7) Dataset statistics & sanity checks**

  * Per-split label distribution, sequence length distribution, missingness patterns
  * Detect extreme values / outliers (normalization decisions)

* **8) Class imbalance strategy**

  * Weighted loss and/or sampling (mild oversampling + regularization)

* **9) Tokenization & batching policy**

  * Correct tokenizer/alphabet; set **max_len**, padding, truncation/cropping, bucketing

* **10) Model head + objective**

  * Correct pooling, correct loss (multi-label != softmax CE), correct masking

* **11) Training loop plumbing**

  * Mixed precision / grad accumulation / clipping / checkpointing / deterministic seeds

* **12) Canary batch + short dev run**

  * One forward+backward sanity, then a small run to validate the pipeline

* **13) Hyperparameter tuning (core knobs)**

  * LR/warmup, batch/accum, max_len, regularization, imbalance parameters

* **14) Full training with structured tracking**

  * TrackIO logging, metrics, dataset drift checks, runtime performance

* **15) Evaluation + error analysis + calibration**

  * Per-class metrics, confidence intervals, thresholding strategies

* **16) Export + ship**

  * Save model + tokenizer + preprocessing + label mapping + config + inference snippet
  * Version artifacts and store durably

* **17) Final test once**

  * Do not select checkpoints based on test results; then ship + archive logs cleanly

---

## Dos and Don'ts + Tips list by layer

#### 1) Task definition and labeling

* **Do**

  * Write down *exactly* what “positive”, “negative”, and “unknown” mean (e.g., `1`, `0`, `-1`) and enforce it everywhere.
  * Choose metrics that match imbalance realities:

    * **PR-AUC**, **macro-F1**, **balanced accuracy**, per-label ROC/PR curves.
  * Decide early for sequence-level tasks:

    * pooling choice: **mean** vs **[CLS]** vs attention pooling
    * truncation policy: head/tail/random/centered crop
* **Don’t**

  * Don’t treat **unknown** as negative by default.
  * Don’t change label encoding mid-pipeline without versioning and explicit migration.

#### 2) Data processing and dataset statistics (ACE2 binding pain points)

* **Do**

  * Compute and log dataset stats **before training**:

    * per-label positives, missingness (`-1` count), multi-positive frequency (“binds to >1 animal”), co-occurrence matrix
    * length histogram; min/median/max; “long tail” outliers
    * for numeric features: max vs mean (max huge → consider normalization/outlier handling)
  * Keep a **discard pile**: file of removed rows + reason (you will need this later).
  * Add explicit checks that catch label mistake:

    * assert that labels are only in `{ -1, 0, 1 }` (or expected set)
    * assert that **positives exist** per label after preprocessing
* **Don’t**

  * Don’t “rename” labels by intuition (e.g., converting `-1→0` and `0→1` without verifying existence of `1`s).
  * Don’t silently drop extreme lengths—either enforce max_len or explicitly log and justify dropping.

#### 3) Split without leakage (the big one)

* **Do**

  * Split to reflect deployment: new proteins, new species, new family, etc.
  * Prefer **clustered split by sequence identity** (e.g., 30–70% depending on difficulty target).
  * Store cluster IDs and ensure **no cluster crosses splits**.
  * Ensure augmentation (mutations/crops) happens **after** the split.
* **Don’t**

  * Don’t random split proteins as a default—too often leaks homologs.
  * Don’t let cleaning/normalization differ between train and eval paths.

#### 4) Class imbalance handling (ACE2 multi-animal binding reality)

* **Do**

  * Start with the safest baseline: **weighted loss** (often simplest and stable).
  * If sampling is needed:

    * **mild oversampling + strong regularization** tends to beat “oversample like crazy”
    * keep validation distribution realistic; don’t “fix” val via sampling unless you explicitly want that
  * Track these every run:

    * per-label positive rate in train/val/test
    * sampler effective distribution (what the model actually sees)
* **Don’t**

  * Don’t rely on accuracy with imbalanced multi-label.
  * Don’t oversample so hard that you train on near-duplicates and overfit quickly.

**weighted sampling approach (if data imbalance is present)**

* **Do**

  * Weight samples using:

    * (a) “mixed-label” boost (examples with >1 observed positive label)
    * (b) inverse frequency of positive labels
  * Use `WeightedRandomSampler(replacement=True)` and explicitly control `num_samples`.
* **Don’t**

  * Don’t set `num_samples` huge without monitoring diversity; it can turn training into “repeat the same minority examples”.


```python

def compute_sample_weights(df, label_cols, mixed_boost, label_boost):
    labels_raw = df[label_cols].values.astype(np.float32)
    observed = labels_raw != -1
    positives = (labels_raw == 1) & observed
    pos_counts = positives.sum(axis=0)
    pos_counts = np.clip(pos_counts, 1.0, None)
    inv_freq = positives * (1.0 / pos_counts)
    label_weight = inv_freq.sum(axis=1)
    mixed_flag = (df["no_labels"] > 1).astype(np.float32).to_numpy()
    weights = 1.0 + mixed_boost * mixed_flag + label_boost * label_weight
    return torch.tensor(weights, dtype=torch.double)
# [...]
sampler = WeightedRandomSampler(
        weights=sample_weights,
        # num_samples=len(sample_weights),
        num_samples=int(1e6), 
        replacement=True,
    )
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    collate_fn=collate_fn,
)
```

* `compute_sample_weights`: inverse-frequency per label + mixed-positives boost
* `WeightedRandomSampler(... num_samples=int(1e6), replacement=True)` to upsample rare positives

#### 5) Tokenization, max length, truncation, batching

* **Do**

  * Confirm the model’s **max context window** and tokenizer/alphabet match the PLM.
  * Decide max length strategy:

    * fixed max_len + truncation is simplest
    * dynamic padding is usually faster if batching is well-behaved
  * For long proteins:

    * chunking / sliding window + pooling across chunks
  * Use bucketing by length for variable-length proteins (big throughput win).
* **Don’t**

  * Don’t reuse token caches if **tokenizer or max_len changed** (cache keys must include them).
  * Don’t let padding tokens contribute to loss (mask them out).

#### 6) Model head, objective, and loss correctness

* **Do**

  * For multi-label classification: use **sigmoid + BCEWithLogitsLoss** (or equivalent), not softmax CE.
  * Mask padding tokens in loss and metrics if any token-level component exists.
  * If using [CLS], verify the architecture actually supports/uses it meaningfully.
* **Don’t**

  * Don’t compute metrics on logits when probabilities are expected (or vice versa) without being explicit.

#### 7) Hyperparameters (measure, don’t eyeball)

* **Do**

  * Use reasonable starting ranges:
    * full fine-tune LR: ~`1e-5` to `5e-5`
    * LoRA/adapters: often higher (~`1e-4`)
  * Batch size: as large as fits **without excessive truncation** (watch long outliers).
  * Regularization:
    * weight decay ~0.01 if overfitting
    * dropout ~0.1 if overfitting
  * Stability:
    * gradient clipping 0.5–1.0 if unstable
    * warmup schedule if needed
* **Don’t**

  * Don’t change 5 knobs at once.
  * Don’t compare runs with different splits (invalid comparison).

#### 8) Training loop and logging (trackio / wandb)

* **Do**

  * Log as first-class outputs (structured, consistent keys):

    * train/val loss
    * task metrics (macro-F1, PR-AUC, Spearman, etc.)
    * LR, grad norm, steps/sec, GPU utilization
    * % truncated + length distribution per split
    * label distribution per split
  * Always run a **canary batch**:

    * 1 forward + backward
    * loss finite, gradients nonzero, shapes correct
    * assertions: no NaNs/inf, labels in range, padding mask correct
  * Do a short dev run (few thousand steps) to validate pipeline before full training.
* **Don’t**

  * Don’t evaluate in train mode (dropout on).
  * Don’t forget accumulation logic / `zero_grad` correctness.

#### 9) Performance + systems knobs

* **Do**

  * Mixed precision (fp16/bf16) where appropriate; consider grad scaling/clipping if overflow.
  * Gradient accumulation to simulate larger batch.
  * Gradient checkpointing to save memory (accept slower).
  * Use pin_memory and non-blocking transfers; tune workers/prefetch.
  * Consider flash attention / optimized kernels if available.
* **Don’t**

  * Don’t crank dataloader workers on network storage (can be slower).
  * Don’t ignore OOM caused by rare long sequences—cap max_len or drop extreme outliers explicitly.

#### 10) Evaluation and reporting (be strict, be honest)

* **Do**

  * Evaluate on the **strict split** you defined (clustered, family, etc.).
  * Report:

    * primary metric + (if possible) bootstrap CIs
    * per-class metrics for imbalanced labels
    * calibration / threshold selection strategy for multi-label
  * Keep a “frozen” test set and touch it once at the end.
* **Don’t**

  * Don’t choose “best checkpoint” on test (that’s training on test).
  * Don’t change clustering thresholds across experiments without recording it.

#### 11) Storage layout, caching, reproducibility

* **Do**

  * Use a predictable layout:

    * `data/raw/` immutable inputs
    * `data/processed/` cleaned + split + metadata
    * `artifacts/tokenized_cache/` (optional)
    * `runs/<date>/<experiment_name>/` with `config.yaml`, `metrics.jsonl`, `checkpoints/`, `logs/`, `predictions/`
  * Save exact:

    * config, label mapping, preprocessing transforms, normalization constants
    * git commit hash + package versions + seeds
* **Don’t**

  * Don’t store checkpoints only on ephemeral disks.
  * Don’t save weights without the preprocessing rules (can’t reproduce inference).

---

### Known error list (Last Updated 28.01.26)
* Train/val/test leakage via homologs (protein) (most common, most damaging) 
    * Near-duplicate sequences (high identity)
    * Same protein family with conserved domains
    * Local homology: even if overall identity is low, short regions can still match strongly and leak signal (common with domains/motifs).
    * In genomes: repeats/duplicated segments spanning splits can leak.
    * -> homolog aware splits
* Label leakage via metadata correlated with label
* Duplicate sequences across splits under different IDs
* Incorrect masking (padding tokens included in loss)
* Silent truncation (model never sees important regions; e.g., C-terminus)
* Wrong objective (CE for multi-label; wrong pooling)
* Caching mismatch (token cache built with different max_len/tokenizer)
* Augmentation before split → siblings leak into test
* Seeds not fixed → “improvements” that don’t reproduce
* Different preprocessing in train vs eval code paths
* Sampler breaks validation distribution unintentionally
* Not having enough logging, progress cues (silent errors or code is hanging) → need proper cues to make good choices 
* Plots are not accurate or reliable (legend overlaps plot, not the correct values are plotted)
* obvious error : AttributeError: 'int' object has no attribute 'sqrt'
    The above exception was the direct cause of the following exception:
    Traceback (most recent call last):
        mcc = (tp * tn - fp * fn) / np.sqrt(denom)
                                ^^^^^^^^^^^^^
    TypeError: loop of ufunc does not support argument 0 of type int which has no callable sqrt method 
* Not making the code modular enough if you do not prompt it to. If you do not specify anything the model will just keep on modifying / appending on the main existing script instead of modularizing the full code. (i.e when implementing multiple models or training strategies, separate model classes from train.py and so on)
* When doing refactoring, agent forgot some imports and therefore the code execution would fail.

---

### Minimal checklist

* Define task + metrics + deployment assumptions
* Choose base model + tuning method (full vs PEFT)
* Lock data schema + version raw data
* Clean sequences + clean labels (log every filter)
* Deduplicate + cluster sequences (store cluster IDs)
* Split train/val/test with leakage-safe rules
* Decide imbalance strategy (weights/sampler)
* Implement tokenization + padding + truncation policy
* Implement head + correct loss + masking
* Canary batch + assertions
* Short dev run to validate pipeline
* Tune core hyperparams (LR/warmup/batch/max_len)
* Full training with TrackIO logging + checkpoints
* Evaluate + error analysis + calibration
* Export artifact: weights + tokenizer + preprocessing + label map + config
* Final test once, then ship
* Archive logs cleanly per task for auditing

