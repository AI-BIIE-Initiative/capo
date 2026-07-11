"""leakage_firewall.py — deterministic Stage-2 external evaluation.

Stage 1 (preprocessing) produces candidate outputs without any access to the
held-out golden set. Stage 2 (this module) loads the golden set, joins on
predefined stable keys, and computes all result tables.

The leakage guarantee comes from the call graph, not from hashing:

* datasets is imported lazily inside GoldEvaluator.load_gold — no
  gold data can enter this process before that call runs.
* The harness only calls Stage 2 after Stage 1 returns, so there is no
  code path where gold can leak into preprocessing.
* write_processed_examples rejects gold-derived columns at write time,
  so accidentally including gold info in a Stage-1 artifact is caught early.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Pandas and pyarrow are required for the evaluation phase. They are not lazy
# because every public function in this module depends on them. The HuggingFace
# datasets library, by contrast, is imported lazily inside GoldEvaluator
# so that the firewall module can be imported without pulling in any HF code.
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical schemas
# ---------------------------------------------------------------------------

# Columns the processed_examples.parquet artifact MUST contain. Stage 1 writes
# these; Stage 2 reads them. Gold-derived columns (gold_label, annotation_*,
# match_status) are explicitly forbidden in Stage 1 and only appear in
# gold_alignment.parquet emitted by Stage 2.
CANDIDATE_SCHEMA: tuple[str, ...] = (
    "run_id",
    "system_name",
    "processed_example_id",
    "raw_file_id",
    "library_id",
    "species_raw",
    "species_normalized",
    "sequence",
    "sequence_hash",
    "predicted_label",
    "label_source",
    "qc_status",
    "deduplication_status",
    "processing_stage",
    "error_code",
)

# Columns that must NOT appear in a Stage-1 artifact. If any of these are
# present in processed_examples.parquet the firewall raises.
GOLD_DERIVED_FORBIDDEN: tuple[str, ...] = (
    "gold_label",
    "gold_example_id",
    "annotation_correct",
    "match_status",
)

# ---------------------------------------------------------------------------
# Species normalization
# ---------------------------------------------------------------------------

# Map raw species strings (filename fragments, agent-emitted labels) to the
# canonical species name used in BIIE-AI/ace2_binding. Conservative — only
# entries we actually expect to see. Anything unmapped flows through unchanged
# so reviewers can spot it in error_analysis.csv.
_SPECIES_NORMALIZATION: dict[str, str] = {
    "human": "human",
    "humannew": "human_new",
    "human_new": "human_new",
    "mouse": "mouse",
    "mus": "mouse",
    "rat": "rat",
    "dog": "dog",
    "cat": "cat",
    "cattle": "cattle",
    "cow": "cattle",
    "bovine": "cattle",
    "horse": "horse",
    "equine": "horse",
    "monkey": "monkey",
    "macaque": "monkey",
    "mink": "mink",
    "bat": "bat",
    "ihbat": "ihbat",
    "ih_bat": "ihbat",
    "pangolin": "pangolin",
    "hamster": "hamster",
}


def canonicalize_species(raw: str | None) -> str:
    """Normalize a raw species string to the BIIE-AI/ace2_binding canonical name.

    Unknown values are lowercased and stripped but otherwise passed through, so
    a mismatch shows up as species_misnormalization in error analysis rather
    than being silently dropped.
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    s_compact = re.sub(r"[\s\-]+", "_", s)
    return _SPECIES_NORMALIZATION.get(s_compact, s_compact)


def sequence_hash(seq: str | None) -> str:
    """Stable short hash for a sequence — first 12 hex chars of SHA256."""
    if seq is None:
        return ""
    return hashlib.sha256(str(seq).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Stage-1 artifact writers
# ---------------------------------------------------------------------------

def write_evaluation_config(out_dir: Path, cfg: dict) -> Path:
    """Write the frozen evaluation contract as YAML.

    This is the pre-registered contract. It MUST be written before any system
    is launched and MUST NOT be modified after gold loading.
    """
    import yaml

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def write_run_metadata(out_dir: Path, metadata: dict) -> Path:
    """Write run_metadata.json with full provenance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=False), encoding="utf-8")
    return path


def write_raw_data_manifest(out_dir: Path, rows: Sequence[dict]) -> Path:
    """Write raw_data_manifest.csv listing each input file with checksum."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raw_data_manifest.csv"
    cols = [
        "raw_file_id",
        "file_path_or_uri",
        "library_id",
        "species_expected",
        "read_count",
        "file_size_bytes",
        "sha256",
        "included_in_run",
        "exclusion_reason",
    ]
    df = pd.DataFrame(list(rows), columns=cols) if rows else pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)
    return path


def write_processed_examples(
    out_dir: Path,
    df: pd.DataFrame,
    sample_n: int = 200,
    sample_seed: int = 0,
) -> tuple[Path, Path]:
    """Write the canonical processed_examples.parquet plus a CSV sample.

    Raises ValueError if the frame is missing any column from
    CANDIDATE_SCHEMA or contains any column from GOLD_DERIVED_FORBIDDEN.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [c for c in CANDIDATE_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(
            f"processed_examples is missing required columns: {missing}. "
            f"Stage 1 must produce the full canonical schema; see CANDIDATE_SCHEMA."
        )
    leaks = [c for c in GOLD_DERIVED_FORBIDDEN if c in df.columns]
    if leaks:
        raise ValueError(
            f"processed_examples contains gold-derived columns {leaks}. "
            f"These may only appear in gold_alignment.parquet."
        )

    df_canonical = df[list(CANDIDATE_SCHEMA)].copy()
    parquet_path = out_dir / "processed_examples.parquet"
    df_canonical.to_parquet(parquet_path, index=False)

    n = min(sample_n, len(df_canonical))
    sample = (
        df_canonical.sample(n=n, random_state=sample_seed)
        if n > 0 else df_canonical.head(0)
    )
    sample_path = out_dir / "processed_examples_sample.csv"
    sample.to_csv(sample_path, index=False)
    return parquet_path, sample_path


# ---------------------------------------------------------------------------
# Candidate adapter — agent CSV -> canonical schema
# ---------------------------------------------------------------------------

@dataclass
class CandidateAdapter:
    """Translate a system's native cleaned CSV into the canonical schema.

    The agents (CAPO and the General Coding Agent) emit a per-species CSV with
    columns like species, aa_seq, sort, binding — the shape of
    that file is system-specific. The adapter is the single place where that
    raw output is lifted to the canonical CANDIDATE_SCHEMA so downstream
    evaluation has a stable contract.

    Stage-1 only: the adapter must not touch the golden set.
    """
    run_id: str
    system_name: str
    sequence_col: str = "aa_seq"
    species_col: str = "species"
    binding_col: str = "binding"
    sort_col: str | None = "sort"
    library_col: str | None = None
    raw_file_col: str | None = None

    # Tolerated alternative names for each canonical input column. The agents
    # have historically drifted on case (Sequence vs aa_seq) and on
    # synonyms, so the adapter case-insensitively resolves these before
    # failing. Order matters — first match wins, so AA-specific names come
    # before the generic sequence (which is often DNA in yeast-display
    # outputs and would silently corrupt the join).
    _SEQUENCE_ALIASES = (
        "aa_seq", "aa_sequence", "protein_seq", "protein_sequence",
        "amino_acid_sequence", "aaseq", "aa", "sequence", "seq",
    )
    _SPECIES_ALIASES = ("species", "species_name", "organism", "host", "ortholog")
    _BINDING_ALIASES = ("binding", "label", "binder", "bind_label", "class")

    # Heuristic: if ≥99% of sampled chars from the resolved sequence column
    # fall in {A,C,G,T,N}, treat as DNA and refuse — the resolved column was
    # the wrong one (almost certainly a generic sequence column shadowing
    # the real aa_sequence).
    _DNA_ALPHABET = set("ACGTNacgtn")

    @staticmethod
    def _resolve_column(df: pd.DataFrame, preferred: str, aliases: tuple[str, ...]) -> str | None:
        """Case-insensitively find preferred (or any alias) in df columns.

        Returns the actual column name as it appears in df, or None if
        none matched.
        """
        lower_map = {c.lower(): c for c in df.columns}
        for cand in (preferred, *aliases):
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=list(CANDIDATE_SCHEMA))

        seq_col = self._resolve_column(df, self.sequence_col, self._SEQUENCE_ALIASES)
        sp_col = self._resolve_column(df, self.species_col, self._SPECIES_ALIASES)
        bind_col = self._resolve_column(df, self.binding_col, self._BINDING_ALIASES)

        # Fail if neither sequence nor species could be resolved
        missing = []
        if seq_col is None:
            missing.append(f"sequence (looked for: {self.sequence_col!r}, "
                           f"aliases: {self._SEQUENCE_ALIASES})")
        if sp_col is None:
            missing.append(f"species (looked for: {self.species_col!r}, "
                           f"aliases: {self._SPECIES_ALIASES})")
        if missing:
            raise ValueError(
                f"CandidateAdapter could not resolve required columns "
                f"[{'; '.join(missing)}] in agent CSV. "
                f"Columns found in input: {sorted(df.columns.tolist())}"
            )

        species_raw = df[sp_col].astype(str)
        species_normalized = species_raw.map(canonicalize_species)
        sequence = df[seq_col].astype(str)

        # Refuse DNA in the sequence column. The agents emit a separate DNA
        # Sequence and AA aa_sequence; if the alias resolver landed on
        # the DNA one (because aa_sequence wasn't present), downstream
        # exact-string merge against gold's AA RBDs would silently produce 0
        # matches. Sample up to 200 non-empty rows.
        sample = [s for s in sequence.head(1000) if s and s != "nan"][:200]
        if sample:
            joined = "".join(sample)
            if joined:
                frac_dna = sum(c in self._DNA_ALPHABET for c in joined) / len(joined)
                if frac_dna >= 0.99:
                    raise ValueError(
                        f"CandidateAdapter resolved sequence column to {seq_col!r}, "
                        f"but its contents look like DNA "
                        f"(~{frac_dna:.1%} ACGTN chars). Stage-2 matches against gold's "
                        f"amino-acid sequences — DNA would produce 0 matches. "
                        f"Rename the AA column to aa_sequence or pass "
                        f"sequence_col=<name> to CandidateAdapter. "
                        f"Columns found in input: {sorted(df.columns.tolist())}"
                    )
        binding = df[bind_col].astype(str).str.lower() if bind_col else ""

        # Predicted binary label: 1 = binder, 0 = non-binder. Anything else
        # (NaN, "ambiguous", etc.) is encoded as -1 so downstream error analysis
        # can pick it up as ambiguous_annotation.
        if isinstance(binding, pd.Series):
            predicted_label = binding.map(
                lambda v: 1 if v == "bind" else (0 if v in ("non", "nonbind", "non_bind") else -1)
            )
        else:
            predicted_label = -1

        n = len(df)
        out = pd.DataFrame({
            "run_id": [self.run_id] * n,
            "system_name": [self.system_name] * n,
            "processed_example_id": [f"{self.system_name}-{i:08d}" for i in range(n)],
            "raw_file_id": (
                df[self.raw_file_col].astype(str) if self.raw_file_col and self.raw_file_col in df.columns
                else [""] * n
            ),
            "library_id": (
                df[self.library_col].astype(str) if self.library_col and self.library_col in df.columns
                else [""] * n
            ),
            "species_raw": species_raw if isinstance(species_raw, pd.Series) else [""] * n,
            "species_normalized": species_normalized if isinstance(species_normalized, pd.Series) else [""] * n,
            "sequence": sequence if isinstance(sequence, pd.Series) else [""] * n,
            "sequence_hash": [sequence_hash(s) for s in (sequence if isinstance(sequence, pd.Series) else [""] * n)],
            "predicted_label": predicted_label if isinstance(predicted_label, pd.Series) else [-1] * n,
            "label_source": [f"sort_{self.sort_col}"] * n if self.sort_col else ["agent"] * n,
            "qc_status": ["pass"] * n,
            "deduplication_status": ["unique"] * n,
            "processing_stage": ["final"] * n,
            "error_code": [""] * n,
        })
        return out


# ---------------------------------------------------------------------------
# System run descriptor
# ---------------------------------------------------------------------------

@dataclass
class SystemRun:
    """One frozen Stage-1 result, ready for Stage-2 evaluation."""
    name: str                       # "CAPO" | "General Coding Agent"
    setting: str                    # "budget_matched" | "best_effort"
    run_dir: Path                   # frozen Stage-1 artifact directory
    runtime_seconds: float
    cost_usd: float
    raw_reads_processed: int | None = None
    reads_retained_after_qc: int | None = None
    failure_count: int = 0
    total_attempts: int = 1

    @property
    def runtime_hours(self) -> float:
        return self.runtime_seconds / 3600.0


# ---------------------------------------------------------------------------
# Gold evaluator
# ---------------------------------------------------------------------------

class GoldEvaluator:
    """Loads gold and computes all Stage-2 result tables.

    Side-effect-light by design: nothing is loaded from HuggingFace until
    load_gold() is called. The lazy import of datasets is the
    leakage boundary — no gold data can enter this process before that call.
    """

    def __init__(
        self,
        systems: Sequence[SystemRun],
        eval_config: dict,
        out_dir: Path,
        rng_seed: int = 0,
    ):
        if not systems:
            raise ValueError("GoldEvaluator requires at least one SystemRun")
        self.systems = list(systems)
        self.eval_config = eval_config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng_seed = rng_seed
        self._gold: pd.DataFrame | None = None

    # -- gold loading --------------------------------------------------------

    def load_gold(self) -> pd.DataFrame:
        """Load the golden set from HuggingFace.

        The datasets library is imported lazily here — that's the actual
        leakage boundary. Stage 1 cannot reach this code path because the
        harness only calls Stage 2 after Stage 1 returns.
        """
        if self._gold is not None:
            return self._gold

        gold_cfg = self.eval_config.get("gold_set", {})
        dataset_ref = gold_cfg.get("dataset_ref") or self.eval_config.get("gold_dataset_name")
        if not dataset_ref:
            raise ValueError("eval_config must specify gold_set.dataset_ref")
        split = gold_cfg.get("split", "test")
        species_filter = gold_cfg.get("species_filter")  # list[str] | None
        single_label_only = bool(gold_cfg.get("single_label_only", True))

        # Lazy import — keeps the firewall module importable without HF deps
        # and makes the leakage boundary explicit in the call graph.
        from datasets import load_dataset  # noqa: WPS433 — intentional lazy import

        ds = load_dataset(dataset_ref, split=split)
        df = ds.to_pandas()

        # Single-label-only mirrors the standing rule in the ACE2 prompts.
        if single_label_only and "no_labels" in df.columns:
            df = df[df["no_labels"] == 1].copy()

        # Reshape from wide (one column per species) to long (one row per
        # gold (sequence, species) pair). The species columns are everything
        # that is neither metadata nor the sequence itself; the eval_config
        # may override this with an explicit species_columns list.
        species_cols: list[str] = gold_cfg.get("species_columns") or [
            c for c in df.columns if c not in {
                "aa_seq", "len", "ed", "sort", "no_labels",
                "split", "pca_kmeans_split", "cluster",
            } and df[c].dtype in (int, "int64", "Int64", "int32", "Int32")
        ]
        if not species_cols:
            raise ValueError(
                "Could not infer species columns from gold set. Set "
                "eval_config.gold_set.species_columns explicitly."
            )

        long = df[["aa_seq", *species_cols]].melt(
            id_vars=["aa_seq"],
            value_vars=species_cols,
            var_name="species_gold",
            value_name="gold_label",
        )

        # Drop missing labels (-1). The user spec treats those as
        # ambiguous_annotation when they appear after alignment, but for the
        # gold coverage denominator we want labeled examples only.
        long = long[long["gold_label"].isin([0, 1])].copy()

        if species_filter:
            wanted = {canonicalize_species(s) for s in species_filter}
            long["species_gold_norm"] = long["species_gold"].map(canonicalize_species)
            long = long[long["species_gold_norm"].isin(wanted)].copy()
        else:
            long["species_gold_norm"] = long["species_gold"].map(canonicalize_species)

        long["gold_example_id"] = (
            "gold-"
            + long["species_gold_norm"]
            + "-"
            + long["aa_seq"].map(sequence_hash)
        )
        long = long.reset_index(drop=True)
        self._gold = long
        return long

    def _auto_restrict_gold_to_candidate_species(self) -> None:
        """Filter self._gold to species the candidates actually processed.

        Called when species_filter is null in the eval contract. Scans every
        system's frozen parquet for its unique species_normalized values,
        unions them across systems, then restricts the in-memory gold so
        downstream macro-averages are taken over the species both sides cover.
        Empty / nan species values are ignored.
        """
        if self._gold is None:
            return

        candidate_species: set[str] = set()
        for s in self.systems:
            parquet_path = s.run_dir / "processed_examples.parquet"
            if not parquet_path.exists():
                continue
            try:
                cand = pd.read_parquet(parquet_path, columns=["species_normalized"])
            except Exception:
                continue
            for v in cand["species_normalized"].dropna().unique():
                v_str = str(v).strip()
                if v_str and v_str.lower() != "nan":
                    candidate_species.add(v_str)

        if not candidate_species:
            print(
                "[gold-auto-restrict] no candidate species found across systems; "
                "keeping full gold panel."
            )
            return

        before = len(self._gold)
        gold_species_all = sorted(self._gold["species_gold_norm"].unique().tolist())
        overlap = sorted(candidate_species & set(gold_species_all))
        if not overlap:
            print(
                f"[gold-auto-restrict] candidate species {sorted(candidate_species)} "
                f"have zero overlap with gold species {gold_species_all}. "
                f"Keeping full gold panel (every system will score 0)."
            )
            return

        self._gold = self._gold[self._gold["species_gold_norm"].isin(overlap)].copy()
        after = len(self._gold)
        print(
            f"[gold-auto-restrict] species_filter was null → restricted gold to "
            f"{len(overlap)} candidate species: {overlap}.\n"
            f"  gold rows: {before:,} → {after:,} "
            f"(macro-averages will be taken over these {len(overlap)} species, "
            f"not the full {len(gold_species_all)}-species panel)"
        )

    # -- alignment + metrics -------------------------------------------------

    def align_system(self, system: SystemRun) -> pd.DataFrame:
        """Join one system's frozen candidate output to gold."""
        if self._gold is None:
            self.load_gold()
        gold = self._gold
        assert gold is not None  # for type checkers

        processed_path = system.run_dir / "processed_examples.parquet"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"{processed_path} missing for system {system.name!r}"
            )
        cand = pd.read_parquet(processed_path)

        # Forbid gold-derived columns at the join boundary too — defence in
        # depth in case someone hand-edits the frozen artifact.
        leaks = [c for c in GOLD_DERIVED_FORBIDDEN if c in cand.columns]
        if leaks:
            raise RuntimeError(
                f"System {system.name!r} candidate file contains gold-derived "
                f"columns {leaks} — frozen artifact has been tampered with."
            )

        self._print_alignment_diagnostic(system, gold, cand)

        # Per-species candidate aggregation: collapse duplicate
        # (species_normalized, sequence) keys, preferring rows labeled as
        # binders so a sequence detected in any binder library counts as
        # predicted=1. This is the only "duplicate handling" rule the harness
        # applies — it does NOT change which gold examples are evaluated.
        cand_g = (
            cand[cand["predicted_label"].isin([0, 1])]
            .sort_values("predicted_label", ascending=False)
            .drop_duplicates(subset=["species_normalized", "sequence"], keep="first")
            [["species_normalized", "sequence", "predicted_label", "processed_example_id"]]
            .rename(columns={
                "species_normalized": "species_predicted",
                "sequence": "aa_seq",
                "predicted_label": "predicted_label",
            })
        )

        merged = gold.merge(
            cand_g,
            left_on=["species_gold_norm", "aa_seq"],
            right_on=["species_predicted", "aa_seq"],
            how="left",
        )

        merged["matched_gold_example_id"] = merged["gold_example_id"]
        merged["match_status"] = merged["predicted_label"].apply(
            lambda v: "matched" if pd.notna(v) else "unmatched_gold"
        )
        merged["label_correct"] = (
            merged["predicted_label"].fillna(-2).astype(int) == merged["gold_label"].astype(int)
        )
        # annotation_exact_match is, for the ACE2 task, the same as
        # label_correct on matched rows. We keep the columns distinct so the
        # framework can be extended to richer annotations later.
        merged["annotation_exact_match"] = merged["label_correct"] & merged["match_status"].eq("matched")
        merged["species_correct"] = merged["species_gold_norm"] == merged["species_predicted"].fillna("")
        merged["sequence_exact_match"] = merged["match_status"].eq("matched")
        merged["error_type"] = merged.apply(_classify_error, axis=1)

        merged.insert(0, "system_name", system.name)
        merged.insert(0, "run_id", system.run_dir.name)
        return merged

    @staticmethod
    def _print_alignment_diagnostic(
        system: SystemRun,
        gold: pd.DataFrame,
        cand: pd.DataFrame,
    ) -> None:
        """Print a side-by-side gold/candidate health summary before joining.

        This is the load-bearing observability hook for "why is coverage 0?".
        It surfaces three failure modes that the exact-match merge would
        otherwise silently absorb:

        * Stage-1 garbage: the candidate parquet has rows but key fields are
          empty (the CAPO-style silent CandidateAdapter fallback).
        * Length-distribution mismatch: gold has full-length sequences but
          candidate has truncated read windows (no exact match possible).
        * Species-set non-overlap: gold and candidate share no species after
          canonicalization.
        """
        def _len_stats(series: pd.Series) -> dict[str, float]:
            lens = series.astype(str).str.len()
            if lens.empty:
                return {"n": 0, "min": 0, "p50": 0, "p99": 0, "max": 0}
            return {
                "n": int(lens.count()),
                "min": int(lens.min()),
                "p50": int(lens.quantile(0.5)),
                "p99": int(lens.quantile(0.99)),
                "max": int(lens.max()),
            }

        cand_nonempty_seq = cand["sequence"].astype(str).str.len() > 0
        cand_nonempty_sp = cand["species_normalized"].astype(str).str.len() > 0
        gold_lens = _len_stats(gold["aa_seq"])
        cand_lens = _len_stats(cand.loc[cand_nonempty_seq, "sequence"])

        cand_species = set(cand.loc[cand_nonempty_sp, "species_normalized"].unique())
        gold_species = set(gold["species_gold_norm"].unique())
        species_overlap = cand_species & gold_species

        print(f"\n[stage-2 diagnostic] system={system.name}")
        print(f"  candidate rows                : {len(cand):>10,}")
        print(f"  candidate rows w/ sequence    : {int(cand_nonempty_seq.sum()):>10,}")
        print(f"  candidate rows w/ species     : {int(cand_nonempty_sp.sum()):>10,}")
        print(f"  gold sequence length (n/min/p50/p99/max): "
              f"{gold_lens['n']}/{gold_lens['min']}/{gold_lens['p50']}/"
              f"{gold_lens['p99']}/{gold_lens['max']}")
        print(f"  cand sequence length (n/min/p50/p99/max): "
              f"{cand_lens['n']}/{cand_lens['min']}/{cand_lens['p50']}/"
              f"{cand_lens['p99']}/{cand_lens['max']}")
        print(f"  gold species   : {sorted(gold_species)}")
        print(f"  cand species   : {sorted(cand_species)}")
        print(f"  species overlap: {sorted(species_overlap) or '[]'}")

        # Explicitly name each failure mode so the cause of an upcoming
        # 0-coverage result is unmissable.
        warnings = []
        if cand_nonempty_seq.sum() == 0 or cand_nonempty_sp.sum() == 0:
            warnings.append(
                "Stage-1 candidate has empty species/sequence — Stage-1 freeze "
                "produced garbage (likely a CandidateAdapter column-name miss)."
            )
        if not species_overlap:
            warnings.append(
                "Gold and candidate share zero species after canonicalization — "
                "the join will produce 100% unmatched_gold."
            )
        # Length distributions: warn if max candidate length is strictly less
        # than min gold length, OR if median candidate length differs from
        # median gold length by more than 20 AA.
        if cand_lens["n"] > 0 and gold_lens["n"] > 0:
            if cand_lens["max"] < gold_lens["min"]:
                warnings.append(
                    f"All candidate sequences are shorter than every gold "
                    f"sequence (cand max={cand_lens['max']} < gold min="
                    f"{gold_lens['min']}). Exact-string merge cannot match. "
                    "Candidate looks like read-window fragments; gold stores "
                    "full-length canonical sequences."
                )
            elif abs(cand_lens["p50"] - gold_lens["p50"]) > 20:
                warnings.append(
                    f"Median sequence length differs by "
                    f"{abs(cand_lens['p50'] - gold_lens['p50'])} AA "
                    f"(gold p50={gold_lens['p50']}, cand p50={cand_lens['p50']}). "
                    "Exact-string merge will likely produce 0 matches even if "
                    "species overlap."
                )

        for w in warnings:
            print(f"  WARNING: {w}")
        if not warnings:
            print("  diagnostic: gold/candidate shapes look compatible.")

    # -- compute all metrics -------------------------------------------------

    def run(self) -> dict[str, Path]:
        """Produce every Stage-2 artifact and return a dict of output paths."""
        if self._gold is None:
            self.load_gold()

        # If the eval contract did not specify an explicit species_filter,
        # restrict gold to the union of species that the candidates actually
        # processed. Without this, macro-averaged coverage gets divided by the
        # full 21-species gold panel and craters whenever the agents only
        # process a subset (e.g. one species per dataset).
        explicit_filter = self.eval_config.get("gold_set", {}).get("species_filter")
        if not explicit_filter:
            self._auto_restrict_gold_to_candidate_species()

        per_system_align: dict[str, pd.DataFrame] = {}
        full_align_frames: list[pd.DataFrame] = []
        for s in self.systems:
            a = self.align_system(s)
            per_system_align[s.name] = a
            full_align_frames.append(a)
        full_align = pd.concat(full_align_frames, ignore_index=True)

        # Persist alignment artifacts
        align_parquet = self.out_dir / "gold_alignment.parquet"
        review_cols = [
            "run_id", "system_name", "processed_example_id",
            "matched_gold_example_id", "match_status",
            "species_predicted", "species_gold_norm",
            "predicted_label", "gold_label",
            "species_correct", "label_correct",
            "sequence_exact_match", "annotation_exact_match",
            "error_type",
        ]
        align_for_parquet = full_align.copy()
        align_for_parquet.to_parquet(align_parquet, index=False)
        review_path = self.out_dir / "gold_alignment_review.csv"
        full_align[review_cols].to_csv(review_path, index=False)

        # Headline summary + per-species + efficiency + errors
        summary_path = self._write_summary_metrics(per_system_align)
        per_species_path = self._write_per_species_metrics(per_system_align)
        efficiency_path = self._write_efficiency_metrics(per_system_align)
        errors_path = self._write_error_analysis(per_system_align)
        stats_path = self._write_statistical_tests(per_system_align)

        return {
            "gold_alignment_parquet": align_parquet,
            "gold_alignment_review_csv": review_path,
            "summary_metrics_csv": summary_path,
            "per_species_metrics_csv": per_species_path,
            "efficiency_metrics_csv": efficiency_path,
            "error_analysis_csv": errors_path,
            "statistical_tests_csv": stats_path,
        }

    # -- writers -------------------------------------------------------------

    def _write_summary_metrics(self, aligns: dict[str, pd.DataFrame]) -> Path:
        rows: list[dict] = []
        for s in self.systems:
            a = aligns[s.name]
            macro = _macro_metrics(a)
            micro = _micro_metrics(a)
            for weighting, metrics in (("macro", macro), ("micro", micro)):
                rows.append({
                    "system_name": s.name,
                    "comparison_setting": s.setting,
                    "species_weighting": weighting,
                    "num_gold_examples": int(len(a)),
                    "gold_coverage": metrics["coverage"],
                    "annotation_exact_match": metrics["annotation_exact_match"],
                    "label_accuracy": metrics["label_accuracy"],
                    "field_precision": metrics["precision"],
                    "field_recall": metrics["recall"],
                    "field_f1": metrics["f1"],
                    "invalid_schema_rate": metrics["invalid_schema_rate"],
                    "unmatched_gold_rate": metrics["unmatched_gold_rate"],
                    "runtime_hours": round(s.runtime_hours, 4),
                    "estimated_cost_usd": round(s.cost_usd, 4),
                })
        path = self.out_dir / "summary_metrics.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_per_species_metrics(self, aligns: dict[str, pd.DataFrame]) -> Path:
        rows: list[dict] = []
        for s in self.systems:
            a = aligns[s.name]
            for species, sub in a.groupby("species_gold_norm"):
                metrics = _basic_metrics(sub)
                # Most frequent error mode for this species
                err = sub.loc[sub["annotation_exact_match"] == False, "error_type"]  # noqa: E712
                main_error = err.value_counts().index[0] if len(err) else ""
                rows.append({
                    "system_name": s.name,
                    "comparison_setting": s.setting,
                    "species": species,
                    "num_gold_examples": int(len(sub)),
                    "gold_coverage": metrics["coverage"],
                    "annotation_exact_match": metrics["annotation_exact_match"],
                    "label_accuracy": metrics["label_accuracy"],
                    "field_f1": metrics["f1"],
                    "invalid_schema_rate": metrics["invalid_schema_rate"],
                    "main_error_mode": main_error,
                })
        path = self.out_dir / "per_species_metrics.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_efficiency_metrics(self, aligns: dict[str, pd.DataFrame]) -> Path:
        rows: list[dict] = []
        for s in self.systems:
            a = aligns[s.name]
            # Cleaned candidate dataset size, sourced from the frozen parquet
            cand_path = s.run_dir / "processed_examples.parquet"
            cand = pd.read_parquet(cand_path) if cand_path.exists() else pd.DataFrame()
            unique_ready = int(cand[["species_normalized", "sequence"]].drop_duplicates().shape[0]) if not cand.empty else 0

            raw = s.raw_reads_processed
            retained = s.reads_retained_after_qc
            runtime = s.runtime_seconds or 0.0
            cost = s.cost_usd or 0.0

            rows.append({
                "system_name": s.name,
                "comparison_setting": s.setting,
                "raw_reads_processed": raw if raw is not None else "",
                "reads_retained_after_qc": retained if retained is not None else "",
                "unique_model_ready_sequences": unique_ready,
                "runtime_hours": round(runtime / 3600.0, 4),
                "runtime_seconds": int(runtime),
                "estimated_cost_usd": round(cost, 4),
                "raw_reads_per_second": (raw / runtime) if (raw and runtime) else "",
                "model_ready_records_per_second": (unique_ready / runtime) if (unique_ready and runtime) else "",
                "cost_per_million_raw_reads": (cost / (raw / 1_000_000.0)) if (raw and cost) else "",
                "cost_per_million_model_ready_records": (cost / (unique_ready / 1_000_000.0)) if (unique_ready and cost) else "",
                "compression_ratio": (raw / unique_ready) if (raw and unique_ready) else "",
                "failure_rate": (s.failure_count / s.total_attempts) if s.total_attempts else 0.0,
                "candidate_dataset_size": int(cand.shape[0]) if not cand.empty else 0,
                "gold_match_fraction_of_candidates": (
                    float(a["match_status"].eq("matched").sum()) / float(unique_ready)
                    if unique_ready else ""
                ),
            })
        path = self.out_dir / "efficiency_metrics.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_error_analysis(self, aligns: dict[str, pd.DataFrame]) -> Path:
        rows: list[dict] = []
        for s in self.systems:
            a = aligns[s.name]
            for species_label, sub in [("all", a)] + list(a.groupby("species_gold_norm")):
                counts = sub["error_type"].value_counts()
                total = int(len(sub))
                for err_type, n in counts.items():
                    if not err_type or err_type == "ok":
                        continue
                    rows.append({
                        "system_name": s.name,
                        "comparison_setting": s.setting,
                        "species": species_label,
                        "error_type": err_type,
                        "error_count": int(n),
                        "total_examples": total,
                        "error_rate": float(n) / total if total else 0.0,
                    })
        path = self.out_dir / "error_analysis.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_statistical_tests(self, aligns: dict[str, pd.DataFrame]) -> Path:
        """Paired tests + bootstrap CIs for system-vs-system comparisons.

        For each ordered pair (A, B), produce one row per metric:
        bootstrap CI on A-B (by example, paired on gold_example_id) plus a
        McNemar test for paired annotation exact-match.
        """
        rows: list[dict] = []
        if len(self.systems) < 2:
            pd.DataFrame(rows).to_csv(self.out_dir / "statistical_tests.csv", index=False)
            return self.out_dir / "statistical_tests.csv"

        # Pair up CAPO vs each other system. Convention: index 0 is treated
        # as "CAPO" (primary). If the harness is configured differently the
        # caller can still read the table — A/B names are explicit.
        a_sys = self.systems[0]
        a = aligns[a_sys.name].set_index("gold_example_id")
        for b_sys in self.systems[1:]:
            b = aligns[b_sys.name].set_index("gold_example_id")
            common = a.index.intersection(b.index)
            a_pair = a.loc[common]
            b_pair = b.loc[common]

            for metric in ("annotation_exact_match", "label_correct", "sequence_exact_match"):
                a_vec = a_pair[metric].astype(int).to_numpy()
                b_vec = b_pair[metric].astype(int).to_numpy()
                ci_lo, ci_hi = _bootstrap_diff_ci(a_vec, b_vec, n_boot=1000, seed=self.rng_seed)
                a_score = float(a_vec.mean()) if len(a_vec) else float("nan")
                b_score = float(b_vec.mean()) if len(b_vec) else float("nan")
                abs_diff = a_score - b_score
                rel_diff = (abs_diff / b_score) if b_score else float("nan")

                # McNemar only for binary correctness metrics
                p_val = _mcnemar_pvalue(a_vec, b_vec) if metric != "label_correct" else _mcnemar_pvalue(a_vec, b_vec)
                rows.append({
                    "metric": metric,
                    "system_a": a_sys.name,
                    "system_b": b_sys.name,
                    "comparison_setting": b_sys.setting,
                    "species_weighting": "micro",
                    "a_score": a_score,
                    "b_score": b_score,
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff,
                    "ci_lower_95": ci_lo,
                    "ci_upper_95": ci_hi,
                    "test_name": "bootstrap_diff_95 + mcnemar",
                    "p_value": p_val,
                    "n_paired": int(len(common)),
                })

        path = self.out_dir / "statistical_tests.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _classify_error(row: pd.Series) -> str:
    if row["match_status"] == "unmatched_gold":
        return "unmatched_gold"
    if row["annotation_exact_match"]:
        return "ok"
    # Matched but wrong
    pred = row["predicted_label"]
    gold = row["gold_label"]
    if pd.isna(pred) or pred == -1:
        return "ambiguous_annotation"
    if int(pred) != int(gold):
        return "label_flip"
    if not row["species_correct"]:
        return "species_misnormalization"
    return "sequence_mismatch"


def _basic_metrics(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    if n == 0:
        return {
            "coverage": 0.0,
            "annotation_exact_match": 0.0,
            "label_accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "invalid_schema_rate": 0.0,
            "unmatched_gold_rate": 0.0,
        }
    matched_mask = df["match_status"].eq("matched")
    coverage = float(matched_mask.sum()) / n
    aem = float(df["annotation_exact_match"].sum()) / n
    label_acc = float(
        (matched_mask & df["label_correct"]).sum()
    ) / max(int(matched_mask.sum()), 1)

    # Binder precision/recall — for the ACE2 task the "field" is the
    # binder-vs-nonbinder annotation.
    matched = df[matched_mask]
    if len(matched):
        tp = int(((matched["predicted_label"] == 1) & (matched["gold_label"] == 1)).sum())
        fp = int(((matched["predicted_label"] == 1) & (matched["gold_label"] == 0)).sum())
        fn = int(((matched["predicted_label"] == 0) & (matched["gold_label"] == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    else:
        precision = recall = f1 = 0.0

    invalid = float((df["predicted_label"] == -1).sum()) / n
    unmatched = 1.0 - coverage
    return {
        "coverage": coverage,
        "annotation_exact_match": aem,
        "label_accuracy": label_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "invalid_schema_rate": invalid,
        "unmatched_gold_rate": unmatched,
    }


def _micro_metrics(df: pd.DataFrame) -> dict[str, float]:
    return _basic_metrics(df)


def _macro_metrics(df: pd.DataFrame) -> dict[str, float]:
    per_species = []
    for _, sub in df.groupby("species_gold_norm"):
        per_species.append(_basic_metrics(sub))
    if not per_species:
        return _basic_metrics(df)
    keys = per_species[0].keys()
    return {k: float(sum(p[k] for p in per_species) / len(per_species)) for k in keys}


def _bootstrap_diff_ci(
    a: "Iterable[int]",
    b: "Iterable[int]",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    import numpy as np

    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    n = len(a_arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(a_arr[idx].mean() - b_arr[idx].mean())
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def _mcnemar_pvalue(a: "Iterable[int]", b: "Iterable[int]") -> float:
    """Two-sided McNemar test on paired binary correctness.

    Uses the exact binomial when the off-diagonal sum is small, and the
    continuity-corrected chi-square otherwise.
    """
    import numpy as np

    a_arr = np.asarray(list(a), dtype=int)
    b_arr = np.asarray(list(b), dtype=int)
    # n10 = A correct, B wrong; n01 = A wrong, B correct
    n10 = int(((a_arr == 1) & (b_arr == 0)).sum())
    n01 = int(((a_arr == 0) & (b_arr == 1)).sum())
    n_disc = n10 + n01
    if n_disc == 0:
        return 1.0
    if n_disc < 25:
        # Exact two-sided binomial test against p=0.5
        k = min(n10, n01)
        # P(X <= k) under Binomial(n_disc, 0.5), doubled
        from math import comb
        cdf = sum(comb(n_disc, i) for i in range(k + 1)) / (2.0 ** n_disc)
        return float(min(1.0, 2.0 * cdf))
    # Continuity-corrected chi-square
    chi2 = (abs(n10 - n01) - 1) ** 2 / n_disc
    # 1 - CDF of chi2 with df=1
    from math import erf, sqrt
    # Survival function for chi2_1 = 2 * (1 - Phi(sqrt(chi2)))
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(sqrt(chi2) / sqrt(2))))
    return float(max(0.0, min(1.0, p)))


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def run_stage2_evaluation(
    systems: Sequence[SystemRun],
    eval_config: dict,
    out_dir: Path,
    rng_seed: int = 0,
) -> dict[str, Path]:
    """End-to-end Stage-2 driver — load gold, write all CSVs.

    Returns a dict of output paths matching the framework spec.
    """
    evaluator = GoldEvaluator(systems=systems, eval_config=eval_config, out_dir=out_dir, rng_seed=rng_seed)
    evaluator.load_gold()
    return evaluator.run()
