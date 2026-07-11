"""Resolve the user's dataset specification → hf / local / uri / named.

CAPO historically assumed every ``dataset_ref`` is a HuggingFace Hub id
(``owner/name``) that the remote instance downloads with ``datasets.load_dataset``.
This module classifies ``dataset_ref`` into one of four kinds so a dataset that
is NOT on the Hub can still be used:

- **hf**    — an ``owner/name`` Hub id → **identity**: ``effective_ref`` equals the
  input verbatim, so every existing consumer of ``dataset_ref`` is byte-identical
  (zero regression). This is the only kind that leaves the ref unchanged.
- **local** — a filesystem path to a CSV/TSV/parquet/JSON/FASTA on the user's
  machine → the orchestrator stages it into ``runs/<run_id>/data/<basename>``,
  the existing run-dir rsync carries it to ``~/capo_runs/<run_id>/data/``, and
  probe/train load it as a local file. ``effective_ref`` becomes the relative
  ``data/<basename>`` (resolves the same after ``cd`` into the run dir on both
  the local profiler host and the remote).
- **uri**   — an ``http(s)://`` / ``gs://`` / ``s3://`` URL → the main agent fetches
  it into ``data/`` on the instance before the probe; ``effective_ref`` is the
  derived ``data/<url-basename>``.
- **named** — a bare label whose fetch instructions live in the task.md prose →
  the agent reads task.md, fetches into ``data/``, and substitutes the path.
  ``effective_ref`` is left equal to the label (no filename to derive).

Pure classification (mirrors ``capo.utils.model_resolution``): the only I/O is
optional existence checks used to disambiguate a relative path from an ``owner/name``
Hub id. Staging (the file copy) lives in the orchestrator, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Schemes the agent can fetch on the remote instance.
_URI_RE = re.compile(r"^(?:https?|gs|s3|ftp|file)://", re.I)

# Data-file extension → canonical loader-format token. A trailing ``.gz`` is
# stripped before lookup so ``foo.csv.gz`` still sniffs as ``csv``.
_FORMAT_BY_EXT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".fasta": "fasta",
    ".fastq": "fastq",
    ".fa": "fasta",
    ".faa": "fasta",
    ".fna": "fasta",
}

# Local files above this size stage slowly through rsync — recommend a private
# HF repo but never block (the user chose a local file deliberately).
_LARGE_FILE_BYTES = 2 * 1024**3  # ~2 GB


@dataclass(frozen=True)
class DatasetSource:
    """Outcome of classifying the user's ``dataset_ref``."""

    kind: str                          # "hf" | "local" | "uri" | "named"
    original_ref: str
    effective_ref: str                 # hf: == original; local/uri: "data/<basename>"; named: == original
    local_path: str | None = None       # absolute source path on the user's machine (local only)
    staged_rel_path: str | None = None  # "data/<basename>" for local/uri
    file_format: str | None = None      # csv|tsv|parquet|json|fasta|None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "original_ref": self.original_ref,
            "effective_ref": self.effective_ref,
            "local_path": self.local_path,
            "staged_rel_path": self.staged_rel_path,
            "file_format": self.file_format,
            "notes": self.notes,
        }


def _sniff_format(name: str) -> str | None:
    """Canonical loader format from a filename's extension (``.gz`` stripped)."""
    n = name.strip().lower().split("?", 1)[0]  # drop any URL query string
    if n.endswith(".gz"):
        n = n[:-3]
    return _FORMAT_BY_EXT.get(Path(n).suffix)


def _has_data_ext(ref: str) -> bool:
    return _sniff_format(ref) is not None


def _looks_pathish(ref: str) -> bool:
    """A ref that is unambiguously a filesystem path (not an ``owner/name`` id)."""
    return ref.startswith(("/", "~", "./", "../")) or ref in (".", "..")


def _is_hf_id(ref: str) -> bool:
    """An ``owner/name`` Hub id: exactly one slash, both parts non-empty, no spaces.

    Mirrors the disambiguation in ``capo.cli.intent_prompt`` — a ref preceded by
    ``/ ~ .`` (path-like) or carrying a data extension is handled as ``local``
    *before* this check runs, so here a lone ``owner/name`` is safely an HF id.
    """
    if any(ws in ref for ws in (" ", "\t", "\n")):
        return False
    parts = ref.split("/")
    return len(parts) == 2 and all(parts)


def _safe_basename(name: str) -> str:
    """Filesystem-plain basename (strips directory components + whitespace)."""
    return Path(name.strip()).name


def _uri_basename(ref: str) -> str:
    tail = ref.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return _safe_basename(tail) or "dataset"


def _resolve_local_path(ref: str, base_dir: str | Path | None) -> Path:
    """Absolute source path for a local ref, preferring one that actually exists.

    Order: an absolute/``~`` path as-is; else ``base_dir/ref`` when it exists;
    else cwd-relative. When nothing exists we still return a resolved path so the
    orchestrator raises a clear FileNotFoundError at copy time.
    """
    expanded = Path(ref).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir).expanduser() / ref)
    candidates.append(expanded)  # cwd-relative
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[-1].resolve()


def resolve_dataset_source(
    dataset_ref: str | None,
    *,
    base_dir: str | Path | None = None,
) -> DatasetSource:
    """Classify ``dataset_ref`` into hf / local / uri / named.

    ``base_dir`` is an extra root against which a relative local path is checked
    for existence (e.g. the repo root). Pure aside from existence probes.
    """
    ref = (dataset_ref or "").strip()
    if not ref:
        return DatasetSource(
            kind="named", original_ref=ref, effective_ref=ref,
            notes="empty dataset_ref",
        )

    # 1. Remote URI the agent fetches on the instance.
    if _URI_RE.match(ref):
        base = _uri_basename(ref)
        return DatasetSource(
            kind="uri", original_ref=ref, effective_ref=f"data/{base}",
            staged_rel_path=f"data/{base}", file_format=_sniff_format(base),
            notes="remote URI — agent fetches into data/ on the instance before the probe",
        )

    # 2/3. Local file — a data extension, an explicit path, or an existing path.
    exists_somewhere = (
        Path(ref).expanduser().exists()
        or (base_dir is not None and (Path(base_dir).expanduser() / ref).exists())
    )
    if _has_data_ext(ref) or _looks_pathish(ref) or exists_somewhere:
        abs_path = _resolve_local_path(ref, base_dir)
        basename = _safe_basename(abs_path.name)
        notes = ""
        try:
            if abs_path.is_file() and abs_path.stat().st_size > _LARGE_FILE_BYTES:
                notes = (
                    "local file exceeds ~2 GB; consider a private HF dataset repo "
                    "for faster transfer — proceeding with rsync staging"
                )
        except OSError:
            pass
        return DatasetSource(
            kind="local", original_ref=ref, effective_ref=f"data/{basename}",
            local_path=str(abs_path), staged_rel_path=f"data/{basename}",
            file_format=_sniff_format(basename), notes=notes,
        )

    # 4. HF Hub owner/name — IDENTITY (this is the zero-regression path).
    if _is_hf_id(ref):
        return DatasetSource(kind="hf", original_ref=ref, effective_ref=ref)

    # 5. Bare label — fetch instructions live in task.md prose.
    return DatasetSource(
        kind="named", original_ref=ref, effective_ref=ref,
        notes="named dataset — agent reads task.md for fetch instructions",
    )
