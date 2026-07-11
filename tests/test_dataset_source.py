"""Tests for capo.utils.dataset_source — hf / local / uri / named classification.

The load-bearing invariant is the **hf identity**: for every ``owner/name`` Hub
id the resolver must return ``effective_ref`` byte-identical to the input, so the
orchestrator only ever mutates ``self.dataset_ref`` for non-hf kinds and the
existing HF pipeline is provably unchanged.
"""

from __future__ import annotations

import json

import pytest

from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator
from capo.utils.dataset_source import DatasetSource, resolve_dataset_source

# ---------------------------------------------------------------------------
# hf — owner/name Hub ids resolve to IDENTITY (zero regression)
# ---------------------------------------------------------------------------

_HF_IDS = [
    "facebook/esm2_t6_8M_UR50D",
    "theoschiff-biie/ace2_binding",
    "BIIE-AI/ace2_binding",
    "dair-ai/emotion",
    "some-org/private-plm",
]


@pytest.mark.parametrize("ref", _HF_IDS)
def test_hf_id_is_identity(ref):
    r = resolve_dataset_source(ref)
    assert r.kind == "hf"
    assert r.effective_ref == ref  # byte-identical — the regression guard
    assert r.local_path is None
    assert r.staged_rel_path is None


def test_hf_id_with_base_dir_still_identity(tmp_path):
    # base_dir must not turn an owner/name id into a local path.
    r = resolve_dataset_source("facebook/esm2_t6_8M_UR50D", base_dir=tmp_path)
    assert r.kind == "hf"
    assert r.effective_ref == "facebook/esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# local — filesystem paths stage into data/<basename>
# ---------------------------------------------------------------------------

def test_absolute_path_is_local(tmp_path):
    f = tmp_path / "assay.csv"
    f.write_text("seq,label\nMK,1\n")
    r = resolve_dataset_source(str(f))
    assert r.kind == "local"
    assert r.effective_ref == "data/assay.csv"
    assert r.staged_rel_path == "data/assay.csv"
    assert r.local_path == str(f.resolve())
    assert r.file_format == "csv"


def test_dot_relative_path_is_local():
    r = resolve_dataset_source("./data/my.parquet")
    assert r.kind == "local"
    assert r.effective_ref == "data/my.parquet"
    assert r.file_format == "parquet"


def test_home_relative_path_is_local():
    r = resolve_dataset_source("~/datasets/train.jsonl")
    assert r.kind == "local"
    assert r.effective_ref == "data/train.jsonl"
    assert r.file_format == "json"  # jsonl → json builder


def test_data_extension_without_path_prefix_is_local():
    # `owner/name.csv` looks like owner/name but the .csv extension means it is a
    # file, not a Hub repo — the data-extension rule wins.
    r = resolve_dataset_source("owner/data.csv")
    assert r.kind == "local"
    assert r.effective_ref == "data/data.csv"


def test_relative_path_resolved_against_base_dir(tmp_path):
    (tmp_path / "data").mkdir()
    f = tmp_path / "data" / "x.tsv"
    f.write_text("a\tb\n1\t2\n")
    r = resolve_dataset_source("data/x.tsv", base_dir=tmp_path)
    assert r.kind == "local"
    assert r.local_path == str(f.resolve())
    assert r.file_format == "tsv"


def test_fasta_format_sniffed():
    for ref in ("./seqs.fasta", "/tmp/proteins.fa", "~/x.faa", "reads.fna"):
        r = resolve_dataset_source(ref)
        assert r.kind == "local", ref
        assert r.file_format == "fasta", ref


def test_gz_suffix_stripped_before_sniff():
    r = resolve_dataset_source("./big.csv.gz")
    assert r.kind == "local"
    assert r.file_format == "csv"


def test_large_file_gets_note(tmp_path, monkeypatch):
    import capo.utils.dataset_source as ds

    f = tmp_path / "huge.parquet"
    f.write_bytes(b"0")
    monkeypatch.setattr(ds, "_LARGE_FILE_BYTES", 0)  # force the branch
    r = resolve_dataset_source(str(f))
    assert r.kind == "local"
    assert "HF dataset repo" in r.notes


# ---------------------------------------------------------------------------
# uri — remote URLs the agent fetches on the instance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ref,basename,fmt",
    [
        ("https://example.com/data/train.parquet", "train.parquet", "parquet"),
        ("http://host/x.csv?token=abc", "x.csv", "csv"),
        ("gs://bucket/path/set.jsonl", "set.jsonl", "json"),
        ("s3://bucket/proteins.fasta", "proteins.fasta", "fasta"),
    ],
)
def test_uri_kinds(ref, basename, fmt):
    r = resolve_dataset_source(ref)
    assert r.kind == "uri"
    assert r.effective_ref == f"data/{basename}"
    assert r.staged_rel_path == f"data/{basename}"
    assert r.file_format == fmt
    assert r.local_path is None


# ---------------------------------------------------------------------------
# named — bare labels (fetch instructions live in task.md)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ref", ["my-assay-data", "internal_kinase_panel", "cohort2024"])
def test_named_kind(ref):
    r = resolve_dataset_source(ref)
    assert r.kind == "named"
    assert r.effective_ref == ref  # no filename to derive → left as-is
    assert r.staged_rel_path is None


def test_empty_ref_is_named():
    r = resolve_dataset_source("")
    assert r.kind == "named"
    assert resolve_dataset_source(None).kind == "named"


# ---------------------------------------------------------------------------
# serialization + basename safety
# ---------------------------------------------------------------------------

def test_to_dict_roundtrips_fields(tmp_path):
    f = tmp_path / "d.parquet"
    f.write_bytes(b"x")
    d = resolve_dataset_source(str(f)).to_dict()
    assert set(d) == {
        "kind", "original_ref", "effective_ref", "local_path",
        "staged_rel_path", "file_format", "notes",
    }
    assert d["kind"] == "local"


def test_staged_basename_has_no_directory_components(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    f = nested / "deep.csv"
    f.write_text("x\n")
    r = resolve_dataset_source(str(f))
    assert r.effective_ref == "data/deep.csv"  # flattened, no a/b/c
    assert "/" not in r.effective_ref.removeprefix("data/")


def test_dataset_source_is_frozen():
    r = resolve_dataset_source("facebook/esm2_t6_8M_UR50D")
    with pytest.raises(Exception):
        r.kind = "local"  # frozen dataclass
    assert isinstance(r, DatasetSource)


# ---------------------------------------------------------------------------
# Orchestrator staging — _stage_dataset_source copies + rewrites the ref
# ---------------------------------------------------------------------------


def _orch(dataset_ref: str) -> FineTuningOrchestrator:
    return FineTuningOrchestrator(
        key_path="/tmp/nonexistent_key",
        ssh_key_name="k",
        model_id="facebook/esm2_t6_8M_UR50D",
        fine_tune_strategy="linear-probe",
        dataset_ref=dataset_ref,
        tolerance_threshold=0.1,
        enable_hf_research=False,
        enable_memory=False,
    )


def _run_dir(tmp_path):
    d = tmp_path / "run"
    (d / "reports").mkdir(parents=True)
    return d


def test_stage_hf_leaves_ref_identical(tmp_path):
    orch = _orch("theoschiff-biie/ace2_binding")
    src = orch._stage_dataset_source(_run_dir(tmp_path))
    assert src.kind == "hf"
    assert orch.dataset_ref == "theoschiff-biie/ace2_binding"  # untouched
    assert not (tmp_path / "run" / "data").exists()  # nothing staged
    meta = json.loads((tmp_path / "run" / "reports" / "dataset_source.json").read_text())
    assert meta["kind"] == "hf"


def test_stage_local_copies_and_rewrites(tmp_path):
    data = tmp_path / "assay.parquet"
    data.write_bytes(b"PARQUET-BYTES")
    orch = _orch(str(data))
    src = orch._stage_dataset_source(_run_dir(tmp_path))
    assert src.kind == "local"
    assert orch.dataset_ref == "data/assay.parquet"  # rewritten to staged rel path
    staged = tmp_path / "run" / "data" / "assay.parquet"
    assert staged.exists() and staged.read_bytes() == b"PARQUET-BYTES"
    meta = json.loads((tmp_path / "run" / "reports" / "dataset_source.json").read_text())
    assert meta["kind"] == "local" and meta["file_format"] == "parquet"


def test_stage_local_idempotent_on_resume_even_if_source_gone(tmp_path):
    data = tmp_path / "x.csv"
    data.write_text("seq,label\nMK,1\n")
    orch = _orch(str(data))
    orch._stage_dataset_source(_run_dir(tmp_path))
    data.unlink()  # simulate the original source being gone on resume
    # second call must NOT raise — the staged copy already exists.
    src2 = orch._stage_dataset_source(tmp_path / "run")
    assert src2.kind == "local"
    assert orch.dataset_ref == "data/x.csv"


def test_stage_missing_local_file_raises(tmp_path):
    orch = _orch(str(tmp_path / "does_not_exist.csv"))
    with pytest.raises(FileNotFoundError):
        orch._stage_dataset_source(_run_dir(tmp_path))


def test_stage_uri_sets_ref_without_copy(tmp_path):
    orch = _orch("https://example.com/data/train.parquet")
    src = orch._stage_dataset_source(_run_dir(tmp_path))
    assert src.kind == "uri"
    assert orch.dataset_ref == "data/train.parquet"
    assert not (tmp_path / "run" / "data").exists()  # not fetched locally


def test_staged_run_dir_passes_structure_validator(tmp_path):
    """data/ + reports/dataset_source.json must not trip the run validator."""
    from capo.utils.checks import validate_run_structure

    data = tmp_path / "d.csv"
    data.write_text("seq,label\nMK,1\n")
    # Build a full canonical run dir, then stage into it.
    orch = _orch(str(data))
    run_dir = orch._setup_run_dir(run_id="ft-test", output_dir=tmp_path / "run", require_existing=False)[0]
    orch._stage_dataset_source(run_dir)
    report = validate_run_structure(run_dir, stage="preflight")
    # data/ and reports/dataset_source.json are not flagged as forbidden.
    assert "data/" not in report.forbidden_items
    assert not any("dataset_source" in f for f in report.forbidden_items)
