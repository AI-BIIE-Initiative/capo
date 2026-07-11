"""Tests for capo.utils.model_resolution — exact / architecture / custom / auto modes."""

from __future__ import annotations

from capo.utils.model_resolution import (
    build_model_selection_json,
    load_registry,
    match_architecture,
    resolve_model,
)

# A small synthetic registry covering the cases.
_REGISTRY = [
    {"registry_id": "esm2_t6_8m", "hf_repo_id": "facebook/esm2_t6_8M_UR50D",
     "family": "ESM", "architecture_type": "encoder_mlm", "parameter_count": 8_000_000},
    {"registry_id": "esm2_t12_35m", "hf_repo_id": "facebook/esm2_t12_35M_UR50D",
     "family": "ESM", "architecture_type": "encoder_mlm", "parameter_count": 35_000_000},
    {"registry_id": "ankh_base", "hf_repo_id": "ElnaggarLab/ankh-base",
     "family": "Ankh", "architecture_type": "encoder_decoder", "parameter_count": 450_000_000},
]


# ---------------------------------------------------------------------------
# Case A — exact model bypasses the selector
# ---------------------------------------------------------------------------

def test_exact_hf_id_bypasses():
    r = resolve_model("facebook/esm2_t6_8M_UR50D", fine_tune_strategy="linear-probe",
                      registry=_REGISTRY)
    assert r.mode == "exact"
    assert r.bypass_selection is True
    assert r.resolved_model_id == "facebook/esm2_t6_8M_UR50D"
    assert r.registry_entry is not None  # found in registry


def test_exact_hf_id_not_in_registry_still_bypasses():
    r = resolve_model("some-org/private-plm", registry=_REGISTRY)
    assert r.mode == "exact"
    assert r.bypass_selection is True
    assert r.registry_entry is None
    assert r.needs_validation is True


def test_exact_registry_id_bypasses():
    r = resolve_model("esm2_t6_8m", registry=_REGISTRY)
    assert r.mode == "exact"
    assert r.bypass_selection is True
    assert r.resolved_model_id == "facebook/esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# Case B — architecture token
# ---------------------------------------------------------------------------

def test_architecture_single_match_bypasses():
    r = resolve_model("ankh", registry=_REGISTRY)
    assert r.mode == "architecture"
    assert r.bypass_selection is True
    assert r.resolved_model_id == "ElnaggarLab/ankh-base"


def test_architecture_multiple_matches_runs_selector():
    r = resolve_model("esm2", registry=_REGISTRY)
    assert r.mode == "architecture"
    assert r.bypass_selection is False
    assert len(r.candidates) == 2


def test_architecture_no_match_runs_selector():
    r = resolve_model("doesnotexist", registry=_REGISTRY)
    assert r.mode == "auto"
    assert r.bypass_selection is False


def test_explicit_architecture_field():
    r = resolve_model(None, architecture="esm2", registry=_REGISTRY)
    assert r.mode == "architecture"
    assert r.bypass_selection is False
    assert len(r.candidates) == 2


# ---------------------------------------------------------------------------
# Case C — custom
# ---------------------------------------------------------------------------

def test_custom_bypasses_registry():
    r = resolve_model("custom", fine_tune_strategy="lora", registry=_REGISTRY)
    assert r.mode == "custom"
    assert r.bypass_selection is True
    assert r.resolved_model_id is None
    assert r.needs_validation is True


# ---------------------------------------------------------------------------
# Case D — auto
# ---------------------------------------------------------------------------

def test_empty_triggers_selection():
    for spec in (None, "", "auto", "null"):
        r = resolve_model(spec, registry=_REGISTRY)
        assert r.mode == "auto"
        assert r.bypass_selection is False


# ---------------------------------------------------------------------------
# build_model_selection_json
# ---------------------------------------------------------------------------

def test_build_selection_json_exact():
    r = resolve_model("facebook/esm2_t6_8M_UR50D", fine_tune_strategy="linear-probe",
                      registry=_REGISTRY)
    js = build_model_selection_json(r)
    assert js["selection_bypassed"] is True
    assert js["preferred"] == "best_fit"
    assert js["best_fit"]["model_id"] == "facebook/esm2_t6_8M_UR50D"
    assert js["best_fit"]["fine_tune_strategy"] == "linear-probe"
    assert js["best_fit"]["registry_id"] == "esm2_t6_8m"


def test_build_selection_json_custom():
    r = resolve_model("custom", fine_tune_strategy="full-finetune", registry=_REGISTRY)
    js = build_model_selection_json(r)
    assert js["best_fit"]["model_id"] == "custom"
    assert "custom" in js["best_fit"]["flags"]


def test_match_architecture_helper():
    assert len(match_architecture(_REGISTRY, "esm")) == 2
    assert match_architecture(_REGISTRY, "encoder_decoder")[0]["registry_id"] == "ankh_base"


# ---------------------------------------------------------------------------
# Real registry loads and the default ESM2 8M model bypasses
# ---------------------------------------------------------------------------

def test_real_registry_loads_and_default_model_bypasses():
    reg = load_registry()
    assert len(reg) > 0
    r = resolve_model("facebook/esm2_t6_8M_UR50D", fine_tune_strategy="linear-probe",
                      registry=reg)
    assert r.bypass_selection is True
    assert r.resolved_model_id == "facebook/esm2_t6_8M_UR50D"
