"""Tests for hf_research.py — ResearchFindings parsing and rendering."""

import json
import pytest

from capo.research.hf_research import (
    ResearchFindings,
    _extract_json_object,
    _parse_findings,
)


# ---------------------------------------------------------------------------
# _parse_findings
# ---------------------------------------------------------------------------

def test_parse_findings_valid_json():
    data = {
        "training_datasets": [
            {"name": "ACE2 Binding", "hf_id": "biie/ace2", "size": "10k", "notes": "binding data", "url": None}
        ],
        "eval_benchmarks": [
            {"name": "ProteinGym", "metrics": ["spearman_rho", "MCC"], "url": None, "notes": "DMS benchmark"}
        ],
        "hyperparameters": {
            "learning_rate": "1e-4",
            "optimizer": "AdamW",
            "notes": "From ESM2 model card",
        },
        "summary": "Use AdamW at 1e-4 with cosine decay.",
    }
    findings = _parse_findings(json.dumps(data))

    assert len(findings.training_datasets) == 1
    assert findings.training_datasets[0]["name"] == "ACE2 Binding"
    assert len(findings.eval_benchmarks) == 1
    assert "spearman_rho" in findings.eval_benchmarks[0]["metrics"]
    assert findings.hyperparameters["learning_rate"] == "1e-4"
    assert findings.summary == "Use AdamW at 1e-4 with cosine decay."
    assert not findings.is_empty()


def test_parse_findings_fenced_json():
    data = {
        "training_datasets": [],
        "eval_benchmarks": [],
        "hyperparameters": {"learning_rate": "5e-5"},
        "summary": "minimal",
    }
    raw = f"```json\n{json.dumps(data)}\n```"
    findings = _parse_findings(raw)
    assert findings.hyperparameters["learning_rate"] == "5e-5"
    assert findings.summary == "minimal"


def test_parse_findings_plain_fenced_json():
    data = {"training_datasets": [], "eval_benchmarks": [], "hyperparameters": {}, "summary": "ok"}
    raw = f"```\n{json.dumps(data)}\n```"
    findings = _parse_findings(raw)
    assert findings.summary == "ok"


def test_parse_findings_invalid_json_graceful_fallback():
    # Invalid JSON → empty findings; raw is preserved for debugging but
    # summary stays empty so garbage never leaks into the main prompt.
    raw = "not json at all"
    findings = _parse_findings(raw)
    assert findings.raw == raw
    assert findings.summary == ""
    assert findings.is_empty()


def test_parse_findings_empty_string():
    findings = _parse_findings("")
    assert findings.is_empty()
    assert findings.raw == ""


def test_parse_findings_policy_error_is_sanitized():
    # Claude Code policy refusals must produce empty findings — never leak into
    # the main Orchestrator prompt where they would cascade a second refusal.
    error = (
        "API Error: Claude Code is unable to respond to this request, "
        "which appears to violate our Usage Policy (https://www.anthropic.com/legal/aup)."
    )
    findings = _parse_findings(error)
    assert findings.is_empty()
    assert findings.raw == error
    assert findings.to_prompt_section() == "No prior research available."


def test_parse_findings_missing_keys_use_defaults():
    # Missing keys -> msgspec falls back to the Struct's declared defaults.
    data = {"summary": "partial"}
    findings = _parse_findings(json.dumps(data))
    assert findings.summary == "partial"
    assert findings.training_datasets == []
    assert findings.eval_benchmarks == []
    assert findings.hyperparameters == {}


def test_parse_findings_validation_error_returns_empty():
    # Wrong-typed value (null where array expected) is caught by the schema
    # decode, logged with a precise JSON path, and the run falls through to
    # the empty-fallback path. This is the durable-execution win — silent
    # coercion is replaced by fail-fast.
    raw = '{"summary": "partial", "training_datasets": null}'
    findings = _parse_findings(raw)
    assert findings.is_empty()
    assert findings.raw == raw


# ---------------------------------------------------------------------------
# ResearchFindings.is_empty
# ---------------------------------------------------------------------------

def test_is_empty_default():
    assert ResearchFindings().is_empty()


def test_is_empty_with_summary():
    assert not ResearchFindings(summary="something").is_empty()


def test_is_empty_with_datasets():
    assert not ResearchFindings(training_datasets=[{"name": "foo"}]).is_empty()


# ---------------------------------------------------------------------------
# ResearchFindings.to_prompt_section
# ---------------------------------------------------------------------------

def test_to_prompt_section_empty():
    section = ResearchFindings().to_prompt_section()
    assert "No prior research available" in section


def test_to_prompt_section_with_datasets():
    findings = ResearchFindings(
        training_datasets=[
            {"name": "ACE2", "hf_id": "biie/ace2", "size": "10k", "notes": "binding"}
        ],
        eval_benchmarks=[
            {"name": "ProteinGym", "metrics": ["MCC"], "url": None, "notes": "DMS"}
        ],
        hyperparameters={"learning_rate": "1e-4", "optimizer": "AdamW"},
        summary="Use AdamW.",
    )
    section = findings.to_prompt_section()
    assert "ACE2" in section
    assert "biie/ace2" in section
    assert "ProteinGym" in section
    assert "MCC" in section
    assert "learning_rate" in section
    assert "1e-4" in section
    assert "Use AdamW." in section


def test_to_prompt_section_no_benchmarks_fallback():
    findings = ResearchFindings(
        training_datasets=[{"name": "foo", "hf_id": None, "size": "1k", "notes": ""}],
    )
    section = findings.to_prompt_section()
    assert "train/val/test split" in section


def test_to_prompt_section_no_hyperparameters_fallback():
    findings = ResearchFindings(summary="hi")
    section = findings.to_prompt_section()
    assert "model-card defaults" in section


# ---------------------------------------------------------------------------
# ResearchFindings.to_dict
# ---------------------------------------------------------------------------

def test_to_dict_excludes_raw():
    findings = ResearchFindings(summary="test", raw="raw text")
    d = findings.to_dict()
    assert d["summary"] == "test"
    assert "raw" not in d


def test_to_dict_is_json_serializable():
    findings = ResearchFindings(
        training_datasets=[{"name": "foo"}],
        eval_benchmarks=[{"name": "bar", "metrics": ["MCC"]}],
        hyperparameters={"lr": "1e-4"},
        summary="ok",
    )
    serialized = json.dumps(findings.to_dict())
    assert "foo" in serialized


# ---------------------------------------------------------------------------
# Richer schema: entity_frame, structured hyperparameters, dataset flags,
# benchmark source_type
# ---------------------------------------------------------------------------

def test_parse_findings_populates_entity_frame():
    data = {
        "entity_frame": {
            "modality": "protein",
            "assay": "binding",
            "organism": "SARS-CoV-2",
            "label_type": "multi_label",
            "model_family": "ESM",
        },
        "training_datasets": [],
        "eval_benchmarks": [],
        "hyperparameters": {},
        "summary": "ok",
    }
    findings = _parse_findings(json.dumps(data))
    assert findings.entity_frame["modality"] == "protein"
    assert findings.entity_frame["assay"] == "binding"


def test_parse_findings_missing_entity_frame_defaults_to_empty():
    data = {"summary": "no frame", "training_datasets": []}
    findings = _parse_findings(json.dumps(data))
    assert findings.entity_frame == {}


def test_to_prompt_section_renders_entity_frame():
    findings = ResearchFindings(
        entity_frame={"modality": "protein", "assay": "binding", "organism": "unknown"},
        summary="hi",
    )
    section = findings.to_prompt_section()
    assert "Entity frame" in section
    assert "modality=protein" in section
    assert "assay=binding" in section
    # "unknown" values must be filtered out of the compact line
    assert "organism=unknown" not in section


def test_to_prompt_section_handles_structured_hyperparameters():
    findings = ResearchFindings(
        hyperparameters={
            "learning_rate": {"value": "1e-4", "provenance": "model_card"},
            "batch_size": {"value": "32", "provenance": "family_heuristic"},
            "lora_r": {"value": "n/a", "provenance": "default"},
            "notes": "ESM2 linear-probe regime",
        },
        summary="ok",
    )
    section = findings.to_prompt_section()
    assert "learning_rate" in section
    assert "1e-4" in section
    assert "model_card" in section
    assert "batch_size" in section
    assert "32" in section
    # n/a values should be skipped
    assert "lora_r" not in section
    assert "ESM2 linear-probe regime" in section


def test_to_prompt_section_dataset_flags_are_surfaced():
    findings = ResearchFindings(
        training_datasets=[
            {
                "name": "GatedDataset",
                "hf_id": "org/gated",
                "size": "5k",
                "recommended_use": "reject",
                "access": "gated",
                "notes": "gated access; cannot fetch",
            }
        ],
        summary="ok",
    )
    section = findings.to_prompt_section()
    assert "use=reject" in section
    assert "access=gated" in section


def test_to_prompt_section_dataset_default_flags_are_hidden():
    findings = ResearchFindings(
        training_datasets=[
            {
                "name": "CleanDataset",
                "hf_id": "org/clean",
                "size": "10k",
                "recommended_use": "primary",
                "access": "public",
                "notes": "standard public dataset",
            }
        ],
        summary="ok",
    )
    section = findings.to_prompt_section()
    # No flag stripe should appear when every flag is at its default
    assert "[" not in section.split("CleanDataset")[1].split("\n")[0]


def test_to_prompt_section_silently_ignores_unknown_dataset_fields():
    # If the agent emits richer fields (legacy or speculative), the renderer
    # must not crash and must not leak them into the markdown.
    findings = ResearchFindings(
        training_datasets=[
            {
                "name": "RichDataset",
                "hf_id": "org/rich",
                "size": "10k",
                "recommended_use": "primary",
                "access": "public",
                "leakage_risk": "high",
                "license_risk": "non_commercial",
                "compatibility": {"input_type": "sequence"},
                "notes": "extra fields present",
            }
        ],
        summary="ok",
    )
    section = findings.to_prompt_section()
    assert "RichDataset" in section
    assert "leakage" not in section
    assert "license_risk" not in section
    assert "input_type" not in section


def test_to_prompt_section_benchmark_source_type_canonical_fallback():
    findings = ResearchFindings(
        eval_benchmarks=[
            {
                "name": "ProteinGym",
                "metrics": ["Spearman", "MCC"],
                "source_type": "canonical_fallback",
                "notes": "DMS variant effect",
            }
        ],
        summary="ok",
    )
    section = findings.to_prompt_section()
    assert "ProteinGym" in section
    assert "Spearman" in section
    assert "[canonical_fallback]" in section


def test_to_prompt_section_benchmark_source_type_default_hidden():
    findings = ResearchFindings(
        eval_benchmarks=[
            {"name": "CustomBench", "metrics": ["MCC"], "source_type": "hf_dataset", "notes": "x"}
        ],
        summary="ok",
    )
    section = findings.to_prompt_section()
    # hf_dataset is the default → no [tag]
    assert "[hf_dataset]" not in section


def test_parse_findings_recovers_json_with_leading_prose():
    # Real-world failure mode (2026-04-29): the agent prepended
    # "Now I have enough verified data. Let me compile the final JSON."
    # before the JSON object, breaking strict json.loads. Defensive parser
    # should still extract the JSON.
    data = {
        "entity_frame": {"modality": "protein"},
        "training_datasets": [{"name": "ACE2", "hf_id": "biie/ace2"}],
        "eval_benchmarks": [],
        "hyperparameters": {"lr": "1e-4"},
        "summary": "ok",
    }
    raw = "Now I have enough verified data. Let me compile the final JSON." + json.dumps(data)
    findings = _parse_findings(raw)
    assert findings.training_datasets[0]["hf_id"] == "biie/ace2"
    assert findings.entity_frame["modality"] == "protein"
    assert findings.summary == "ok"


def test_parse_findings_recovers_json_with_trailing_prose():
    data = {"training_datasets": [], "eval_benchmarks": [], "hyperparameters": {}, "summary": "x"}
    raw = json.dumps(data) + "\n\nThat is the full report. Let me know if you need more."
    findings = _parse_findings(raw)
    assert findings.summary == "x"


def test_parse_findings_recovers_json_with_prose_on_both_sides():
    data = {"training_datasets": [], "eval_benchmarks": [], "hyperparameters": {}, "summary": "y"}
    raw = "Here you go:\n" + json.dumps(data) + "\nDone."
    findings = _parse_findings(raw)
    assert findings.summary == "y"


def test_extract_json_object_handles_braces_in_strings():
    # Naive regex would mis-balance on '}' inside string literals. The scanner
    # must respect string boundaries.
    text = 'preamble {"a": "}{"} trailer'
    extracted = _extract_json_object(text)
    assert extracted == '{"a": "}{"}'
    assert json.loads(extracted) == {"a": "}{"}


def test_extract_json_object_handles_escaped_quotes():
    text = 'lead {"a": "he said \\"hi\\" to {him}"} tail'
    extracted = _extract_json_object(text)
    assert extracted is not None
    assert json.loads(extracted)["a"] == 'he said "hi" to {him}'


def test_extract_json_object_handles_nested_objects():
    text = 'X {"outer": {"inner": {"deep": 1}}} Y'
    extracted = _extract_json_object(text)
    assert extracted == '{"outer": {"inner": {"deep": 1}}}'


def test_extract_json_object_returns_none_when_no_brace():
    assert _extract_json_object("no json here at all") is None


def test_parse_findings_extracted_path_does_not_swallow_refusal():
    # If the response is a refusal AND happens to contain a stray '{...}',
    # we should not silently parse the brace, the refusal-detection path
    # below must still fire on the original raw text. The agent emits valid
    # findings as the FIRST and only `{}`; refusals don't.
    refusal = (
        "API Error: Claude Code is unable to respond to this request, "
        "which appears to violate our Usage Policy."
    )
    findings = _parse_findings(refusal)
    assert findings.is_empty()
    assert findings.to_prompt_section() == "No prior research available."


def test_to_prompt_section_renders_unknown_hf_id():
    # Canonical fallback benchmarks where the agent could not verify a Hub ID
    # must still render — the human-readable name remains the load-bearing
    # signal, and "unknown" tells the downstream agent to search itself.
    findings = ResearchFindings(
        eval_benchmarks=[
            {
                "name": "ProteinGym",
                "hf_id": "unknown",
                "metrics": ["Spearman", "MCC"],
                "source_type": "canonical_fallback",
                "notes": "DMS variant effect; Hub ID unverified — search by name",
            }
        ],
        summary="ok",
    )
    section = findings.to_prompt_section()
    assert "ProteinGym" in section
    assert "Spearman" in section
    assert "[canonical_fallback]" in section


def test_system_prompt_forbids_hallucinated_hf_ids():
    # Regression guard for the verification rule. If this test fails after a
    # prompt edit, the verification scaffolding has been weakened.
    from capo.research.hf_research import _RESEARCH_SYSTEM_PROMPT
    assert "Hub-verified" in _RESEARCH_SYSTEM_PROMPT
    assert "MANDATORY" in _RESEARCH_SYSTEM_PROMPT
    # The two real-world hallucinations that motivated this rule must remain
    # listed as explicit do-not-emit examples.
    assert "proteingym/substitutions" in _RESEARCH_SYSTEM_PROMPT
    assert "flips/FLIP" in _RESEARCH_SYSTEM_PROMPT


def test_entity_frame_in_to_dict_round_trip():
    findings = ResearchFindings(
        entity_frame={"modality": "antibody"},
        hyperparameters={"lr": {"value": "1e-4", "provenance": "model_card"}},
    )
    d = findings.to_dict()
    assert d["entity_frame"]["modality"] == "antibody"
    assert d["hyperparameters"]["lr"]["provenance"] == "model_card"
    json.dumps(d)  # round-trips cleanly
