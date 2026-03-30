import json
from pathlib import Path

from src.evaluation.golden_set import (
    GoldenPhoneCase,
    GoldenPipelineCase,
    evaluate_phone_golden_set,
    evaluate_pipeline_golden_set,
)


def test_phone_golden_set_release_gate_passes():
    metrics = evaluate_phone_golden_set(
        rows_by_case_id={
            "case-1": {
                "Top_Number_1": "+4930123456",
                "Final_Row_Outcome_Reason": "Contact_Successfully_Extracted",
            }
        },
        cases=[
            GoldenPhoneCase(
                case_id="case-1",
                expected_top_number="+4930123456",
                required_outcome="Contact_Successfully_Extracted",
                forbidden_numbers=["+4412345"],
            )
        ],
    )

    assert metrics["release_gate_pass"] is True


def test_pipeline_golden_set_release_gate_passes():
    metrics = evaluate_pipeline_golden_set(
        rows_by_case_id={
            "case-1": {
                "matched_golden_partner": "Partner A",
                "match_confidence": "strong",
                "match_acceptance_reason": "accepted_strong_match",
                "sales_pitch": "Concise pitch text",
            }
        },
        cases=[
            GoldenPipelineCase(
                case_id="case-1",
                expected_partner="Partner A",
                expected_confidence="strong",
                requires_pitch=True,
            )
        ],
    )

    assert metrics["release_gate_pass"] is True
    assert metrics["top1_acceptable_accuracy"] == 1.0
    assert metrics["acceptance_reason_counts"]["accepted_strong_match"] == 1


def test_pipeline_golden_set_accepts_allowed_partner_list():
    metrics = evaluate_pipeline_golden_set(
        rows_by_case_id={
            "case-1": {
                "matched_golden_partner": "Partner B",
                "match_confidence": "weak",
                "shortlisted_partner_names": ["Partner A", "Partner B"],
                "sales_pitch": "Concise pitch text",
            }
        },
        cases=[
            GoldenPipelineCase(
                case_id="case-1",
                allowed_partners=["Partner A", "Partner B"],
                expected_confidence="weak",
                requires_pitch=True,
                judgement_note="Either shortlisted partner is acceptable for this judged case.",
            )
        ],
    )

    assert metrics["release_gate_pass"] is True
    assert metrics["shortlist_recall_at_k"] == 1.0


def test_pipeline_golden_set_reads_semicolon_shortlist_strings():
    metrics = evaluate_pipeline_golden_set(
        rows_by_case_id={
            "case-1": {
                "matched_golden_partner": "Partner B",
                "match_confidence": "weak",
                "shortlisted_partner_names": "Partner A; Partner B; Partner C",
                "sales_pitch": "Concise pitch text",
            }
        },
        cases=[
            GoldenPipelineCase(
                case_id="case-1",
                allowed_partners=["Partner B"],
                expected_confidence="weak",
                requires_pitch=True,
            )
        ],
    )

    assert metrics["shortlist_recall_at_k"] == 1.0


def test_pipeline_golden_set_matches_partners_case_insensitively():
    metrics = evaluate_pipeline_golden_set(
        rows_by_case_id={
            "case-1": {
                "matched_golden_partner": "uberblick.io",
                "match_confidence": "weak",
                "shortlisted_partner_names": "PNP Media; Uberblick.io; Partner C",
                "sales_pitch": "Concise pitch text",
            }
        },
        cases=[
            GoldenPipelineCase(
                case_id="case-1",
                allowed_partners=["Uberblick.io"],
                expected_confidence="weak",
                requires_pitch=True,
            )
        ],
    )

    assert metrics["release_gate_pass"] is True
    assert metrics["top1_acceptable_accuracy"] == 1.0
    assert metrics["shortlist_recall_at_k"] == 1.0


def test_pipeline_golden_set_tracks_no_match_quality():
    metrics = evaluate_pipeline_golden_set(
        rows_by_case_id={
            "case-no-match": {
                "matched_golden_partner": "",
                "match_confidence": "no_match",
                "match_acceptance_reason": "explicit_no_match",
                "sales_pitch": "Neutral pitch text",
            }
        },
        cases=[
            GoldenPipelineCase(
                case_id="case-no-match",
                expect_no_match=True,
                expected_confidence="no_match",
                requires_pitch=True,
            )
        ],
    )

    assert metrics["release_gate_pass"] is True
    assert metrics["no_match_precision"] == 1.0
    assert metrics["no_match_recall"] == 1.0


def test_partner_match_quality_fixture_is_well_formed():
    fixture_path = Path("tests/fixtures/partner_match_quality_cases.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert len(payload) >= 8
    for case in payload:
        assert case["case_id"]
        assert case["judgement_note"]
        assert "requires_pitch" in case
        if case.get("expect_no_match"):
            assert case.get("expected_confidence") == "no_match"
