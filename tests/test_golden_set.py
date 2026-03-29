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
                "sales_pitch": "Concise pitch text",
            }
        },
        cases=[
            GoldenPipelineCase(
                case_id="case-1",
                expected_partner="Partner A",
                requires_pitch=True,
            )
        ],
    )

    assert metrics["release_gate_pass"] is True
