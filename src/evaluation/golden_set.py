from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class GoldenPhoneCase:
    case_id: str
    expected_top_number: Optional[str] = None
    required_outcome: Optional[str] = None
    forbidden_numbers: List[str] = field(default_factory=list)


@dataclass
class GoldenPipelineCase:
    case_id: str
    expected_partner: Optional[str] = None
    requires_pitch: bool = False


def evaluate_phone_golden_set(
    *,
    rows_by_case_id: Mapping[str, Mapping[str, Any]],
    cases: Iterable[GoldenPhoneCase],
) -> Dict[str, Any]:
    cases = list(cases)
    total_cases = len(cases)
    top_number_passes = 0
    outcome_passes = 0
    forbidden_number_failures: List[str] = []
    missing_cases: List[str] = []

    for case in cases:
        row = rows_by_case_id.get(case.case_id)
        if row is None:
            missing_cases.append(case.case_id)
            continue
        top_number = str(row.get("Top_Number_1") or "").strip()
        if case.expected_top_number is None or top_number == case.expected_top_number:
            top_number_passes += 1
        outcome = str(row.get("Final_Row_Outcome_Reason") or "").strip()
        if case.required_outcome is None or outcome == case.required_outcome:
            outcome_passes += 1
        forbidden_present = {
            value
            for value in case.forbidden_numbers
            if value and value in {
                str(row.get("Top_Number_1") or "").strip(),
                str(row.get("Top_Number_2") or "").strip(),
                str(row.get("Top_Number_3") or "").strip(),
            }
        }
        if forbidden_present:
            forbidden_number_failures.append(case.case_id)

    evaluated_cases = total_cases - len(missing_cases)
    return {
        "total_cases": total_cases,
        "evaluated_cases": evaluated_cases,
        "missing_cases": missing_cases,
        "top_number_pass_rate": (top_number_passes / total_cases) if total_cases else 0.0,
        "outcome_pass_rate": (outcome_passes / total_cases) if total_cases else 0.0,
        "forbidden_number_failures": forbidden_number_failures,
        "release_gate_pass": not missing_cases and not forbidden_number_failures and top_number_passes == total_cases and outcome_passes == total_cases,
    }


def evaluate_pipeline_golden_set(
    *,
    rows_by_case_id: Mapping[str, Mapping[str, Any]],
    cases: Iterable[GoldenPipelineCase],
) -> Dict[str, Any]:
    cases = list(cases)
    total_cases = len(cases)
    partner_passes = 0
    pitch_passes = 0
    missing_cases: List[str] = []

    for case in cases:
        row = rows_by_case_id.get(case.case_id)
        if row is None:
            missing_cases.append(case.case_id)
            continue
        matched_partner = str(row.get("matched_golden_partner") or row.get("matched_partner_name") or "").strip()
        if case.expected_partner is None or matched_partner == case.expected_partner:
            partner_passes += 1
        pitch = str(row.get("sales_pitch") or row.get("phone_sales_line") or "").strip()
        if not case.requires_pitch or bool(pitch):
            pitch_passes += 1

    return {
        "total_cases": total_cases,
        "missing_cases": missing_cases,
        "partner_pass_rate": (partner_passes / total_cases) if total_cases else 0.0,
        "pitch_pass_rate": (pitch_passes / total_cases) if total_cases else 0.0,
        "release_gate_pass": not missing_cases and partner_passes == total_cases and pitch_passes == total_cases,
    }
