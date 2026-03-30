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
    allowed_partners: List[str] = field(default_factory=list)
    forbidden_partners: List[str] = field(default_factory=list)
    expected_confidence: Optional[str] = None
    expect_no_match: bool = False
    requires_pitch: bool = False
    judgement_note: Optional[str] = None


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


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
    confidence_passes = 0
    missing_cases: List[str] = []
    false_positive_cases: List[str] = []
    false_negative_cases: List[str] = []
    no_match_true_positives = 0
    no_match_false_positives = 0
    no_match_false_negatives = 0
    shortlist_recall_hits = 0
    shortlist_recall_total = 0
    acceptance_reason_counts: Dict[str, int] = {}

    for case in cases:
        row = rows_by_case_id.get(case.case_id)
        if row is None:
            missing_cases.append(case.case_id)
            continue
        matched_partner = str(row.get("matched_golden_partner") or row.get("matched_partner_name") or "").strip()
        matched_partner_normalized = _normalize_label(matched_partner)
        match_confidence = str(row.get("match_confidence") or row.get("match_confidence_tier") or "").strip().lower()
        allowed_partners = [p for p in case.allowed_partners if p]
        allowed_partner_labels = {_normalize_label(p) for p in allowed_partners if _normalize_label(p)}
        reason = str(row.get("match_acceptance_reason") or row.get("acceptance_reason") or "").strip()
        if reason:
            acceptance_reason_counts[reason] = acceptance_reason_counts.get(reason, 0) + 1

        is_no_match_output = (
            not matched_partner
            or matched_partner.lower() == "no suitable match found"
            or match_confidence == "no_match"
        )
        partner_pass = False
        if case.expect_no_match:
            partner_pass = is_no_match_output
            if is_no_match_output:
                no_match_true_positives += 1
            else:
                false_positive_cases.append(case.case_id)
                no_match_false_positives += 1
        else:
            if allowed_partners:
                partner_pass = matched_partner_normalized in allowed_partner_labels
            elif case.expected_partner is None or matched_partner_normalized == _normalize_label(case.expected_partner):
                partner_pass = True
            forbidden_partner_labels = {_normalize_label(p) for p in case.forbidden_partners if _normalize_label(p)}
            if forbidden_partner_labels and matched_partner_normalized in forbidden_partner_labels:
                partner_pass = False
            if partner_pass:
                partner_passes += 1
            else:
                false_negative_cases.append(case.case_id)
                if is_no_match_output:
                    no_match_false_negatives += 1
        if case.expect_no_match and partner_pass:
            partner_passes += 1

        if case.expected_confidence is None or match_confidence == case.expected_confidence.lower():
            confidence_passes += 1

        pitch = str(row.get("sales_pitch") or row.get("phone_sales_line") or "").strip()
        if not case.requires_pitch or bool(pitch):
            pitch_passes += 1

        shortlist_raw = row.get("shortlisted_partner_names") or row.get("shortlisted_partners") or row.get("shortlisted_partner_ids")
        shortlist_values: List[str] = []
        if isinstance(shortlist_raw, str):
            shortlist_values = [item.strip() for item in shortlist_raw.split(";") if item.strip()]
        elif isinstance(shortlist_raw, list):
            shortlist_values = [str(item).strip() for item in shortlist_raw if str(item).strip()]
        shortlist_labels = {_normalize_label(item) for item in shortlist_values if _normalize_label(item)}
        if allowed_partners:
            shortlist_recall_total += 1
            if any(item in shortlist_labels for item in allowed_partner_labels):
                shortlist_recall_hits += 1

    return {
        "total_cases": total_cases,
        "missing_cases": missing_cases,
        "partner_pass_rate": (partner_passes / total_cases) if total_cases else 0.0,
        "top1_acceptable_accuracy": (partner_passes / total_cases) if total_cases else 0.0,
        "pitch_pass_rate": (pitch_passes / total_cases) if total_cases else 0.0,
        "confidence_pass_rate": (confidence_passes / total_cases) if total_cases else 0.0,
        "no_match_precision": (
            no_match_true_positives / (no_match_true_positives + no_match_false_positives)
            if (no_match_true_positives + no_match_false_positives) else 0.0
        ),
        "no_match_recall": (
            no_match_true_positives / (no_match_true_positives + no_match_false_negatives)
            if (no_match_true_positives + no_match_false_negatives) else 0.0
        ),
        "false_positive_rate": (
            len(false_positive_cases) / len([case for case in cases if case.expect_no_match])
            if any(case.expect_no_match for case in cases) else 0.0
        ),
        "false_negative_rate": (
            len(false_negative_cases) / len([case for case in cases if not case.expect_no_match])
            if any(not case.expect_no_match for case in cases) else 0.0
        ),
        "shortlist_recall_at_k": (shortlist_recall_hits / shortlist_recall_total) if shortlist_recall_total else 0.0,
        "acceptance_reason_counts": acceptance_reason_counts,
        "false_positive_cases": false_positive_cases,
        "false_negative_cases": false_negative_cases,
        "release_gate_pass": (
            not missing_cases
            and partner_passes == total_cases
            and pitch_passes == total_cases
            and (not false_positive_cases)
            and (not false_negative_cases)
        ),
    }
