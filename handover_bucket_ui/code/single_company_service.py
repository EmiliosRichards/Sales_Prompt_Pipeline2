"""
Single-company service wrapper for UI integration (description-driven).

This calls the same underlying LLM tasks as the batch pipeline, but avoids
DataFrame + batch output concerns and returns structured JSON.

Assumptions for UI:
- UI provides description inputs (description / keywords / reasoning).
- Phone extraction is optional (not enabled in this minimal wrapper by default).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.config import AppConfig
from src.core.schemas import WebsiteTextSummary
from src.data_handling.partner_data_handler import load_golden_partners, summarize_golden_partner
from src.extractors.llm_tasks.extract_attributes_task import extract_detailed_attributes
from src.extractors.llm_tasks.match_partner_task import match_partner
from src.extractors.llm_tasks.generate_sales_pitch_task import generate_sales_pitch
from src.extractors.llm_tasks.german_short_summary_from_description_task import (
    generate_german_short_summary_from_description,
)
from src.llm_clients.gemini_client import GeminiClient
from src.phone_retrieval.retrieval_wrapper import retrieve_phone_numbers_for_url
from src.utils.helpers import sanitize_filename_component


def build_description_blob(*, description: str, keywords: Optional[str] = None, reasoning: Optional[str] = None) -> str:
    parts: List[str] = []
    if description and str(description).strip():
        parts.append(f"Short description:\n{str(description).strip()}")
    if keywords and str(keywords).strip():
        parts.append(f"Keywords:\n{str(keywords).strip()}")
    if reasoning and str(reasoning).strip():
        parts.append(f"Reasoning:\n{str(reasoning).strip()}")
    return "\n\n".join(parts).strip()


@dataclass
class SingleCompanyResult:
    short_german_description: Optional[str]
    attributes: Optional[Dict[str, Any]]
    partner_match: Optional[Dict[str, Any]]
    sales_pitch: Optional[Dict[str, Any]]
    phones: Optional[Dict[str, Any]]
    errors: List[str]


def run_single_company_sales_pitch(
    *,
    company_url: str,
    company_name: str = "",
    description: str,
    keywords: Optional[str] = None,
    reasoning: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    run_phone_extraction: bool = False,
) -> SingleCompanyResult:
    cfg = AppConfig()
    client = GeminiClient(config=cfg)

    errors: List[str] = []
    phones_payload: Optional[Dict[str, Any]] = None

    blob = build_description_blob(description=description, keywords=keywords, reasoning=reasoning)
    if not blob:
        return SingleCompanyResult(
            short_german_description=None,
            attributes=None,
            partner_match=None,
            sales_pitch=None,
            phones=None,
            errors=["No description/keywords/reasoning provided (empty blob)."],
        )

    # Artifact dirs (optional but helpful for UI audit/debug)
    if artifacts_dir:
        ctx = os.path.join(artifacts_dir, "llm_context")
        req = os.path.join(artifacts_dir, "llm_requests")
        os.makedirs(ctx, exist_ok=True)
        os.makedirs(req, exist_ok=True)
    else:
        # If not provided, use temporary in-memory-ish convention (still required by tasks)
        # Use current working directory to keep it simple.
        ctx = os.path.normpath("ui_single_company_llm_context")
        req = os.path.normpath("ui_single_company_llm_requests")
        os.makedirs(ctx, exist_ok=True)
        os.makedirs(req, exist_ok=True)

    # Load golden partners
    golden_path = cfg.PATH_TO_GOLDEN_PARTNERS_DATA
    partners_raw = load_golden_partners(golden_path)
    partner_summaries = [summarize_golden_partner(p) for p in partners_raw] if partners_raw else []

    # Per-request prefix used by artifact files
    prefix = sanitize_filename_component(f"UI_{company_name[:20]}_{str(time.time())[-5:]}", max_len=50)

    # Optional: phone extraction (runs the existing phone retrieval wrapper for one URL)
    if run_phone_extraction:
        try:
            numbers, status, meta = retrieve_phone_numbers_for_url(
                url=company_url,
                company_name=company_name or "Unknown",
                app_config=cfg,
                run_output_dir=os.path.join(artifacts_dir or "output_data/ui_single_company_requests", "phone_retrieval"),
                llm_context_dir=os.path.join(ctx, "phone_retrieval"),
                run_id=prefix,
            )
            phones_payload = {
                "status": status,
                "meta": meta,
                "consolidated_numbers_count": len(numbers) if numbers else 0,
            }
        except Exception as e:
            errors.append(f"Phone extraction failed: {type(e).__name__}: {e}")
            phones_payload = {"status": "Error", "meta": {}, "consolidated_numbers_count": 0}

    # 1) Short German summary (≤100 words) for UI display
    german_summary, german_raw, _tok = generate_german_short_summary_from_description(
        gemini_client=client,
        config=cfg,
        description_text=blob,
        llm_context_dir=ctx,
        llm_requests_dir=req,
        file_identifier_prefix=prefix,
        triggering_input_row_id="UI",
        triggering_company_name=company_name or "Unknown",
    )
    if german_summary is None:
        errors.append(f"German short summary failed: {german_raw}")

    # 2) WebsiteTextSummary object for attribute extraction (reuse the blob as summary)
    summary_obj = WebsiteTextSummary(
        original_url=company_url,
        summary=str(blob),
        extracted_company_name_from_summary=company_name or None,
        key_topics_mentioned=[],
    )

    # 3) Attribute extraction
    attrs_obj, attrs_raw, _tok2 = extract_detailed_attributes(
        gemini_client=client,
        config=cfg,
        summary_obj=summary_obj,
        llm_context_dir=ctx,
        llm_requests_dir=req,
        file_identifier_prefix=prefix,
        triggering_input_row_id="UI",
        triggering_company_name=company_name or "Unknown",
    )
    if not attrs_obj:
        errors.append(f"Attribute extraction failed: {attrs_raw}")
        return SingleCompanyResult(
            short_german_description=german_summary,
            attributes=None,
            partner_match=None,
            sales_pitch=None,
            phones=phones_payload,
            errors=errors,
        )

    # 4) Partner match
    pm_obj, pm_raw, _tok3 = match_partner(
        gemini_client=client,
        config=cfg,
        target_attributes=attrs_obj,
        golden_partner_summaries=partner_summaries,
        llm_context_dir=ctx,
        llm_requests_dir=req,
        file_identifier_prefix=prefix,
        triggering_input_row_id="UI",
        triggering_company_name=company_name or "Unknown",
    )
    if not pm_obj:
        errors.append(f"Partner match failed: {pm_raw}")
        return SingleCompanyResult(
            short_german_description=german_summary,
            attributes=attrs_obj.model_dump(),
            partner_match=None,
            sales_pitch=None,
            phones=phones_payload,
            errors=errors,
        )

    matched_name = getattr(pm_obj, "matched_partner_name", None)
    # For parity with the main pipeline: pass the summarized partner dict into the sales pitch generator.
    matched_partner_summary = None
    if matched_name and partner_summaries:
        for p in partner_summaries:
            if str(p.get("name")) == str(matched_name):
                matched_partner_summary = p
                break
    if not matched_partner_summary:
        errors.append(f"Matched partner data not found for: {matched_name}")
        return SingleCompanyResult(
            short_german_description=german_summary,
            attributes=attrs_obj.model_dump(),
            partner_match=pm_obj.model_dump(),
            sales_pitch=None,
            phones=phones_payload,
            errors=errors,
        )

    # 5) Sales pitch
    sp_obj, sp_raw, _tok4 = generate_sales_pitch(
        gemini_client=client,
        config=cfg,
        target_attributes=attrs_obj,
        matched_partner=matched_partner_summary,
        website_summary_obj=summary_obj,
        previous_match_rationale=pm_obj.match_rationale_features or [],
        llm_context_dir=ctx,
        llm_requests_dir=req,
        file_identifier_prefix=prefix,
        triggering_input_row_id="UI",
        triggering_company_name=company_name or "Unknown",
    )
    if not sp_obj:
        errors.append(f"Sales pitch failed: {sp_raw}")
        return SingleCompanyResult(
            short_german_description=german_summary,
            attributes=attrs_obj.model_dump(),
            partner_match=pm_obj.model_dump(),
            sales_pitch=None,
            phones=phones_payload,
            errors=errors,
        )

    return SingleCompanyResult(
        short_german_description=german_summary,
        attributes=attrs_obj.model_dump(),
        partner_match=pm_obj.model_dump(),
        sales_pitch=sp_obj.model_dump(),
        phones=phones_payload,
        errors=errors,
    )

