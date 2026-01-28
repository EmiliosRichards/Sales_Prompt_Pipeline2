"""
Handles the generation of CSV reports for prospect analysis.

This module provides functions to take structured prospect analysis data,
flatten it, and write it to a CSV file in a specified output directory.
It includes logic for handling nested data structures and ensuring
consistent output for easier consumption and review.
"""
import os
import logging
import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd

from ..core.schemas import GoldenPartnerMatchOutput, DetailedCompanyAttributes

logger = logging.getLogger(__name__)
def _parse_contacts_block(text: Optional[str]) -> str:
    """Extract 'Name | Role | Phone | Email' tuples from noisy contact text.

    Strategy:
    - Ignore boilerplate/nav lines (e.g., "Mehr Lesen", "Zum Produkt", ...).
    - Prefer parsing within contact sections (e.g., 'Mitarbeiter', 'Ansprechpartner', 'Contacts').
    - Detect names as 2–4 capitalized words; capture nearby role (non-phone/non-email)
      and phone/email within the next few lines.
    - Return unique entries joined by ' || '.
    """
    if not text or not isinstance(text, str):
        return ""

    raw_lines = [ln.strip() for ln in text.splitlines()]
    # Drop obvious boilerplate tokens
    drop_prefixes = {
        "mehr lesen", "download", "zum hallenplan", "produkte", "zum produkt", "keywords",
        "wir bieten", "branche", "über uns", "halles", "standnummer", "hersteller",
        "video", "kontaktinformation", "sprache", "kontakt per e-mail", "zum ausstellerverzeichnis",
        "hallenplan", "programm", "news", "newsletter", "impressum", "datenschutz", "agb"
    }
    lines: List[str] = []
    for ln in raw_lines:
        low = ln.lower().strip(': ')
        if not ln:
            continue
        if any(low.startswith(pfx) for pfx in drop_prefixes):
            continue
        lines.append(ln)

    # Regexes
    phone_re = re.compile(r"(\+\d[\d\s()\-]+\d)")
    tel_re = re.compile(r"tel\.?\s*[:\-]?\s*(\+?\d[\d\s()\-]+\d)", re.IGNORECASE)
    email_re = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
    name_re = re.compile(r"^[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß\-']+(\s+[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß\-']+){1,3}$")

    # Only parse after a contacts section marker if present; otherwise parse whole text
    markers = {"mitarbeiter", "ansprechpartner", "contacts", "kontakt", "team"}
    start_idx = 0
    for idx, ln in enumerate(lines):
        if ln.lower() in markers:
            start_idx = idx + 1
            break

    entries: List[str] = []
    seen = set()
    i = start_idx
    while i < len(lines):
        ln = lines[i]
        # Remove "E-Mail senden" placeholder-only lines
        if ln.lower() == "e-mail senden" or ln.lower() == "email senden":
            i += 1
            continue

        if name_re.match(ln):
            name = ln
            role = ""
            phone = ""
            email = ""
            j = i + 1
            # collect role if the next non-empty line is not a phone/email
            while j < len(lines) and lines[j] == "":
                j += 1
            if j < len(lines) and not (phone_re.search(lines[j]) or tel_re.search(lines[j]) or email_re.search(lines[j])):
                # avoid picking obvious non-role markers
                cand_role = lines[j]
                if not any(tok in cand_role.lower() for tok in ("zum produkt", "download", "hallenplan")):
                    role = cand_role
                j += 1

            # search a few lines for phone/email
            k = j
            while k < len(lines) and k < j + 8:
                if not phone and (m := phone_re.search(lines[k]) or tel_re.search(lines[k])):
                    # support both regexes
                    m1 = phone_re.search(lines[k])
                    m2 = tel_re.search(lines[k])
                    phone = (m1.group(1) if m1 else m2.group(1)) if (m1 or m2) else ""
                if not email and (em := email_re.search(lines[k])):
                    email = em.group(0)
                if phone and email:
                    break
                k += 1

            parts = [p for p in [name, role, phone, email] if p]
            if name and (phone or email):
                key = (name, role, phone, email)
                if key not in seen:
                    seen.add(key)
                    entries.append(" | ".join(parts))
            i = max(k, j, i + 1)
        else:
            i += 1

    return " || ".join(entries)


def write_sales_outreach_report(
    output_data: List[GoldenPartnerMatchOutput],
    output_dir: str,
    run_id: str,
    original_df: pd.DataFrame,
    sales_prompt_path: Optional[str] = None,
    output_format: str = 'csv'
) -> Optional[str]:
    """
    Writes the sales outreach data to a CSV or Excel file.

    Args:
        output_data (List[GoldenPartnerMatchOutput]): A list of GoldenPartnerMatchOutput objects.
        output_dir (str): The directory where the file will be saved.
        run_id (str): The unique identifier for the current run.
        original_df (pd.DataFrame): The original input DataFrame.
        output_format (str): The output format, either 'csv' or 'excel'.

    Returns:
        Optional[str]: The full path to the saved file, or None if an error occurred.
    """
    if not output_data:
        logger.warning("No output data provided to write_sales_outreach_report. Skipping CSV generation.")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        if output_format == 'excel':
            filename = f"SalesOutreachReport_{run_id}.xlsx"
        else:
            filename = f"SalesOutreachReport_{run_id}.csv"
        full_path = os.path.join(output_dir, filename)

        report_data = []
        

        # Create a mapping from URL to original row for efficient lookup
        original_df_map = {}
        for i in range(len(original_df)):
            row_series = original_df.iloc[i]
            row_dict = row_series.to_dict()
            # Adding 2 because header is 1 and index is 0-based
            row_dict['original_row_number'] = i + 2
            url = row_series.get('GivenURL')
            if url:
                original_df_map[url] = row_dict

        for item in output_data:
            original_url = item.analyzed_company_url
            original_row_data = original_df_map.get(original_url)

            attrs = item.analyzed_company_attributes if item else None
            
            # Initialize a base row structure
            row = {
                'Company Name': None, 'Original_Number': None, 'URL': original_url,
                'CanonicalEntryURL': None,
                'PhoneNumber_Status': None,
                'is_b2b': None, 'is_b2b_reason': None,
                'serves_1000': None, 'serves_1000_reason': None,
                'found_number': None,
                # Top callable numbers (LLM-ranked when phone retrieval ran; otherwise may be input-provided)
                'Top_Number_1': None, 'Top_Type_1': None, 'Top_SourceURL_1': None,
                'Top_Number_2': None, 'Top_Type_2': None, 'Top_SourceURL_2': None,
                'Top_Number_3': None, 'Top_Type_3': None, 'Top_SourceURL_3': None,
                # Person-associated contact fields (optional; populated by phone extraction when present)
                'BestPersonContactName': None,
                'BestPersonContactRole': None,
                'BestPersonContactDepartment': None,
                'BestPersonContactNumber': None,
                'PersonContacts': None,
                # Main office / switchboard backup + LLM ranking trace
                'MainOffice_Number': None,
                'MainOffice_Type': None,
                'MainOffice_SourceURL': None,
                'LLMPhoneRanking': None,
                'LLMPhoneRankingError': None,
                # Callable-but-deprioritized + suspected other-org buckets (second-stage phone reranker)
                'DeprioritizedNumbers': None,
                'SuspectedOtherOrgNumbers': None,
                'sales_pitch': None, 'Short German Description': None, 'matched_golden_partner': None,
                'match_reasoning': None,
                'Industry': None,
                'Matched Partner Description': '', 'Avg Leads Per Day': '',
                'Rank': '', 'B2B Indicator': '',
                'Phone Outreach Suitability': '', 'Target Group Size Assessment': '',
                'Products/Services Offered': '', 'USP/Key Selling Points': '',
                'Customer Target Segments': '', 'Business Model': '',
                'Company Size Inferred': '', 'Innovation Level Indicators': '',
                'Website Clarity Notes': '',
                'Original Row Number': None,
                'ScrapeStatus': None
            }

            parsed_contacts_val = ""
            if original_row_data is not None:
                row.update({
                    'Company Name': original_row_data.get('CompanyName'),
                    'Original_Number': original_row_data.get('Original_Number') or original_row_data.get('PhoneNumber') or original_row_data.get('Number'),
                    'CanonicalEntryURL': original_row_data.get('CanonicalEntryURL'),
                    'PhoneNumber_Status': original_row_data.get('PhoneNumber_Status'),
                    # Prefer the pipeline-produced short German description if present; fall back to input Description.
                    'Short German Description': original_row_data.get('Short German Description') or original_row_data.get('Description'),
                    'Industry': original_row_data.get('Industry'),
                    'is_b2b': original_row_data.get('is_b2b'),
                    'is_b2b_reason': original_row_data.get('is_b2b_reason'),
                    'serves_1000': original_row_data.get('serves_1000'),
                    'serves_1000_reason': original_row_data.get('serves_1000_reason'),
                    'found_number': original_row_data.get('found_number'),
                    'Top_Number_1': original_row_data.get('Top_Number_1'),
                    'Top_Type_1': original_row_data.get('Top_Type_1'),
                    'Top_SourceURL_1': original_row_data.get('Top_SourceURL_1'),
                    'Top_Number_2': original_row_data.get('Top_Number_2'),
                    'Top_Type_2': original_row_data.get('Top_Type_2'),
                    'Top_SourceURL_2': original_row_data.get('Top_SourceURL_2'),
                    'Top_Number_3': original_row_data.get('Top_Number_3'),
                    'Top_Type_3': original_row_data.get('Top_Type_3'),
                    'Top_SourceURL_3': original_row_data.get('Top_SourceURL_3'),
                    'BestPersonContactName': original_row_data.get('BestPersonContactName'),
                    'BestPersonContactRole': original_row_data.get('BestPersonContactRole'),
                    'BestPersonContactDepartment': original_row_data.get('BestPersonContactDepartment'),
                    'BestPersonContactNumber': original_row_data.get('BestPersonContactNumber'),
                    'PersonContacts': original_row_data.get('PersonContacts'),
                    'MainOffice_Number': original_row_data.get('MainOffice_Number'),
                    'MainOffice_Type': original_row_data.get('MainOffice_Type'),
                    'MainOffice_SourceURL': original_row_data.get('MainOffice_SourceURL'),
                    'LLMPhoneRanking': original_row_data.get('LLMPhoneRanking'),
                    'LLMPhoneRankingError': original_row_data.get('LLMPhoneRankingError'),
                    'DeprioritizedNumbers': original_row_data.get('DeprioritizedNumbers'),
                    'SuspectedOtherOrgNumbers': original_row_data.get('SuspectedOtherOrgNumbers'),
                    'Original Row Number': original_row_data.get('original_row_number'),
                    'ScrapeStatus': original_row_data.get('ScrapingStatus'),
                })
                # Parse contacts if present
                contacts_text = original_row_data.get('contacts') if isinstance(original_row_data, dict) else None
                if contacts_text:
                    parsed_contacts_val = _parse_contacts_block(contacts_text)

            if item:
                row.update({
                    'Short German Description': item.summary or row.get('Short German Description'),
                    'sales_pitch': item.phone_sales_line.replace('{programmatic placeholder}', f"{item.avg_leads_per_day:.0f}") if item.phone_sales_line and item.avg_leads_per_day is not None else item.phone_sales_line,
                    'match_reasoning': "; ".join(item.match_rationale_features) if item.match_rationale_features else "",
                    'matched_golden_partner': item.matched_partner_name or '',
                    'Matched Partner Description': item.matched_partner_description or '',
                    'Avg Leads Per Day': item.avg_leads_per_day if item.avg_leads_per_day is not None else '',
                    'Rank': item.rank if item.rank is not None else '',
                })

            if attrs:
                row.update({
                    'Industry': attrs.industry or row.get('Industry'),
                    'B2B Indicator': attrs.b2b_indicator or '',
                    'Phone Outreach Suitability': attrs.phone_outreach_suitability or '',
                    'Target Group Size Assessment': attrs.target_group_size_assessment or '',
                    'Products/Services Offered': "; ".join(attrs.products_services_offered) if attrs.products_services_offered else '',
                    'USP/Key Selling Points': "; ".join(attrs.usp_key_selling_points) if attrs.usp_key_selling_points else '',
                    'Customer Target Segments': "; ".join(attrs.customer_target_segments) if attrs.customer_target_segments else '',
                    'Business Model': attrs.business_model or '',
                    'Company Size Inferred': attrs.company_size_category_inferred or '',
                    'Innovation Level Indicators': attrs.innovation_level_indicators_text or '',
                    'Website Clarity Notes': attrs.website_clarity_notes or ''
                })
            
            # Optionally attach Parsed_Contacts column when available
            if parsed_contacts_val:
                row['Parsed_Contacts'] = parsed_contacts_val

            # Ensure PersonContacts is serialized to a stable string for CSV/Excel friendliness.
            # (JSONL outputs elsewhere preserve native objects.)
            try:
                if isinstance(row.get("PersonContacts"), (list, dict)):
                    row["PersonContacts"] = json.dumps(row["PersonContacts"], ensure_ascii=False)
            except Exception:
                pass

            # Same for second-stage ranking trace (if present)
            try:
                if isinstance(row.get("LLMPhoneRanking"), (list, dict)):
                    row["LLMPhoneRanking"] = json.dumps(row["LLMPhoneRanking"], ensure_ascii=False)
            except Exception:
                pass
            report_data.append(row)

        if not report_data:
            logger.warning(f"No data to write for sales outreach report. Run ID: {run_id}")
            return None

        df = pd.DataFrame(report_data)
        if output_format == 'excel':
            df.to_excel(full_path, index=False)
            logger.info(f"Successfully wrote sales outreach report to Excel: {full_path}")
        else:
            df.to_csv(full_path, index=False, encoding='utf-8-sig')
            logger.info(f"Successfully wrote sales outreach report to CSV: {full_path}")
        return full_path

    except Exception as e:
        logger.error(f"Error writing sales outreach report to CSV for run_id {run_id}: {e}", exc_info=True)
        return None