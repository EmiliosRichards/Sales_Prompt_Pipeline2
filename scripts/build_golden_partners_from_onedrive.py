import argparse
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `import src...` works when running as
# `python scripts\build_golden_partners_from_onedrive.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.config import AppConfig
from src.llm_clients.gemini_client import GeminiClient
from src.utils.llm_processing_helpers import extract_json_from_text

from google.genai import types as genai_types


DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _illegal_excel_chars_re() -> re.Pattern:
    # openpyxl uses this same character class.
    return re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


_ILLEGAL_XLSX_RE = _illegal_excel_chars_re()


def sanitize_excel_cell(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = _ILLEGAL_XLSX_RE.sub("", s)
    return s.strip()


def extract_docx_text(docx_path: Path) -> str:
    """
    Extract reasonably clean text from a .docx without extra dependencies.
    Keeps paragraph boundaries.
    """
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_bytes = z.read("word/document.xml")
    except Exception as e:
        raise RuntimeError(f"Failed to read docx zip/document.xml: {docx_path} ({e})")

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to parse document.xml: {docx_path} ({e})")

    lines: List[str] = []
    for p in root.findall(".//w:p", DOCX_NS):
        texts = [t.text for t in p.findall(".//w:t", DOCX_NS) if t.text]
        line = "".join(texts).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def extract_pdf_text(pdf_path: Path) -> str:
    """
    PDF extraction using pypdf (required dependency).
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        raise RuntimeError(
            "PDF support requires 'pypdf'. Install it (e.g. `pip install -r requirements.txt`)."
        )

    try:
        reader = PdfReader(str(pdf_path))
        parts: List[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def build_partner_docs_bundle(partner_name: str, files: List[Path], max_chars: int) -> str:
    chunks: List[str] = []
    for fp in sorted(files, key=lambda p: p.name.lower()):
        suffix = fp.suffix.lower()
        text = ""
        if suffix == ".docx":
            text = extract_docx_text(fp)
        elif suffix == ".pdf":
            text = extract_pdf_text(fp)
        else:
            continue

        text = sanitize_excel_cell(text)
        if not text:
            continue

        chunks.append(f"=== FILE: {fp.name} ===\n{text}")

    bundle = "\n\n".join(chunks).strip()
    if not bundle:
        return ""

    if len(bundle) > max_chars:
        bundle = bundle[:max_chars] + "\n\n[TRUNCATED]"
    return bundle


CANONICAL_COLUMNS: List[str] = [
    "Company Name",
    "Industry",
    "Industry Category",
    "Products/Services Offered",
    "USP (Unique Selling Proposition) / Key Selling Points",
    "Customer Target Segments",
    "Customer Target Segments Category",
    "Business Model",
    "Business Model Category",
    "Company Size Indicators",
    "Company Size Category",
    "Innovation Level Indicators",
    "Geographic Reach",
    "Geographic Reach Category",
    "Website",
    "Email",
    "Phone",
    "Source Document Section/Notes",
    "Is Successful Partner",
    "Partner_Targets_B2B",
    "Partner_Target_Audience_Size_Over_1000",
    "Targets_Specific_Industry_Type",
    "Is_Startup",
    "Is_AI_Software",
    "Is_Innovative_Product",
    "Is_Disruptive_Product",
    "Is_VC_Funded",
    "Is_SaaS_Software",
    "Is_Complex_Solution",
    "Is_Investment_Product",
    "Avg Leads Per Day",
    "Rank (1-47)",
]


def coerce_row_to_schema(row: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure all canonical columns exist and are safe for Excel.
    out: Dict[str, Any] = {}
    for col in CANONICAL_COLUMNS:
        out[col] = sanitize_excel_cell(row.get(col, ""))
    return out


def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def llm_extract_partner_row(
    gemini_client: GeminiClient,
    app_config: AppConfig,
    prompt_template: str,
    partner_name: str,
    docs_bundle: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Returns (row_dict, raw_response_text).
    """
    prompt = prompt_template.replace("{{PARTNER_DOCS_BUNDLE}}", docs_bundle)

    system_instruction = (
        "Your entire response MUST be a single valid JSON object and nothing else."
    )

    generation_config_dict = {
        "response_mime_type": "text/plain",
        "candidate_count": 1,
        # Keep this reasonably small; the extraction JSON should be compact.
        "max_output_tokens": min(int(app_config.llm_max_tokens or 1024), 2048),
        "temperature": float(getattr(app_config, "llm_temperature_extraction", 0.2) or 0.2),
    }
    if hasattr(app_config, "llm_top_k") and app_config.llm_top_k is not None:
        generation_config_dict["top_k"] = app_config.llm_top_k
    if hasattr(app_config, "llm_top_p") and app_config.llm_top_p is not None:
        generation_config_dict["top_p"] = app_config.llm_top_p
    generation_config = genai_types.GenerateContentConfig(**generation_config_dict)

    response = gemini_client.generate_content_with_retry(
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        generation_config=generation_config,
        system_instruction=system_instruction,
        file_identifier_prefix=f"GP_{partner_name}",
        triggering_input_row_id="golden_partner_build",
        triggering_company_name=partner_name,
    )

    raw = response.text if response and hasattr(response, "text") else ""
    json_str = extract_json_from_text(raw) or ""
    if not json_str:
        raise RuntimeError(f"LLM did not return parseable JSON for partner '{partner_name}'.")
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise RuntimeError(f"LLM JSON was not an object for partner '{partner_name}'.")
    return data, raw


def merge_into_existing(existing_df: pd.DataFrame, new_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge by exact Company Name match (case-insensitive trim). New rows overwrite existing values
    when the new value is non-empty.
    """
    df = existing_df.copy()
    if "Company Name" not in df.columns:
        raise ValueError("Existing golden partner sheet missing 'Company Name' column.")

    # Ensure all canonical columns exist
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    def key(s: Any) -> str:
        return str(s or "").strip().lower()

    idx_by_name = {key(n): i for i, n in enumerate(df["Company Name"].tolist()) if key(n)}

    for r in new_rows:
        name = key(r.get("Company Name", ""))
        if not name:
            continue
        if name in idx_by_name:
            i = idx_by_name[name]
            for col in CANONICAL_COLUMNS:
                nv = sanitize_excel_cell(r.get(col, ""))
                if nv:
                    df.at[i, col] = nv
        else:
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)

    # Preserve a stable ordering if Rank exists and is numeric-like.
    if "Rank (1-47)" in df.columns:
        rank_num = pd.to_numeric(df["Rank (1-47)"], errors="coerce")
        df["_rank_sort"] = rank_num.fillna(1e9)
        df = df.sort_values(by=["_rank_sort", "Company Name"], kind="stable").drop(columns=["_rank_sort"])

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/merge Golden Partner rows from OneDrive partner documents."
    )
    parser.add_argument(
        "--onedrive-root",
        type=str,
        default=r"OneDrive_2026-03-10\Golden Partner",
        help="Root folder containing partner subfolders.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=r"prompts\golden_partner_profile_from_docs_prompt.txt",
        help="Prompt template path (repo-relative).",
    )
    parser.add_argument(
        "--out-xlsx",
        type=str,
        default="",
        help="Output XLSX path. Defaults to data/kgs_from_onedrive_<YYYYMMDD>.xlsx",
    )
    parser.add_argument(
        "--merge-into",
        type=str,
        default="",
        help="Optional existing golden partners XLSX to merge into (updates existing rows, appends new).",
    )
    parser.add_argument(
        "--partners",
        type=str,
        nargs="*",
        default=[],
        help="Optional list of partner folder names to process (defaults to all).",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=30000,
        help="Maximum chars of combined docs sent to the LLM per partner.",
    )
    args = parser.parse_args()

    load_dotenv(override=False)
    app_config = AppConfig()
    gemini_client = GeminiClient(config=app_config)

    onedrive_root = Path(args.onedrive_root)
    if not onedrive_root.exists():
        raise FileNotFoundError(f"OneDrive root not found: {onedrive_root}")

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    prompt_template = load_prompt(prompt_path)

    out_xlsx = args.out_xlsx.strip()
    if not out_xlsx:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_xlsx = str(Path("data") / f"kgs_from_onedrive_{stamp}.xlsx")
    out_xlsx_path = Path(out_xlsx)
    out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing output: if it exists, add a timestamp suffix.
    if out_xlsx_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_xlsx_path = out_xlsx_path.with_name(f"{out_xlsx_path.stem}__{stamp}{out_xlsx_path.suffix}")

    partner_dirs = [p for p in onedrive_root.iterdir() if p.is_dir()]
    if args.partners:
        allow = set(args.partners)
        partner_dirs = [p for p in partner_dirs if p.name in allow]

    new_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for pdir in sorted(partner_dirs, key=lambda p: p.name.lower()):
        partner_name = pdir.name
        files = [f for f in pdir.rglob("*") if f.is_file() and f.suffix.lower() in {".docx", ".pdf"}]
        if not files:
            failures.append({"partner": partner_name, "error": "No docx/pdf files found"})
            continue

        bundle = build_partner_docs_bundle(partner_name, files, max_chars=int(args.max_input_chars))
        if not bundle:
            failures.append({"partner": partner_name, "error": "No extractable text from files"})
            continue

        try:
            data, _raw = llm_extract_partner_row(
                gemini_client=gemini_client,
                app_config=app_config,
                prompt_template=prompt_template,
                partner_name=partner_name,
                docs_bundle=bundle,
            )
            # Coerce to schema + prefer LEGAL name from docs.
            legal_name = sanitize_excel_cell(data.get("Company Name") or "")
            if not legal_name:
                # Fallback: keep the folder name but flag it for review.
                legal_name = partner_name
                existing_notes = sanitize_excel_cell(data.get("Source Document Section/Notes") or "")
                flag = "WARNING: Legal name not found in docs; used OneDrive folder name."
                data["Source Document Section/Notes"] = (flag + ("\n" + existing_notes if existing_notes else "")).strip()
            data["Company Name"] = legal_name
            # Always embed the folder name for traceability (harmless if duplicated).
            notes_now = sanitize_excel_cell(data.get("Source Document Section/Notes") or "")
            folder_tag = f"OneDrive folder: {partner_name}"
            if folder_tag not in notes_now:
                data["Source Document Section/Notes"] = (folder_tag + ("\n" + notes_now if notes_now else "")).strip()
            row = coerce_row_to_schema(data)
            new_rows.append(row)
        except Exception as e:
            failures.append({"partner": partner_name, "error": str(e)})

    # Build output DF
    df_out = pd.DataFrame(new_rows)
    for c in CANONICAL_COLUMNS:
        if c not in df_out.columns:
            df_out[c] = ""
    df_out = df_out[CANONICAL_COLUMNS]

    if args.merge_into:
        existing_path = Path(args.merge_into)
        # Never write output over the input golden-partner file.
        try:
            if existing_path.resolve() == out_xlsx_path.resolve():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_xlsx_path = out_xlsx_path.with_name(f"{out_xlsx_path.stem}__copy_{stamp}{out_xlsx_path.suffix}")
        except Exception:
            pass
        existing_df = pd.read_excel(existing_path) if existing_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(existing_path)
        df_out = merge_into_existing(existing_df=existing_df, new_rows=new_rows)
        # Ensure canonical columns ordering
        for c in CANONICAL_COLUMNS:
            if c not in df_out.columns:
                df_out[c] = ""
        df_out = df_out[CANONICAL_COLUMNS]

    df_out.to_excel(out_xlsx_path, index=False)

    # Sidecar report for review
    report_path = out_xlsx_path.with_suffix(".build_report.json")
    report = {
        "generated_at": datetime.now().isoformat(),
        "onedrive_root": str(onedrive_root),
        "partners_processed": len(partner_dirs),
        "rows_written": len(df_out),
        "failures": failures,
        "output_xlsx": str(out_xlsx_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {out_xlsx_path}")
    print(f"Report: {report_path}")
    if failures:
        print(f"Failures: {len(failures)} (see report)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

