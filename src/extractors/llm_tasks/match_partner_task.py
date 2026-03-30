import json
import logging
import re
import unicodedata
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

from sklearn.decomposition import TruncatedSVD  # pyright: ignore[reportMissingModuleSource]
from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics.pairwise import cosine_similarity  # pyright: ignore[reportMissingModuleSource]

from ...core.config import AppConfig
from ...core.schemas import DetailedCompanyAttributes, PartnerMatchOnlyOutput
from ...matching.taxonomy import infer_match_taxonomy
from ...utils.helpers import sanitize_filename_component
from ...utils.llm_processing_helpers import (
    load_prompt_template,
    save_llm_artifact,
)

logger = logging.getLogger(__name__)

_GENERIC_RATIONALE_MARKERS = (
    "digitale lösungen",
    "digitale software",
    "softwarelösungen",
    "digitalisierungslösungen",
    "plattformen",
    "technologie",
    "automatisierung",
    "effizienzsteigerung",
    "prozessoptimierung",
    "transparenz",
    "innovation",
    "innovativ",
    "künstliche intelligenz",
    "ki ",
)
_SPECIFIC_RATIONALE_MARKERS = (
    "gemeinsame zielgruppe",
    "zielgruppe",
    "buyer",
    "buying center",
    "shared audience",
    "shared target audience",
    "same audience",
    "gemeinsame ansprechpartner",
    "gemeinsame entscheider",
    "ähnliche produkte",
    "ähnliche services",
    "ähnliche produkt",
    "gemeinsame branche",
    "überschneidende marktsegmente",
    "ähnliche branchenanforderungen",
    "gemeinsame kunden",
    "anwendungsfälle",
    "workflow",
    "use cases",
    "einzelhändler",
    "law firms",
    "kanzlei",
    "marken",
    "logistik",
    "rfid",
    "e-commerce",
    "vertrieb",
)
_SHORTLIST_KEEP_SHORT_TOKENS = {
    "ai", "bi", "crm", "erp", "hr", "it", "ot", "rfid", "seo", "sea", "api",
    "edi", "gps", "gnss", "plm", "eam", "mde", "cms", "b2b", "b2c", "iot",
}
_SHORTLIST_STOPWORDS = {
    "and", "the", "for", "with", "from", "into", "that", "this", "these", "those",
    "their", "your", "they", "them", "already", "very", "more", "less", "also",
    "our", "ours", "you", "wir", "und", "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "eines", "einem", "einen", "mit", "auf", "von", "für",
    "fur", "aus", "bei", "als", "auch", "noch", "durch", "uber", "über", "im",
    "in", "am", "an", "to", "of", "or", "is", "are", "be", "can", "will", "via",
    "company", "companies", "business", "businesses", "kunden", "customer", "customers",
    "solutions", "solution", "software", "services", "service", "plattform", "platform",
    "digital", "digitale", "technology", "technologies", "technologie",
    "innovative", "innovation", "innovativ", "system", "systems", "general", "specific",
    "focus", "fokus", "support", "offered", "offering", "offerings", "angebot", "angebote",
}
_GENERIC_AUDIENCE_TAGS = {"sme_buyers"}
_GENERIC_WORKFLOW_TAGS = {"workflow_automation"}
_GENERIC_DOMAIN_TAGS: set[str] = set()
_DOMAIN_MISMATCH_PENALTIES: Dict[str, Dict[str, int]] = {
    "construction": {"document_software": 22, "healthcare": 26, "digital_agency": 16},
    "defense_simulation": {"document_software": 28, "digital_agency": 24, "healthcare": 28},
    "digital_agency": {"document_software": 18, "industrial_maintenance": 20, "logistics": 16, "mobility": 16},
    "ecommerce_retail": {"healthcare": 24, "industrial_maintenance": 20, "defense_simulation": 28},
    "mobility": {"document_software": 26, "digital_agency": 18, "healthcare": 24, "law_tax": 24},
    "procurement": {"healthcare": 20, "digital_agency": 14, "mobility": 14},
}


def _normalize_match_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _coerce_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item or "").strip()]
    text = str(value).strip()
    return [text] if text else []


def _tokenize_match_text(value: Any) -> List[str]:
    tokens: List[str] = []
    for token in _normalize_match_text(value).split():
        if len(token) <= 2 and token not in _SHORTLIST_KEEP_SHORT_TOKENS:
            continue
        if token in _SHORTLIST_STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _weighted_terms(values: List[str], weight: int) -> Counter[str]:
    terms: Counter[str] = Counter()
    for value in values:
        for token in _tokenize_match_text(value):
            terms[token] += weight
    return terms


def _extract_phrases(values: List[str], limit: int = 8) -> List[str]:
    phrases: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_match_text(value)
        if not normalized:
            continue
        for part in re.split(r"\s*;\s*|\s*,\s*", normalized):
            words = [w for w in part.split() if w not in _SHORTLIST_STOPWORDS]
            if len(words) < 2:
                continue
            phrase = " ".join(words[:8]).strip()
            if len(phrase) < 8 or phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text or text.lower() in {"none", "null", "nan", "n/a", "na"}:
        return ""
    return text


def _normalize_match_confidence(value: Optional[str], match_score: Optional[str] = None) -> str:
    normalized = _normalize_match_text(value)
    if normalized in {"strong", "high"}:
        return "strong"
    if normalized in {"weak", "medium"}:
        return "weak"
    if normalized in {"no match", "nomatch", "no_match", "low"}:
        return "no_match"
    score = _normalize_match_score(match_score)
    if score == "high":
        return "strong"
    if score == "medium":
        return "weak"
    if score == "low":
        return "no_match"
    return ""


def _normalize_overlap_type(value: Optional[str]) -> str:
    normalized = _normalize_match_text(value)
    if normalized in {"audience", "buyer", "target audience"}:
        return "audience"
    if normalized in {"use case", "usecase"}:
        return "use_case"
    if normalized in {"workflow", "process"}:
        return "workflow"
    if normalized == "industry":
        return "industry"
    if normalized in {"mixed", "audience use case", "audience workflow"}:
        return "mixed"
    return normalized or "mixed"


def _normalize_match_score(value: Optional[str]) -> str:
    low = (value or "").strip().lower()
    if low in {"high", "hoch"}:
        return "high"
    if low in {"medium", "mittel"}:
        return "medium"
    if low in {"low", "niedrig"}:
        return "low"
    return ""


def _normalize_rationale_features(features: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for item in features or []:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_evidence_list(values: Optional[List[str]]) -> List[str]:
    evidence: List[str] = []
    for value in values or []:
        cleaned = _clean_text(value)
        if cleaned:
            evidence.append(cleaned)
    return evidence


def _is_specific_rationale_feature(feature: str) -> bool:
    low = feature.lower()
    return any(marker in low for marker in _SPECIFIC_RATIONALE_MARKERS)


def _is_generic_rationale_feature(feature: str) -> bool:
    low = feature.lower()
    return any(marker in low for marker in _GENERIC_RATIONALE_MARKERS)


def _profile_texts_from_match_profile(match_profile: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    profile = match_profile or {}
    audience_texts = _coerce_text_list(profile.get("target_audience"))
    use_case_texts = (
        _coerce_text_list(profile.get("services_products"))
        + _coerce_text_list(profile.get("usp"))
    )
    industry_texts = _coerce_text_list(profile.get("industry"))
    business_model_texts = _coerce_text_list(profile.get("business_model"))
    return {
        "audience_texts": audience_texts,
        "use_case_texts": use_case_texts,
        "industry_texts": industry_texts,
        "business_model_texts": business_model_texts,
    }


def _weighted_text(values: List[str], repeat_count: int) -> str:
    parts: List[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned:
            continue
        parts.extend([cleaned] * max(1, repeat_count))
    return " ".join(parts)


def _build_target_match_profile(target_attributes: DetailedCompanyAttributes) -> Dict[str, Any]:
    audience_texts = _coerce_text_list(getattr(target_attributes, "customer_target_segments", None))
    use_case_texts = (
        _coerce_text_list(getattr(target_attributes, "products_services_offered", None))
        + _coerce_text_list(getattr(target_attributes, "usp_key_selling_points", None))
        + _coerce_text_list(getattr(target_attributes, "website_clarity_notes", None))
    )
    industry_texts = _coerce_text_list(getattr(target_attributes, "industry", None))
    business_model_texts = _coerce_text_list(getattr(target_attributes, "business_model", None))
    match_taxonomy = infer_match_taxonomy(
        *audience_texts,
        *use_case_texts,
        *industry_texts,
        *business_model_texts,
    )
    dense_text = " ".join(
        part for part in [
            _weighted_text(audience_texts, 4),
            _weighted_text(use_case_texts, 3),
            _weighted_text(industry_texts, 2),
            _weighted_text(business_model_texts, 2),
        ] if part
    )
    return {
        "audience_texts": audience_texts,
        "use_case_texts": use_case_texts,
        "industry_texts": industry_texts,
        "business_model_texts": business_model_texts,
        "audience_terms": _weighted_terms(audience_texts, 6),
        "use_case_terms": _weighted_terms(use_case_texts, 4),
        "industry_terms": _weighted_terms(industry_texts, 2),
        "business_model_terms": _weighted_terms(business_model_texts, 2),
        "audience_phrases": _extract_phrases(audience_texts, limit=8),
        "use_case_phrases": _extract_phrases(use_case_texts, limit=8),
        "match_taxonomy": match_taxonomy,
        "dense_text": dense_text,
        "all_text": " ".join(audience_texts + use_case_texts + industry_texts + business_model_texts),
    }


def _build_partner_match_profile(partner_summary: Dict[str, Any]) -> Dict[str, Any]:
    match_profile = partner_summary.get("match_profile") if isinstance(partner_summary.get("match_profile"), dict) else None
    if not match_profile:
        match_profile = {
            "target_audience": partner_summary.get("target_audience"),
            "services_products": partner_summary.get("services_products"),
            "usp": partner_summary.get("usp"),
            "industry": partner_summary.get("industry"),
            "business_model": partner_summary.get("business_model"),
        }
    profile_texts = _profile_texts_from_match_profile(match_profile)
    alias_texts = _coerce_text_list(partner_summary.get("partner_aliases"))
    match_taxonomy = partner_summary.get("match_taxonomy") if isinstance(partner_summary.get("match_taxonomy"), dict) else None
    if not match_taxonomy:
        match_taxonomy = infer_match_taxonomy(
            *profile_texts["audience_texts"],
            *profile_texts["use_case_texts"],
            *profile_texts["industry_texts"],
            *profile_texts["business_model_texts"],
        )
    all_texts = (
        profile_texts["audience_texts"]
        + profile_texts["use_case_texts"]
        + profile_texts["industry_texts"]
        + profile_texts["business_model_texts"]
        + alias_texts
    )
    dense_text = " ".join(
        part for part in [
            _weighted_text(profile_texts["audience_texts"], 4),
            _weighted_text(profile_texts["use_case_texts"], 3),
            _weighted_text(profile_texts["industry_texts"], 2),
            _weighted_text(profile_texts["business_model_texts"], 2),
            _weighted_text(alias_texts, 1),
        ] if part
    )
    return {
        "audience_texts": profile_texts["audience_texts"],
        "use_case_texts": profile_texts["use_case_texts"],
        "industry_texts": profile_texts["industry_texts"],
        "business_model_texts": profile_texts["business_model_texts"],
        "alias_texts": alias_texts,
        "audience_tokens": set(_tokenize_match_text(" ".join(profile_texts["audience_texts"]))),
        "use_case_tokens": set(_tokenize_match_text(" ".join(profile_texts["use_case_texts"]))),
        "industry_tokens": set(_tokenize_match_text(" ".join(profile_texts["industry_texts"]))),
        "business_model_tokens": set(_tokenize_match_text(" ".join(profile_texts["business_model_texts"]))),
        "all_tokens": set(_tokenize_match_text(" ".join(all_texts))),
        "audience_blob": _normalize_match_text(" ".join(profile_texts["audience_texts"])),
        "use_case_blob": _normalize_match_text(" ".join(profile_texts["use_case_texts"])),
        "all_blob": _normalize_match_text(" ".join(all_texts)),
        "match_taxonomy": match_taxonomy,
        "dense_text": dense_text,
    }


def _taxonomy_tag_weight(category: str, tag: str) -> int:
    if category == "audience":
        return 10 if tag in _GENERIC_AUDIENCE_TAGS else 26
    if category == "workflow":
        return 8 if tag in _GENERIC_WORKFLOW_TAGS else 18
    return 6 if tag in _GENERIC_DOMAIN_TAGS else 14


def _format_taxonomy_label(value: str) -> str:
    return value.replace("_", " ")


def _score_taxonomy_alignment(
    target_profile: Dict[str, Any],
    partner_profile: Dict[str, Any],
) -> Dict[str, Any]:
    target_taxonomy = target_profile.get("match_taxonomy") or {}
    partner_taxonomy = partner_profile.get("match_taxonomy") or {}
    shared_tags: Dict[str, List[str]] = {}
    overlap_score = 0

    for category in ("audience", "workflow", "domain"):
        target_tags = set(target_taxonomy.get(category) or [])
        partner_tags = set(partner_taxonomy.get(category) or [])
        shared = sorted(target_tags.intersection(partner_tags))
        if not shared:
            continue
        shared_tags[category] = shared
        overlap_score += sum(_taxonomy_tag_weight(category, tag) for tag in shared)

    shared_audience = shared_tags.get("audience", [])
    shared_workflow = shared_tags.get("workflow", [])
    shared_domain = shared_tags.get("domain", [])
    penalty = 0
    penalty_reasons: List[str] = []
    if not shared_audience and not shared_workflow:
        target_domains = set(target_taxonomy.get("domain") or [])
        partner_domains = set(partner_taxonomy.get("domain") or [])
        for target_domain in sorted(target_domains):
            mismatch_map = _DOMAIN_MISMATCH_PENALTIES.get(target_domain, {})
            for partner_domain in sorted(partner_domains):
                penalty_value = mismatch_map.get(partner_domain)
                if not penalty_value:
                    continue
                penalty += penalty_value
                penalty_reasons.append(
                    f"Domain mismatch: target {_format_taxonomy_label(target_domain)} vs partner {_format_taxonomy_label(partner_domain)}"
                )

    evidence: List[str] = []
    if shared_audience:
        evidence.append(
            "Shared audience tags: " + ", ".join(_format_taxonomy_label(tag) for tag in shared_audience[:4])
        )
    if shared_workflow:
        evidence.append(
            "Shared workflow tags: " + ", ".join(_format_taxonomy_label(tag) for tag in shared_workflow[:4])
        )
    if shared_domain:
        evidence.append(
            "Shared domain tags: " + ", ".join(_format_taxonomy_label(tag) for tag in shared_domain[:4])
        )
    evidence.extend(penalty_reasons[:2])
    return {
        "score": overlap_score - penalty,
        "shared_tags": shared_tags,
        "penalty": penalty,
        "evidence": evidence,
    }


def _score_term_overlap(target_terms: Counter[str], partner_tokens: set[str], multiplier: int) -> Tuple[int, List[str]]:
    overlap = sorted(
        [term for term in target_terms if term in partner_tokens],
        key=lambda term: (-target_terms[term], term),
    )
    score = sum(target_terms[term] * multiplier for term in overlap)
    return score, overlap


def _partner_rank_value(partner_summary: Dict[str, Any]) -> float:
    try:
        raw = str(partner_summary.get("rank") or "").strip()
        return float(raw) if raw else 9999.0
    except Exception:
        return 9999.0


def _evidence_line(prefix: str, values: List[str], limit: int = 5) -> Optional[str]:
    if not values:
        return None
    return f"{prefix}: {', '.join(values[:limit])}"


def _score_partner_shortlist_sparse(
    target_profile: Dict[str, Any],
    partner_summary: Dict[str, Any],
) -> Dict[str, Any]:
    partner_profile = _build_partner_match_profile(partner_summary)
    taxonomy_alignment = _score_taxonomy_alignment(target_profile, partner_profile)

    audience_exact_score, audience_exact_terms = _score_term_overlap(
        target_profile["audience_terms"], partner_profile["audience_tokens"], 5
    )
    audience_all_score, audience_all_terms = _score_term_overlap(
        target_profile["audience_terms"], partner_profile["all_tokens"], 3
    )
    use_case_exact_score, use_case_exact_terms = _score_term_overlap(
        target_profile["use_case_terms"], partner_profile["use_case_tokens"], 3
    )
    use_case_all_score, use_case_all_terms = _score_term_overlap(
        target_profile["use_case_terms"], partner_profile["all_tokens"], 2
    )
    industry_score, industry_terms = _score_term_overlap(
        target_profile["industry_terms"], partner_profile["all_tokens"], 1
    )
    business_model_score, business_model_terms = _score_term_overlap(
        target_profile["business_model_terms"], partner_profile["all_tokens"], 1
    )

    phrase_score = 0
    phrase_evidence: List[str] = []
    for phrase in target_profile["audience_phrases"]:
        if phrase and phrase in partner_profile["audience_blob"]:
            phrase_score += 18
            phrase_evidence.append(f"Shared target audience phrase: {phrase}")
        elif phrase and phrase in partner_profile["all_blob"]:
            phrase_score += 8
            phrase_evidence.append(f"Shared audience hint: {phrase}")
    for phrase in target_profile["use_case_phrases"]:
        if phrase and phrase in partner_profile["use_case_blob"]:
            phrase_score += 10
            phrase_evidence.append(f"Shared use-case phrase: {phrase}")
        elif phrase and phrase in partner_profile["all_blob"]:
            phrase_score += 5
            phrase_evidence.append(f"Shared use-case hint: {phrase}")

    evidence: List[str] = []
    for candidate in (
        _evidence_line("Shared target audience terms", audience_exact_terms or audience_all_terms),
        _evidence_line("Shared use-case terms", use_case_exact_terms or use_case_all_terms),
        _evidence_line("Shared industry terms", industry_terms),
        _evidence_line("Shared business-model terms", business_model_terms),
    ):
        if candidate:
            evidence.append(candidate)
    for item in phrase_evidence[:3]:
        if item not in evidence:
            evidence.append(item)
    for item in taxonomy_alignment["evidence"]:
        if item not in evidence:
            evidence.append(item)

    total_score = (
        audience_exact_score
        + audience_all_score
        + use_case_exact_score
        + use_case_all_score
        + industry_score
        + business_model_score
        + phrase_score
        + taxonomy_alignment["score"]
    )
    return {
        "partner": partner_summary,
        "score": total_score,
        "rank_value": _partner_rank_value(partner_summary),
        "evidence": evidence[:4],
        "profile": partner_profile,
        "shared_taxonomy_tags": taxonomy_alignment["shared_tags"],
        "taxonomy_penalty": taxonomy_alignment["penalty"],
    }


def _dense_similarity_scores(
    target_profile: Dict[str, Any],
    partner_summaries: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not partner_summaries:
        return {}
    documents = [target_profile["dense_text"]] + [
        _build_partner_match_profile(partner)["dense_text"]
        for partner in partner_summaries
    ]
    if not any(_clean_text(doc) for doc in documents):
        return {
            str(partner.get("partner_id") or partner.get("name") or ""): 0.0
            for partner in partner_summaries
        }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {
            str(partner.get("partner_id") or partner.get("name") or ""): 0.0
            for partner in partner_summaries
        }
    n_docs = matrix.shape[0]
    n_features = matrix.shape[1]

    if n_docs >= 4 and n_features >= 4:
        n_components = max(2, min(32, n_docs - 1, n_features - 1))
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(matrix)
        sims = cosine_similarity(reduced[0:1], reduced[1:]).flatten()
    else:
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    scores: Dict[str, float] = {}
    for partner_summary, score in zip(partner_summaries, sims):
        partner_id = str(partner_summary.get("partner_id") or partner_summary.get("name") or "")
        scores[partner_id] = float(score)
    return scores


def _rrf_score(rank_position: Optional[int], rrf_k: int) -> float:
    if not rank_position or rank_position <= 0:
        return 0.0
    return 1.0 / float(rank_position + rrf_k)


def _candidate_prompt_payload(partner_summary: Dict[str, Any]) -> Dict[str, Any]:
    match_profile = partner_summary.get("match_profile") if isinstance(partner_summary.get("match_profile"), dict) else {}
    match_taxonomy = partner_summary.get("match_taxonomy") if isinstance(partner_summary.get("match_taxonomy"), dict) else {}
    return {
        "partner_id": partner_summary.get("partner_id"),
        "name": partner_summary.get("name"),
        "rank": partner_summary.get("rank"),
        "target_audience": match_profile.get("target_audience") or partner_summary.get("target_audience") or "",
        "services_products": match_profile.get("services_products") or partner_summary.get("services_products") or "",
        "usp": match_profile.get("usp") or partner_summary.get("usp") or "",
        "industry": match_profile.get("industry") or partner_summary.get("industry") or "",
        "business_model": match_profile.get("business_model") or partner_summary.get("business_model") or "",
        "match_candidate_summary": partner_summary.get("match_candidate_summary") or "",
        "normalized_audience_tags": list(match_taxonomy.get("audience") or []),
        "normalized_workflow_tags": list(match_taxonomy.get("workflow") or []),
        "normalized_domain_tags": list(match_taxonomy.get("domain") or []),
    }


def build_partner_shortlist(
    *,
    target_attributes: DetailedCompanyAttributes,
    golden_partner_summaries: List[Dict[str, Any]],
    sparse_top_k: int,
    dense_top_k: int,
    fused_top_k: int,
    rrf_k: int,
) -> List[Dict[str, Any]]:
    if not golden_partner_summaries:
        return []

    target_profile = _build_target_match_profile(target_attributes)
    sparse_candidates = [
        _score_partner_shortlist_sparse(target_profile, partner_summary)
        for partner_summary in golden_partner_summaries
    ]
    sparse_candidates.sort(
        key=lambda item: (
            -item["score"],
            item["rank_value"],
            str(item["partner"].get("name") or "").lower(),
        )
    )
    sparse_rank_lookup = {
        str(item["partner"].get("partner_id") or item["partner"].get("name") or ""): idx
        for idx, item in enumerate(sparse_candidates[:max(1, sparse_top_k)], start=1)
    }
    sparse_payload_lookup = {
        str(item["partner"].get("partner_id") or item["partner"].get("name") or ""): item
        for item in sparse_candidates
    }

    dense_scores = _dense_similarity_scores(target_profile, golden_partner_summaries)
    dense_candidates = sorted(
        golden_partner_summaries,
        key=lambda partner: (
            -dense_scores.get(str(partner.get("partner_id") or partner.get("name") or ""), 0.0),
            _partner_rank_value(partner),
            str(partner.get("name") or "").lower(),
        )
    )
    dense_rank_lookup = {
        str(partner.get("partner_id") or partner.get("name") or ""): idx
        for idx, partner in enumerate(dense_candidates[:max(1, dense_top_k)], start=1)
    }

    fused_items: List[Tuple[float, float, Dict[str, Any], Dict[str, Any]]] = []
    for partner_summary in golden_partner_summaries:
        partner_id = str(partner_summary.get("partner_id") or partner_summary.get("name") or "")
        sparse_rank = sparse_rank_lookup.get(partner_id)
        dense_rank = dense_rank_lookup.get(partner_id)
        fused_score = _rrf_score(sparse_rank, rrf_k) + _rrf_score(dense_rank, rrf_k)
        sparse_payload = sparse_payload_lookup.get(partner_id, {})
        fused_items.append(
            (
                fused_score,
                dense_scores.get(partner_id, 0.0),
                partner_summary,
                sparse_payload,
            )
        )

    fused_items.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            _partner_rank_value(item[2]),
            str(item[2].get("name") or "").lower(),
        )
    )

    shortlisted: List[Dict[str, Any]] = []
    for shortlist_position, (fused_score, dense_score, partner_summary, sparse_payload) in enumerate(
        fused_items[:max(1, min(fused_top_k, len(fused_items)))],
        start=1,
    ):
        partner_id = str(partner_summary.get("partner_id") or partner_summary.get("name") or "")
        channels: List[str] = []
        if partner_id in sparse_rank_lookup:
            channels.append("sparse")
        if partner_id in dense_rank_lookup:
            channels.append("dense")

        enriched_partner = dict(partner_summary)
        enriched_partner["shortlist_score"] = round(fused_score, 6)
        enriched_partner["shortlist_rank_position"] = shortlist_position
        enriched_partner["shortlist_sparse_rank"] = sparse_rank_lookup.get(partner_id)
        enriched_partner["shortlist_dense_rank"] = dense_rank_lookup.get(partner_id)
        enriched_partner["shortlist_sparse_score"] = sparse_payload.get("score", 0)
        enriched_partner["shortlist_dense_score"] = round(dense_score, 6)
        enriched_partner["shortlist_retrieval_channels"] = channels
        enriched_partner["shortlist_overlap_features"] = list(sparse_payload.get("evidence") or [])
        enriched_partner["shortlist_shared_taxonomy_tags"] = dict(sparse_payload.get("shared_taxonomy_tags") or {})
        enriched_partner["shortlist_taxonomy_penalty"] = sparse_payload.get("taxonomy_penalty", 0)
        enriched_partner["shortlist_prompt_payload"] = _candidate_prompt_payload(partner_summary)
        shortlisted.append(enriched_partner)
    return shortlisted


def _canonicalize_partner_selection(
    partner_match_output: PartnerMatchOnlyOutput,
    partner_candidates: List[Dict[str, Any]],
) -> PartnerMatchOnlyOutput:
    id_lookup = {
        str(candidate.get("partner_id") or "").strip().lower(): candidate
        for candidate in partner_candidates
        if str(candidate.get("partner_id") or "").strip()
    }
    name_lookup = {
        str(candidate.get("name") or "").strip().lower(): candidate
        for candidate in partner_candidates
        if str(candidate.get("name") or "").strip()
    }
    alias_lookup = {}
    for candidate in partner_candidates:
        for alias in candidate.get("partner_aliases") or []:
            alias_lookup[str(alias).strip().lower()] = candidate

    matched_partner_id = _clean_text(partner_match_output.matched_partner_id).lower()
    matched_partner_name = _clean_text(partner_match_output.matched_partner_name)
    normalized_name = matched_partner_name.lower()
    candidate = None
    if matched_partner_id:
        candidate = id_lookup.get(matched_partner_id)
    if candidate is None and normalized_name:
        candidate = name_lookup.get(normalized_name) or alias_lookup.get(normalized_name)
    if candidate is None and normalized_name:
        normalized_raw = _normalize_match_text(normalized_name)
        for item in partner_candidates:
            if _normalize_match_text(item.get("name")) == normalized_raw:
                candidate = item
                break
    if candidate:
        partner_match_output.matched_partner_id = candidate.get("partner_id")
        partner_match_output.matched_partner_name = candidate.get("name")

    runner_up_id = _clean_text(partner_match_output.runner_up_partner_id).lower()
    runner_up_name = _clean_text(partner_match_output.runner_up_partner_name)
    runner_up_candidate = None
    if runner_up_id:
        runner_up_candidate = id_lookup.get(runner_up_id)
    if runner_up_candidate is None and runner_up_name:
        runner_up_candidate = name_lookup.get(runner_up_name.lower()) or alias_lookup.get(runner_up_name.lower())
    if runner_up_candidate:
        partner_match_output.runner_up_partner_id = runner_up_candidate.get("partner_id")
        partner_match_output.runner_up_partner_name = runner_up_candidate.get("name")
    return partner_match_output


def _salvage_partial_partner_match(raw_text: str) -> Optional[PartnerMatchOnlyOutput]:
    if not raw_text:
        return None
    match_score_match = re.search(r'"match_score"\s*:\s*"([^"]+)"', raw_text)
    partner_name_match = re.search(r'"matched_partner_name"\s*:\s*"([^"]+)"', raw_text)
    if not match_score_match and not partner_name_match:
        return None
    confidence_match = re.search(r'"match_confidence"\s*:\s*"([^"]+)"', raw_text)
    overlap_match = re.search(r'"overlap_type"\s*:\s*"([^"]+)"', raw_text)
    partner_id_match = re.search(r'"matched_partner_id"\s*:\s*"([^"]+)"', raw_text)
    return PartnerMatchOnlyOutput(
        match_score=match_score_match.group(1).strip() if match_score_match else None,
        match_confidence=confidence_match.group(1).strip() if confidence_match else None,
        overlap_type=overlap_match.group(1).strip() if overlap_match else None,
        matched_partner_id=partner_id_match.group(1).strip() if partner_id_match else None,
        matched_partner_name=partner_name_match.group(1).strip() if partner_name_match else None,
        match_rationale_features=[],
        target_evidence=[],
        partner_evidence=[],
    )


def _partner_lookup_texts(matched_partner_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not matched_partner_data:
        return {"target_audience": "", "services_products": "", "usp": "", "industry": "", "business_model": ""}
    match_profile = matched_partner_data.get("match_profile") if isinstance(matched_partner_data.get("match_profile"), dict) else {}
    return {
        "target_audience": _clean_text(match_profile.get("target_audience") or matched_partner_data.get("target_audience")),
        "services_products": _clean_text(match_profile.get("services_products") or matched_partner_data.get("services_products")),
        "usp": _clean_text(match_profile.get("usp") or matched_partner_data.get("usp")),
        "industry": _clean_text(match_profile.get("industry") or matched_partner_data.get("industry")),
        "business_model": _clean_text(match_profile.get("business_model") or matched_partner_data.get("business_model")),
    }


def _target_lookup_texts(target_attributes: Optional[DetailedCompanyAttributes]) -> Dict[str, str]:
    if not target_attributes:
        return {"audience": "", "use_case": "", "industry": "", "business_model": ""}
    audience = "; ".join(_coerce_text_list(getattr(target_attributes, "customer_target_segments", None)))
    use_case = "; ".join(
        _coerce_text_list(getattr(target_attributes, "products_services_offered", None))
        + _coerce_text_list(getattr(target_attributes, "usp_key_selling_points", None))
    )
    return {
        "audience": _clean_text(audience),
        "use_case": _clean_text(use_case),
        "industry": _clean_text(getattr(target_attributes, "industry", None)),
        "business_model": _clean_text(getattr(target_attributes, "business_model", None)),
    }


def _evidence_supported(evidence: str, haystacks: List[str]) -> bool:
    normalized_evidence = _normalize_match_text(evidence)
    if not normalized_evidence:
        return False
    evidence_tokens = set(_tokenize_match_text(evidence))
    if not evidence_tokens:
        return False
    for haystack in haystacks:
        normalized_haystack = _normalize_match_text(haystack)
        if not normalized_haystack:
            continue
        if normalized_evidence in normalized_haystack:
            return True
        haystack_tokens = set(_tokenize_match_text(haystack))
        if evidence_tokens and len(evidence_tokens.intersection(haystack_tokens)) >= min(2, len(evidence_tokens)):
            return True
    return False


def assess_partner_match(
    partner_match_output: Optional[PartnerMatchOnlyOutput],
    matched_partner_data: Optional[Dict[str, Any]],
    target_attributes: Optional[DetailedCompanyAttributes] = None,
) -> Dict[str, Any]:
    decision = {
        "decision": "no_match",
        "reason": "missing_partner_match_output",
        "match_confidence": "no_match",
        "overlap_type": None,
        "target_evidence": [],
        "partner_evidence": [],
        "runner_up_partner_id": None,
        "runner_up_partner_name": None,
    }
    if not partner_match_output:
        return decision

    decision["runner_up_partner_id"] = partner_match_output.runner_up_partner_id
    decision["runner_up_partner_name"] = partner_match_output.runner_up_partner_name

    matched_name = _clean_text(partner_match_output.matched_partner_name)
    if not matched_name:
        decision["reason"] = "missing_partner_name"
        return decision

    confidence = _normalize_match_confidence(
        partner_match_output.match_confidence,
        partner_match_output.match_score,
    ) or "weak"
    overlap_type = _normalize_overlap_type(partner_match_output.overlap_type)
    rationale_features = _normalize_rationale_features(partner_match_output.match_rationale_features)
    if matched_name.lower() == "no suitable match found" or confidence == "no_match":
        decision["reason"] = "explicit_no_match"
        return decision
    if not matched_partner_data:
        decision["reason"] = "matched_partner_data_missing"
        return decision

    target_lookup = _target_lookup_texts(target_attributes)
    partner_lookup = _partner_lookup_texts(matched_partner_data)
    target_evidence = [
        item for item in _normalize_evidence_list(partner_match_output.target_evidence)
        if _evidence_supported(item, list(target_lookup.values()))
    ]
    partner_evidence = [
        item for item in _normalize_evidence_list(partner_match_output.partner_evidence)
        if _evidence_supported(item, list(partner_lookup.values()))
    ]

    if not target_evidence and rationale_features:
        target_evidence = [item for item in rationale_features if _evidence_supported(item, list(target_lookup.values()))][:2]
    if not partner_evidence and rationale_features:
        partner_evidence = [item for item in rationale_features if _evidence_supported(item, list(partner_lookup.values()))][:2]

    shortlist_score = 0.0
    try:
        shortlist_score = float(matched_partner_data.get("shortlist_score") or 0.0)
    except Exception:
        shortlist_score = 0.0
    shortlist_features = [
        _clean_text(feature)
        for feature in (matched_partner_data.get("shortlist_overlap_features") or [])
        if _clean_text(feature)
    ]
    specific_count = sum(1 for feature in rationale_features if _is_specific_rationale_feature(feature))
    generic_count = sum(1 for feature in rationale_features if _is_generic_rationale_feature(feature))

    decision.update({
        "match_confidence": confidence,
        "overlap_type": overlap_type,
        "target_evidence": target_evidence,
        "partner_evidence": partner_evidence,
    })

    if confidence == "strong":
        if target_evidence and partner_evidence:
            decision["decision"] = "strong_match"
            decision["reason"] = "accepted_strong_validated_evidence"
            return decision
        if shortlist_score >= 0.03 and shortlist_features and specific_count >= 1:
            decision["decision"] = "weak_match"
            decision["match_confidence"] = "weak"
            decision["reason"] = "downgraded_strong_missing_direct_evidence"
            return decision
        decision["match_confidence"] = "no_match"
        decision["reason"] = "rejected_strong_missing_evidence"
        return decision

    if confidence == "weak":
        if target_evidence or partner_evidence or shortlist_features:
            decision["decision"] = "weak_match"
            decision["reason"] = "accepted_weak_match"
            return decision
        decision["match_confidence"] = "no_match"
        decision["reason"] = "rejected_weak_missing_evidence"
        return decision

    if generic_count >= 2 and specific_count == 0:
        decision["match_confidence"] = "no_match"
        decision["reason"] = "generic_overlap_only"
        return decision
    return decision


def should_accept_partner_match(
    partner_match_output: Optional[PartnerMatchOnlyOutput],
    matched_partner_data: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    decision = assess_partner_match(partner_match_output, matched_partner_data, target_attributes=None)
    if decision["decision"] == "strong_match":
        return True, decision["reason"]

    # Backward-compatible fallback for legacy unit tests and call sites that do
    # not provide target attributes. The runtime pipeline now uses
    # `assess_partner_match()` instead.
    if not partner_match_output:
        return False, "missing_partner_match_output"

    matched_name = (partner_match_output.matched_partner_name or "").strip()
    if not matched_name:
        return False, "missing_partner_name"
    if matched_name.lower() == "no suitable match found":
        return False, "explicit_no_match"
    if not matched_partner_data:
        return False, "matched_partner_data_missing"

    score = _normalize_match_score(partner_match_output.match_score)
    rationale_features = _normalize_rationale_features(partner_match_output.match_rationale_features)
    if score == "low":
        return False, "low_match_score"
    if not rationale_features:
        return False, "missing_match_rationale"

    specific_count = sum(1 for feature in rationale_features if _is_specific_rationale_feature(feature))
    generic_count = sum(1 for feature in rationale_features if _is_generic_rationale_feature(feature))
    if score == "high":
        if specific_count == 0 and generic_count >= 2:
            return False, "high_match_but_rationale_too_generic"
        return True, "accepted_high_match"
    if score == "medium":
        if specific_count >= 1 and not (specific_count == 0 and generic_count >= 2):
            return True, "accepted_medium_match_with_specific_rationale"
        return False, "medium_match_too_generic"
    return False, decision["reason"]


def match_partner(
    gemini_client: Any,
    config: AppConfig,
    target_attributes: DetailedCompanyAttributes,
    golden_partner_summaries: List[Dict[str, Any]],
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    triggering_input_row_id: Any,
    triggering_company_name: str
) -> Tuple[Optional[PartnerMatchOnlyOutput], Optional[str], Optional[Dict[str, int]]]:
    """
    Identifies the best golden partner match for a target company using an LLM.
    """
    log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: PartnerMatch]"
    logger.info(f"{log_prefix} Starting partner matching.")

    prompt_template_path: str = "Path not initialized"
    shortlisted_partners: List[Dict[str, Any]] = []

    try:
        prompt_template_path = config.PROMPT_PATH_GERMAN_PARTNER_MATCHING
        prompt_template = load_prompt_template(prompt_template_path)
        target_attributes_json = target_attributes.model_dump_json(indent=2)
        sparse_top_k = int(getattr(config, "partner_match_sparse_top_k", 15) or 15)
        dense_top_k = int(getattr(config, "partner_match_dense_top_k", 15) or 15)
        fused_top_k = int(getattr(config, "partner_match_fused_top_k", getattr(config, "MAX_GOLDEN_PARTNERS_IN_PROMPT", 10)) or 10)
        rrf_k = int(getattr(config, "partner_match_rrf_k", 60) or 60)
        shortlisted_partners = build_partner_shortlist(
            target_attributes=target_attributes,
            golden_partner_summaries=golden_partner_summaries,
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
            fused_top_k=min(fused_top_k, int(getattr(config, "MAX_GOLDEN_PARTNERS_IN_PROMPT", fused_top_k) or fused_top_k)),
            rrf_k=rrf_k,
        )
        partner_summaries_str = "\n".join(
            [f"{i+1}. {json.dumps(item.get('shortlist_prompt_payload') or _candidate_prompt_payload(item))}" for i, item in enumerate(shortlisted_partners)]
        )
        formatted_prompt = prompt_template.replace("{{TARGET_COMPANY_ATTRIBUTES_JSON_PLACEHOLDER}}", target_attributes_json)
        formatted_prompt = formatted_prompt.replace("{{GOLDEN_PARTNER_SUMMARIES_PLACEHOLDER}}", partner_summaries_str)
        formatted_prompt = formatted_prompt.replace("{{SHORTLIST_DEBUG_PLACEHOLDER}}", "")
        logger.info(
            "%s Shortlisted %s/%s partners for rerank: %s",
            log_prefix,
            len(shortlisted_partners),
            len(golden_partner_summaries),
            ", ".join(str(item.get("name")) for item in shortlisted_partners[:5]),
        )
    except Exception as e:
        logger.error(f"{log_prefix} Failed to load/format partner matching prompt: {e}", exc_info=True)
        return None, f"Error: Failed to load/format prompt - {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    s_file_id_prefix = sanitize_filename_component(file_identifier_prefix, max_len=15)
    s_row_id = sanitize_filename_component(str(triggering_input_row_id), max_len=8)
    s_comp_name = sanitize_filename_component(
        triggering_company_name,
        max_len=config.filename_company_name_max_len if hasattr(config, 'filename_company_name_max_len') and config.filename_company_name_max_len is not None and config.filename_company_name_max_len <= 20 else 20,
    )

    prompt_filename_base = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}"
    prompt_filename_with_suffix = f"{prompt_filename_base}_partner_match_prompt.txt"
    try:
        save_llm_artifact(
            content=formatted_prompt,
            directory=llm_requests_dir,
            filename=prompt_filename_with_suffix,
            log_prefix=log_prefix
        )
    except Exception as e_save_prompt:
         logger.error(f"{log_prefix} Failed to save formatted prompt artifact '{prompt_filename_with_suffix}': {e_save_prompt}", exc_info=True)

    system_instruction_text = (
        "You match the target company to the best golden partner from the provided shortlist. "
        "Return only valid JSON matching the requested schema."
    )
    request_payload_to_log = {
        "model_name": getattr(config, "llm_model_name", ""),
        "system_instruction": system_instruction_text,
        "schema_name": "partner_match_only",
        "response_model": "PartnerMatchOnlyOutput",
        "max_output_tokens": config.llm_max_tokens,
        "temperature": config.llm_temperature_extraction,
        "shortlisted_partner_names": [item.get("name") for item in shortlisted_partners],
        "shortlist_debug": [
            {
                "partner_id": item.get("partner_id"),
                "name": item.get("name"),
                "shortlist_score": item.get("shortlist_score"),
                "shortlist_sparse_rank": item.get("shortlist_sparse_rank"),
                "shortlist_dense_rank": item.get("shortlist_dense_rank"),
                "shortlist_overlap_features": item.get("shortlist_overlap_features"),
            }
            for item in shortlisted_partners
        ],
        "user_prompt": formatted_prompt,
    }
    request_payload_filename = f"{prompt_filename_base}_partner_match_request_payload.json"
    try:
        save_llm_artifact(
            content=json.dumps(request_payload_to_log, indent=2),
            directory=llm_requests_dir,
            filename=request_payload_filename,
            log_prefix=log_prefix
        )
    except Exception as e_save_payload:
        logger.error(f"{log_prefix} Failed to save request payload artifact: {e_save_payload}", exc_info=True)

    try:
        llm_result = gemini_client.generate_structured_output_with_retry(
            user_prompt=formatted_prompt,
            response_model=PartnerMatchOnlyOutput,
            schema_name="partner_match_only",
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            system_prompt=system_instruction_text,
            temperature=config.llm_temperature_extraction,
            max_output_tokens=config.llm_max_tokens,
        )
        token_stats = llm_result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if llm_result.raw_text:
            response_filename = f"{prompt_filename_base}_partner_match_response.txt"
            try:
                save_llm_artifact(
                    content=llm_result.raw_text,
                    directory=llm_context_dir,
                    filename=response_filename,
                    log_prefix=log_prefix,
                )
            except Exception as e_save_resp:
                logger.error(f"{log_prefix} Failed to save raw LLM response artifact: {e_save_resp}", exc_info=True)
        if llm_result.parsed_output:
            parsed_output = _canonicalize_partner_selection(llm_result.parsed_output, shortlisted_partners)
            parsed_output.shortlisted_partner_ids = [
                str(item.get("partner_id") or "").strip()
                for item in shortlisted_partners
                if str(item.get("partner_id") or "").strip()
            ]
            parsed_output.shortlisted_partner_names = [
                str(item.get("name") or "").strip()
                for item in shortlisted_partners
                if str(item.get("name") or "").strip()
            ]
            logger.info(f"{log_prefix} Successfully parsed PartnerMatchOnlyOutput.")
            return parsed_output, llm_result.raw_text, token_stats
        salvaged_output = _salvage_partial_partner_match(llm_result.raw_text or "")
        if salvaged_output is not None:
            salvaged_output = _canonicalize_partner_selection(salvaged_output, shortlisted_partners)
            salvaged_output.shortlisted_partner_ids = [
                str(item.get("partner_id") or "").strip()
                for item in shortlisted_partners
                if str(item.get("partner_id") or "").strip()
            ]
            salvaged_output.shortlisted_partner_names = [
                str(item.get("name") or "").strip()
                for item in shortlisted_partners
                if str(item.get("name") or "").strip()
            ]
            logger.warning(
                f"{log_prefix} Salvaged partial PartnerMatchOnlyOutput from malformed structured response."
            )
            return salvaged_output, llm_result.raw_text, token_stats
        error_message = llm_result.provider_error or (
            f"Model refusal: {llm_result.refusal}" if llm_result.refusal else "Failed to parse partner matching structured output."
        )
        logger.error(f"{log_prefix} {error_message}")
        return None, llm_result.raw_text or error_message, token_stats
    except Exception as e_gen:
        logger.error(f"{log_prefix} Unexpected error during partner matching: {e_gen}", exc_info=True)
        raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
        return None, raw_llm_response_str, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}