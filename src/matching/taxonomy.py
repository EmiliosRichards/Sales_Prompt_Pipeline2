from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List


_TAG_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "audience": {
        "construction_buyers": [
            "construction company", "construction companies", "civil engineering", "underground construction",
            "site manager", "bauleiter", "baustelle", "quarry", "mining", "rohstoff", "recyclingbetrieb",
            "sandwerk", "kies", "tiefbau",
        ],
        "industrial_ops_buyers": [
            "industrial companies", "manufacturing companies", "production", "factory", "technical plants",
            "asset management", "maintenance", "instandhaltung", "anlagen", "plant operators",
        ],
        "procurement_buyers": [
            "procurement", "purchasing", "sourcing", "supplier", "lieferanten", "beschaffung", "einkauf",
            "buying organizations",
        ],
        "retail_ecommerce_buyers": [
            "brand", "brands", "retail", "retailer", "retailers", "merchant", "merchants",
            "marketplace", "marketplaces", "e commerce", "ecommerce", "digital shelf", "shop",
            "online shop", "consumer goods",
        ],
        "law_tax_buyers": [
            "law firm", "law firms", "legal department", "legal departments", "tax consultant", "tax consultants",
            "auditors", "datev", "notary", "notaries", "kanzlei", "steuerberater",
        ],
        "healthcare_buyers": [
            "hospital", "hospitals", "clinic", "clinics", "care organization", "care organizations",
            "nursing", "medical professionals", "healthcare facilities",
        ],
        "hr_workforce_buyers": [
            "hr", "human resources", "people ops", "recruiting", "workforce", "operations managers",
            "hospital administration", "talent", "personnel",
        ],
        "mobility_buyers": [
            "mobility manager", "fleet manager", "fleet management", "carsharing", "bike sharing",
            "shared mobility", "vehicle booking", "fuhrpark", "poolfahrzeuge", "municipal utilities",
        ],
        "sme_buyers": [
            "small and medium", "smes", "smaller businesses", "mittelstand", "entrepreneurs",
        ],
        "defense_public_buyers": [
            "defense", "military", "maritime defense", "national security", "government", "public sector",
        ],
    },
    "workflow": {
        "construction_site_ops": [
            "construction plans", "surveying", "gnss", "georeferenced", "site documentation", "mass flow",
            "construction site", "baustell", "quarry", "mining", "civil engineering", "bautagesbericht",
        ],
        "document_workflow": [
            "document management", "documents", "sorting", "tagging", "archiv", "archive", "signature",
            "eidas", "forms", "contracts", "procedural documentation",
        ],
        "digital_marketing_web": [
            "web design", "website", "websites", "drupal", "seo", "sea", "online marketing",
            "performance marketing", "branding", "web development", "ux", "ui",
        ],
        "procurement_ops": [
            "procurement", "sourcing", "supplier integration", "catalog", "invoice automation",
            "e procurement", "purchase order", "supplier management",
        ],
        "logistics_ops": [
            "logistics", "transport", "warehouse", "shipping", "route optimization", "tracking",
            "supply chain", "freight",
        ],
        "maintenance_asset_ops": [
            "asset management", "maintenance", "predictive maintenance", "eam", "asset lifecycle",
            "service support", "dismantling", "maintenance 4 0",
        ],
        "mobility_sharing_ops": [
            "carsharing", "bike sharing", "shared mobility", "vehicle booking", "fleet management",
            "driver app", "telematics", "vehicle provisioning", "mobility",
        ],
        "commerce_analytics": [
            "digital shelf", "price monitoring", "competitor analysis", "marketplace rankings",
            "map rrp", "review monitoring", "e commerce analytics", "product content monitoring",
        ],
        "workflow_automation": [
            "workflow", "process optimization", "automation", "integrations", "compliance", "back office",
            "administrative effort", "media breaks",
        ],
        "defense_simulation_ops": [
            "simulation", "battlefield twin", "decision support", "engagement modeling", "hedging strategies",
            "autonomous sea mine", "training environment",
        ],
    },
    "domain": {
        "construction": [
            "construction", "civil engineering", "underground construction", "baustelle", "bau", "quarry",
            "mining", "rohstoff", "proptech",
        ],
        "industrial_maintenance": [
            "maintenance", "asset management", "eam", "technical plants", "predictive maintenance", "manufacturing",
        ],
        "procurement": [
            "procurement", "e procurement", "supplier", "purchasing", "sourcing", "catalog cloud",
        ],
        "ecommerce_retail": [
            "e commerce", "ecommerce", "retail", "marketplace", "digital shelf", "brand protection",
        ],
        "digital_agency": [
            "digital agency", "marketing agency", "seo", "sea", "web development", "drupal", "branding",
        ],
        "document_software": [
            "document management", "dms", "document organization", "e signature", "tagging", "archive",
        ],
        "logistics": [
            "logistics", "transport", "warehouse", "supply chain", "shipping",
        ],
        "mobility": [
            "mobility", "carsharing", "bike sharing", "fleet", "vehicle booking", "telematics",
        ],
        "healthcare": [
            "healthcare", "hospital", "clinic", "medical", "care facility", "nursing",
        ],
        "law_tax": [
            "law firm", "legal", "tax", "datev", "notary", "kanzlei", "steuerberater",
        ],
        "defense_simulation": [
            "defense", "military", "simulation", "battlefield", "maritime defense", "national security",
        ],
    },
}


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _iter_texts(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text:
            out.append(text)
    return out


def infer_match_taxonomy(*values: Any) -> Dict[str, List[str]]:
    texts = _iter_texts(values)
    normalized_blob = " ".join(_normalize_text(value) for value in texts if _normalize_text(value))
    result: Dict[str, List[str]] = {"audience": [], "workflow": [], "domain": []}
    if not normalized_blob:
        return result

    for category, tag_map in _TAG_KEYWORDS.items():
        matches: List[str] = []
        for tag, keywords in tag_map.items():
            if any(_normalize_text(keyword) in normalized_blob for keyword in keywords):
                matches.append(tag)
        result[category] = matches
    return result

