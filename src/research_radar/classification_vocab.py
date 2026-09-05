"""Closed vocabulary for scoring v3 classify + independence paper_kind.

VOCAB_PROVISIONAL: replace with the vocabulary derived for the main pipeline
before treating these enums as stable cross-repo keys. Two independently derived
enums produce silent join failures (e.g. healthcare vs life_sciences).
Backlog: align with main pipeline vocabulary derivation.
"""

from __future__ import annotations

# VOCAB_PROVISIONAL — see module docstring.
APPLICATION_DOMAINS: tuple[str, ...] = (
    "general_method",
    "healthcare_life_sciences",
    "financial_services",
    "manufacturing_industrial",
    "energy_utilities",
    "mining_resources",
    "retail_ecommerce",
    "transport_logistics",
    "agriculture",
    "telecom_networks",
    "public_sector_govtech",
    "legal_compliance",
    "education",
    "media_creative",
    "scientific_research",
    "security_defence",
    "other",
)

AUDIENCE_RELEVANCE: tuple[str, ...] = (
    "practitioner",
    "technical_leadership",
    "enterprise_adoption",
    "student",
)

PAPER_KINDS: tuple[str, ...] = (
    "method",
    "empirical_study",
    "benchmark_dataset",
    "survey_review",
    "theory",
    "position",
    "negative_result",
    "system_infrastructure",
)

GEOGRAPHY_FOCUS: tuple[str, ...] = (
    "none",
    "us",
    "china",
    "eu",
    "india",
    "other",
)
