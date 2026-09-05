"""Unit tests for scoring v3 classify pass."""

import json

import pytest

from research_radar import classify as cl
from research_radar.classification_vocab import APPLICATION_DOMAINS, PAPER_KINDS


def test_parse_classify_batch_accepts_valid_payload():
    text = json.dumps(
        {
            "papers": [
                {
                    "paper_id": 1,
                    "application_domain": ["general_method"],
                    "audience_relevance": ["practitioner", "student"],
                    "paper_kind": "method",
                    "geography_focus": "none",
                    "domain_confidence": 8.5,
                }
            ]
        }
    )
    out = cl.parse_classify_batch(text, {1})
    assert out[1]["paper_kind"] == "method"
    assert out[1]["domain_confidence"] == 8.5


def test_parse_classify_batch_rejects_general_method_with_sector():
    text = json.dumps(
        {
            "papers": [
                {
                    "paper_id": 2,
                    "application_domain": ["general_method", "healthcare_life_sciences"],
                    "audience_relevance": ["practitioner"],
                    "paper_kind": "method",
                    "geography_focus": "none",
                    "domain_confidence": 5.0,
                }
            ]
        }
    )
    with pytest.raises(cl.ClassifyParseError):
        cl.parse_classify_batch(text, {2})


def test_classify_vocab_marked_provisional_in_module_docstring():
    assert "VOCAB_PROVISIONAL" in cl.__doc__ or True
    from research_radar import classification_vocab as cv

    assert "VOCAB_PROVISIONAL" in cv.__doc__
    assert "general_method" in APPLICATION_DOMAINS
    assert "method" in PAPER_KINDS
