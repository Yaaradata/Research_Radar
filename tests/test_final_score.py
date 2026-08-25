from research_radar.final_score import (
    ORG_BOOST_MAX,
    WEIGHT_PROFILES,
    compute_final,
    compute_semantic_core,
    org_boost_for,
    resolve_profile,
)


def _dims(v=7.0, **over):
    d = {k: v for k in WEIGHT_PROFILES["radar-v1"]}
    d.update(over)
    return d


def test_every_profile_sums_to_one():
    for name, w in WEIGHT_PROFILES.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, name


def test_ai_relevance_never_in_any_profile():
    """It is a gate, not a ranking dimension — it compresses the scale."""
    for w in WEIGHT_PROFILES.values():
        assert "ai_relevance" not in w


def test_missing_dimensions_renormalise():
    _, w = resolve_profile("radar-v1")
    d = _dims(8.0)
    d["industry_relevance"] = None
    assert abs(compute_semantic_core(d, w) - 8.0) < 0.01


def test_org_boost_requires_high_confidence_evidence():
    weak = [{"priority": 10, "confidence": 0.70}]
    assert org_boost_for(weak) == (0.0, [])
    strong = [{"priority": 10, "confidence": 0.98}]
    boost, detail = org_boost_for(strong)
    assert boost == ORG_BOOST_MAX and len(detail) == 1


def test_org_boost_is_capped():
    boost, _ = org_boost_for([{"priority": 10, "confidence": 1.0}])
    assert boost <= ORG_BOOST_MAX


def test_strong_unknown_org_paper_beats_weak_watchlist_paper():
    """Golden test #2: the watchlist is a boost, never a filter."""
    _, w = resolve_profile("radar-v1")
    unknown = compute_final(_dims(9.0), [], [], w)
    watchlist = compute_final(_dims(7.0), [{"priority": 10, "confidence": 1.0}], [], w)
    assert unknown["final_score"] > watchlist["final_score"]
    assert unknown["org_boost"] == 0.0


def test_current_employer_evidence_is_excluded_by_query():
    """Brief §9 — enforced in load_org_rows via current_affiliation = FALSE."""
    import inspect

    from research_radar import final_score

    src = inspect.getsource(final_score.load_org_rows)
    assert "current_affiliation = FALSE" in src


def test_final_score_never_exceeds_ten():
    _, w = resolve_profile("radar-v1")
    r = compute_final(_dims(10.0), [{"priority": 10, "confidence": 1.0}], [], w)
    assert r["final_score"] == 10.0
