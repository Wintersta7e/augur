"""Tests for _derive_domain — round-3 fix for all-correlated non-chess sessions."""


def test_derive_domain_falls_back_to_correlated_when_no_standalone():
    """Round-3 fix: all-correlated typing session should NOT fall back to chess."""
    feedback = {
        "advice_events": [
            {"domain": "typing", "correlation_found": True},
            {"domain": "typing", "correlation_found": True},
            {"domain": "chess", "correlation_found": True},
        ]
    }
    from disciplina.reflection_engine import _derive_domain

    assert _derive_domain(feedback) == "typing"


def test_derive_domain_prefers_standalone_when_present():
    """Standalone events take priority over correlated events for scope."""
    feedback = {
        "advice_events": [
            {"domain": "typing", "correlation_found": True},
            {"domain": "typing", "correlation_found": True},
            {"domain": "chess", "correlation_found": False},  # the only standalone
        ]
    }
    from disciplina.reflection_engine import _derive_domain

    assert _derive_domain(feedback) == "chess"
