"""The enrichment helper resolves + sets ctx['app_descriptor'] for activity events."""

from unittest.mock import MagicMock

from reasoning.augur_advisor import (
    enrich_activity_descriptor,
    enrich_payload_descriptors,
)


def test_enrich_sets_descriptor_from_os_identity():
    pm = MagicMock()
    lane = MagicMock()
    anomaly = {
        "domain": "activity_intensity",
        "entity": "alpha_app",
        "context": {"app_identity": "Alpha Browser"},
    }
    enrich_activity_descriptor(pm, lane, anomaly)
    assert anomaly["context"]["app_descriptor"] == "Alpha Browser"
    lane.enqueue.assert_not_called()


def test_enrich_enqueues_on_miss():
    pm = MagicMock()
    pm.load_app_descriptor.return_value = None
    lane = MagicMock()
    anomaly = {"domain": "activity_focus", "entity": "gamma_app", "context": {}}
    enrich_activity_descriptor(pm, lane, anomaly)
    assert anomaly["context"].get("app_descriptor") is None
    lane.enqueue.assert_called_once_with("gamma_app")


def test_enrich_skips_non_activity_domain():
    pm = MagicMock()
    lane = MagicMock()
    anomaly = {"domain": "typing", "entity": "user", "context": {}}
    enrich_activity_descriptor(pm, lane, anomaly)
    assert "app_descriptor" not in anomaly["context"]
    pm.load_app_descriptor.assert_not_called()
    lane.enqueue.assert_not_called()


def test_enrich_tolerates_missing_context():
    pm = MagicMock()
    pm.load_app_descriptor.return_value = "cached editor"
    lane = MagicMock()
    anomaly = {"domain": "activity_intensity", "entity": "beta_app"}
    enrich_activity_descriptor(pm, lane, anomaly)
    assert anomaly["context"]["app_descriptor"] == "cached editor"


def test_enrich_payload_correlation_enriches_primary_and_correlated():
    pm = MagicMock()
    pm.load_app_descriptor.side_effect = lambda e: {
        "alpha_app": "Alpha Browser",
        "beta_app": "Beta Editor",
    }.get(e)
    lane = MagicMock()
    payload = {
        "primary_anomaly": {
            "domain": "activity_intensity",
            "entity": "alpha_app",
            "context": {},
        },
        "correlated_events": [
            {"domain": "activity_focus", "entity": "beta_app", "context": {}},
            {"domain": "typing", "entity": "user", "context": {}},
        ],
    }
    enrich_payload_descriptors(pm, lane, "correlation", payload)
    assert payload["primary_anomaly"]["context"]["app_descriptor"] == "Alpha Browser"
    assert payload["correlated_events"][0]["context"]["app_descriptor"] == "Beta Editor"
    assert (
        "app_descriptor" not in payload["correlated_events"][1]["context"]
    )  # typing skipped


def test_enrich_payload_non_correlation_enriches_primary_only():
    pm = MagicMock()
    pm.load_app_descriptor.return_value = "Alpha Browser"
    lane = MagicMock()
    payload = {
        "primary_anomaly": {
            "domain": "activity_intensity",
            "entity": "alpha_app",
            "context": {},
        },
        "correlated_events": [
            {"domain": "activity_focus", "entity": "beta_app", "context": {}}
        ],
    }
    enrich_payload_descriptors(pm, lane, "primary", payload)
    assert payload["primary_anomaly"]["context"]["app_descriptor"] == "Alpha Browser"
    # correlated NOT enriched on the non-correlation path
    assert "app_descriptor" not in payload["correlated_events"][0]["context"]
