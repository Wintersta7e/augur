"""End-to-end: an activity anomaly with OS identity is cached + injected.

Uses real Redis (fast integration tier). The fixture isolates the descriptor
hash key so it does not require a full FLUSHALL.
"""

import pytest

from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.persistence import PersistenceManager
from reasoning.app_descriptor import ClassifierLane
from reasoning.augur_advisor import enrich_activity_descriptor


@pytest.fixture
def pm():
    config = AugurConfig.from_env()
    client = connect_redis(config)
    client.delete("augur:config:app_descriptors")
    yield PersistenceManager(client)
    client.delete("augur:config:app_descriptors")


def test_os_identity_cached_and_injected(pm):
    cfg = AugurConfig()
    lane = ClassifierLane(pm, None, cfg)  # not exercised on the OS path
    anomaly = {
        "domain": "activity_intensity",
        "entity": "alpha_app",
        "context": {"app_identity": "Alpha Browser"},
    }
    enrich_activity_descriptor(pm, lane, anomaly)
    # injected into context for the prompt...
    assert anomaly["context"]["app_descriptor"] == "Alpha Browser"
    # ...and persisted (decoded) for future events.
    assert pm.load_app_descriptor("alpha_app") == "Alpha Browser"
    assert pm.load_app_descriptors() == {"alpha_app": "Alpha Browser"}


def test_llm_fallback_does_not_clobber_os_identity(pm):
    pm.save_app_descriptor("alpha_app", "Alpha Browser", overwrite=True)
    pm.save_app_descriptor("alpha_app", "some llm guess", overwrite=False)
    assert pm.load_app_descriptor("alpha_app") == "Alpha Browser"
