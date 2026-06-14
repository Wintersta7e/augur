import fakeredis
from tabula.persistence import PersistenceManager


def _pm() -> PersistenceManager:
    return PersistenceManager(fakeredis.FakeStrictRedis(decode_responses=False))


def test_auspices_round_trip():
    pm = _pm()
    assert pm.load_auspices() is None
    snap = {"schema_version": 1, "generated_at": 1.0, "salience": {"value": 0.5}}
    pm.save_auspices(snap)
    assert pm.load_auspices() == snap


def test_self_model_round_trip():
    pm = _pm()
    assert pm.load_self_model() is None
    snap = {"schema_version": 1, "competence": {"value": 0.7}}
    pm.save_self_model(snap)
    assert pm.load_self_model() == snap
