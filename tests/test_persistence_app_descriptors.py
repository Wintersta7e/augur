"""Unit tests for the app-descriptor hash methods on PersistenceManager."""

from unittest.mock import MagicMock

from tabula.persistence import PersistenceManager

KEY = "augur:config:app_descriptors"


def test_save_app_descriptor_overwrite_uses_hset():
    r = MagicMock()
    pm = PersistenceManager(r)
    pm.save_app_descriptor("alpha_app", "Alpha Browser", overwrite=True)
    r.hset.assert_called_once_with(KEY, "alpha_app", "Alpha Browser")
    r.hsetnx.assert_not_called()


def test_save_app_descriptor_no_overwrite_uses_hsetnx():
    r = MagicMock()
    pm = PersistenceManager(r)
    pm.save_app_descriptor("beta_app", "beta editor", overwrite=False)
    r.hsetnx.assert_called_once_with(KEY, "beta_app", "beta editor")
    r.hset.assert_not_called()


def test_load_app_descriptor_decodes_bytes():
    r = MagicMock()
    r.hget.return_value = b"Alpha Browser"
    pm = PersistenceManager(r)
    assert pm.load_app_descriptor("alpha_app") == "Alpha Browser"
    r.hget.assert_called_once_with(KEY, "alpha_app")


def test_load_app_descriptor_missing_returns_none():
    r = MagicMock()
    r.hget.return_value = None
    pm = PersistenceManager(r)
    assert pm.load_app_descriptor("ghost_app") is None


def test_load_app_descriptors_decodes_keys_and_values():
    r = MagicMock()
    r.hgetall.return_value = {
        b"alpha_app": b"Alpha Browser",
        b"beta_app": b"beta editor",
    }
    pm = PersistenceManager(r)
    assert pm.load_app_descriptors() == {
        "alpha_app": "Alpha Browser",
        "beta_app": "beta editor",
    }


def test_load_app_descriptors_empty():
    r = MagicMock()
    r.hgetall.return_value = {}
    pm = PersistenceManager(r)
    assert pm.load_app_descriptors() == {}


def test_save_app_descriptor_drops_new_entity_when_full():
    r = MagicMock()
    r.hexists.return_value = False
    r.hlen.return_value = 2000  # MAX_APP_DESCRIPTORS
    pm = PersistenceManager(r)
    pm.save_app_descriptor("new_app", "X", overwrite=False)
    r.hsetnx.assert_not_called()
    r.hset.assert_not_called()


def test_save_app_descriptor_allows_update_when_full():
    r = MagicMock()
    r.hexists.return_value = True
    r.hlen.return_value = 2000
    pm = PersistenceManager(r)
    pm.save_app_descriptor("existing_app", "Y", overwrite=True)
    r.hset.assert_called_once_with("augur:config:app_descriptors", "existing_app", "Y")


def test_save_app_descriptor_drops_new_entity_when_full_overwrite_true():
    r = MagicMock()
    r.hexists.return_value = False
    r.hlen.return_value = 2000  # == MAX_APP_DESCRIPTORS
    pm = PersistenceManager(r)
    pm.save_app_descriptor("brand_new_app", "OS Name", overwrite=True)
    r.hset.assert_not_called()
    r.hsetnx.assert_not_called()
