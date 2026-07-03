from tabula.config import AugurConfig


def test_dialogue_defaults():
    c = AugurConfig()
    assert c.dialogue_enabled is True
    assert c.dialogue_model == "qwen2.5:32b"
    assert c.dialogue_confirmed_apply_enabled is True
    assert c.imperator_ii_apply_enabled is False  # watch-first untouched
    assert 0.0 <= c.dialogue_temperature <= 2.0


def test_dialogue_env_override(monkeypatch):
    monkeypatch.setenv("AUGUR_DIALOGUE_ENABLED", "false")
    monkeypatch.setenv("AUGUR_DIALOGUE_NUM_PREDICT", "256")
    c = AugurConfig.from_env()
    assert c.dialogue_enabled is False
    assert c.dialogue_num_predict == 256
