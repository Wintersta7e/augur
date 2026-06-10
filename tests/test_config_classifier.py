"""Config defaults + env coercion for the Ollama classifier fields."""

from tabula.config import AugurConfig


def test_classifier_defaults():
    c = AugurConfig()
    assert c.ollama_classifier_model == "qwen2.5:1.5b"
    assert c.ollama_classifier_enabled is True


def test_classifier_model_env_override(monkeypatch):
    monkeypatch.setenv("AUGUR_OLLAMA_CLASSIFIER_MODEL", "qwen2.5:3b")
    c = AugurConfig.from_env()
    assert c.ollama_classifier_model == "qwen2.5:3b"


def test_classifier_enabled_bool_coercion(monkeypatch):
    monkeypatch.setenv("AUGUR_OLLAMA_CLASSIFIER_ENABLED", "false")
    c = AugurConfig.from_env()
    assert c.ollama_classifier_enabled is False
