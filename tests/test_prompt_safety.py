from tabula.config import AugurConfig
from consilium.prompt_safety import violates_forbidden_patterns, is_prompt_acceptable

CFG = AugurConfig.from_env()


def test_violates():
    assert violates_forbidden_patterns("As an AI, take a break", CFG) is True
    assert violates_forbidden_patterns("Consider castling to safety.", CFG) is False


def test_acceptable():
    assert (
        is_prompt_acceptable("Consider developing your pieces toward the center.", CFG)
        is True
    )
    assert is_prompt_acceptable("short", CFG) is False
    assert is_prompt_acceptable("", CFG) is False
    assert is_prompt_acceptable("As an AI " + "x" * 50, CFG) is False


def test_non_string_is_unacceptable():
    # A non-string (e.g. a malformed LLM action.text) must be rejected, not crash.
    assert is_prompt_acceptable(None, CFG) is False
    assert is_prompt_acceptable({"oops": 1}, CFG) is False
    assert is_prompt_acceptable(123, CFG) is False
