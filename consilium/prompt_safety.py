"""Shared prompt-safety checks (extracted from Disciplina's mutate_prompt guard).

Used by Disciplina (prompt mutation) and Imperator II (proposal apply).
"""

from __future__ import annotations

from tabula.config import AugurConfig


def violates_forbidden_patterns(prompt_text: str, config: AugurConfig) -> bool:
    """True if the text reintroduces a forbidden valence/meta pattern."""
    low = prompt_text.lower()
    return any(pat.lower() in low for pat in config.prompt_forbidden_patterns)


def is_prompt_acceptable(prompt_text: str, config: AugurConfig) -> bool:
    """Non-empty, >= min_prompt_len, and free of forbidden patterns."""
    if not prompt_text or len(prompt_text.strip()) < config.min_prompt_len:
        return False
    return not violates_forbidden_patterns(prompt_text, config)
