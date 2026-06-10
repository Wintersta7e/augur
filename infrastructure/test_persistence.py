"""Exercise all PersistenceManager methods and confirm round-trip correctness."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis

from tabula.contracts import PerceptionEvent
from tabula.persistence import PersistenceManager

# Use a test-specific key prefix to avoid polluting real data.
# We clean up all keys at the end.
TEST_KEYS: list[str] = []


def track(key: str) -> str:
    TEST_KEYS.append(key)
    return key


def main() -> int:
    r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=5)
    r.ping()
    pm = PersistenceManager(r)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    print("=== Baseline persistence ===")
    track("augur:vigil:profile:test:white")
    track("augur:vigil:profile:test:black")

    state = {"ewma_mean": 3.5, "ewma_var": 0.8, "move_count": 12}
    pm.save_baseline("test", "white", state)
    loaded = pm.load_baseline("test", "white")
    check("save/load round-trip", loaded == state, f"got {loaded}")

    check("load missing returns None", pm.load_baseline("test", "black") is None)

    print("\n=== Event history ===")
    track("augur:vigil:history:test")

    events = []
    for i in range(5):
        e = PerceptionEvent(
            domain="test",
            stream_id="test_stream",
            entity="white",
            event_type="move",
            value=float(i),
            unit="seconds",
            context={"move_number": i + 1},
            timestamp=f"2026-03-07T00:00:0{i}+00:00",
            session_id="test-session",
        )
        events.append(e)
        pm.append_event(e)

    history = pm.get_history("test", limit=100)
    check("history length", len(history) == 5, f"got {len(history)}")
    # lpush means newest first
    check(
        "history order (newest first)",
        history[0]["value"] == 4.0,
        f"got {history[0]['value']}",
    )
    check(
        "history preserves fields",
        history[0]["domain"] == "test" and history[0]["session_id"] == "test-session",
    )

    limited = pm.get_history("test", limit=3)
    check("history limit works", len(limited) == 3, f"got {len(limited)}")

    print("\n=== Feedback storage ===")
    track("augur:feedback:sess-001")
    track("augur:feedback:sess-002")
    track("augur:feedback:_index")

    pm.save_feedback("sess-001", {"rating": 5, "comment": "great advice"})
    pm.save_feedback("sess-002", {"rating": 2, "comment": "not helpful"})

    fb1 = pm.get_feedback("sess-001")
    check("feedback round-trip", fb1 is not None and fb1["rating"] == 5)
    check("feedback has timestamp", "timestamp" in fb1)

    check("missing feedback returns None", pm.get_feedback("nonexistent") is None)

    all_fb = pm.get_all_feedback(limit=10)
    check("get_all_feedback returns records", len(all_fb) == 2, f"got {len(all_fb)}")
    check(
        "get_all_feedback includes session_id", all(("session_id" in f) for f in all_fb)
    )

    print("\n=== Prompt versioning ===")
    track("augur:prompts:test:current")
    track("augur:prompts:test:history")

    check("load missing prompt returns None", pm.load_prompt("test") is None)

    pm.save_prompt("test", "You are a chess advisor v1", score=0.6)
    check("save/load prompt v1", pm.load_prompt("test") == "You are a chess advisor v1")

    pm.save_prompt("test", "You are a chess advisor v2", score=0.8)
    check("save/load prompt v2", pm.load_prompt("test") == "You are a chess advisor v2")

    pm.save_prompt("test", "You are a chess advisor v3", score=0.3)
    check("save/load prompt v3", pm.load_prompt("test") == "You are a chess advisor v3")

    hist = pm.get_prompt_history("test", limit=10)
    check("prompt history has 2 entries", len(hist) == 2, f"got {len(hist)}")
    check(
        "prompt history newest first", hist[0]["prompt"] == "You are a chess advisor v2"
    )
    check(
        "prompt history has scores", hist[0]["score"] == 0.8 and hist[1]["score"] == 0.6
    )

    # Rollback: v3 (score 0.3) is current, v2 (score 0.8) should be restored
    ok = pm.rollback_prompt("test")
    check("rollback succeeds", ok is True)
    check(
        "rollback restores v2", pm.load_prompt("test") == "You are a chess advisor v2"
    )

    # v3 should now be at the end of history, v1 at position 0
    hist_after = pm.get_prompt_history("test", limit=10)
    check(
        "rollback archives bad version",
        any(h["prompt"] == "You are a chess advisor v3" for h in hist_after),
    )

    # Rollback again to v1
    pm.rollback_prompt("test")
    check(
        "second rollback restores v1",
        pm.load_prompt("test") == "You are a chess advisor v1",
    )

    print("\n=== Threshold config ===")
    track("augur:vigil:thresholds:test")

    thresholds = {"sigma": 2.0, "hst": 0.7, "min_moves": 3}
    pm.save_thresholds("test", thresholds)
    loaded_th = pm.load_thresholds("test")
    check("threshold round-trip", loaded_th == thresholds, f"got {loaded_th}")

    check("missing thresholds returns None", pm.load_thresholds("nonexistent") is None)

    # Cleanup test keys
    print("\n=== Cleanup ===")
    for key in TEST_KEYS:
        r.delete(key)
    print(f"  Cleaned up {len(TEST_KEYS)} test keys")

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 40}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
