"""Tier classification + the deterministic Memoria sweep planner.

Pure: takes plain state dicts + observed patterns + config, returns a
SweepPlan that Disciplina applies atomically via PersistenceManager.
See docs/superpowers/specs/2026-06-10-memoria-memory-spine.md §4-§6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoria.fsrs import review, retrievability


def is_floor_protected(state: dict) -> bool:
    """HIGH memories are never pruned by R (mirror Limen HIGH-bypass): either
    the combined origin severity is HIGH, or an endpoint was HIGH (inferred
    from rule_key, e.g. "HIGH+LOW"). They may still be archived by cap
    enforcement (kept, recoverable) — but this implementation refuses new
    creates before evicting protected memories (see plan_sweep §3b)."""
    if state.get("origin_severity") == "HIGH":
        return True
    rule_key = (state.get("pattern") or {}).get("rule_key") or ""
    return "HIGH" in rule_key


def classify(state: dict, active_session: int, cfg) -> str:
    """Return 'promote' | 'demote' | 'prune' | 'keep' for one memory.

    Order matters: a cold, decayed, low-S memory DEMOTES (cold→warm, toward
    prune-eligibility); a warm decayed memory PRUNES. The cold-demote check
    therefore lives INSIDE the prune branch, before the generic prune —
    otherwise demote is unreachable.
    """
    r = retrievability(state, active_session, cfg)
    s = state["S"]
    if state["tier"] == "warm" and s >= cfg.memory_promote_s:
        return "promote"
    if r < cfg.memory_prune_r and not is_floor_protected(state):
        if state["tier"] == "cold" and s < cfg.memory_promote_s:
            return "demote"
        return "prune"
    return "keep"


@dataclass
class SweepPlan:
    creates: list[dict] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    promotions: list[dict] = field(default_factory=list)  # final states, tier=cold
    demotions: list[dict] = field(default_factory=list)  # final states, tier=warm
    prunes: list[dict] = field(default_factory=list)  # states to archive
    reviewed_count: int = 0  # total reviews (incl. reviewed-and-promoted/demoted)
    refused: int = 0  # new creates refused at cap (logged, never silent)

    def counts(self) -> dict:
        return {
            "created": len(self.creates),
            "reviewed": self.reviewed_count,
            "promoted": len(self.promotions),
            "demoted": len(self.demotions),
            "archived": len(self.prunes),
            "refused": self.refused,
        }


def _new_memory(pattern: dict, active_session: int, session_id: str) -> dict:
    """Fresh warm episodic memory from an observed pattern."""
    return {
        "memory_id": pattern["memory_id"],
        "pattern": {k: pattern[k] for k in ("kind", "domains", "rule_key", "severity")},
        "S": 1.0,
        "D": 5.0,
        "last_review_session": active_session,
        "tier": "warm",
        "status": "active",
        "origin_severity": pattern["severity"],
        "memory_kind": "episodic",
        "source_sessions": [session_id],
    }


def plan_sweep(states, observed_patterns, active_session, session_id, cfg) -> SweepPlan:
    """Deterministic plan: ingest unseen, review recurring, decay-classify the
    rest, then enforce MAX_MEMORY_ITEMS — archive lowest-R EXISTING non-protected
    survivors first, then REFUSE excess new creates (logged). Each existing
    memory lands in exactly one bucket; `survivors` holds existing only."""
    plan = SweepPlan()
    by_id = {s["memory_id"]: s for s in states}
    obs_by_id = {p["memory_id"]: p for p in observed_patterns}

    # 1. create unseen observed patterns
    for mid in obs_by_id.keys() - by_id.keys():
        plan.creates.append(_new_memory(obs_by_id[mid], active_session, session_id))

    # 2. existing memories: optional review, then decay-classify the result
    survivors: list[dict] = []
    for mid, st in by_id.items():
        cur = review(st, active_session, session_id, cfg) if mid in obs_by_id else st
        if cur is not st:  # review actually changed it
            plan.reviewed_count += 1
        action = classify(cur, active_session, cfg)
        if action == "prune":
            plan.prunes.append(cur)  # cur tier == original tier here
            continue
        if action == "promote":
            cur = {**cur, "tier": "cold"}
            plan.promotions.append(cur)
        elif action == "demote":
            cur = {**cur, "tier": "warm"}
            plan.demotions.append(cur)
        elif cur is not st:
            plan.reviews.append(cur)
        survivors.append(cur)

    # 3. cap enforcement
    if len(survivors) + len(plan.creates) > cfg.max_memory_items:
        # 3a. archive lowest-R existing non-protected survivors first.
        # A survivor may carry a planned tier change (promote/demote); SREM must
        # target its CURRENT redis tier, so prune with the ORIGINAL tier.
        prunable = sorted(
            (s for s in survivors if not is_floor_protected(s)),
            key=lambda s: retrievability(s, active_session, cfg),
        )
        need = len(survivors) + len(plan.creates) - cfg.max_memory_items
        for s in prunable[:need]:
            orig_tier = by_id[s["memory_id"]]["tier"]
            plan.prunes.append({**s, "tier": orig_tier})
            for bucket in (plan.reviews, plan.promotions, plan.demotions):
                if s in bucket:
                    bucket.remove(s)
            survivors.remove(s)
        # 3b. still over cap (all-protected survivors, or creates alone overflow)
        # → refuse the lowest-priority new creates (logged via plan.refused).
        remaining = len(survivors) + len(plan.creates) - cfg.max_memory_items
        if remaining > 0:
            plan.refused = min(remaining, len(plan.creates))
            if plan.refused:
                plan.creates = plan.creates[: len(plan.creates) - plan.refused]
    return plan
