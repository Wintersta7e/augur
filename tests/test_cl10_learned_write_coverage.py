"""CL10 — every Redis-mutating function carries a provenance marker (spec §7).

Behavioural discovery, NOT a hand-maintained list (which would have the same
failure mode as the contamination bug: someone adds a write and forgets the
list). This walks the AST of every production module, finds each function that
calls a Redis mutator — directly on a redis handle, or via a PersistenceManager
write helper — and requires it to be one of:

- **marked** with ``@learned_write`` or ``@non_learning_write``;
- an **exempt** write helper (``_set_json`` / ``_hash_save`` — pure plumbing; the
  provenance gate lives on the public method that calls it);
- on the explicit **PENDING_MIGRATION** backlog.

PENDING_MIGRATION is a RATCHET: it may only shrink. A stale or already-marked
entry fails the suite (forcing removal as writers are migrated), and a NEW
unmarked writer that is not listed fails too (so contamination cannot silently
return). The flip to ``ProvenanceMode.ENFORCE`` (spec §10 step 5.5) is gated on
this backlog reaching **empty** — until then the decorators run in OFF/REPORT and
change nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROD_DIRS = [
    "tabula",
    "nexus",
    "consilium",
    "vigil",
    "responsum",
    "disciplina",
    "imperator",
    "praefectus",
    "memoria",
    "conscientia",
    "praesagium",
    "limen",
    "vox",
    "augur_mcp",
    "sensus",
]

# Redis write-command method names. Reads (get/hget/zrange/...) are excluded.
MUTATORS = {
    "set",
    "setex",
    "psetex",
    "setnx",
    "mset",
    "msetnx",
    "getset",
    "hset",
    "hmset",
    "hsetnx",
    "hincrby",
    "hincrbyfloat",
    "hdel",
    "lpush",
    "rpush",
    "lpushx",
    "rpushx",
    "lset",
    "ltrim",
    "lrem",
    "lpop",
    "rpop",
    "sadd",
    "srem",
    "spop",
    "smove",
    "sinterstore",
    "sunionstore",
    "zadd",
    "zrem",
    "zincrby",
    "zpopmin",
    "zpopmax",
    "zremrangebyrank",
    "zremrangebyscore",
    "zremrangebylex",
    "delete",
    "unlink",
    "expire",
    "pexpire",
    "expireat",
    "persist",
    "rename",
    "renamenx",
    "move",
    "incr",
    "incrby",
    "incrbyfloat",
    "decr",
    "decrby",
    "setbit",
    "setrange",
    "flushdb",
    "flushall",
    "copy",
    "restore",
    "geoadd",
    "xadd",
    "pfadd",
}

# PersistenceManager private write helpers: their callers are the real writers.
WRITE_HELPERS = {"_set_json", "_delete_json", "_hash_save"}

# Pure write plumbing — the gate lives on the public method that calls it, not here.
EXEMPT = {
    "tabula/persistence.py::PersistenceManager._set_json",
    "tabula/persistence.py::PersistenceManager._hash_save",
}

# The migration backlog (spec §10 step 5.4). Remove a key when its writer is
# marked @learned_write / @non_learning_write. RATCHET: this set only shrinks.
# Empty: every Redis writer is now MARKED (@learned_write / @non_learning_write).
# A newly-added writer must be marked or listed here — the ratchet keeps it honest.
# NB: CL10 asserts the MARKER exists; threading a LearnContext to each learned
# writer's call sites (so ENFORCE has provenance in hand) is a separate concern,
# verified per-faculty and gated at the flip (spec §10 step 5.4/5.5).
PENDING_MIGRATION = frozenset()

_MARKERS = {"learned_write", "non_learning_write"}


def _is_redis_receiver(value: ast.expr) -> bool:
    if (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "self"
        and value.attr in {"_r", "_redis", "_redis_client", "redis"}
    ):
        return True
    if isinstance(value, ast.Name) and value.id in {
        "pipe",
        "pipeline",
        "pipe2",
        "r",
        "rc",
        "redis_client",
        "client",
        "conn",
    }:
        return True
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        return _is_redis_receiver(value.func)
    return False


def _mutates_redis(func_node: ast.AST) -> bool:
    for node in ast.walk(func_node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr = node.func.attr
        if attr in WRITE_HELPERS:
            v = node.func.value
            if isinstance(v, ast.Name) and v.id == "self":
                return True
            continue
        if attr in MUTATORS and _is_redis_receiver(node.func.value):
            return True
    return False


def _is_marked(func_node) -> bool:
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in _MARKERS:
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            if dec.func.id in _MARKERS:
                return True
        if isinstance(dec, ast.Attribute) and dec.attr in _MARKERS:
            return True
    return False


class _Collector(ast.NodeVisitor):
    def __init__(self, relpath: str):
        self.relpath = relpath
        self.stack: list[str] = []
        self.writers: dict[str, bool] = {}  # key -> is_marked

    def _visit_func(self, node):
        self.stack.append(node.name)
        if _mutates_redis(node):
            key = f"{self.relpath}::{'.'.join(self.stack)}"
            self.writers[key] = _is_marked(node)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _discover() -> dict[str, bool]:
    writers: dict[str, bool] = {}
    for d in PROD_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            c = _Collector(str(path.relative_to(ROOT)))
            c.visit(tree)
            writers.update(c.writers)
    return writers


def test_writer_surface_is_nonempty() -> None:
    # Guard the guard: a broken enumerator must fail loudly, not vacuously pass.
    assert len(_discover()) >= 55


def test_no_unmarked_unlisted_redis_writer() -> None:
    unaccounted = []
    for key, marked in _discover().items():
        if marked or key in EXEMPT or key in PENDING_MIGRATION:
            continue
        unaccounted.append(key)
    assert not unaccounted, (
        "these functions mutate Redis but carry no @learned_write / "
        "@non_learning_write marker and are not on PENDING_MIGRATION — a new "
        f"unguarded learned-write surface: {sorted(unaccounted)}"
    )


def test_pending_backlog_only_shrinks() -> None:
    discovered = _discover()
    stale = []
    for key in PENDING_MIGRATION:
        if key not in discovered:
            stale.append(f"{key} (no longer a Redis writer)")
        elif discovered[key]:
            stale.append(f"{key} (now marked — remove from PENDING_MIGRATION)")
        elif key in EXEMPT:
            stale.append(f"{key} (exempt — remove from PENDING_MIGRATION)")
    assert not stale, (
        "PENDING_MIGRATION is a ratchet and must only shrink; these entries are "
        f"stale and must be removed: {sorted(stale)}"
    )


def test_exempt_helpers_are_real_writers() -> None:
    # Exemptions must correspond to real write plumbing, not typos that would
    # silently excuse a genuine writer.
    discovered = _discover()
    assert EXEMPT <= set(discovered), (
        f"EXEMPT names that are not discovered writers: {EXEMPT - set(discovered)}"
    )
