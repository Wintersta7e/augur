"""Integration test: correlator session-end graph flush.

Starts detector + correlator, injects two cross-domain anomalies to
create a correlation edge, publishes augur.session.end, and verifies
the graph lands in Redis at augur:correlation:graph:<session_id>.

No Ollama — advisor is not in the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


def _make_perception_event(
    domain: str,
    entity: str,
    event_type: str,
    value: float,
    unit: str,
    context: dict,
    session_id: str,
    ts: datetime,
) -> bytes:
    return json.dumps(
        {
            "domain": domain,
            "stream_id": f"{domain}_injected",
            "entity": entity,
            "event_type": event_type,
            "value": value,
            "unit": unit,
            "context": context,
            "timestamp": ts.isoformat(),
            "session_id": session_id,
        }
    ).encode()


@pytest.mark.parametrize("pipeline", [["detector", "correlator"]], indirect=True)
async def test_session_end_flushes_graph_to_redis(
    pipeline,
    redis_client,
    nats_conn,
) -> None:
    """Correlation event + session.end → graph in Redis."""
    sid = str(uuid.uuid4())

    # Warm chess baseline. Must be long enough that the detector's
    # min_observations gate flips to trained — pad to (gate + 5) so the test
    # adapts to future config bumps without rewriting fixed-length arrays.
    from tabula.config import AugurConfig  # local import — test-only

    _warm = AugurConfig().min_observations + 5
    _chess_pool = [5.2, 6.8, 7.4, 5.9, 8.1, 6.3, 7.0, 5.5, 6.7, 7.8]
    chess_base = [_chess_pool[i % len(_chess_pool)] for i in range(_warm)]
    for v in chess_base:
        await nats_conn.publish(
            "augur.perception.chess",
            _make_perception_event(
                "chess",
                "white",
                "move",
                v,
                "seconds",
                {"move_san": "e4", "move_number": 1},
                sid,
                datetime.now(timezone.utc),
            ),
        )
        await asyncio.sleep(0.02)

    # Warm typing baseline (same adaptive pattern)
    _typing_pool = [2.8, 3.4, 2.9, 3.6, 3.1, 2.7, 3.3, 2.5, 3.2, 3.0]
    typing_base = [_typing_pool[i % len(_typing_pool)] for i in range(_warm)]
    for v in typing_base:
        await nats_conn.publish(
            "augur.perception.typing",
            _make_perception_event(
                "typing",
                "user",
                "pause",
                v,
                "seconds",
                {"avg_wpm": 60},
                sid,
                datetime.now(timezone.utc),
            ),
        )
        await asyncio.sleep(0.02)

    # Let the detector absorb baselines
    await asyncio.sleep(1.5)

    # Outlier 1: chess anomaly
    await nats_conn.publish(
        "augur.perception.chess",
        _make_perception_event(
            "chess",
            "white",
            "move",
            22.0,
            "seconds",
            {"move_san": "Nf3", "move_number": 11},
            sid,
            datetime.now(timezone.utc),
        ),
    )
    await asyncio.sleep(1.0)

    # Outlier 2: typing anomaly within window
    await nats_conn.publish(
        "augur.perception.typing",
        _make_perception_event(
            "typing",
            "user",
            "pause",
            9.0,
            "seconds",
            {"avg_wpm": 30},
            sid,
            datetime.now(timezone.utc),
        ),
    )

    # Give the correlator time to receive both anomalies and build edges
    await asyncio.sleep(2.0)

    # Publish session.end and wait for flush
    await nats_conn.publish(
        "augur.session.end",
        json.dumps(
            {
                "session_id": sid,
                "domain": "integration_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ).encode(),
    )

    # Poll Redis for the graph
    key = f"augur:correlation:graph:{sid}"
    graph_raw = None
    for _ in range(30):
        graph_raw = redis_client.get(key)
        if graph_raw is not None:
            break
        await asyncio.sleep(0.2)
    assert graph_raw is not None, f"Correlation graph not found at {key} within 6s"

    graph_data = json.loads(graph_raw)

    # Shape assertions — node_link_data format (NetworkX 3.4+ uses "edges")
    assert graph_data["directed"] is True
    assert "nodes" in graph_data
    assert "edges" in graph_data

    # At least one cross-domain edge should exist
    assert len(graph_data["nodes"]) >= 2
    assert len(graph_data["edges"]) >= 1

    # Find the chess-typing edge specifically. Other tests in this suite can
    # leak edges into the correlator's in-memory session graph; assert on the
    # edge this test is actually responsible for, not on graph_data["edges"][0].
    chess_typing = [
        e for e in graph_data["edges"] if set(e["domains"]) == {"chess", "typing"}
    ]
    assert chess_typing, f"No chess↔typing edge in graph; edges: {graph_data['edges']}"
    edge = chess_typing[0]
    assert "temporal_lag" in edge
    assert "escalation_rule" in edge
    assert "combined_severity" in edge
    assert "domains" in edge
    # domains is a list after JSON round-trip (was tuple in memory)
    assert isinstance(edge["domains"], list)


@pytest.mark.parametrize("pipeline", [["detector", "correlator"]], indirect=True)
async def test_session_end_with_no_correlations_still_saves_empty_graph(
    pipeline,
    redis_client,
    nats_conn,
) -> None:
    """A session with no cross-domain correlations still persists an
    empty graph so consumers can distinguish 'session ended cleanly with
    zero correlations' from 'session never existed'."""
    sid = str(uuid.uuid4())

    # Publish session.end immediately — no perception events at all
    await nats_conn.publish(
        "augur.session.end",
        json.dumps(
            {
                "session_id": sid,
                "domain": "integration_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ).encode(),
    )

    key = f"augur:correlation:graph:{sid}"
    graph_raw = None
    for _ in range(30):
        graph_raw = redis_client.get(key)
        if graph_raw is not None:
            break
        await asyncio.sleep(0.2)
    assert graph_raw is not None, f"Empty graph not persisted at {key} within 6s"

    graph_data = json.loads(graph_raw)
    assert graph_data["directed"] is True
    assert graph_data["nodes"] == []
    assert graph_data["edges"] == []
