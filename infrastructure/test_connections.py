"""Verify that Redis and NATS infrastructure services are reachable."""

import asyncio
import sys

import redis
import nats


def test_redis(host: str = "localhost", port: int = 6379) -> bool:
    try:
        client = redis.Redis(host=host, port=port, socket_connect_timeout=5)
        client.ping()
        print(f"Redis: OK ({host}:{port})")
        return True
    except redis.ConnectionError as exc:
        print(f"Redis: FAILED ({host}:{port}) - {exc}")
        return False


async def test_nats(url: str = "nats://localhost:4222") -> bool:
    try:
        nc = await nats.connect(url, connect_timeout=5)
        print(f"NATS: OK ({url})")
        await nc.close()
        return True
    except Exception as exc:
        print(f"NATS: FAILED ({url}) - {exc}")
        return False


async def main() -> int:
    redis_ok = test_redis()
    nats_ok = await test_nats()

    if redis_ok and nats_ok:
        print("\nAll services reachable.")
        return 0
    else:
        print("\nSome services unreachable.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
