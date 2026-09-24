"""Shared TestClient factory for tests/test_api_*.py."""

from fastapi.testclient import TestClient

from api_world import ApiWorld
from experionyx.api.app import create_app


def client(world: ApiWorld) -> TestClient:
    return TestClient(create_app(str(world.ws)))


__all__ = ["client"]
