from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from backend.services.engine.routers import alpha_agent


def test_universe_resolution_forwards_owner_to_custom_pool(monkeypatch):
    calls = {}
    symbols = ["sh600000", "sz000001"]

    def resolve_custom_pool(universe, *, tenant_id=None, user_id=None):
        calls.update(
            universe=universe,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return symbols

    qlib = ModuleType("qlib")
    qlib_data = ModuleType("qlib.data")
    qlib_data.D = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data)
    monkeypatch.setattr(
        alpha_agent, "_resolve_custom_pool_instruments", resolve_custom_pool
    )

    resolved = alpha_agent._resolve_instruments_for_universe(
        "CN",
        "pool:private-test",
        tenant_id="tenant-test",
        user_id="user-test",
    )

    assert resolved == symbols
    assert calls == {
        "universe": "pool:private-test",
        "tenant_id": "tenant-test",
        "user_id": "user-test",
    }
