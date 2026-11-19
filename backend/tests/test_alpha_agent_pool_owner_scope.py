from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_alpha_router(monkeypatch):
    """Import the router without initializing the optional Qlib service facade."""
    services_name = "backend.services.engine.qlib_app.services"
    services_path = (
        Path(__file__).resolve().parents[1]
        / "services"
        / "engine"
        / "qlib_app"
        / "services"
    )
    services_package = types.ModuleType(services_name)
    services_package.__path__ = [str(services_path)]
    monkeypatch.setitem(sys.modules, services_name, services_package)
    monkeypatch.delitem(
        sys.modules, f"{services_name}.rd_agent_persistence", raising=False
    )
    return importlib.import_module("backend.services.engine.routers.alpha_agent")


def test_universe_helpers_and_alphasage_selection_forward_owner_scope(monkeypatch):
    alpha_router = _load_alpha_router(monkeypatch)
    resolver = importlib.import_module("backend.shared.stock_pool.resolver")

    monkeypatch.setattr(alpha_router, "_VALID_CN_UNIVERSES", ["builtin"])
    calls = []

    def resolve(ref, **kwargs):
        calls.append((ref, kwargs))
        return SimpleNamespace(unfiltered=False, symbols=["600000.SH"])

    monkeypatch.setattr(resolver, "resolve_pool_sync", resolve)

    assert alpha_router._universe_is_valid(
        "private-list", tenant_id="tenant-a", user_id="user-a"
    )
    assert alpha_router._resolve_custom_pool_instruments(
        "private-list", tenant_id="tenant-a", user_id="user-a"
    ) == ["SH600000"]
    assert alpha_router._alphasage_instrument_selection(
        "private-list", tenant_id="tenant-a", user_id="user-a"
    ) == (
        {"instruments": ["sh600000"]},
        {"resolved_instruments": ["sh600000"]},
    )

    assert calls == [
        ("pool:private-list", {"tenant_id": "tenant-a", "user_id": "user-a"}),
        ("pool:private-list", {"tenant_id": "tenant-a", "user_id": "user-a"}),
        ("pool:private-list", {"tenant_id": "tenant-a", "user_id": "user-a"}),
    ]


def test_pool_helpers_reject_partial_owner_scope_even_for_builtins(monkeypatch):
    alpha_router = _load_alpha_router(monkeypatch)
    monkeypatch.setattr(alpha_router, "_VALID_CN_UNIVERSES", ["builtin"])

    assert not alpha_router._universe_is_valid("builtin", tenant_id="tenant-a")
    with pytest.raises(ValueError, match="Both owner dimensions"):
        alpha_router._resolve_custom_pool_instruments(
            "private-list", tenant_id="tenant-a"
        )
    with pytest.raises(ValueError, match="Both owner dimensions"):
        alpha_router._alphasage_instrument_selection(
            "private-list", user_id="user-a"
        )


def test_pool_helpers_preserve_unscoped_legacy_lookup(monkeypatch):
    alpha_router = _load_alpha_router(monkeypatch)
    resolver = importlib.import_module("backend.shared.stock_pool.resolver")

    monkeypatch.setattr(alpha_router, "_VALID_CN_UNIVERSES", ["builtin"])
    calls = []

    def resolve(ref, **kwargs):
        calls.append((ref, kwargs))
        return SimpleNamespace(unfiltered=False, symbols=["600000.SH"])

    monkeypatch.setattr(resolver, "resolve_pool_sync", resolve)

    assert alpha_router._universe_is_valid("private-list")
    assert alpha_router._resolve_custom_pool_instruments("private-list") == [
        "SH600000"
    ]
    assert calls == [
        ("pool:private-list", {}),
        ("pool:private-list", {}),
    ]
