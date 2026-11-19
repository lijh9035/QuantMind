from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest


def _load_persistence_module(monkeypatch):
    """Load this DB repository module without importing the Qlib services facade."""
    package_name = "backend.services.engine.qlib_app.services"
    package_path = (
        Path(__file__).resolve().parents[1]
        / "services"
        / "engine"
        / "qlib_app"
        / "services"
    )
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.delitem(
        sys.modules, f"{package_name}.rd_agent_persistence", raising=False
    )
    return importlib.import_module(f"{package_name}.rd_agent_persistence")


def _api_factor_revision_no(factor):
    module_name = "backend.services.engine.alpha_agent.factor_job_api"
    previous = sys.modules.get(module_name)
    module = importlib.import_module(module_name)
    try:
        return module._factor_revision_no(factor)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
            parent = sys.modules.get("backend.services.engine.alpha_agent")
            if parent is not None and getattr(parent, "factor_job_api", None) is module:
                delattr(parent, "factor_job_api")
        else:
            sys.modules[module_name] = previous


class _Result:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class _Session:
    def __init__(self, row):
        self.row = row
        self.statement = None
        self.params = None

    async def execute(self, statement, params):
        self.statement = str(statement)
        self.params = params
        return _Result(self.row)


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_args):
        return False


def _factor_row(*, revision_no=7):
    return {
        "factor_id": "factor-a",
        "factor_name": "Momentum",
        "factor_code": "fm:Ref($close, 1)",
        "status": "pending",
        "ic_value": None,
        "sharpe_ratio": None,
        "annual_return": None,
        "max_drawdown": None,
        "rank_ic": None,
        "user_id": "user-a",
        "market": "a_share",
        "universe": "csi300",
        "factor_formulation": "Ref($close, 1)",
        "data_source": "qlib_bin",
        "date_range": "2020-01-01~2020-12-31",
        "metadata_json": {"source": "test"},
        "created_at": None,
        "updated_at": None,
        "factor_revision_no": revision_no,
    }


def test_owner_scoped_factor_read_supplies_current_revision_for_materialization(
    monkeypatch,
):
    rd_agent_persistence = _load_persistence_module(monkeypatch)
    session = _Session(_factor_row())
    calls = []

    def get_session(*, read_only=False):
        calls.append(read_only)
        return _SessionContext(session)

    monkeypatch.setattr(rd_agent_persistence, "get_session", get_session)

    factor = asyncio.run(
        rd_agent_persistence.RDAgentFactorPersistence().get_factor(
            "factor-a", tenant_id="tenant-a", user_id="user-a"
        )
    )

    assert factor is not None
    assert _api_factor_revision_no(factor) == 7
    assert factor["metadata"] == {"source": "test"}
    assert calls == [True]
    assert "factor.tenant_id=:tenant_id AND factor.user_id=:user_id" in session.statement
    assert "MAX(revision.revision_no)" in session.statement
    assert session.params == {
        "factor_id": "factor-a",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
    }


def test_legacy_factor_read_remains_available_without_owner_scope(monkeypatch):
    rd_agent_persistence = _load_persistence_module(monkeypatch)

    session = _Session(_factor_row(revision_no=3))

    def get_session(*, read_only=False):
        assert read_only is True
        return _SessionContext(session)

    monkeypatch.setattr(rd_agent_persistence, "get_session", get_session)

    factor = asyncio.run(
        rd_agent_persistence.RDAgentFactorPersistence().get_factor("factor-a")
    )

    assert factor is not None
    assert factor["factor_revision_no"] == 3
    assert "factor.tenant_id=:tenant_id" not in session.statement
    assert session.params == {"factor_id": "factor-a"}


def test_factor_read_rejects_partial_owner_scope(monkeypatch):
    rd_agent_persistence = _load_persistence_module(monkeypatch)

    with pytest.raises(ValueError, match="Both owner dimensions"):
        asyncio.run(
            rd_agent_persistence.RDAgentFactorPersistence().get_factor(
                "factor-a", tenant_id="tenant-a"
            )
        )


def test_materialization_admission_logger_redacts_exception_message(
    monkeypatch, caplog
):
    _load_persistence_module(monkeypatch)
    from backend.services.engine.alpha_agent.factor_job_api import (
        _log_materialization_admission_failure,
    )

    caplog.set_level(logging.ERROR)
    sensitive_message = "password=" + "very-secret" + " " + "C:\\data\\qlib_provider"
    try:
        error = RuntimeError(sensitive_message)
        error.orig = types.SimpleNamespace(pgcode="42P01")
        raise error
    except Exception as exc:
        _log_materialization_admission_failure(exc)

    assert "exception_type=RuntimeError" in caplog.text
    assert "sqlstate=42P01" in caplog.text
    assert "exception details redacted" in caplog.text
    assert "very-secret" not in caplog.text
    assert "C:\\data\\qlib_provider" not in caplog.text
