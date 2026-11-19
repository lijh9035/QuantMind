from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from backend.services.engine.rd_agent.market_adapters.a_share import AShareAdapter
from backend.shared import database_manager_v2
from scripts.alpha_agent import run_rd_agent


def test_base_factors_json_contains_only_qlib_expressions(tmp_path: Path) -> None:
    generated = AShareAdapter().generate_base_factors_json(str(tmp_path))

    assert generated is not None
    factors = json.loads(Path(generated).read_text(encoding="utf-8"))
    assert factors
    assert "Abs($open - $close)" in factors["KUP"]
    assert "Max(" not in factors["KUP"]
    assert "Min(" not in factors["KLOW"]
    assert all(
        not any("\u4e00" <= character <= "\u9fff" for character in expression)
        for expression in factors.values()
    )


def test_master_engine_url_log_masks_password(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FakeConfig:
        pool_size = 2
        max_overflow = 0
        pool_timeout = 1
        pool_recycle = 60
        pool_pre_ping = False
        echo = False
        echo_pool = False
        slave_list: list[tuple[str, int]] = []

        @staticmethod
        def get_master_url() -> str:
            return "postgresql+asyncpg://qm_user:distinct-secret@db.example:5432/quantmind"

    monkeypatch.setattr(
        database_manager_v2, "create_async_engine", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        database_manager_v2, "async_sessionmaker", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        database_manager_v2.DatabaseManager,
        "_setup_engine_events",
        lambda *args, **kwargs: None,
    )
    manager = database_manager_v2.DatabaseManager(FakeConfig())

    async def no_health_check() -> None:
        return None

    monkeypatch.setattr(manager, "health_check", no_health_check)
    with caplog.at_level(logging.INFO, logger=database_manager_v2.__name__):
        asyncio.run(manager.initialize())

    messages = [
        record.getMessage()
        for record in caplog.records
        if "Creating master database engine with URL" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "distinct-secret" not in messages[0]
    assert "qm_user:***@db.example" in messages[0]


def _redirect_ic_temp_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    cache_path = tmp_path / "daily_pv.h5"
    real_path = run_rd_agent.Path

    def mapped_path(value: str | os.PathLike[str]) -> Path:
        if str(value) == "/tmp/daily_pv.h5":
            return cache_path
        return real_path(value)

    monkeypatch.setattr(run_rd_agent, "Path", mapped_path)

    original_named_temp = tempfile.NamedTemporaryFile
    original_mkstemp = tempfile.mkstemp

    def named_temp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return original_named_temp(*args, **kwargs)

    def mkstemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", named_temp)
    monkeypatch.setattr(tempfile, "mkstemp", mkstemp)
    return cache_path


def test_ic_replaces_newer_zero_byte_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.h5"
    source.write_bytes(b"fixture-h5")
    cache = _redirect_ic_temp_paths(monkeypatch, tmp_path)
    cache.write_bytes(b"")
    now = time.time()
    os.utime(source, (now - 20, now - 20))
    os.utime(cache, (now, now))

    def fake_run(command, **kwargs):
        assert cache.read_bytes() == b"fixture-h5"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="IC=0.1000\nRANK_IC=0.0500\nICIR=0.2000\nRANK_ICIR=0.1000\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    metrics = run_rd_agent.compute_factor_ic("factor code", str(source))

    assert metrics == {
        "ic": 0.1,
        "rank_ic": 0.05,
        "icir": 0.2,
        "rank_icir": 0.1,
    }


def test_ic_returns_empty_when_subprocess_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.h5"
    source.write_bytes(b"fixture-h5")
    _redirect_ic_temp_paths(monkeypatch, tmp_path)

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="fixture error")

    monkeypatch.setattr(subprocess, "run", failed_run)

    assert run_rd_agent.compute_factor_ic("factor code", str(source)) == {}


def test_persist_factors_passes_tenant_without_computing_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakePersistence:
        async def ensure_tables(self) -> None:
            return None

        async def save_factor(self, **kwargs) -> None:
            calls.append(kwargs)

    class FixedId:
        @staticmethod
        def hexdigest() -> str:
            return "stub-factor-id"

    real_import = builtins.__import__

    def import_without_digest(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hashlib" and globals and globals.get("__name__") == run_rd_agent.__name__:
            return types.SimpleNamespace(md5=lambda *_args, **_kwargs: FixedId())
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_digest)

    package_names = (
        "backend.services.engine.qlib_app",
        "backend.services.engine.qlib_app.services",
    )
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    persistence_module = types.ModuleType(
        "backend.services.engine.qlib_app.services.rd_agent_persistence"
    )
    persistence_module.RDAgentFactorPersistence = FakePersistence
    monkeypatch.setitem(
        sys.modules,
        "backend.services.engine.qlib_app.services.rd_agent_persistence",
        persistence_module,
    )

    saved = asyncio.run(
        run_rd_agent.persist_factors(
            [{"name": "candidate", "code": "return 1"}],
            task_id="task",
            user_id="user",
            market="a_share",
            tenant_id="tenant-7",
        )
    )

    assert saved == 1
    assert calls[0]["factor_id"] == "stub-factor-id"
    assert calls[0]["tenant_id"] == "tenant-7"
