from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.services.engine.alpha_agent.api_models import MiningSubmitRequest
from backend.services.engine.alpha_agent.task_spec import (
    ALPHASAGE_VERSION,
    MiningTaskSpec,
)


def _payload(**overrides):
    payload = {
        "algorithm": "alphasage",
        "market": "a_share",
        "universe": "csi300",
        "freq": "day",
        "data_source": "qlib_bin",
        "mining_start": "2020-01-01",
        "mining_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "horizon": 5,
        "n_episodes": 100,
        "pool_capacity": 32,
        "encoder_type": "gnn",
        "seed": 7,
    }
    payload.update(overrides)
    return payload


def test_alphasage_request_requires_full_non_overlapping_windows_and_bounds():
    parsed = MiningSubmitRequest.model_validate(_payload())
    assert parsed.algorithm == "alphasage"
    assert parsed.horizon == 5

    with pytest.raises(ValueError):
        MiningSubmitRequest.model_validate(_payload(valid_start=None))
    with pytest.raises(ValueError, match="之后开始"):
        MiningSubmitRequest.model_validate(_payload(valid_start="2020-12-31"))
    with pytest.raises(ValueError):
        MiningSubmitRequest.model_validate(_payload(pool_capacity=0))
    for unsupported in (
        {"horizon": 21},
        {"n_episodes": 15},
        {"n_episodes": 2_001},
        {"pool_capacity": 33},
        {"seed": 2_147_483_648},
    ):
        with pytest.raises(ValueError):
            MiningSubmitRequest.model_validate(_payload(**unsupported))


def test_alphasage_task_spec_is_versioned_and_fingerprint_is_stable():
    request = _payload()
    params = {**request, "loop_n": 3, "direction": ""}
    first = MiningTaskSpec.build(
        algorithm="alphasage",
        algorithm_version=ALPHASAGE_VERSION,
        request=request,
        params=params,
        data_snapshot={},
        request_key="alpha-submit-1",
    )
    retry = MiningTaskSpec.build(
        algorithm="alphasage",
        algorithm_version=ALPHASAGE_VERSION,
        request=request,
        params=params,
        data_snapshot={},
        request_key="alpha-submit-1",
    )

    assert first.algorithm == "alphasage"
    assert first.execution_fingerprint == retry.execution_fingerprint


def test_worker_result_is_fenced_to_frozen_identity_and_windows():
    from backend.services.engine.alpha_agent.alphasage.worker import _validate_result

    params = {
        "mining_start": "2020-01-01",
        "mining_end": "2020-12-31",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "horizon": 5,
    }
    result = {
        "schema_version": 1,
        "task_id": "task-a",
        "algorithm": "alphasage",
        "algorithm_version": ALPHASAGE_VERSION,
        "execution_fingerprint": "a" * 64,
        "training_window": {
            "requested_start": "2020-01-01",
            "requested_end": "2020-12-31",
            "mature_label_end": "2020-12-23",
            "horizon": 5,
        },
        "validation_window": {
            "start": "2021-01-01",
            "end": "2021-12-31",
            "evaluated": False,
        },
        "candidate_count": 1,
        "candidates": [{"source_index": 0, "raw_expression": "Add($close,1)"}],
    }
    calendar_fn = lambda _start, _end: [  # noqa: E731
        "2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24",
        "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31",
    ]

    candidates = _validate_result(
        result,
        task_id="task-a",
        fingerprint="a" * 64,
        params=params,
        calendar_fn=calendar_fn,
    )
    assert len(candidates) == 1

    with pytest.raises(ValueError, match="identity or status"):
        _validate_result(
            {**result, "execution_fingerprint": "b" * 64},
            task_id="task-a",
            fingerprint="a" * 64,
            params=params,
            calendar_fn=calendar_fn,
        )
    with pytest.raises(ValueError, match="validation data"):
        _validate_result(
            {
                **result,
                "validation_window": {**result["validation_window"], "evaluated": True},
            },
            task_id="task-a",
            fingerprint="a" * 64,
            params=params,
            calendar_fn=calendar_fn,
        )

    with pytest.raises(ValueError, match="Qlib training limit"):
        _validate_result(
            {
                **result,
                "training_window": {
                    **result["training_window"],
                    "mature_label_end": "2020-12-29",
                },
            },
            task_id="task-a",
            fingerprint="a" * 64,
            params=params,
            calendar_fn=calendar_fn,
        )


def test_launcher_routes_alphasage_to_isolated_worker_adapter(monkeypatch):
    from backend.services.engine.alpha_agent import launcher as launcher_module
    from backend.services.engine.alpha_agent.alphasage import worker

    calls = []

    async def fake_run(record):
        calls.append(record)

    monkeypatch.setattr(worker, "run_alphasage_queued", fake_run)
    record = {"task_id": "task-a", "algorithm": "alphasage"}
    launcher = object.__new__(launcher_module.AlphaAgentLauncher)

    asyncio.run(launcher.run_queued(record))

    assert calls == [record]


def test_trusted_task_recomputes_fingerprint_with_frozen_custom_instruments():
    from backend.services.engine.alpha_agent.alphasage.worker import _trusted_task

    request = _payload()
    params = {**request, "loop_n": 3, "direction": ""}
    data_snapshot = {"resolved_instruments": ["sh600000", "sz000001"]}
    spec = MiningTaskSpec.build(
        algorithm="alphasage",
        algorithm_version=ALPHASAGE_VERSION,
        request=request,
        params=params,
        data_snapshot=data_snapshot,
    )
    record = {
        "task_id": "task-a",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "algorithm": "alphasage",
        "algorithm_version": ALPHASAGE_VERSION,
        "request_snapshot": request,
        "params_snapshot": params,
        "data_snapshot": data_snapshot,
        "execution_fingerprint": spec.execution_fingerprint,
    }

    trusted = _trusted_task(record)

    assert trusted[3] == data_snapshot
    assert trusted[4] == spec.execution_fingerprint

    with pytest.raises(ValueError, match="execution fingerprint"):
        _trusted_task({**record, "data_snapshot": {}})


def test_orphan_recovery_treats_worker_404_as_not_running(monkeypatch):
    from backend.services.engine.alpha_agent import resource_budget
    from backend.services.engine.alpha_agent.alphasage.client import (
        AlphaSageWorkerNotFound,
    )
    from backend.services.engine.alpha_agent.alphasage.worker import (
        recover_orphaned_execution,
    )
    from backend.shared.owner_scope import OwnerScope

    class FakeStore:
        status = "running"

        async def get(self, *_args, **_kwargs):
            return {"status": self.status}

        async def request_cancel(self, *_args, **_kwargs):
            self.status = "stopping"
            return self.status

        async def update(self, *_args, expected_status, status, **_kwargs):
            assert self.status in expected_status
            self.status = status
            return True

        async def confirm_cancelled(self, *_args, **_kwargs):
            self.status = "cancelled"
            return True

    class MissingWorker:
        async def cancel(self, _task_id):
            raise AlphaSageWorkerNotFound("missing", status_code=404)

    store = FakeStore()
    finishes = []
    monkeypatch.setattr(
        resource_budget,
        "finish_execution",
        lambda *args: finishes.append(args) or True,
    )

    state = asyncio.run(
        recover_orphaned_execution(
            store,
            "task-a",
            OwnerScope("tenant-a", "user-a"),
            client=MissingWorker(),
        )
    )

    assert state == "failed"
    assert store.status == "failed"
    assert len(finishes) == 1
    assert finishes[0][-1] == "failed"


def test_orphan_recovery_retains_lease_after_worker_network_error(monkeypatch):
    from backend.services.engine.alpha_agent import resource_budget
    from backend.services.engine.alpha_agent.alphasage.client import (
        AlphaSageWorkerUnavailable,
    )
    from backend.services.engine.alpha_agent.alphasage.worker import (
        recover_orphaned_execution,
    )
    from backend.shared.owner_scope import OwnerScope

    class FakeStore:
        status = "running"

        async def get(self, *_args, **_kwargs):
            return {"status": self.status}

        async def request_cancel(self, *_args, **_kwargs):
            self.status = "stopping"
            return self.status

    class OfflineWorker:
        async def cancel(self, _task_id):
            raise AlphaSageWorkerUnavailable("offline", status_code=503)

    store = FakeStore()
    finishes = []
    monkeypatch.setattr(
        resource_budget,
        "finish_execution",
        lambda *args: finishes.append(args) or True,
    )

    state = asyncio.run(
        recover_orphaned_execution(
            store,
            "task-a",
            OwnerScope("tenant-a", "user-a"),
            client=OfflineWorker(),
        )
    )

    assert state is None
    assert store.status == "stopping"
    assert finishes == []


def test_alpha_cancel_confirms_missing_worker_task_as_stopped():
    from backend.services.engine.alpha_agent.alphasage.client import (
        AlphaSageWorkerNotFound,
    )
    from backend.services.engine.alpha_agent.alphasage.worker import (
        _cancel_and_confirm,
    )
    from backend.shared.owner_scope import OwnerScope

    class FakeStore:
        cancelled = False

        async def confirm_cancelled(self, *_args, **_kwargs):
            self.cancelled = True
            return True

    class MissingWorker:
        async def cancel(self, _task_id):
            raise AlphaSageWorkerNotFound("missing", status_code=404)

    store = FakeStore()
    state = asyncio.run(
        _cancel_and_confirm(
            MissingWorker(), store, OwnerScope("tenant-a", "user-a"), "task-a"
        )
    )

    assert state == "cancelled"
    assert store.cancelled is True


def test_owner_scoped_log_can_be_read_before_task_is_in_memory(tmp_path):
    from backend.services.engine.alpha_agent.launcher import AlphaAgentLauncher

    launcher = object.__new__(AlphaAgentLauncher)
    launcher._log_dir = tmp_path
    launcher._tasks = {}
    log_dir = launcher.task_dir("task-a", "tenant-a", "user-a")
    log_dir.mkdir(parents=True)
    (log_dir / "subprocess_stdout.log").write_text("queued\n", encoding="utf-8")

    content = asyncio.run(
        launcher.get_task_log(
            "task-a", tenant_id="tenant-a", user_id="user-a"
        )
    )
    foreign_content = asyncio.run(
        launcher.get_task_log(
            "task-a", tenant_id="tenant-b", user_id="user-a"
        )
    )

    assert content == "queued\n"
    assert foreign_content is None


def test_submit_admits_owner_scoped_alphasage_to_durable_queue(monkeypatch):
    pytest.importorskip("qlib")
    from backend.services.engine.alpha_agent import resource_budget
    from backend.services.engine.routers import alpha_agent
    from backend.services.engine.qlib_app.services import rd_agent_persistence

    class FakeStore:
        def __init__(self):
            self.record = None
            self.specification = None

        async def get_by_request_key(self, key, *, tenant_id, user_id):
            assert (key, tenant_id, user_id) == ("retry-key", "tenant-a", "user-a")
            return None

        async def create(self, task_id, tenant_id, user_id, *, specification, **fields):
            assert (tenant_id, user_id) == ("tenant-a", "user-a")
            assert fields["status"] == "queued"
            assert fields["data_source"] == "qlib_bin"
            self.specification = specification
            self.record = {
                "task_id": task_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "algorithm": "alphasage",
                "algorithm_version": ALPHASAGE_VERSION,
                "execution_fingerprint": specification.execution_fingerprint,
                "request_snapshot": json.loads(specification.request_json),
                "params_snapshot": json.loads(specification.params_json),
                "status": "queued",
            }
            return task_id

        async def get_for_owner_with_spec(
            self, task_id, *, tenant_id, user_id, consistent=False
        ):
            assert self.record["task_id"] == task_id
            assert (tenant_id, user_id) == ("tenant-a", "user-a")
            return self.record

        async def queue_position(self, task_id, *, tenant_id, user_id):
            return 1

    store = FakeStore()
    monkeypatch.setattr(
        rd_agent_persistence, "AlphaMiningTaskPersistence", lambda: store
    )
    monkeypatch.setattr(
        alpha_agent,
        "get_authenticated_identity",
        lambda _request: ("user-a", "tenant-a"),
    )
    monkeypatch.setattr(
        alpha_agent, "_universe_is_valid", lambda _universe, **_kwargs: True
    )
    monkeypatch.setattr(resource_budget, "submit_budget", lambda *_args: object())

    response = asyncio.run(
        alpha_agent.submit_mining_task(_payload(), object(), "retry-key")
    )

    assert response.status_code == 202
    receipt = json.loads(response.body)["data"]
    assert receipt["algorithm"] == "alphasage"
    assert receipt["status"] == "queued"
    assert receipt["queue_position"] == 1


def test_submit_rejects_reused_key_for_different_frozen_request(monkeypatch):
    pytest.importorskip("qlib")
    from backend.services.engine.alpha_agent import resource_budget
    from backend.services.engine.routers import alpha_agent
    from backend.services.engine.qlib_app.services import rd_agent_persistence

    class FakeStore:
        async def get_by_request_key(self, *_args, **_kwargs):
            return {"task_id": "existing-task"}

        async def get_for_owner_with_spec(self, *_args, **_kwargs):
            return {
                "task_id": "existing-task",
                "algorithm": "alphasage",
                "algorithm_version": ALPHASAGE_VERSION,
                "execution_fingerprint": "a" * 64,
                "request_snapshot": _payload(horizon=10),
                "status": "completed",
            }

    monkeypatch.setattr(
        rd_agent_persistence, "AlphaMiningTaskPersistence", FakeStore
    )
    monkeypatch.setattr(
        alpha_agent,
        "get_authenticated_identity",
        lambda _request: ("user-a", "tenant-a"),
    )
    monkeypatch.setattr(
        alpha_agent, "_universe_is_valid", lambda _universe, **_kwargs: True
    )
    monkeypatch.setattr(resource_budget, "submit_budget", lambda *_args: object())

    with pytest.raises(HTTPException) as error:
        asyncio.run(alpha_agent.submit_mining_task(_payload(), object(), "retry-key"))
    assert error.value.status_code == 409


def test_owned_task_can_be_read_from_durable_queue_after_restart(monkeypatch):
    pytest.importorskip("qlib")
    from backend.services.engine.routers import alpha_agent
    from backend.services.engine.qlib_app.services import rd_agent_persistence

    record = {
        "task_id": "durable-task",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "algorithm": "alphasage",
        "algorithm_version": ALPHASAGE_VERSION,
        "status": "running",
        "phase": "train",
        "progress": "GPU search",
        "progress_pct": 42,
        "loop_n": 3,
        "market": "a_share",
        "universe": "csi300",
        "data_source": "qlib_bin",
        "params_snapshot": {
            "mining_start": "2020-01-01",
            "mining_end": "2020-12-31",
            "valid_start": "2021-01-01",
            "valid_end": "2021-12-31",
        },
        "executor_json": {"progress": {"candidate_count": 4}},
    }

    class FakeStore:
        async def get_for_owner_with_spec(self, task_id, *, tenant_id, user_id):
            assert (task_id, tenant_id, user_id) == (
                "durable-task", "tenant-a", "user-a"
            )
            return record

    launcher = SimpleNamespace(get_task_status=lambda _task_id: asyncio.sleep(0, result=None))
    monkeypatch.setattr(alpha_agent, "get_launcher", lambda: launcher)
    monkeypatch.setattr(
        alpha_agent,
        "get_authenticated_identity",
        lambda _request: ("user-a", "tenant-a"),
    )
    monkeypatch.setattr(
        rd_agent_persistence, "AlphaMiningTaskPersistence", FakeStore
    )

    status = asyncio.run(alpha_agent._require_owned_task("durable-task", object()))

    assert status["algorithm"] == "alphasage"
    assert status["status"] == "running"
    assert status["progress_pct"] == 42
    assert status["mining_progress"] == {"candidate_count": 4}
