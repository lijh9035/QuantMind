from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services.engine.alpha_agent import factor_job_runner_api as runner_api


JOB_ID = "factor-job-test"
DISPATCH_TOKEN = "dispatch-token-test"
OWNER_HEADERS = {
    "X-Internal-Call": "test-internal-secret",
    "X-Tenant-Id": "tenant-test",
    "X-User-Id": "user-test",
}
RUN_BODY = {
    "job_id": JOB_ID,
    "execution_generation": 1,
    "dispatch_token": DISPATCH_TOKEN,
}


class FakeStore:
    def __init__(self, *, claim: str = "claimed", stopping: bool = False):
        self.claim = claim
        self.stopping = stopping
        self.finished = False

    def get_lease(self, *args, expected_dispatch_token):
        assert expected_dispatch_token == DISPATCH_TOKEN
        return object()

    def task_snapshot(self, *args):
        return {
            "state": "running",
            "stop_requested": self.stopping,
            "executor": {
                "dispatch_token": DISPATCH_TOKEN,
                **({} if self.stopping else {}),
            },
        }

    def claim_engine_execution(self, *args, **kwargs):
        return self.claim

    def transition(self, *args, **kwargs):
        return True

    def mark_engine_request_finished(self, *args, **kwargs):
        self.finished = True
        return True


class FakeProjection:
    def heartbeat(self, lease):
        return True


class FakeJobs:
    def __init__(
        self,
        *,
        status="running",
        kind="materialize",
        race_cancel_on_finish=False,
        fail_finish=False,
    ):
        self.status = status
        self.kind = kind
        self.race_cancel_on_finish = race_cancel_on_finish
        self.fail_finish = fail_finish
        self.result = None
        self.finished_with_error = False
        self.confirmed_cancel = False

    async def get_for_replay(self, job_id, *, dispatch_token):
        assert job_id == JOB_ID
        assert dispatch_token == DISPATCH_TOKEN
        return {
            "tenant_id": "tenant-test",
            "user_id": "user-test",
            "kind": self.kind,
            "status": self.status,
            "execution_generation": 1,
            "request_snapshot": {"request": {"generation_id": "generation-test"}},
        }

    async def get(self, job_id, *, tenant_id, user_id, consistent=False):
        return {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "status": self.status,
            "execution_generation": 1,
        }

    async def update_progress(self, *args, **kwargs):
        return True

    async def finish(self, job_id, *, result=None, error_code=None, **kwargs):
        if self.race_cancel_on_finish:
            self.status = "stopping"
            raise RuntimeError("cancel raced with finish")
        if self.fail_finish:
            raise RuntimeError("finish persistence unavailable")
        self.result = result
        self.finished_with_error = error_code is not None
        self.status = "failed" if error_code is not None else "completed"
        return True

    async def confirm_cancelled(self, *args, **kwargs):
        self.confirmed_cancel = True
        self.status = "cancelled"
        return True


def _client(monkeypatch, jobs: FakeJobs, store: FakeStore) -> TestClient:
    monkeypatch.setattr(
        runner_api, "get_internal_call_secret", lambda: "test-internal-secret"
    )
    monkeypatch.setattr(runner_api, "_jobs", jobs)
    monkeypatch.setattr(
        runner_api,
        "require_ready_budget",
        lambda: SimpleNamespace(store=store, projection=FakeProjection()),
    )
    app = FastAPI()
    app.include_router(runner_api.router)
    return TestClient(app)


def test_runner_requires_internal_secret(monkeypatch):
    client = _client(monkeypatch, FakeJobs(), FakeStore())

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers={"X-Tenant-Id": "tenant-test", "X-User-Id": "user-test"},
    )

    assert response.status_code == 401


def test_runner_replays_materialization_and_finishes(monkeypatch):
    jobs = FakeJobs()
    store = FakeStore()
    client = _client(monkeypatch, jobs, store)
    monkeypatch.setattr(
        "backend.shared.process_witness.capture_process", lambda: {"pid": 123}
    )
    monkeypatch.setattr(
        "backend.services.engine.alpha_agent.factor_job_executor.execute_factor_materialization",
        lambda request, **kwargs: {"status": "materialized", "row_count": 1},
    )

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers=OWNER_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "completed"}
    assert jobs.result == {"status": "materialized", "row_count": 1}
    assert store.finished


def test_runner_does_not_start_a_duplicate_engine_claim(monkeypatch):
    jobs = FakeJobs()
    store = FakeStore(claim="existing_same")
    client = _client(monkeypatch, jobs, store)
    monkeypatch.setattr(
        "backend.shared.process_witness.capture_process", lambda: {"pid": 123}
    )

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers=OWNER_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "running"}
    assert not store.finished


def test_runner_confirms_cancel_only_before_engine_claim(monkeypatch):
    jobs = FakeJobs(status="stopping")
    store = FakeStore(stopping=True)
    client = _client(monkeypatch, jobs, store)

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers=OWNER_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "cancelled"}
    assert jobs.confirmed_cancel


def test_runner_refuses_unsupported_job_kind(monkeypatch):
    client = _client(monkeypatch, FakeJobs(kind="backtest"), FakeStore())

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers=OWNER_HEADERS,
    )

    assert response.status_code == 409


def test_runner_confirms_cancel_that_races_with_finish(monkeypatch):
    jobs = FakeJobs(race_cancel_on_finish=True)
    store = FakeStore()
    client = _client(monkeypatch, jobs, store)
    monkeypatch.setattr(
        "backend.shared.process_witness.capture_process", lambda: {"pid": 123}
    )
    monkeypatch.setattr(
        "backend.services.engine.alpha_agent.factor_job_executor.execute_factor_materialization",
        lambda request, **kwargs: {"status": "materialized", "row_count": 1},
    )

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers=OWNER_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "cancelled"}
    assert jobs.confirmed_cancel


def test_runner_requires_cleanup_and_stop_proof_before_finalizing():
    assert not runner_api._lifecycle_is_safe_to_finalize(
        {
            "intent_recorded": True,
            "cleanup_confirmed": False,
            "create_started": False,
            "confirmed_stopped": True,
        }
    )
    assert not runner_api._lifecycle_is_safe_to_finalize(
        {
            "cleanup_confirmed": True,
            "create_started": True,
            "confirmed_stopped": False,
        }
    )
    assert runner_api._lifecycle_is_safe_to_finalize(
        {
            "cleanup_confirmed": True,
            "create_started": True,
            "confirmed_stopped": True,
        }
    )


def test_runner_keeps_engine_claim_active_without_a_terminal_receipt(monkeypatch):
    jobs = FakeJobs(fail_finish=True)
    store = FakeStore()
    client = _client(monkeypatch, jobs, store)
    monkeypatch.setattr(
        "backend.shared.process_witness.capture_process", lambda: {"pid": 123}
    )
    monkeypatch.setattr(
        "backend.services.engine.alpha_agent.factor_job_executor.execute_factor_materialization",
        lambda request, **kwargs: {"status": "materialized", "row_count": 1},
    )

    response = client.post(
        "/api/v1/internal/alpha-factor-jobs/run",
        json=RUN_BODY,
        headers=OWNER_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "running"}
    assert not store.finished
