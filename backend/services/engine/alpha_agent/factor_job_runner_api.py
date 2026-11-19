"""Internal engine runner for durable owner-scoped factor materialization jobs."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from backend.shared.auth import get_internal_call_secret
from backend.shared.owner_scope import OwnerScope
from backend.shared.secret_redaction import redact_text
from backend.services.engine.alpha_agent.factor_job_budget import (
    POOL,
    require_ready_budget,
)
from backend.services.engine.alpha_agent.factor_jobs import AlphaFactorJobPersistence

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/internal/alpha-factor-jobs", tags=["Internal Factor Jobs"]
)
_jobs = AlphaFactorJobPersistence()
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class FactorJobRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: StrictStr = Field(min_length=1, max_length=128)
    execution_generation: StrictInt = Field(gt=0)
    dispatch_token: StrictStr = Field(min_length=1, max_length=128)


def _owner_from_internal_headers(request: Request) -> OwnerScope:
    supplied = request.headers.get("X-Internal-Call", "")
    expected = get_internal_call_secret()
    if not expected or not hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Internal call required")
    try:
        return OwnerScope(
            request.headers.get("X-Tenant-Id", ""),
            request.headers.get("X-User-Id", ""),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Owner context is required"
        ) from exc


async def _current_job(
    job_id: str, *, owner: OwnerScope, generation: int
) -> dict[str, Any] | None:
    job = await _jobs.get(
        job_id,
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        consistent=True,
    )
    if job is None or job.get("execution_generation") != generation:
        return None
    return job


async def _watch_execution(
    *,
    job_id: str,
    generation: int,
    dispatch_token: str,
    owner: OwnerScope,
    budget,
    lease,
    cancel_event: threading.Event,
    stop_event: asyncio.Event,
) -> None:
    """Observe cancellation and renew the existing lease while the thread runs."""
    ticks = 0
    while not stop_event.is_set():
        try:
            job = await _jobs.get(
                job_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                consistent=True,
            )
            snapshot = await asyncio.to_thread(
                budget.store.task_snapshot,
                POOL,
                job_id,
                owner.tenant_id,
                owner.user_id,
            )
            if (
                job is None
                or job.get("execution_generation") != generation
                or job.get("status") != "running"
                or snapshot is None
                or snapshot.get("state") not in {"running", "suspect"}
                or snapshot.get("stop_requested")
            ):
                cancel_event.set()

            ticks += 1
            if ticks >= 10:
                ticks = 0
                renewed = await asyncio.to_thread(budget.projection.heartbeat, lease)
                durable = False
                if renewed and not cancel_event.is_set():
                    durable = await asyncio.to_thread(
                        budget.store.transition,
                        POOL,
                        lease,
                        "running",
                        expected_dispatch_token=dispatch_token,
                    )
                if not renewed or not durable:
                    cancel_event.set()
                    try:
                        await asyncio.to_thread(
                            budget.store.transition,
                            POOL,
                            lease,
                            "suspect",
                            expected_dispatch_token=dispatch_token,
                        )
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep occupancy and stop new writes on uncertainty
            cancel_event.set()
            logger.warning(
                "[factor-job-runner] monitor failed for %s: %s",
                job_id,
                type(exc).__name__,
            )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue


async def _await_executor(task: asyncio.Task) -> Any:
    """Do not abandon a worker thread when an HTTP caller disconnects."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            # The route remains responsible for the executor until it unwinds.
            continue


async def _terminal_receipt(
    job_id: str, *, owner: OwnerScope, generation: int
) -> dict[str, str] | None:
    job = await _current_job(job_id, owner=owner, generation=generation)
    if job and job.get("status") in _TERMINAL_STATUSES:
        return {"status": str(job["status"])}
    return None


def _lifecycle_is_safe_to_finalize(lifecycle: dict[str, Any]) -> bool:
    """Require cleanup proof for any sandbox intent and stop proof for a started sandbox."""
    return lifecycle.get("cleanup_confirmed", True) and (
        not lifecycle.get("create_started") or lifecycle.get("confirmed_stopped")
    )


async def _settled_receipt(
    *,
    job_id: str,
    generation: int,
    owner: OwnerScope,
    safe_to_finalize: bool,
) -> dict[str, str]:
    """Re-read after finish to catch a concurrent cancellation transition."""
    try:
        job = await _current_job(job_id, owner=owner, generation=generation)
    except Exception as exc:
        logger.warning(
            "[factor-job-runner] settled status lookup failed for %s: %s",
            job_id,
            type(exc).__name__,
        )
        return {"status": "running"}
    if job and job.get("status") in _TERMINAL_STATUSES:
        return {"status": str(job["status"])}
    if job and job.get("status") == "stopping":
        if safe_to_finalize:
            try:
                if await _jobs.confirm_cancelled(
                    job_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    generation=generation,
                ):
                    return {"status": "cancelled"}
            except Exception as exc:
                logger.warning(
                    "[factor-job-runner] raced cancel confirmation failed for %s: %s",
                    job_id,
                    type(exc).__name__,
                )
        return {"status": "stopping"}
    return {"status": "running"}


async def _confirm_unstarted_cancel(
    *,
    job_id: str,
    generation: int,
    owner: OwnerScope,
    lease_snapshot: dict[str, Any] | None,
) -> dict[str, str]:
    """Cancel only when stop is requested and no engine execution was claimed."""
    executor = (lease_snapshot or {}).get("executor") or {}
    if (
        lease_snapshot
        and lease_snapshot.get("state") == "running"
        and lease_snapshot.get("stop_requested")
        and not executor.get("engine_claim_token")
        and not executor.get("engine_process")
    ):
        try:
            if await _jobs.confirm_cancelled(
                job_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                generation=generation,
            ):
                return {"status": "cancelled"}
        except Exception as exc:
            logger.warning(
                "[factor-job-runner] unstarted cancel confirmation failed for %s: %s",
                job_id,
                type(exc).__name__,
            )
    try:
        return await _terminal_receipt(job_id, owner=owner, generation=generation) or {
            "status": "stopping"
        }
    except Exception as exc:
        logger.warning(
            "[factor-job-runner] cancel status lookup failed for %s: %s",
            job_id,
            type(exc).__name__,
        )
        return {"status": "stopping"}


@router.post("/run")
async def run_factor_job(
    payload: FactorJobRunRequest, request: Request
) -> dict[str, str]:
    owner = _owner_from_internal_headers(request)
    try:
        record = await _jobs.get_for_replay(
            payload.job_id, dispatch_token=payload.dispatch_token
        )
    except Exception as exc:
        logger.warning(
            "[factor-job-runner] replay lookup failed for %s: %s",
            payload.job_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503, detail="Factor job lookup is unavailable"
        ) from exc

    if record is None:
        try:
            receipt = await _terminal_receipt(
                payload.job_id,
                owner=owner,
                generation=payload.execution_generation,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Factor job lookup is unavailable"
            ) from exc
        if receipt is not None:
            return receipt
        raise HTTPException(
            status_code=409, detail="Factor job is not active for this dispatch"
        )

    if (
        record.get("tenant_id") != owner.tenant_id
        or record.get("user_id") != owner.user_id
        or record.get("kind") != "materialize"
        or record.get("execution_generation") != payload.execution_generation
    ):
        raise HTTPException(
            status_code=409, detail="Factor job dispatch does not match"
        )
    if record.get("status") not in {"running", "stopping"}:
        raise HTTPException(status_code=409, detail="Factor job is not running")

    request_snapshot = record.get("request_snapshot")
    if isinstance(request_snapshot, str):
        try:
            request_snapshot = json.loads(request_snapshot)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail="Factor job snapshot is invalid"
            ) from exc
    request_body = (
        request_snapshot.get("request") if isinstance(request_snapshot, dict) else None
    )
    if not isinstance(request_body, dict):
        raise HTTPException(status_code=409, detail="Factor job snapshot is invalid")

    try:
        budget = await asyncio.to_thread(require_ready_budget)
        lease = await asyncio.to_thread(
            budget.store.get_lease,
            POOL,
            payload.job_id,
            owner.tenant_id,
            owner.user_id,
            expected_dispatch_token=payload.dispatch_token,
        )
        lease_snapshot = await asyncio.to_thread(
            budget.store.task_snapshot,
            POOL,
            payload.job_id,
            owner.tenant_id,
            owner.user_id,
        )
    except Exception as exc:
        logger.warning(
            "[factor-job-runner] lease lookup failed for %s: %s",
            payload.job_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503, detail="Factor job lease is unavailable"
        ) from exc
    if (
        lease is None
        or lease_snapshot is None
        or lease_snapshot.get("state") != "running"
        or (lease_snapshot.get("executor") or {}).get("dispatch_token")
        != payload.dispatch_token
    ):
        raise HTTPException(status_code=409, detail="Factor job lease does not match")
    if record.get("status") == "stopping" or lease_snapshot.get("stop_requested"):
        return await _confirm_unstarted_cancel(
            job_id=payload.job_id,
            generation=payload.execution_generation,
            owner=owner,
            lease_snapshot=lease_snapshot,
        )

    from backend.services.engine.alpha_agent.factor_job_executor import (
        FactorMaterializationCancelled,
        execute_factor_materialization,
    )
    from backend.services.engine.strategy_lab.runner.container_runner import (
        SandboxExecutionCancelled,
    )
    from backend.shared.process_witness import capture_process

    request_id = uuid.uuid4().hex
    try:
        claim = await asyncio.to_thread(
            budget.store.claim_engine_execution,
            POOL,
            lease,
            dispatch_token=payload.dispatch_token,
            request_id=request_id,
            engine_process=capture_process(),
            execution_generation=payload.execution_generation,
        )
    except Exception as exc:
        logger.warning(
            "[factor-job-runner] execution claim failed for %s: %s",
            payload.job_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503, detail="Factor execution claim is unavailable"
        ) from exc
    if claim == "existing_same":
        return {"status": "running"}
    if claim == "stop_requested":
        fresh_snapshot = await asyncio.to_thread(
            budget.store.task_snapshot,
            POOL,
            payload.job_id,
            owner.tenant_id,
            owner.user_id,
        )
        return await _confirm_unstarted_cancel(
            job_id=payload.job_id,
            generation=payload.execution_generation,
            owner=owner,
            lease_snapshot=fresh_snapshot,
        )
    if claim != "claimed":
        raise HTTPException(status_code=409, detail="Factor execution is fenced")

    loop = asyncio.get_running_loop()
    cancel_event = threading.Event()
    lifecycle: dict[str, Any] = {
        "create_started": False,
        "confirmed_stopped": False,
        "cleanup_confirmed": True,
    }
    stop_monitor = asyncio.Event()

    def thread_progress(snapshot: dict[str, Any]) -> None:
        future = asyncio.run_coroutine_threadsafe(
            _jobs.update_progress(
                payload.job_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                generation=payload.execution_generation,
                progress=snapshot,
            ),
            loop,
        )
        try:
            if not future.result(timeout=15):
                cancel_event.set()
        except Exception as exc:
            cancel_event.set()
            logger.warning(
                "[factor-job-runner] progress update failed for %s: %s",
                payload.job_id,
                type(exc).__name__,
            )

    monitor = asyncio.create_task(
        _watch_execution(
            job_id=payload.job_id,
            generation=payload.execution_generation,
            dispatch_token=payload.dispatch_token,
            owner=owner,
            budget=budget,
            lease=lease,
            cancel_event=cancel_event,
            stop_event=stop_monitor,
        )
    )
    operation = asyncio.create_task(
        asyncio.to_thread(
            execute_factor_materialization,
            request_body,
            tenant_id=owner.tenant_id,
            user_id=owner.user_id,
            lease=lease,
            budget=budget,
            lifecycle=lifecycle,
            progress=thread_progress,
            should_cancel=cancel_event.is_set,
        )
    )
    result: dict[str, Any] | None = None
    execution_error: Exception | None = None
    try:
        try:
            result = await _await_executor(operation)
        except Exception as exc:
            execution_error = exc
    finally:
        stop_monitor.set()
        try:
            await asyncio.wait_for(monitor, timeout=5)
        except asyncio.TimeoutError:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    if execution_error is None and not isinstance(result, dict):
        execution_error = RuntimeError("Factor materialization returned no result")

    safe_to_finalize = _lifecycle_is_safe_to_finalize(lifecycle)
    try:
        current = await _current_job(
            payload.job_id,
            owner=owner,
            generation=payload.execution_generation,
        )
    except Exception as exc:
        current = None
        logger.warning(
            "[factor-job-runner] final status lookup failed for %s: %s",
            payload.job_id,
            type(exc).__name__,
        )
    if current and current.get("status") in _TERMINAL_STATUSES:
        receipt = {"status": str(current["status"])}
    elif current and current.get("status") == "stopping":
        if safe_to_finalize:
            try:
                if await _jobs.confirm_cancelled(
                    payload.job_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    generation=payload.execution_generation,
                ):
                    receipt = {"status": "cancelled"}
                else:
                    receipt = {"status": "stopping"}
            except Exception as exc:
                logger.warning(
                    "[factor-job-runner] cancel confirmation failed for %s: %s",
                    payload.job_id,
                    type(exc).__name__,
                )
                receipt = {"status": "stopping"}
        else:
            receipt = {"status": "stopping"}
    elif not safe_to_finalize:
        # Preserve the lease and engine witness for the existing fail-closed reconciler.
        receipt = {"status": "running"}
    elif execution_error is not None:
        try:
            await _jobs.finish(
                payload.job_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                generation=payload.execution_generation,
                error_code=type(execution_error).__name__,
                error_message=redact_text(str(execution_error))[:2000],
            )
        except Exception as exc:
            logger.warning(
                "[factor-job-runner] failure receipt could not be saved for %s: %s",
                payload.job_id,
                type(exc).__name__,
            )
        receipt = await _settled_receipt(
            job_id=payload.job_id,
            owner=owner,
            generation=payload.execution_generation,
            safe_to_finalize=safe_to_finalize,
        )
    else:
        try:
            await _jobs.finish(
                payload.job_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                generation=payload.execution_generation,
                result=result,
            )
        except Exception as exc:
            logger.warning(
                "[factor-job-runner] completion receipt could not be saved for %s: %s",
                payload.job_id,
                type(exc).__name__,
            )
        receipt = await _settled_receipt(
            job_id=payload.job_id,
            owner=owner,
            generation=payload.execution_generation,
            safe_to_finalize=safe_to_finalize,
        )

    if receipt.get("status") in _TERMINAL_STATUSES:
        try:
            await asyncio.to_thread(
                budget.store.mark_engine_request_finished,
                POOL,
                lease,
                dispatch_token=payload.dispatch_token,
                request_id=request_id,
                execution_generation=payload.execution_generation,
            )
        except Exception as exc:
            logger.warning(
                "[factor-job-runner] execution witness close failed for %s: %s",
                payload.job_id,
                type(exc).__name__,
            )

    return receipt
