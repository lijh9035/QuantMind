"""Trusted QuantMind adapter for the isolated AlphaSAGE HTTP worker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from backend.shared.owner_scope import OwnerScope
from backend.shared.secret_redaction import redact_text
from backend.services.engine.alpha_agent.alphasage.client import (
    AlphaSageClient,
    AlphaSageWorkerError,
    AlphaSageWorkerNotFound,
    AlphaSageWorkerUnavailable,
)
from backend.services.engine.alpha_agent.task_spec import (
    ALPHASAGE_VERSION,
    MiningTaskSpec,
)

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 128
_MAX_EXPRESSION_LENGTH = 8192
_TERMINAL_GPU_STATES = frozenset({"completed", "failed", "cancelled"})
_Qlib_SYMBOL_RE = re.compile(r"(?:sh|sz|bj)[0-9]{6}\Z")


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"Persisted AlphaSAGE {label} is invalid")
    return value


def _trusted_task(
    record: dict[str, Any],
) -> tuple[OwnerScope, dict, dict, dict, str]:
    if not isinstance(record, dict):
        raise ValueError("Persisted AlphaSAGE task is invalid")
    owner = OwnerScope(record.get("tenant_id"), record.get("user_id"))
    if any(record.get(key) != value for key, value in owner.sql_params().items()):
        raise ValueError("Persisted AlphaSAGE owner is not canonical")
    if record.get("algorithm") != "alphasage":
        raise ValueError("Persisted task is not an AlphaSAGE task")
    if record.get("algorithm_version") != ALPHASAGE_VERSION:
        raise ValueError("Persisted AlphaSAGE algorithm version is unsupported")
    request = _json_object(record.get("request_snapshot"), "request snapshot")
    params = _json_object(record.get("params_snapshot"), "parameter snapshot")
    data_snapshot = _json_object(record.get("data_snapshot") or {}, "data snapshot")
    if set(data_snapshot) - {"resolved_instruments"}:
        raise ValueError("Persisted AlphaSAGE data snapshot contains unsupported fields")
    resolved_instruments = data_snapshot.get("resolved_instruments")
    if resolved_instruments is not None and (
        not isinstance(resolved_instruments, list)
        or not 1 <= len(resolved_instruments) <= 500
        or any(
            not isinstance(symbol, str) or not _Qlib_SYMBOL_RE.fullmatch(symbol)
            for symbol in resolved_instruments
        )
        or len(set(resolved_instruments)) != len(resolved_instruments)
        or resolved_instruments != sorted(resolved_instruments)
    ):
        raise ValueError("Persisted AlphaSAGE custom instruments are invalid")
    if request.get("algorithm") != "alphasage" or params.get("algorithm") != "alphasage":
        raise ValueError("Persisted AlphaSAGE specification algorithm does not match")
    spec = MiningTaskSpec.build(
        algorithm="alphasage",
        algorithm_version=ALPHASAGE_VERSION,
        request=request,
        params=params,
        data_snapshot=data_snapshot,
    )
    execution_fingerprint = record.get("execution_fingerprint")
    if (
        not isinstance(execution_fingerprint, str)
        or spec.execution_fingerprint != execution_fingerprint
    ):
        raise ValueError("Persisted AlphaSAGE execution fingerprint does not match")
    if record.get("task_id") is None:
        raise ValueError("Persisted AlphaSAGE task ID is missing")
    return owner, request, params, data_snapshot, execution_fingerprint


def _qlib_provider_uri() -> str:
    """Resolve QuantMind's configured local CN provider; never accept a request path."""
    from backend.services.engine.rd_agent.market_adapters import get_adapter

    uri = str(get_adapter("a_share").get_qlib_provider_uri() or "").strip()
    if not uri or not Path(uri).is_absolute():
        raise ValueError("QuantMind Qlib provider path is unavailable")
    if not Path(uri).is_dir():
        raise ValueError("QuantMind Qlib provider data is unavailable")
    return uri


def _evaluate_quantmind_factor(
    expression: str,
    *,
    params: dict[str, Any],
    qlib_dir: str,
    frozen_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate signed training and validation RankIC with the shared FM runtime."""
    import pandas as pd

    from backend.services.engine.alpha_agent.factorminer import dsl
    from backend.services.engine.alpha_agent.factorminer.runtime import evaluate_window
    from backend.services.engine.alpha_agent.statistical import panel as panel_mod
    from backend.services.engine.alpha_agent.statistical.ic_stats import (
        DEFAULT_MIN_NAMES_PER_DAY,
        daily_ic,
        forward_returns,
    )

    universe = str(params["universe"])
    symbols = frozen_symbols or panel_mod.qlib_instruments(
        market=universe, qlib_dir=qlib_dir
    )
    if not symbols:
        raise ValueError("QuantMind Qlib universe resolved to no instruments")

    mining_start = str(params["mining_start"])
    mining_end = str(params["mining_end"])
    valid_start = str(params["valid_start"])
    valid_end = str(params["valid_end"])
    horizon = int(params["horizon"])
    warmup = dsl.min_history(dsl.parse(expression.removeprefix("fm:")))
    extension_end = (
        date.fromisoformat(valid_end) + timedelta(days=horizon * 4 + 14)
    ).isoformat()
    price_panel = panel_mod.load_price_panel(
        mining_start,
        extension_end,
        market=universe,
        symbols=symbols,
        warmup_bars=warmup,
        qlib_dir=qlib_dir,
    )
    if price_panel.empty:
        raise ValueError("QuantMind Qlib price panel is empty")
    factor_values = evaluate_window(
        expression,
        instruments=symbols,
        start=mining_start,
        end=valid_end,
        qlib_dir=qlib_dir,
    )

    dates = price_panel.index.get_level_values("trade_date")
    training_panel = price_panel.loc[dates <= pd.Timestamp(mining_end)]
    training_returns = forward_returns(training_panel, horizon)
    training_factor = factor_values.reindex(training_returns.index)
    training_mask = (
        (training_returns.index.get_level_values("trade_date") >= pd.Timestamp(mining_start))
        & (training_returns.index.get_level_values("trade_date") <= pd.Timestamp(mining_end))
    )
    training_ic = daily_ic(
        training_factor.loc[training_mask],
        training_returns.loc[training_mask],
        min_names=DEFAULT_MIN_NAMES_PER_DAY,
    )

    validation_returns = forward_returns(price_panel, horizon)
    validation_factor = factor_values.reindex(validation_returns.index)
    validation_mask = (
        (validation_returns.index.get_level_values("trade_date") >= pd.Timestamp(valid_start))
        & (validation_returns.index.get_level_values("trade_date") <= pd.Timestamp(valid_end))
    )
    validation_ic = daily_ic(
        validation_factor.loc[validation_mask],
        validation_returns.loc[validation_mask],
        min_names=DEFAULT_MIN_NAMES_PER_DAY,
    )

    def summarize(frame) -> dict[str, Any]:
        if frame.empty:
            return {"ic": None, "rank_ic": None, "effective_days": 0}
        return {
            "ic": float(frame["ic"].mean()),
            "rank_ic": float(frame["rank_ic"].mean()),
            "effective_days": int(len(frame)),
        }

    return {
        "horizon": horizon,
        "training": summarize(training_ic),
        "validation": summarize(validation_ic),
        "training_window": [mining_start, mining_end],
        "validation_window": [valid_start, valid_end],
        "validation_used_for_search": False,
        "universe": universe,
        "instrument_count": len(symbols),
    }


def _candidate_to_factor(
    *, candidate: dict[str, Any], params: dict[str, Any], task_id: str,
    execution_fingerprint: str, algorithm_version: str, qlib_dir: str,
    frozen_symbols: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Convert and compare one GPU candidate before evaluating or persisting it."""
    from backend.services.engine.alpha_agent.alphasage.expression_converter import (
        convert_expression,
        evaluate_semantic_check,
    )

    source_index = candidate.get("source_index")
    raw_expression = candidate.get("raw_expression")
    if type(source_index) is not int or not 0 <= source_index < _MAX_CANDIDATES:
        return None, {"source_index": source_index, "reason": "invalid_source_index"}
    if (
        not isinstance(raw_expression, str)
        or not raw_expression.strip()
        or len(raw_expression) > _MAX_EXPRESSION_LENGTH
    ):
        return None, {"source_index": source_index, "reason": "invalid_expression"}

    converted = convert_expression(raw_expression)
    converted = asdict(converted) if is_dataclass(converted) else dict(converted)
    canonical = converted.get("canonical_expression")
    if (
        converted.get("structurally_supported", converted.get("accepted"))
        is not True
        or not isinstance(canonical, str)
    ):
        return None, {
            "source_index": source_index,
            "raw_expression": raw_expression[:512],
            "reason": str(converted.get("reject_reason") or "unsupported_expression")[:512],
        }

    source_check = candidate.get("semantic_check")
    if not isinstance(source_check, dict):
        return None, {"source_index": source_index, "reason": "missing_source_values"}
    dates = source_check.get("dates")
    symbols = source_check.get("symbols")
    source_values = source_check.get("source_values")
    try:
        semantic = evaluate_semantic_check(
            canonical,
            dates=dates,
            symbols=symbols,
            source_values=source_values,
            qlib_dir=qlib_dir,
            atol=1e-5,
            rtol=1e-4,
            training_start=str(params["mining_start"]),
            training_end=str(params["mining_end"]),
        )
        semantic = asdict(semantic) if is_dataclass(semantic) else dict(semantic)
    except Exception as exc:  # noqa: BLE001 -- reject malformed or unevaluable candidate only
        return None, {
            "source_index": source_index,
            "raw_expression": raw_expression[:512],
            "reason": f"semantic_evaluation_failed:{type(exc).__name__}",
        }
    if semantic.get("passed") is not True:
        return None, {
            "source_index": source_index,
            "raw_expression": raw_expression[:512],
            "canonical_expression": canonical,
            "reason": str(semantic.get("reject_reason") or "semantic_mismatch")[:512],
            "semantic_check": semantic,
        }

    evaluation = _evaluate_quantmind_factor(
        canonical,
        params=params,
        qlib_dir=qlib_dir,
        frozen_symbols=frozen_symbols,
    )
    training = evaluation["training"]
    factor_id = f"as-{task_id}-{source_index}"
    metadata = {
        "task_id": task_id,
        "algorithm": "alphasage",
        "algorithm_version": algorithm_version,
        "execution_fingerprint": execution_fingerprint,
        "source_index": source_index,
        "source_expression": raw_expression,
        "canonical_expression": canonical,
        "converter_backend": converted.get("backend"),
        "semantic_check": semantic,
        "training": evaluation["training"],
        "validation": evaluation["validation"],
        "horizon": evaluation["horizon"],
        "training_window": evaluation["training_window"],
        "validation_window": evaluation["validation_window"],
        "validation_used_for_search": False,
        "universe": evaluation["universe"],
        "instrument_count": evaluation["instrument_count"],
        "search_parameters": {
            "n_episodes": params["n_episodes"],
            "pool_capacity": params["pool_capacity"],
            "encoder_type": params["encoder_type"],
            "seed": params["seed"],
        },
    }
    factor = {
        "factor_id": factor_id,
        "factor_name": f"AlphaSAGE {source_index + 1}",
        "factor_code": canonical,
        "rank_ic": training.get("rank_ic"),
        "factor_formulation": canonical.removeprefix("fm:"),
        "market": "a_share",
        "universe": str(params["universe"]),
        "date_range": f"{params['mining_start']}~{params['mining_end']}",
        "data_source": "qlib_bin",
        "metadata": metadata,
    }
    return factor, {
        "source_index": source_index,
        "factor_id": factor_id,
        "factor_formulation": factor["factor_formulation"],
        "training": evaluation["training"],
        "validation": evaluation["validation"],
        "semantic_check": semantic,
    }


def _validate_result(
    result: dict[str, Any],
    *,
    task_id: str,
    fingerprint: str,
    params: dict[str, Any],
    qlib_dir: str | None = None,
    calendar_fn=None,
) -> list[dict[str, Any]]:
    if (
        result.get("schema_version") != 1
        or result.get("task_id") != task_id
        or result.get("algorithm") != "alphasage"
        or result.get("execution_fingerprint") != fingerprint
        or result.get("status", "completed") != "completed"
    ):
        raise ValueError("AlphaSAGE result identity or status does not match task")
    if result.get("algorithm_version") != ALPHASAGE_VERSION:
        raise ValueError("AlphaSAGE result algorithm version does not match")
    training = result.get("training_window")
    expected_horizon = params.get("horizon")
    if (
        not isinstance(training, dict)
        or training.get("requested_start") != params["mining_start"]
        or training.get("requested_end") != params["mining_end"]
        or type(training.get("horizon")) is not int
        or training.get("horizon") != expected_horizon
    ):
        raise ValueError("AlphaSAGE training window does not match task")
    mature_label_end = training.get("mature_label_end")
    try:
        mature_date = date.fromisoformat(mature_label_end)
        requested_start = date.fromisoformat(str(params["mining_start"]))
        if calendar_fn is None:
            if not qlib_dir:
                raise ValueError("QuantMind Qlib calendar is required")
            from backend.services.engine.alpha_agent.statistical.panel import (
                qlib_calendar,
            )

            calendar_fn = lambda start, end: qlib_calendar(  # noqa: E731
                start, end, qlib_dir=qlib_dir
            )
        calendar = calendar_fn("1900-01-01", str(params["mining_end"]))
        horizon = int(expected_horizon)
        if len(calendar) <= horizon:
            raise ValueError("Qlib calendar has too few sessions for mature labels")
        mature_upper = date.fromisoformat(
            str(calendar[-horizon - 1])[:10]
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("AlphaSAGE mature label end is invalid") from exc
    if mature_date < requested_start or mature_date > mature_upper:
        raise ValueError("AlphaSAGE mature labels exceed the Qlib training limit")

    expected_validation = [params["valid_start"], params["valid_end"]]
    validation = result.get("validation_window")
    if not isinstance(validation, dict) or (
        validation.get("start"), validation.get("end")
    ) != tuple(expected_validation):
        raise ValueError("AlphaSAGE validation window does not match task")
    if validation.get("evaluated") is not False:
        raise ValueError("AlphaSAGE search must not use validation data")
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > _MAX_CANDIDATES:
        raise ValueError("AlphaSAGE candidate list exceeds its supported bound")
    candidate_count = result.get("candidate_count")
    if (
        type(candidate_count) is not int
        or candidate_count != len(candidates)
    ):
        raise ValueError("AlphaSAGE candidate count does not match result")
    seen_indices: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source_index = candidate.get("source_index")
        if type(source_index) is int and source_index in seen_indices:
            raise ValueError("AlphaSAGE result contains duplicate source indices")
        if type(source_index) is int:
            seen_indices.add(source_index)
    return candidates


def _write_progress_log(launcher, owner: OwnerScope, task_id: str, payload: dict) -> None:
    directory = launcher.task_dir(task_id, owner.tenant_id, owner.user_id)
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with (directory / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with (directory / "subprocess_stdout.log").open("a", encoding="utf-8") as handle:
        handle.write(str(payload.get("message") or payload.get("progress") or "") + "\n")


def _progress_percent(payload: dict[str, Any], previous: int) -> int:
    value = payload.get("progress_pct")
    if type(value) is not int:
        return previous
    return min(99, max(previous, max(0, value)))


async def _record_progress(store, launcher, owner, task_id, payload, previous):
    percent = _progress_percent(payload, previous)
    phase = str(payload.get("phase") or "train")[:64]
    message = str(payload.get("message") or "AlphaSAGE GPU 任务运行中")[:1000]
    snapshot = {
        "algorithm": "alphasage",
        "phase": phase,
        "progress": message,
        "progress_pct": percent,
        "candidate_count": payload.get("candidate_count"),
        "updated_at": payload.get("updated_at"),
    }
    await store.update_progress(
        task_id,
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        snapshot=snapshot,
    )
    await asyncio.to_thread(
        _write_progress_log, launcher, owner, task_id, {**snapshot, "message": message}
    )
    task = launcher._tasks.get(task_id)
    if task is not None:
        task.phase = phase
        task.progress = message
        task.progress_pct = percent
        task.mining_progress = snapshot
    return percent


async def _cancel_and_confirm(client, store, owner, task_id, *, timeout_seconds=90):
    try:
        await client.cancel(task_id)
    except AlphaSageWorkerNotFound:
        # A missing durable worker record means there is no child process left
        # to protect the execution lease from.
        await store.confirm_cancelled(
            task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
        )
        return "cancelled"
    except AlphaSageWorkerError:
        logger.warning("AlphaSAGE cancellation request could not be confirmed for %s", task_id)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            state = await client.status(task_id)
        except AlphaSageWorkerError:
            await asyncio.sleep(2)
            continue
        gpu_status = str(state.get("status") or "").lower()
        if gpu_status in _TERMINAL_GPU_STATES:
            if gpu_status == "cancelled":
                await store.confirm_cancelled(
                    task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
                )
                return "cancelled"
            return gpu_status
        await asyncio.sleep(2)
    return "stopping"


async def recover_orphaned_execution(
    store,
    task_id: str,
    owner: OwnerScope,
    *,
    client: AlphaSageClient | None = None,
) -> str | None:
    """Stop an AlphaSAGE job whose QuantMind engine executor disappeared.

    A failed DELETE/network check is not evidence that the GPU process ended;
    callers must retain the resource lease and retry recovery later.
    """
    from backend.services.engine.alpha_agent.resource_budget import finish_execution

    terminal_states = {"completed", "failed", "cancelled"}
    row = await store.get(
        task_id,
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        consistent=True,
    )
    if row is None:
        return None
    if row.get("status") in terminal_states:
        finished = await asyncio.to_thread(
            finish_execution,
            task_id,
            owner.tenant_id,
            owner.user_id,
            str(row["status"]),
        )
        return str(row["status"]) if finished else None

    prior_status = str(row.get("status") or "")
    if prior_status in {"pending", "queued"}:
        state = await store.request_cancel(
            task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
        )
        if state == "cancelled":
            finished = await asyncio.to_thread(
                finish_execution,
                task_id,
                owner.tenant_id,
                owner.user_id,
                "cancelled",
            )
            return "cancelled" if finished else None
    elif prior_status != "stopping":
        state = await store.request_cancel(
            task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
        )
        if state is None:
            latest = await store.get(
                task_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                consistent=True,
            )
            if latest and latest.get("status") in terminal_states:
                return await recover_orphaned_execution(
                    store, task_id, owner, client=client
                )
            return None

    worker = client or AlphaSageClient()
    try:
        receipt = await worker.cancel(task_id)
    except AlphaSageWorkerNotFound:
        # The worker's persistent jobs directory has no accepted task/process.
        # Preserve an explicit owner stop as cancelled; otherwise record that
        # the engine vanished before the task could produce a result.
        if prior_status == "stopping":
            await store.confirm_cancelled(
                task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
            )
            final_state = "cancelled"
        else:
            await store.update(
                task_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                expected_status=("stopping",),
                status="failed",
                phase="failed",
                progress="AlphaSAGE GPU worker has no accepted task",
                error_message="Engine stopped before the GPU job was accepted",
            )
            final_state = "failed"
    except AlphaSageWorkerError as exc:
        logger.warning(
            "Cannot yet recover AlphaSAGE GPU process %s (%s)",
            task_id,
            type(exc).__name__,
        )
        return None
    else:
        if (
            receipt.get("task_id") != task_id
            or str(receipt.get("status") or "").lower() not in terminal_states
        ):
            return None
        # Engine death loses the in-memory conversion/evaluation context. Even
        # if the worker finished, do not complete or persist its unverified data.
        await store.confirm_cancelled(
            task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
        )
        final_state = "cancelled"

    finished = await asyncio.to_thread(
        finish_execution,
        task_id,
        owner.tenant_id,
        owner.user_id,
        final_state,
    )
    return final_state if finished else None


async def _submit_idempotently(client, payload):
    """Resolve a timed-out POST by reusing the same durable worker task ID."""
    task_id = str(payload["task_id"])
    while True:
        try:
            return await client.submit(payload)
        except AlphaSageWorkerUnavailable:
            await asyncio.sleep(2)
            try:
                state = await client.status(task_id)
            except AlphaSageWorkerError:
                # A 404 means POST may be safely retried with the same task ID.
                continue
            if state.get("task_id") == task_id:
                return state


async def run_alphasage_queued(
    record: dict[str, Any], *, client: AlphaSageClient | None = None
) -> None:
    """Run a previously admitted task through its existing lease and DB queue."""
    from backend.services.engine.alpha_agent.launcher import (
        EvolutionTask,
        TaskStatus,
        get_launcher,
    )
    from backend.services.engine.alpha_agent.resource_budget import (
        begin_execution,
        finish_execution,
    )
    from backend.services.engine.alpha_agent.queue import capture_controller
    from backend.services.engine.qlib_app.services.rd_agent_persistence import (
        AlphaMiningTaskPersistence,
    )

    owner, request, params, data_snapshot, fingerprint = _trusted_task(record)
    task_id = str(record["task_id"])
    store = AlphaMiningTaskPersistence()
    launcher = get_launcher()
    task = EvolutionTask(
        task_id=task_id,
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        market="a_share",
        universe=str(params["universe"]),
        loop_n=3,
        data_source="qlib_bin",
        mining_start=str(params["mining_start"]),
        mining_end=str(params["mining_end"]),
        valid_start=str(params["valid_start"]),
        valid_end=str(params["valid_end"]),
        status=TaskStatus.QUEUED,
        phase="queued",
        progress="等待 AlphaSAGE GPU worker",
    )
    task.algorithm = "alphasage"
    launcher._tasks[task_id] = task
    worker = client or AlphaSageClient()
    qlib_dir = _qlib_provider_uri()
    witness = await asyncio.to_thread(capture_controller)
    previous_progress = 0
    submitted = False

    try:
        with begin_execution(
            task_id,
            owner.tenant_id,
            owner.user_id,
            executor_kind="engine-local",
            witness={**witness, "alphasage_worker": True, "alphasage_task_id": task_id},
        ):
            claimed = await store.update(
                task_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                expected_status=("pending", "queued"),
                status="running",
                phase="starting",
                progress="正在提交 AlphaSAGE GPU 任务",
                progress_pct=1,
                error_message=None,
            )
            if not claimed:
                return
            task.status = TaskStatus.RUNNING
            task.phase = "starting"
            task.progress = "正在提交 AlphaSAGE GPU 任务"
            task.progress_pct = 1
            try:
                await worker.health()
            except AlphaSageWorkerUnavailable as exc:
                await store.update(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    expected_status=("running",),
                    status="failed",
                    error_message=redact_text(str(exc))[:2000],
                    progress="AlphaSAGE GPU worker 不可用",
                    phase="failed",
                )
                task.status = TaskStatus.FAILED
                task.error_message = redact_text(str(exc))[:2000]
                return
            payload = {
                "task_id": task_id,
                "execution_fingerprint": fingerprint,
                "mining_start": params["mining_start"],
                "mining_end": params["mining_end"],
                "valid_start": params["valid_start"],
                "valid_end": params["valid_end"],
                "horizon": params["horizon"],
                "n_episodes": params["n_episodes"],
                "pool_capacity": params["pool_capacity"],
                "encoder_type": params["encoder_type"],
                "seed": params["seed"],
            }
            custom_instruments = data_snapshot.get("resolved_instruments")
            if custom_instruments is not None:
                payload["instruments"] = custom_instruments
            else:
                payload["instrument"] = params["universe"]
            receipt = await _submit_idempotently(worker, payload)
            if receipt.get("task_id") != task_id:
                raise AlphaSageWorkerError("AlphaSAGE worker task ID does not match")
            submitted = True

            while True:
                current = await store.get(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    consistent=True,
                )
                if current is None:
                    await _cancel_and_confirm(worker, store, owner, task_id)
                    raise ValueError("AlphaSAGE owner-scoped task disappeared")
                if current.get("status") == "stopping":
                    terminal = await _cancel_and_confirm(worker, store, owner, task_id)
                    if terminal == "cancelled":
                        task.status = TaskStatus.CANCELLED
                        task.phase = "cancelled"
                        task.progress = "任务已取消，AlphaSAGE GPU 进程已退出"
                        return
                    if terminal in {"completed", "failed"}:
                        # Stop was requested before the result could be committed;
                        # do not persist a result after the owner requested cancel.
                        await store.confirm_cancelled(
                            task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
                        )
                        task.status = TaskStatus.CANCELLED
                        task.phase = "cancelled"
                        return
                    await asyncio.sleep(2)
                    continue

                try:
                    state = await worker.status(task_id)
                except AlphaSageWorkerUnavailable as exc:
                    message = f"等待 AlphaSAGE GPU worker 恢复：{type(exc).__name__}"
                    previous_progress = await _record_progress(
                        store,
                        launcher,
                        owner,
                        task_id,
                        {"phase": "worker_wait", "message": message},
                        previous_progress,
                    )
                    await asyncio.sleep(5)
                    continue
                gpu_status = str(state.get("status") or "").lower()
                previous_progress = await _record_progress(
                    store, launcher, owner, task_id, state, previous_progress
                )
                if gpu_status in {"queued", "running", "stopping"}:
                    await asyncio.sleep(
                        min(30.0, max(1.0, float(os.getenv("ALPHASAGE_POLL_SECONDS", "2"))))
                    )
                    continue
                if gpu_status == "cancelled":
                    await store.update(
                        task_id,
                        tenant_id=owner.tenant_id,
                        user_id=owner.user_id,
                        expected_status=("running", "stopping"),
                        status="cancelled",
                        phase="cancelled",
                        progress="AlphaSAGE GPU 任务已取消",
                        error_message="Cancelled by user",
                    )
                    task.status = TaskStatus.CANCELLED
                    task.phase = "cancelled"
                    return
                if gpu_status == "failed":
                    error = redact_text(str(state.get("error") or "AlphaSAGE GPU task failed"))[:2000]
                    await store.update(
                        task_id,
                        tenant_id=owner.tenant_id,
                        user_id=owner.user_id,
                        expected_status=("running",),
                        status="failed",
                        phase="failed",
                        error_message=error,
                        progress="AlphaSAGE GPU 任务失败",
                    )
                    task.status = TaskStatus.FAILED
                    task.error_message = error
                    return
                if gpu_status != "completed":
                    raise AlphaSageWorkerError("AlphaSAGE worker returned an unknown task state")

                # The result is fetched only after the GPU job is terminal. Numeric
                # semantic comparison uses the training-only worker sample; only
                # QuantMind's own post-search evaluation reads the validation window.
                result = await worker.result(task_id)
                candidates = _validate_result(
                    result,
                    task_id=task_id,
                    fingerprint=fingerprint,
                    params=params,
                    qlib_dir=qlib_dir,
                )
                factors: list[dict[str, Any]] = []
                summaries: list[dict[str, Any]] = []
                rejections: list[dict[str, Any]] = []
                seen_expressions: set[str] = set()
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        rejections.append({"reason": "invalid_candidate_record"})
                        continue
                    factor, summary = await asyncio.to_thread(
                        _candidate_to_factor,
                        candidate=candidate,
                        params=params,
                        task_id=task_id,
                        execution_fingerprint=fingerprint,
                        algorithm_version=ALPHASAGE_VERSION,
                        qlib_dir=qlib_dir,
                        frozen_symbols=custom_instruments,
                    )
                    if factor is None:
                        rejections.append(summary)
                        continue
                    expression = factor["factor_code"]
                    if expression in seen_expressions:
                        rejections.append(
                            {
                                "source_index": candidate.get("source_index"),
                                "reason": "duplicate_canonical_expression",
                            }
                        )
                        continue
                    seen_expressions.add(expression)
                    factors.append(factor)
                    summaries.append(summary)

                result_snapshot = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "algorithm": "alphasage",
                    "algorithm_version": ALPHASAGE_VERSION,
                    "execution_fingerprint": fingerprint,
                    "status": "completed" if factors else "completed_with_zero_factors",
                    "training_window": result["training_window"],
                    "validation_window": result["validation_window"],
                    # Persist only the frozen request fields. The worker's
                    # optional provider/runtime metadata is never copied into
                    # the owner-visible result record.
                    "parameters": {
                        key: params[key]
                        for key in (
                            "universe", "horizon", "n_episodes", "pool_capacity",
                            "encoder_type", "seed", "mining_start", "mining_end",
                            "valid_start", "valid_end",
                        )
                    },
                    "candidate_count": len(candidates),
                    "converted_and_evaluated_count": len(factors),
                    "persisted_count": len(factors),
                    "rejected_count": len(rejections),
                    "candidates": summaries,
                    "rejections": rejections[:_MAX_CANDIDATES],
                    "validation_used_for_search": False,
                }
                # The single transaction verifies task owner, algorithm and
                # execution fingerprint, then inserts factors/result and completes.
                await store.complete_alphasage_with_result(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    algorithm_version=ALPHASAGE_VERSION,
                    execution_fingerprint=fingerprint,
                    result=result_snapshot,
                    factors=factors,
                )
                task.status = TaskStatus.COMPLETED
                task.phase = "completed"
                task.progress = "AlphaSAGE 因子转换、评价与入库完成"
                task.progress_pct = 100
                task.result = result_snapshot
                return
    except Exception as exc:  # noqa: BLE001 -- task errors become durable terminal state
        logger.exception("AlphaSAGE queued task failed: %s", task_id)
        remote_terminal = None
        if submitted:
            # A result/validation exception can happen after GPU completion. Ask
            # the worker to stop and wait for a terminal receipt before releasing
            # the single-slot lease; never infer remote process exit locally.
            while remote_terminal in {None, "stopping"}:
                remote_terminal = await _cancel_and_confirm(
                    worker, store, owner, task_id, timeout_seconds=90
                )
                if remote_terminal == "stopping":
                    await asyncio.sleep(5)
        latest = await store.get(
            task_id, tenant_id=owner.tenant_id, user_id=owner.user_id, consistent=True
        )
        if latest and latest.get("status") == "stopping":
            await store.confirm_cancelled(
                task_id, tenant_id=owner.tenant_id, user_id=owner.user_id
            )
            task.status = TaskStatus.CANCELLED
            task.phase = "cancelled"
            await asyncio.to_thread(
                finish_execution,
                task_id,
                owner.tenant_id,
                owner.user_id,
                "cancelled",
            )
            return
        if latest and latest.get("status") in {"pending", "queued", "running"}:
            message = redact_text(f"{type(exc).__name__}: {exc}")[:2000]
            await store.update(
                task_id,
                tenant_id=owner.tenant_id,
                user_id=owner.user_id,
                expected_status=("pending", "queued", "running"),
                status="failed",
                phase="failed",
                progress="AlphaSAGE 结果校验或入库失败",
                error_message=message,
            )
            task.status = TaskStatus.FAILED
            task.error_message = message
            await asyncio.to_thread(
                finish_execution,
                task_id,
                owner.tenant_id,
                owner.user_id,
                "failed",
            )
