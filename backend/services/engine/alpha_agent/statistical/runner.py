"""Durable controller for the restricted statistical mining worker."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from backend.shared.owner_scope import OwnerScope
from backend.shared.secret_redaction import redact_text

logger = logging.getLogger(__name__)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROGRESS_NAMES = {
    "snapshot": "冻结任务输入快照",
    "data": "读取并整理行情数据",
    "filter": "应用可交易性过滤",
    "evaluate": "计算截面 IC 与稳定性",
    "select": "筛选并保存入选因子",
    "stability": "检查子窗稳定性",
    "dedup": "执行相关性去重",
    "composite": "评估 IC 加权合成因子",
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"Persisted statistical {label} is invalid")
    return value


def _atomic_json(
    path: Path, value: dict[str, Any], *, max_bytes: int = 4 * 1024**2
) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > max_bytes:
        raise ValueError("Statistical controller artifact exceeds its size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, *, max_bytes: int = 4 * 1024**2) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"Statistical artifact is missing or oversized: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Statistical artifact is not an object: {path.name}")
    return value


def _capture_prepared_snapshot(
    manifest: dict[str, Any], *, task_dir: Path, initial_snapshot: dict[str, Any]
) -> dict[str, Any]:
    required_hashes = ("panel_sha256", "metadata_sha256", "symbols_sha256")
    for key in required_hashes:
        value = manifest.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"Statistical worker produced an invalid {key}")
    if manifest.get("status") != "prepared":
        raise ValueError("Statistical worker did not prepare a frozen input panel")
    if type(manifest.get("panel_rows")) is not int or manifest["panel_rows"] < 1:
        raise ValueError("Statistical worker produced an empty price panel")
    if type(manifest.get("n_symbols")) is not int or manifest["n_symbols"] < 1:
        raise ValueError("Statistical worker produced an empty instrument set")

    from .worker import digest_file

    panel_path = task_dir / "statistical_panel.parquet"
    metadata_path = task_dir / "statistical_metadata.parquet"
    if digest_file(panel_path, 2 * 1024**3) != manifest["panel_sha256"]:
        raise ValueError("Prepared statistical panel failed its parent hash check")
    if digest_file(metadata_path, 64 * 1024**2) != manifest["metadata_sha256"]:
        raise ValueError("Prepared statistical metadata failed its parent hash check")
    return {**initial_snapshot, **manifest, "status": "frozen"}


def _finite_number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _required_text(value: Any, label: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"Statistical result has an invalid {label}")
    return value.strip()


def _candidate_factor(
    candidate: dict[str, Any],
    *,
    task_id: str,
    owner: OwnerScope,
    fingerprint: str,
    algorithm_version: str,
    params: dict[str, Any],
    expression: Any = None,
    formulation: Any = None,
    candidate_key: Any = None,
    composite: bool = False,
) -> dict[str, Any]:
    from .dedup import stable_factor_id

    key = _required_text(candidate_key or candidate.get("key"), "candidate key", 256)
    name = _required_text(candidate.get("name"), "factor name", 256)
    code = _required_text(
        expression or candidate.get("expression"), "factor expression", 8192
    )
    formula = _required_text(
        formulation or candidate.get("formulation") or f"统计量价候选：{key}",
        "factor formulation",
        2000,
    )
    factor_id, identity_scheme = stable_factor_id(
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        market="a_share",
        universe=str(params["universe"]),
        candidate_key=key,
        task_fingerprint=fingerprint,
    )
    evidence = {
        field: candidate.get(field)
        for field in (
            "family",
            "source",
            "best_horizon",
            "rank_ic_mean",
            "icir",
            "t_stat",
            "p_value",
            "fdr_pass",
            "fdr_horizons",
            "oos_verified",
            "oos_horizon",
            "oos_rank_ic",
            "oos_n_days",
            "quantile_monotonicity",
            "spread_mean",
            "selection_score",
            "fold_sign_consistency",
            "max_abs_corr",
            "corr_warn",
            "eligible",
            "components",
            "component_rank_ics",
            "horizon",
            "z_window",
        )
        if field in candidate
    }
    metadata = {
        "algorithm": "statistical",
        "algorithm_version": algorithm_version,
        "task_id": task_id,
        "execution_fingerprint": fingerprint,
        "candidate_key": key,
        "identity_scheme": identity_scheme,
        "composite": composite,
        "evidence": evidence,
    }
    return {
        "factor_id": factor_id,
        "factor_name": name,
        "factor_code": code,
        "rank_ic": _finite_number(candidate.get("rank_ic_mean")),
        "factor_formulation": formula,
        "market": "a_share",
        "universe": str(params["universe"]),
        "date_range": f"{params['mining_start']}~{params['mining_end']}",
        "data_source": "qlib_bin",
        "metadata": metadata,
    }


def _factor_rows(
    result: dict[str, Any],
    *,
    task_id: str,
    owner: OwnerScope,
    fingerprint: str,
    algorithm_version: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = result.get("selected")
    candidates = result.get("candidates")
    if not isinstance(selected, list) or not isinstance(candidates, list):
        raise ValueError("Statistical result is missing its selected candidate list")
    by_key: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("key"), str):
            if candidate["key"] in by_key:
                raise ValueError("Statistical result contains duplicate candidate keys")
            by_key[candidate["key"]] = candidate

    if (
        len(selected) > 32
        or any(not isinstance(key, str) for key in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Statistical result contains an invalid Top-K selection")
    rows: list[dict[str, Any]] = []
    for key in selected:
        candidate = by_key.get(key) if isinstance(key, str) else None
        if candidate is None or candidate.get("selected") is not True:
            raise ValueError("Statistical selected candidate is absent or unverified")
        rows.append(
            _candidate_factor(
                candidate,
                task_id=task_id,
                owner=owner,
                fingerprint=fingerprint,
                algorithm_version=algorithm_version,
                params=params,
            )
        )

    composite = result.get("composite")
    if isinstance(composite, dict) and composite.get("eligible") is True:
        components = composite.get("components")
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("Statistical composite has invalid component evidence")
        rows.append(
            _candidate_factor(
                composite,
                task_id=task_id,
                owner=owner,
                fingerprint=fingerprint,
                algorithm_version=algorithm_version,
                params=params,
                candidate_key=composite.get("key"),
                formulation=(
                    "IC 加权合成：" + " + ".join(str(item) for item in components)
                ),
                composite=True,
            )
        )
    return rows


async def _record_progress(
    store,
    launcher,
    task,
    owner: OwnerScope,
    progress_path: Path,
    previous_percent: int,
    *,
    stage_override: str | None = None,
    detail_override: str | None = None,
) -> int:
    payload = (
        _read_json(progress_path, max_bytes=64 * 1024)
        if progress_path.is_file() and not progress_path.is_symlink()
        else {}
    )
    stage = stage_override or str(payload.get("stage") or "running")[:64]
    detail = (
        detail_override
        or str(
            payload.get("detail") or _PROGRESS_NAMES.get(stage) or "统计量价任务运行中"
        )[:1000]
    )
    raw_percent = payload.get("progress_pct")
    percent = (
        max(previous_percent, min(99, max(0, raw_percent)))
        if type(raw_percent) is int
        else previous_percent
    )
    snapshot = {
        "algorithm": "statistical",
        "phase": stage,
        "progress": detail,
        "progress_pct": percent,
        "updated_at": time.time(),
    }
    await store.update_progress(
        task.task_id,
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        snapshot=snapshot,
    )
    task.phase = stage
    task.progress = detail
    task.progress_pct = percent
    task.mining_progress = snapshot
    launcher._persist_task(task)
    return percent


async def _stop_worker_process_group(launcher, process: subprocess.Popen) -> bool:
    """Stop and reap the complete worker session before settling its task lease."""
    try:
        if launcher._process_group_has_live_process(process.pid):
            stopped = await launcher._terminate_process_group(process.pid)
            if not stopped:
                return False
    except OSError:
        logger.exception(
            "Could not verify statistical worker process group %s", process.pid
        )
        return False
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None


async def run_statistical_queued(record: dict[str, Any]) -> None:
    """Execute an admitted statistical task with frozen inputs and atomic result persistence."""
    from backend.services.engine.alpha_agent.launcher import (
        EvolutionTask,
        TaskStatus,
        get_launcher,
    )
    from backend.services.engine.alpha_agent.resource_budget import (
        ExecutionCapacityFull,
        begin_execution,
        record_process_group,
    )
    from backend.services.engine.alpha_agent.queue import capture_controller
    from backend.services.engine.alpha_agent.task_spec import (
        MiningTaskSpec,
        STATISTICAL_VERSION,
    )
    from backend.services.engine.qlib_app.services.rd_agent_persistence import (
        AlphaMiningTaskPersistence,
    )
    from backend.shared.process_witness import capture_process

    owner = OwnerScope(record.get("tenant_id"), record.get("user_id"))
    task_id = str(record.get("task_id") or "")
    store = AlphaMiningTaskPersistence()
    launcher = get_launcher()
    task: EvolutionTask | None = None
    process: subprocess.Popen | None = None
    worker_group_stopped: bool | None = None
    log_stream = None
    try:
        request_snapshot = _mapping(record.get("request_snapshot"), "request snapshot")
        params = _mapping(record.get("params_snapshot"), "parameter snapshot")
        initial_snapshot = _mapping(record.get("data_snapshot"), "data snapshot")
        if (
            record.get("algorithm") != "statistical"
            or record.get("algorithm_version") != STATISTICAL_VERSION
            or request_snapshot.get("algorithm") != "statistical"
            or params.get("algorithm") != "statistical"
            or initial_snapshot.get("status") != "preparing"
        ):
            raise ValueError("Persisted statistical task specification is unsupported")
        spec = MiningTaskSpec.build(
            algorithm="statistical",
            algorithm_version=STATISTICAL_VERSION,
            request=request_snapshot,
            params=params,
            data_snapshot=initial_snapshot,
        )
        fingerprint = record.get("execution_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or spec.execution_fingerprint != fingerprint
        ):
            raise ValueError("Persisted statistical task fingerprint does not match")

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
            valid_start=params.get("valid_start"),
            valid_end=params.get("valid_end"),
            status=TaskStatus.QUEUED,
            phase="queued",
            progress="等待统计量价任务执行",
        )
        task.algorithm = "statistical"
        launcher._tasks[task_id] = task

        task_dir = launcher.task_dir(task_id, owner.tenant_id, owner.user_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        request_path = task_dir / "statistical_request.json"
        result_path = task_dir / "statistical_result.json"
        progress_path = task_dir / "statistical_progress.json"
        prepared_path = task_dir / "statistical_prepared.json"
        sealed_path = task_dir / "statistical_sealed.json"
        log_path = task_dir / "subprocess_stdout.log"
        for stale_path in (result_path, progress_path, prepared_path, sealed_path):
            if stale_path.exists() or stale_path.is_symlink():
                raise ValueError(
                    "Statistical task directory already contains worker artifacts"
                )
        _atomic_json(
            request_path,
            {
                "algorithm": "statistical",
                "params": params,
                "data_snapshot": initial_snapshot,
                "panel_path": str(task_dir / "statistical_panel.parquet"),
                "metadata_path": str(task_dir / "statistical_metadata.parquet"),
            },
        )

        witness = await asyncio.to_thread(capture_controller)
        try:
            with begin_execution(
                task_id,
                owner.tenant_id,
                owner.user_id,
                executor_kind="engine-local",
                witness={**witness, "statistical_worker": True},
            ):
                claimed = await store.update(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    expected_status=("pending", "queued"),
                    status="running",
                    phase="starting",
                    progress="正在准备统计量价挖掘",
                    progress_pct=2,
                    error_message=None,
                )
                if not claimed:
                    return
                task.status = TaskStatus.RUNNING
                task.phase = "starting"
                task.progress = "正在准备统计量价挖掘"
                task.progress_pct = 2
                launcher._persist_task(task)

                environment = dict(os.environ)
                python_paths = [
                    item
                    for item in (
                        os.getenv("PYTHONPATH", "").split(os.pathsep) + ["/app"]
                    )
                    if item
                ]
                environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
                environment["LOG_TRACE_PATH"] = str(task_dir)
                command = [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "backend.services.engine.alpha_agent.statistical.worker",
                    str(request_path),
                    str(result_path),
                    str(progress_path),
                ]
                log_stream = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=task_dir,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                task.process = process
                launcher._persist_task(task)
                try:
                    leader = await asyncio.to_thread(capture_process, process.pid)
                    await asyncio.to_thread(
                        record_process_group,
                        task_id,
                        owner.tenant_id,
                        owner.user_id,
                        pgid=process.pid,
                        leader=leader,
                    )
                except (OSError, ValueError, IndexError):
                    logger.info(
                        "Statistical worker %s exited before its process witness was recorded",
                        task_id,
                    )

                percent = 2
                sealed_fingerprint: str | None = None
                frozen_snapshot: dict[str, Any] | None = None
                last_progress_token: tuple[int, int] | None = None
                while process.poll() is None:
                    latest = await store.get(
                        task_id,
                        tenant_id=owner.tenant_id,
                        user_id=owner.user_id,
                    )
                    if latest and latest.get("status") == "stopping":
                        worker_group_stopped = await _stop_worker_process_group(
                            launcher, process
                        )
                        if not worker_group_stopped:
                            task.status = TaskStatus.STOPPING
                            task.phase = "stopping"
                            task.progress = "正在确认统计 worker 已退出"
                            task.error_message = "取消请求已接收，仍在等待 worker 退出"
                            launcher._persist_task(task)
                            return
                        await store.confirm_cancelled(
                            task_id,
                            tenant_id=owner.tenant_id,
                            user_id=owner.user_id,
                        )
                        task.status = TaskStatus.CANCELLED
                        task.phase = "cancelled"
                        task.progress = "统计量价任务已取消"
                        task.error_message = "Cancelled by user"
                        launcher._persist_task(task)
                        return

                    if prepared_path.is_file() and sealed_fingerprint is None:
                        manifest = _read_json(prepared_path, max_bytes=128 * 1024)
                        frozen_snapshot = _capture_prepared_snapshot(
                            manifest,
                            task_dir=task_dir,
                            initial_snapshot=initial_snapshot,
                        )
                        sealed_fingerprint = await store.seal_data_snapshot(
                            task_id,
                            tenant_id=owner.tenant_id,
                            user_id=owner.user_id,
                            data_snapshot=frozen_snapshot,
                        )
                        sealed_record = await store.get_for_owner_with_spec(
                            task_id,
                            tenant_id=owner.tenant_id,
                            user_id=owner.user_id,
                            consistent=True,
                        )
                        if (
                            not sealed_record
                            or sealed_record.get("algorithm") != "statistical"
                        ):
                            raise ValueError(
                                "Frozen statistical task could not be reloaded"
                            )
                        frozen_snapshot = _mapping(
                            sealed_record.get("data_snapshot"), "frozen data snapshot"
                        )
                        if (
                            frozen_snapshot.get("status") != "frozen"
                            or sealed_record.get("execution_fingerprint")
                            != sealed_fingerprint
                        ):
                            raise ValueError(
                                "Frozen statistical task verification failed"
                            )
                        _atomic_json(sealed_path, frozen_snapshot, max_bytes=256 * 1024)
                        percent = await _record_progress(
                            store,
                            launcher,
                            task,
                            owner,
                            progress_path,
                            percent,
                            stage_override="snapshot",
                            detail_override="行情面板和变体记忆已冻结",
                        )

                    if progress_path.is_file() and not progress_path.is_symlink():
                        progress_stat = progress_path.stat()
                        progress_token = (
                            progress_stat.st_mtime_ns,
                            progress_stat.st_size,
                        )
                        if progress_token != last_progress_token:
                            percent = await _record_progress(
                                store, launcher, task, owner, progress_path, percent
                            )
                            last_progress_token = progress_token
                    await asyncio.sleep(1)

                return_code = process.wait(timeout=30)
                worker_group_stopped = await _stop_worker_process_group(
                    launcher, process
                )
                if not worker_group_stopped:
                    raise RuntimeError("统计 worker 子进程组退出未能确认")
                if return_code != 0:
                    tail = "".join(
                        log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines(keepends=True)[-80:]
                    )
                    raise RuntimeError(
                        redact_text(
                            tail.strip()
                            or f"Statistical worker exited with code {return_code}"
                        )[:2000]
                    )
                if sealed_fingerprint is None:
                    raise ValueError(
                        "Statistical worker exited before its input snapshot was sealed"
                    )
                result = _read_json(result_path, max_bytes=4 * 1024**2)
                if frozen_snapshot is None:
                    raise ValueError("Statistical frozen snapshot is unavailable")
                factors = _factor_rows(
                    result,
                    task_id=task_id,
                    owner=owner,
                    fingerprint=sealed_fingerprint,
                    algorithm_version=STATISTICAL_VERSION,
                    params=params,
                )
                result_snapshot = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "algorithm": "statistical",
                    "algorithm_version": STATISTICAL_VERSION,
                    "execution_fingerprint": sealed_fingerprint,
                    "status": "completed",
                    "selected_count": len(result.get("selected") or []),
                    "persisted_count": len(factors),
                    "parameters": params,
                    "data_snapshot": {
                        key: frozen_snapshot.get(key)
                        for key in (
                            "panel_sha256",
                            "metadata_sha256",
                            "symbols_sha256",
                            "panel_rows",
                            "n_symbols",
                            "fetch_start",
                            "fetch_end",
                            "variant_memory",
                        )
                    },
                    "statistics": result,
                }
                completed = await store.complete_with_result(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    execution_fingerprint=sealed_fingerprint,
                    result=result_snapshot,
                    factors=factors,
                )
                if not completed:
                    latest = await store.get(
                        task_id,
                        tenant_id=owner.tenant_id,
                        user_id=owner.user_id,
                        consistent=True,
                    )
                    if latest and latest.get("status") == "stopping":
                        await store.confirm_cancelled(
                            task_id,
                            tenant_id=owner.tenant_id,
                            user_id=owner.user_id,
                        )
                        task.status = TaskStatus.CANCELLED
                        task.phase = "cancelled"
                        return
                    raise RuntimeError(
                        "Statistical task left running state before result commit"
                    )
                task.status = TaskStatus.COMPLETED
                task.phase = "completed"
                task.progress = f"统计量价挖掘完成，保存 {len(factors)} 个入选因子"
                task.progress_pct = 100
                task.result = result_snapshot
                task.mining_progress = {
                    "algorithm": "statistical",
                    "phase": "completed",
                    "progress": task.progress,
                    "progress_pct": 100,
                    "selected_count": len(result.get("selected") or []),
                    "persisted_count": len(factors),
                }
                launcher._persist_task(task)
        except ExecutionCapacityFull:
            return
    except asyncio.CancelledError:
        if process is not None and process.poll() is None:
            worker_group_stopped = await _stop_worker_process_group(launcher, process)
        raise
    except Exception as exc:  # noqa: BLE001 -- the queue records a durable terminal state
        message = redact_text(f"{type(exc).__name__}: {exc}")[:2000]
        latest = await store.get(
            task_id,
            tenant_id=owner.tenant_id,
            user_id=owner.user_id,
            consistent=True,
        )
        if latest and latest.get("status") == "stopping":
            worker_group_stopped = process is None or (
                worker_group_stopped is True
                or await _stop_worker_process_group(launcher, process)
            )
            if worker_group_stopped:
                await store.confirm_cancelled(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                )
                if task:
                    task.status = TaskStatus.CANCELLED
                    task.phase = "cancelled"
                    task.progress = "统计量价任务已取消"
                    task.error_message = "Cancelled by user"
                    launcher._persist_task(task)
            elif task:
                task.status = TaskStatus.STOPPING
                task.phase = "stopping"
                task.progress = "正在确认统计 worker 已退出"
                task.error_message = "取消请求已接收，仍在等待 worker 退出"
                launcher._persist_task(task)
                logger.error(
                    "Statistical worker process group still active after stop request: %s",
                    task_id,
                )
            return
        if latest and latest.get("status") in {"pending", "queued", "running"}:
            worker_group_stopped = process is None or (
                worker_group_stopped is True
                or await _stop_worker_process_group(launcher, process)
            )
            if worker_group_stopped:
                await store.update(
                    task_id,
                    tenant_id=owner.tenant_id,
                    user_id=owner.user_id,
                    expected_status=("pending", "queued", "running"),
                    status="failed",
                    phase="failed",
                    progress="统计量价任务失败",
                    error_message=message,
                )
            else:
                logger.error(
                    "Statistical task %s failed but its worker process group is still active",
                    task_id,
                )
        if task:
            task.status = TaskStatus.FAILED
            task.phase = "failed"
            task.progress = "统计量价任务失败"
            task.error_message = message
            launcher._persist_task(task)
        logger.exception("Statistical mining task failed: %s", task_id)
    finally:
        process_stopped = process is None or worker_group_stopped is True
        if process is not None and not process_stopped:
            process_stopped = await _stop_worker_process_group(launcher, process)
        if task is not None:
            task.process = None if process_stopped else process
            launcher._persist_task(task)
        if log_stream is not None:
            log_stream.close()
