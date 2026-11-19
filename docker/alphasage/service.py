from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import signal
import sys
import tempfile
from bisect import bisect_left, bisect_right
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
SYMBOL_RE = re.compile(r"^(sh|sz|bj)[0-9]{6}$")
QLIB_INSTRUMENTS = frozenset(
    {"all", "all_a", "csi300", "csi500", "csi800", "csi1000", "sse50", "gem", "star"}
)
TERMINAL = frozenset({"completed", "failed", "cancelled"})
ALGORITHM_VERSION = "quantmind-alphasage/1.0"
JOBS_ROOT = Path(os.getenv("QM_ALPHASAGE_JOBS", "/jobs"))
QLIB_ROOT = Path(os.getenv("QM_ALPHASAGE_QLIB_DB", "/qlib"))
TOKEN = os.getenv("QM_ALPHASAGE_TOKEN", "").strip()
MAX_ACTIVE = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Annotated[str, Field(min_length=1, max_length=64, pattern=TASK_ID_RE.pattern)]
    execution_fingerprint: Annotated[str, Field(pattern=FINGERPRINT_RE.pattern)]
    instrument: str | None = Field(default=None, min_length=1, max_length=32)
    instruments: list[str] | None = Field(default=None, min_length=1, max_length=500)
    mining_start: date
    mining_end: date
    valid_start: date
    valid_end: date
    horizon: int = Field(ge=1, le=20, strict=True)
    n_episodes: int = Field(ge=16, le=2000, strict=True)
    pool_capacity: int = Field(ge=1, le=32, strict=True)
    encoder_type: Literal["gnn"] = "gnn"
    seed: int = Field(default=0, ge=0, le=2147483647, strict=True)

    @model_validator(mode="after")
    def validate_inputs(self):
        if self.mining_start >= self.mining_end:
            raise ValueError("mining_start must be before mining_end")
        if self.valid_start <= self.mining_end or self.valid_end < self.valid_start:
            raise ValueError("validation window must start after mining_end")
        if bool(self.instrument) == bool(self.instruments):
            raise ValueError("Provide exactly one Qlib instrument ID or resolved instrument list")
        if self.instrument and self.instrument not in QLIB_INSTRUMENTS:
            raise ValueError("Unsupported Qlib instrument ID")
        if self.instruments and (
            len(set(self.instruments)) != len(self.instruments)
            or any(not SYMBOL_RE.fullmatch(symbol) for symbol in self.instruments)
        ):
            raise ValueError("Resolved instruments must be unique Qlib symbols")
        return self


_lock = asyncio.Lock()
_processes: dict[str, asyncio.subprocess.Process] = {}
_runners: dict[str, asyncio.Task] = {}


def _job_dir(task_id: str) -> Path:
    if not TASK_ID_RE.fullmatch(task_id) or task_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid task_id")
    return JOBS_ROOT / task_id


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _state_path(task_id: str) -> Path:
    return _job_dir(task_id) / "status.json"


def _get_state(task_id: str) -> dict | None:
    return _read_json(_state_path(task_id))


def _save_state(task_id: str, **changes) -> dict:
    current = _get_state(task_id) or {"task_id": task_id}
    current.update(changes)
    _write_json_atomic(_state_path(task_id), current)
    return current


def _public_state(state: dict) -> dict:
    allowed = (
        "task_id",
        "status",
        "progress_pct",
        "phase",
        "message",
        "candidate_count",
        "error",
        "updated_at",
    )
    return {key: state[key] for key in allowed if key in state}


def _authorize(authorization: str | None) -> None:
    if not TOKEN:
        raise HTTPException(status_code=503, detail="AlphaSAGE internal token is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(supplied, TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _training_dates(request: JobRequest) -> tuple[str, str]:
    calendar_path = QLIB_ROOT / "calendars" / "day.txt"
    if not calendar_path.is_file() or not (QLIB_ROOT / "features").is_dir():
        raise RuntimeError("The read-only Qlib provider is unavailable")
    days = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed_days = [date.fromisoformat(value) for value in days]
    start_index = bisect_left(parsed_days, request.mining_start)
    mining_end_index = bisect_right(parsed_days, request.mining_end) - 1
    if start_index >= len(parsed_days) or mining_end_index < start_index:
        raise ValueError("The mining window is outside the Qlib calendar")
    if start_index < 100:
        raise ValueError("The selected mining_start lacks 100 Qlib backtrack sessions")
    mature_end_index = mining_end_index - request.horizon
    if mature_end_index < start_index:
        raise ValueError("The mining window is shorter than its prediction horizon")
    if mature_end_index - start_index + 1 < 60:
        raise ValueError("At least 60 mature Qlib sessions are required for semantic verification")
    return parsed_days[start_index].isoformat(), parsed_days[mature_end_index].isoformat()


def _validate_provider_instrument(request: JobRequest) -> None:
    if request.instrument:
        if not (QLIB_ROOT / "instruments" / f"{request.instrument}.txt").is_file():
            raise ValueError("The requested Qlib instrument list is unavailable")
        return
    missing = [
        symbol
        for symbol in request.instruments or []
        if not (QLIB_ROOT / "features" / symbol / "close.day.bin").is_file()
    ]
    if missing:
        raise ValueError(f"Qlib data is unavailable for {len(missing)} requested instruments")


def _read_result(task_id: str) -> dict | None:
    return _read_json(_job_dir(task_id) / "result.json")


def _valid_semantic_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value in {"inf", "-inf"}
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


async def _terminate_process(task_id: str, process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _run_job(request: JobRequest, train_start: str, train_end: str) -> None:
    task_id = request.task_id
    job_dir = _job_dir(task_id)
    training_dir = job_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    instrument_arg = request.instrument or "all_a"
    instruments_file = ""
    if request.instruments:
        instruments_file = str(job_dir / "instruments.json")
        _write_json_atomic(Path(instruments_file), {"symbols": request.instruments})

    update_freq = max(2, min(32, request.n_episodes))
    progress_every = max(1, request.n_episodes // 20)
    command = [
        sys.executable,
        "/opt/alphasage/train_gfn.py",
        "--qlib-path",
        str(QLIB_ROOT),
        "--train-start",
        train_start,
        "--train-end",
        train_end,
        "--horizon",
        str(request.horizon),
        "--device",
        "cuda:0",
        "--output-dir",
        str(training_dir),
        "--instrument",
        instrument_arg,
        "--seed",
        str(request.seed),
        "--pool_capacity",
        str(request.pool_capacity),
        "--log_freq",
        str(request.n_episodes),
        "--update_freq",
        str(update_freq),
        "--n_episodes",
        str(request.n_episodes),
        "--encoder_type",
        request.encoder_type,
        "--progress-every",
        str(progress_every),
    ]
    if instruments_file:
        command.extend(["--instruments-file", instruments_file])

    _save_state(
        task_id,
        status="running",
        progress_pct=1,
        phase="training",
        message="AlphaSAGE GNN training started",
        candidate_count=0,
        updated_at=_timestamp(),
        error=None,
    )
    log_path = job_dir / "train.log"
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd="/opt/alphasage",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        _processes[task_id] = process
        with log_path.open("ab") as log_file:
            while True:
                chunk = await process.stdout.readline()
                if not chunk:
                    break
                log_file.write(chunk)
                log_file.flush()
                text = chunk.decode("utf-8", errors="replace")
                marker = text.rfind("{")
                if marker >= 0:
                    try:
                        event = json.loads(text[marker:].strip())
                    except ValueError:
                        event = None
                    if isinstance(event, dict) and event.get("event") == "progress":
                        total = max(1, int(event.get("total_episodes", 1)))
                        done = max(0, min(total, int(event.get("episode", 0))))
                        _save_state(
                            task_id,
                            progress_pct=5 + int(85 * done / total),
                            phase="training",
                            message=f"训练中 {done}/{total} episodes",
                            candidate_count=int(event.get("pool_size", 0)),
                            updated_at=_timestamp(),
                        )
        return_code = await process.wait()
        latest = _get_state(task_id) or {}
        if latest.get("status") in {"stopping", "cancelled"}:
            _save_state(
                task_id,
                status="cancelled",
                progress_pct=int(latest.get("progress_pct", 0)),
                phase="cancelled",
                message="GPU process group stopped",
                updated_at=_timestamp(),
            )
            return
        if return_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            _save_state(
                task_id,
                status="failed",
                phase="failed",
                error=tail or f"AlphaSAGE exited with code {return_code}",
                message="AlphaSAGE training failed",
                updated_at=_timestamp(),
            )
            return

        pool_path = training_dir / f"pool_{request.n_episodes}.json"
        pool_payload = _read_json(pool_path)
        if pool_payload is None or not isinstance(pool_payload.get("exprs"), list):
            raise RuntimeError("AlphaSAGE did not produce a valid final pool")
        semantic_payload = _read_json(training_dir / "semantic_samples.json") or {}
        semantic_dates = semantic_payload.get("dates")
        semantic_symbols = semantic_payload.get("symbols")
        semantic_records = {
            item.get("source_index"): item
            for item in semantic_payload.get("candidates", [])
            if isinstance(item, dict) and isinstance(item.get("source_index"), int)
        }
        candidates = []
        for index, expression in enumerate(pool_payload["exprs"][: request.pool_capacity]):
            if isinstance(expression, str) and expression.strip() and len(expression) <= 8192:
                candidate = {"source_index": index, "raw_expression": expression.strip()}
                sample = semantic_records.get(index)
                source_values = sample.get("source_values") if isinstance(sample, dict) else None
                valid_sample = (
                    isinstance(semantic_dates, list)
                    and len(semantic_dates) >= 60
                    and all(isinstance(value, str) for value in semantic_dates)
                    and isinstance(semantic_symbols, list)
                    and len(semantic_symbols) >= 1
                    and len(set(semantic_symbols)) == len(semantic_symbols)
                    and all(isinstance(value, str) and SYMBOL_RE.fullmatch(value) for value in semantic_symbols)
                    and isinstance(source_values, list)
                    and len(source_values) == len(semantic_dates)
                    and all(isinstance(row, list) and len(row) == len(semantic_symbols) for row in source_values)
                    and all(_valid_semantic_scalar(value) for row in source_values for value in row)
                )
                if valid_sample:
                    candidate["semantic_check"] = {
                        "dates": semantic_dates,
                        "symbols": semantic_symbols,
                        "source_values": source_values,
                    }
                elif isinstance(sample, dict) and isinstance(sample.get("source_error"), str):
                    candidate["semantic_error"] = sample["source_error"][:500]
                else:
                    candidate["semantic_error"] = "AlphaSAGE source values are unavailable or malformed"
                candidates.append(candidate)

        result = {
            "schema_version": 1,
            "task_id": task_id,
            "algorithm": "alphasage",
            "algorithm_version": ALGORITHM_VERSION,
            "execution_fingerprint": request.execution_fingerprint,
            "training_window": {
                "requested_start": request.mining_start.isoformat(),
                "requested_end": request.mining_end.isoformat(),
                "mature_label_end": train_end,
                "horizon": request.horizon,
            },
            "validation_window": {
                "start": request.valid_start.isoformat(),
                "end": request.valid_end.isoformat(),
                "evaluated": False,
            },
            "parameters": {
                "instrument": request.instrument,
                "instruments": request.instruments,
                "n_episodes": request.n_episodes,
                "pool_capacity": request.pool_capacity,
                "encoder_type": request.encoder_type,
                "seed": request.seed,
                "device": "cuda:0",
            },
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        _write_json_atomic(job_dir / "result.json", result)
        _save_state(
            task_id,
            status="completed",
            progress_pct=100,
            phase="completed",
            message=f"训练完成，生成 {len(candidates)} 个候选表达式",
            candidate_count=len(candidates),
            updated_at=_timestamp(),
            error=None,
        )
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_process(task_id, process)
        latest = _get_state(task_id) or {}
        if latest.get("status") not in TERMINAL:
            _save_state(
                task_id,
                status="cancelled",
                phase="cancelled",
                message="GPU job cancelled",
                updated_at=_timestamp(),
            )
        raise
    except Exception as exc:  # noqa: BLE001 - persist remote execution failures
        if process is not None and process.returncode is None:
            await _terminate_process(task_id, process)
        latest = _get_state(task_id) or {}
        if latest.get("status") not in {"stopping", "cancelled"}:
            _save_state(
                task_id,
                status="failed",
                phase="failed",
                error=f"{type(exc).__name__}: {exc}"[:2000],
                message="AlphaSAGE training failed",
                updated_at=_timestamp(),
            )
    finally:
        _processes.pop(task_id, None)
        _runners.pop(task_id, None)


def _gpu_details() -> dict:
    import torch

    available = bool(torch.cuda.is_available())
    details: dict = {
        "gpu_available": available,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    if available:
        try:
            device = torch.device("cuda:0")
            probe = torch.tensor([1.0, 2.0], device=device)
            details.update(
                {
                    "device_name": torch.cuda.get_device_name(device),
                    "compute_capability": ".".join(
                        str(part) for part in torch.cuda.get_device_capability(device)
                    ),
                    "kernel_probe": float((probe * 2).sum().item()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            details["gpu_available"] = False
            details["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return details


async def _mark_interrupted_jobs() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    for entry in JOBS_ROOT.iterdir():
        if entry.is_symlink() or not entry.is_dir() or not TASK_ID_RE.fullmatch(entry.name):
            continue
        state = _read_json(entry / "status.json")
        if state and state.get("status") in {"queued", "running", "stopping"}:
            _write_json_atomic(
                entry / "status.json",
                {
                    **state,
                    "status": "failed",
                    "phase": "failed",
                    "message": "GPU service restarted before job completion",
                    "error": "The training process did not survive the container restart",
                    "updated_at": _timestamp(),
                },
            )


async def _stop_all_jobs() -> None:
    for task_id, process in list(_processes.items()):
        _save_state(task_id, status="stopping", phase="stopping", message="GPU service stopping")
        await _terminate_process(task_id, process)
    for task in list(_runners.values()):
        task.cancel()
    if _runners:
        await asyncio.gather(*list(_runners.values()), return_exceptions=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _mark_interrupted_jobs()
    yield
    await _stop_all_jobs()


app = FastAPI(title="QuantMind AlphaSAGE GPU service", version="1", lifespan=lifespan)


@app.get("/health")
async def health(authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    details = _gpu_details()
    provider_ready = (QLIB_ROOT / "calendars" / "day.txt").is_file() and (QLIB_ROOT / "features").is_dir()
    active = any((_get_state(task_id) or {}).get("status") in {"queued", "running", "stopping"} for task_id in _runners)
    return {
        "status": "ready" if details["gpu_available"] and provider_ready else "unavailable",
        "provider_ready": provider_ready,
        "active_job": active,
        **details,
    }


@app.post("/jobs")
async def submit_job(request: JobRequest, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    details = _gpu_details()
    if not details["gpu_available"]:
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")
    try:
        train_start, train_end = _training_dates(request)
        _validate_provider_instrument(request)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with _lock:
        existing = _get_state(request.task_id)
        if existing is not None:
            stored_request = _read_json(_job_dir(request.task_id) / "request.json") or {}
            if stored_request.get("execution_fingerprint") != request.execution_fingerprint:
                raise HTTPException(status_code=409, detail="task_id is already bound to another execution")
            return _public_state(existing)
        if _runners:
            raise HTTPException(status_code=409, detail="GPU is busy with another durable-queue task")
        job_dir = _job_dir(request.task_id)
        if job_dir.exists():
            raise HTTPException(status_code=409, detail="task output already exists without a valid status")
        job_dir.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(job_dir / "request.json", request.model_dump(mode="json"))
        _save_state(
            request.task_id,
            status="queued",
            progress_pct=0,
            phase="queued",
            message="Accepted by GPU service",
            candidate_count=0,
            updated_at=_timestamp(),
            error=None,
        )
        runner = asyncio.create_task(_run_job(request, train_start, train_end))
        _runners[request.task_id] = runner
        return _public_state(_get_state(request.task_id) or {})


@app.get("/jobs/{task_id}")
async def get_job(task_id: str, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    _job_dir(task_id)
    state = _get_state(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _public_state(state)


@app.get("/jobs/{task_id}/result")
async def get_job_result(task_id: str, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    _job_dir(task_id)
    state = _get_state(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job has no completed result")
    result = _read_result(task_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Completed job result is unavailable")
    return result


@app.delete("/jobs/{task_id}")
async def cancel_job(task_id: str, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    _job_dir(task_id)
    async with _lock:
        state = _get_state(task_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if state.get("status") in TERMINAL:
            return _public_state(state)
        _save_state(
            task_id,
            status="stopping",
            phase="stopping",
            message="Waiting for GPU process group to stop",
        )
        process = _processes.get(task_id)
        runner = _runners.get(task_id)
    if process is not None:
        await _terminate_process(task_id, process)
    if runner is not None and not runner.done():
        try:
            await asyncio.wait_for(asyncio.shield(runner), timeout=20)
        except TimeoutError:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
    latest = _get_state(task_id) or state
    if latest.get("status") not in TERMINAL:
        latest = _save_state(
            task_id,
            status="cancelled",
            phase="cancelled",
            message="GPU process group stopped",
        )
    return _public_state(latest)
