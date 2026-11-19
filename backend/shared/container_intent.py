"""Validated persistence for owner-fenced Docker sandbox recovery metadata."""

from __future__ import annotations

import json
from typing import Any


def _labels_match_owner(
    labels: Any, *, lease, pool: str, role: str | None = None
) -> bool:
    if not isinstance(labels, dict) or not labels:
        return False
    if any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in labels.items()
    ):
        return False
    expected = {
        "quantmind.job_id": str(lease.task_id),
        "quantmind.tenant_id": str(lease.tenant_id),
        "quantmind.user_id": str(lease.user_id),
        "quantmind.lease_token": str(lease.token),
        "quantmind.resource_pool": pool,
    }
    if any(labels.get(key) != value for key, value in expected.items()):
        return False
    return role is None or labels.get("quantmind.role") == role


def record_container_intent(
    lease,
    *,
    daemon_id: str,
    container_name: str,
    labels: dict[str, str],
    related_containers: list[dict[str, Any]],
    volume_name: str,
    volume_labels: dict[str, str],
    budget,
    container_id: str | None = None,
) -> bool:
    """Record only owner-matched container and volume identifiers on the lease."""
    pool = str(getattr(budget, "pool", "") or "").strip()
    if (
        not pool
        or not isinstance(daemon_id, str)
        or not daemon_id.strip()
        or len(daemon_id) > 256
        or not isinstance(container_name, str)
        or not container_name.strip()
        or len(container_name) > 255
        or not isinstance(volume_name, str)
        or not volume_name.strip()
        or len(volume_name) > 255
        or not _labels_match_owner(labels, lease=lease, pool=pool, role="sandbox")
        or not _labels_match_owner(
            volume_labels, lease=lease, pool=pool, role="output_keeper"
        )
        or not isinstance(related_containers, list)
        or len(related_containers) > 16
    ):
        return False
    run_id = labels.get("quantmind.run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
        or container_name
        not in {
            f"qm-lab-run-{run_id}",
            f"qm-rdagent-run-{run_id}",
        }
        or volume_name != f"qm-output-{run_id}"
        or volume_labels.get("quantmind.run_id") != run_id
    ):
        return False
    related = []
    for item in related_containers:
        if not isinstance(item, dict):
            return False
        name = item.get("name")
        labels_for_item = item.get("labels")
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 255
            or not _labels_match_owner(
                labels_for_item, lease=lease, pool=pool, role="output_keeper"
            )
            or labels_for_item != volume_labels
            or name != f"qm-output-hold-{run_id}"
        ):
            return False
        related.append({"name": name, "labels": labels_for_item})
    if container_id is not None and (
        not isinstance(container_id, str)
        or not container_id.strip()
        or len(container_id) > 256
    ):
        return False

    intent = {
        "daemon_id": daemon_id.strip(),
        "container_name": container_name,
        "container_labels": labels,
        "related_containers": related,
        "volume_name": volume_name,
        "volume_labels": volume_labels,
    }
    if container_id is not None:
        intent["container_id"] = container_id
    try:
        json.dumps(intent, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False
    try:
        snapshot = budget.store.task_snapshot(
            pool, lease.task_id, lease.tenant_id, lease.user_id
        )
    except Exception:
        return False
    if (
        snapshot is None
        or snapshot.get("state") != "running"
        or snapshot.get("stop_requested")
    ):
        return False
    executor = (snapshot or {}).get("executor") or {}
    dispatch_token = executor.get("dispatch_token")
    if executor.get("engine_claim_token") and (
        executor.get("engine_claim_token") != dispatch_token
        or executor.get("engine_request_state") != "active"
    ):
        return False
    return bool(
        budget.store.record_container_intent(
            pool,
            lease,
            intent=intent,
            expected_dispatch_token=dispatch_token,
        )
    )
