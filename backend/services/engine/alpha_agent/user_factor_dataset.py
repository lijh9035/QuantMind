"""Owner-authorized reads for the active private ``rd_mined`` generation."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_PARTITION_RE = re.compile(r"^dt=(\d{8})$")


class UserFactorDatasetUnavailable(RuntimeError):
    """The owner's active private generation cannot be read safely."""


@dataclass(frozen=True)
class ActiveUserFactorDataset:
    tenant_id: str
    user_id: str
    generation_id: str
    feature_keys: tuple[str, ...]
    feature_generation_by_column: dict[str, str]
    root: Path
    partitions: tuple[Path, ...]
    row_count: int
    byte_count: int
    start_date: str | None
    end_date: str | None


def _safe_generation_files(
    *,
    tenant_id: str,
    user_id: str,
    generation_id: str,
    root: Path,
    feature_keys: tuple[str, ...],
    expected_rows: int,
    expected_bytes: int,
    manifest: dict[str, Any],
) -> tuple[Path, ...]:
    from backend.shared.feature_source import quantcustom_root, user_factor_dataset_dir
    from backend.shared.owner_path import user_scoped_dir

    expected_root = user_factor_dataset_dir(
        tenant_id, user_id, "rd_mined", generation_id=generation_id
    )
    if root != expected_root:
        raise UserFactorDatasetUnavailable("Private factor path does not match its owner")

    custom_root = quantcustom_root()
    owner_root = user_scoped_dir(custom_root / "users", tenant_id, user_id)
    guarded_dirs = (
        custom_root,
        custom_root / "users",
        custom_root / "users" / tenant_id,
        owner_root,
        owner_root / "factor_generations",
        owner_root / "factor_generations" / generation_id,
        owner_root / "factor_generations" / generation_id / "6_ml_datasets",
        root,
    )
    for directory in guarded_dirs:
        try:
            mode = os.lstat(directory).st_mode
        except OSError as exc:
            raise UserFactorDatasetUnavailable(
                "Active private factor directory is unavailable"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise UserFactorDatasetUnavailable("Active private factor path is unsafe")

    try:
        resolved_owner = owner_root.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise UserFactorDatasetUnavailable(
            "Active private factor directory is unavailable"
        ) from exc
    if not resolved_root.is_relative_to(resolved_owner):
        raise UserFactorDatasetUnavailable("Active private factor path escaped its owner")

    partitions = tuple(sorted(root.glob("dt=*/data.parquet")))
    if not partitions:
        raise UserFactorDatasetUnavailable("Active private factor has no Parquet partitions")

    import pyarrow.parquet as pq

    total_rows = 0
    total_bytes = 0
    for partition in partitions:
        match = _PARTITION_RE.fullmatch(partition.parent.name)
        if match is None:
            raise UserFactorDatasetUnavailable("Active private factor has an invalid partition")
        try:
            if partition.is_symlink() or partition.parent.is_symlink() or not partition.is_file():
                raise UserFactorDatasetUnavailable("Active private factor file is unsafe")
            if not partition.resolve(strict=True).is_relative_to(resolved_root):
                raise UserFactorDatasetUnavailable("Active private factor file escaped its generation")
            parquet = pq.ParquetFile(str(partition))
            columns = set(parquet.schema_arrow.names)
            if not {"symbol", "trade_date", *feature_keys}.issubset(columns):
                raise UserFactorDatasetUnavailable(
                    "Active private factor schema does not match its feature mapping"
                )
            total_rows += int(parquet.metadata.num_rows)
            total_bytes += partition.stat().st_size
        except UserFactorDatasetUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UserFactorDatasetUnavailable(
                "Active private factor Parquet could not be inspected"
            ) from exc

    expected_partitions = manifest.get("partition_count")
    if type(expected_partitions) is int and expected_partitions != len(partitions):
        raise UserFactorDatasetUnavailable("Active private factor partition count changed")
    if total_rows != expected_rows or total_bytes != expected_bytes:
        raise UserFactorDatasetUnavailable("Active private factor size does not match its manifest")
    return partitions


async def resolve_active_owner_factor_dataset(
    tenant_id: str,
    user_id: str,
    requested_features: Iterable[str] | None = None,
    *,
    expected_generation_map: dict[str, str] | None = None,
    require_match: bool = False,
) -> ActiveUserFactorDataset | None:
    """Resolve only this owner's published active generation and verify its files.

    The owner identity must come from authenticated request state or a trusted
    persisted job. No global ``qm_*`` fallback or archived-generation scan is
    performed.
    """
    from sqlalchemy import text

    from backend.shared.database_manager_v2 import get_session
    from backend.shared.feature_source import user_factor_dataset_dir
    from backend.shared.owner_path import owner_segment

    tenant = owner_segment(tenant_id)
    user = owner_segment(user_id)
    requested = None
    if requested_features is not None:
        requested = list(dict.fromkeys(str(item).strip() for item in requested_features))
        if any(not item or len(item) > 128 for item in requested):
            raise UserFactorDatasetUnavailable("Requested private feature is invalid")

    expected_map: dict[str, str] | None = None
    if expected_generation_map is not None:
        if not isinstance(expected_generation_map, dict) or not expected_generation_map:
            raise UserFactorDatasetUnavailable("Pinned private factor generation map is invalid")
        expected_map = {}
        for feature, generation in expected_generation_map.items():
            feature_key, generation_key = str(feature).strip(), str(generation).strip()
            if not feature_key or len(feature_key) > 128 or not generation_key:
                raise UserFactorDatasetUnavailable("Pinned private factor generation map is invalid")
            expected_map[feature_key] = generation_key

    async with get_session(read_only=True) as session:
        active = (
            await session.execute(
                text(
                    "SELECT g.generation_id,g.status,g.dataset_key,g.row_count,"
                    "g.byte_count,g.manifest_json FROM alpha_user_factor_dataset_active a "
                    "JOIN alpha_user_factor_generations g ON g.tenant_id=a.tenant_id "
                    "AND g.user_id=a.user_id AND g.generation_id=a.generation_id "
                    "WHERE a.tenant_id=:tenant AND a.user_id=:user "
                    "AND a.dataset_key='rd_mined'"
                ),
                {"tenant": tenant, "user": user},
            )
        ).mappings().first()
        if active is None:
            if require_match or expected_map:
                raise UserFactorDatasetUnavailable("Owner has no active private factor generation")
            return None
        if active["status"] != "published" or active["dataset_key"] != "rd_mined":
            raise UserFactorDatasetUnavailable("Owner active private factor is not published")

        rows = (
            await session.execute(
                text(
                    "SELECT feature_key,row_count FROM alpha_user_factor_generation_features "
                    "WHERE tenant_id=:tenant AND user_id=:user AND generation_id=:generation "
                    "ORDER BY feature_key"
                ),
                {
                    "tenant": tenant,
                    "user": user,
                    "generation": active["generation_id"],
                },
            )
        ).mappings().all()

    generation_id = str(active["generation_id"])
    available = {str(row["feature_key"]): int(row["row_count"]) for row in rows}
    if not available:
        raise UserFactorDatasetUnavailable("Owner active private factor has no feature mapping")
    if expected_map:
        if any(value != generation_id for value in expected_map.values()):
            raise UserFactorDatasetUnavailable("Pinned private factor generation is no longer active")
        if not set(expected_map).issubset(available):
            raise UserFactorDatasetUnavailable("Pinned private factor mapping is unavailable")

    selected = (
        list(available)
        if requested is None
        else [feature for feature in requested if feature in available]
    )
    if expected_map:
        selected = [feature for feature in selected if feature in expected_map]
    if require_match and not selected:
        raise UserFactorDatasetUnavailable("Requested features are not in the active generation")
    if expected_map and set(selected) != set(expected_map):
        raise UserFactorDatasetUnavailable("Pinned private factor features are not all requested")
    if not selected:
        return None

    generation_rows = int(active["row_count"])
    if any(available[feature] != generation_rows for feature in selected):
        raise UserFactorDatasetUnavailable(
            "Active private factor feature row counts do not match its generation"
        )

    manifest = active["manifest_json"]
    if isinstance(manifest, str):
        import json

        try:
            manifest = json.loads(manifest)
        except ValueError as exc:
            raise UserFactorDatasetUnavailable("Active private factor manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise UserFactorDatasetUnavailable("Active private factor manifest is invalid")
    manifest_feature = manifest.get("feature_key")
    if manifest_feature and manifest_feature not in available:
        raise UserFactorDatasetUnavailable("Active private factor manifest mapping is inconsistent")

    root = user_factor_dataset_dir(
        tenant, user, "rd_mined", generation_id=generation_id
    )
    partitions = _safe_generation_files(
        tenant_id=tenant,
        user_id=user,
        generation_id=generation_id,
        root=root,
        feature_keys=tuple(selected),
        expected_rows=generation_rows,
        expected_bytes=int(active["byte_count"]),
        manifest=manifest,
    )
    start_date = manifest.get("start_date")
    end_date = manifest.get("end_date")
    return ActiveUserFactorDataset(
        tenant_id=tenant,
        user_id=user,
        generation_id=generation_id,
        feature_keys=tuple(selected),
        feature_generation_by_column={feature: generation_id for feature in selected},
        root=root,
        partitions=partitions,
        row_count=generation_rows,
        byte_count=int(active["byte_count"]),
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
    )
