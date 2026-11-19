from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pandas as pd
import pytest

from backend.shared.feature_source import (
    quantcustom_root,
    user_factor_dataset_dir,
)
from backend.services.engine.alpha_agent.user_factor_dataset import (
    UserFactorDatasetUnavailable,
    _safe_generation_files,
    resolve_active_owner_factor_dataset,
)
from backend.shared.training.schemas import DataCfg, TrainingConfig, dump_contract_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "docker" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.loading import _merge_owner_factor_generation


def test_owner_factor_path_uses_custom_root_and_validated_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QM_QUANTCUSTOM_DATA_DIR", str(tmp_path))

    root = user_factor_dataset_dir(
        "tenant_a", "user_7", "rd_mined", generation_id="gen_123"
    )

    assert quantcustom_root() == tmp_path
    assert root == (
        tmp_path
        / "users"
        / "tenant_a"
        / "user_7"
        / "factor_generations"
        / "gen_123"
        / "6_ml_datasets"
        / "rd_mined"
    )
    with pytest.raises(ValueError):
        user_factor_dataset_dir(
            "tenant_a", "user_7", generation_id="../other"
        )
    with pytest.raises(ValueError):
        user_factor_dataset_dir(
            "tenant_a", "user_7", "l1_factors", generation_id="gen_123"
        )


def test_generation_file_validation_checks_manifest_counts_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QM_QUANTCUSTOM_DATA_DIR", str(tmp_path))
    root = user_factor_dataset_dir(
        "tenant_a", "user_7", generation_id="gen_123"
    )
    partition = root / "dt=20260105" / "data.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ"],
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "factor_test": [0.25, -0.5],
        }
    ).to_parquet(partition, index=False, engine="pyarrow")

    verified = _safe_generation_files(
        tenant_id="tenant_a",
        user_id="user_7",
        generation_id="gen_123",
        root=root,
        feature_keys=("factor_test",),
        expected_rows=2,
        expected_bytes=partition.stat().st_size,
        manifest={"partition_count": 1, "feature_key": "factor_test"},
    )

    assert verified == (partition,)
    with pytest.raises(UserFactorDatasetUnavailable, match="size does not match"):
        _safe_generation_files(
            tenant_id="tenant_a",
            user_id="user_7",
            generation_id="gen_123",
            root=root,
            feature_keys=("factor_test",),
            expected_rows=3,
            expected_bytes=partition.stat().st_size,
            manifest={"partition_count": 1},
        )


def test_training_reader_joins_private_values_by_symbol_and_date(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private_generation"
    partition = root / "dt=20260105" / "data.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ"],
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "factor_test": [0.25, -0.5],
        }
    ).to_parquet(partition, index=False, engine="pyarrow")
    base = pd.DataFrame(
        {
            "symbol": ["SH600000", "000001"],
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "close": [10.0, 20.0],
        }
    )

    merged = _merge_owner_factor_generation(
        base,
        root,
        ["factor_test"],
        start=pd.Timestamp("2026-01-05"),
        end=pd.Timestamp("2026-01-05"),
    )

    assert merged["factor_test"].tolist() == [0.25, -0.5]
    assert merged["close"].tolist() == [10.0, 20.0]


def test_training_reader_fails_closed_when_pinned_generation_is_missing(
    tmp_path: Path,
) -> None:
    base = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": pd.to_datetime(["2026-01-05"]),
            "close": [10.0],
        }
    )

    with pytest.raises(RuntimeError, match="directory is unavailable"):
        _merge_owner_factor_generation(
            base,
            tmp_path / "missing-generation",
            ["factor_test"],
            start=pd.Timestamp("2026-01-05"),
            end=pd.Timestamp("2026-01-05"),
        )


def test_active_generation_resolver_is_owner_scoped_and_pins_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QM_QUANTCUSTOM_DATA_DIR", str(tmp_path))
    root = user_factor_dataset_dir(
        "tenant_a", "user_7", generation_id="gen_123"
    )
    partition = root / "dt=20260105" / "data.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": pd.to_datetime(["2026-01-05"]),
            "factor_test": [0.25],
        }
    ).to_parquet(partition, index=False, engine="pyarrow")
    expected_rows = [
        {
            "feature_key": "factor_test",
            "row_count": 1,
        }
    ]
    active_row = {
        "generation_id": "gen_123",
        "status": "published",
        "dataset_key": "rd_mined",
        "row_count": 1,
        "byte_count": partition.stat().st_size,
        "manifest_json": {
            "feature_key": "factor_test",
            "partition_count": 1,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
        },
    }
    calls: list[dict[str, str]] = []

    class Result:
        def __init__(self, rows, single=None):
            self.rows = rows
            self.single = single

        def mappings(self):
            return self

        def first(self):
            return self.single

        def all(self):
            return self.rows

    class Session:
        async def execute(self, statement, params):
            calls.append(dict(params))
            if "alpha_user_factor_dataset_active" in str(statement):
                row = active_row if params["tenant"] == "tenant_a" else None
                return Result([], single=row)
            return Result(expected_rows)

    class SessionContext:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(
        "backend.shared.database_manager_v2.get_session",
        lambda **_kwargs: SessionContext(),
    )

    resolved = asyncio.run(
        resolve_active_owner_factor_dataset(
            "tenant_a",
            "user_7",
            ["factor_test"],
            expected_generation_map={"factor_test": "gen_123"},
            require_match=True,
        )
    )
    absent = asyncio.run(
        resolve_active_owner_factor_dataset("tenant_b", "user_9", ["factor_test"])
    )
    with pytest.raises(UserFactorDatasetUnavailable, match="no active"):
        asyncio.run(
            resolve_active_owner_factor_dataset(
                "tenant_b",
                "user_9",
                ["factor_test"],
                expected_generation_map={"factor_test": "gen_123"},
                require_match=True,
            )
        )

    assert resolved is not None
    assert resolved.feature_generation_by_column == {"factor_test": "gen_123"}
    assert resolved.partitions == (partition,)
    assert absent is None
    assert calls[0] == {"tenant": "tenant_a", "user": "user_7"}
    assert calls[-1] == {"tenant": "tenant_b", "user": "user_9"}


def test_private_training_config_fields_are_omitted_when_empty_and_kept_when_pinned() -> None:
    empty = dump_contract_dict(TrainingConfig.from_dict({}))
    assert "user_factor_features" not in empty["data"]

    pinned = TrainingConfig(
        data=DataCfg(
            user_factor_source_dir="/private_factors/rd_mined",
            user_factor_features=["factor_test"],
            user_factor_generation_by_column={"factor_test": "gen_123"},
        )
    )
    emitted = dump_contract_dict(pinned)
    assert emitted["data"]["user_factor_source_dir"] == "/private_factors/rd_mined"
    assert emitted["data"]["user_factor_features"] == ["factor_test"]
    assert emitted["data"]["user_factor_generation_by_column"] == {
        "factor_test": "gen_123"
    }
