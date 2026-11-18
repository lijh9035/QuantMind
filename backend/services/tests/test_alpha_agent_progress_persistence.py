"""Focused tests for Alpha Agent phase progress persistence."""

from pathlib import Path
import tempfile
import unittest

from backend.services.engine.alpha_agent.launcher import (
    AlphaAgentLauncher,
    EvolutionTask,
    TaskStatus,
)


class _MemoryProgressStore:
    def __init__(self):
        self.updates = []

    async def update_progress(self, task_id, *, tenant_id, user_id, snapshot):
        self.updates.append(
            {
                "task_id": task_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "snapshot": dict(snapshot),
            }
        )
        return True


class AlphaAgentProgressPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_and_phase_snapshots_are_persisted_only_when_changed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            launcher = AlphaAgentLauncher()
            launcher._log_dir = root / "state"
            launcher._log_dir.mkdir()
            task = EvolutionTask(
                task_id="progress-smoke",
                tenant_id="tenant-a",
                user_id="user-a",
                status=TaskStatus.RUNNING,
                phase="starting",
                progress="starting",
                progress_pct=2,
                loop_n=1,
            )
            log_dir = root / "run"
            log_dir.mkdir()
            store = _MemoryProgressStore()

            local_sig, db_sig = await launcher._refresh_and_persist_progress(
                task, log_dir, task_store=store
            )
            self.assertEqual(len(store.updates), 1)
            self.assertEqual(store.updates[0]["snapshot"]["phase"], "starting")
            self.assertEqual(store.updates[0]["tenant_id"], "tenant-a")
            state_file = launcher._log_dir / task.task_id / "task_state.json"
            self.assertTrue(state_file.is_file())

            local_sig, db_sig = await launcher._refresh_and_persist_progress(
                task,
                log_dir,
                task_store=store,
                local_signature=local_sig,
                database_signature=db_sig,
            )
            self.assertEqual(len(store.updates), 1)

            phase_dir = log_dir / "Loop_0" / "hypothesis generation"
            phase_dir.mkdir(parents=True)
            (phase_dir / "response.txt").write_text("hypothesis generated")
            local_sig, db_sig = await launcher._refresh_and_persist_progress(
                task,
                log_dir,
                task_store=store,
                local_signature=local_sig,
                database_signature=db_sig,
            )
            self.assertEqual(len(store.updates), 2)
            self.assertEqual(store.updates[-1]["snapshot"]["phase"], "hypothesis")
            self.assertGreater(
                store.updates[-1]["snapshot"]["progress_pct"],
                store.updates[0]["snapshot"]["progress_pct"],
            )


if __name__ == "__main__":
    unittest.main()
