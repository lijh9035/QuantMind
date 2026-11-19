from types import SimpleNamespace

from backend.shared.container_intent import record_container_intent


class FakeStore:
    def __init__(self, *, dispatch_token="dispatch-1", claim_state="active"):
        self.dispatch_token = dispatch_token
        self.claim_state = claim_state
        self.intent = None

    def task_snapshot(self, pool, task_id, tenant_id, user_id):
        return {
            "state": "running",
            "executor": {
                "dispatch_token": self.dispatch_token,
                "engine_claim_token": self.dispatch_token,
                "engine_request_state": self.claim_state,
            },
        }

    def record_container_intent(self, pool, lease, *, intent, expected_dispatch_token):
        self.intent = intent
        assert expected_dispatch_token == self.dispatch_token
        return True


def _case(*, tenant_id="tenant-1", dispatch_token="dispatch-1", claim_state="active"):
    lease = SimpleNamespace(
        task_id="job-1",
        tenant_id=tenant_id,
        user_id="user-1",
        token="lease-token-1",
        epoch="epoch-1",
    )
    pool = "alpha_factor_jobs"
    run_id = "ab" * 16
    labels = {
        "quantmind.run_id": run_id,
        "quantmind.job_id": lease.task_id,
        "quantmind.tenant_id": lease.tenant_id,
        "quantmind.user_id": lease.user_id,
        "quantmind.lease_token": lease.token,
        "quantmind.resource_pool": pool,
        "quantmind.role": "sandbox",
    }
    volume_labels = {**labels, "quantmind.role": "output_keeper"}
    store = FakeStore(dispatch_token=dispatch_token, claim_state=claim_state)
    budget = SimpleNamespace(pool=pool, store=store)
    return lease, budget, run_id, labels, volume_labels, store


def test_container_intent_is_fenced_to_the_active_owner_lease():
    lease, budget, run_id, labels, volume_labels, store = _case()

    recorded = record_container_intent(
        lease,
        daemon_id="docker-daemon-1",
        container_name=f"qm-lab-run-{run_id}",
        labels=labels,
        related_containers=[
            {"name": f"qm-output-hold-{run_id}", "labels": volume_labels}
        ],
        volume_name=f"qm-output-{run_id}",
        volume_labels=volume_labels,
        budget=budget,
    )

    assert recorded
    assert store.intent["container_labels"] == labels
    assert store.intent["related_containers"][0]["labels"] == volume_labels


def test_container_intent_rejects_cross_owner_labels():
    lease, budget, run_id, labels, volume_labels, store = _case()
    labels["quantmind.user_id"] = "user-2"

    recorded = record_container_intent(
        lease,
        daemon_id="docker-daemon-1",
        container_name=f"qm-lab-run-{run_id}",
        labels=labels,
        related_containers=[
            {"name": f"qm-output-hold-{run_id}", "labels": volume_labels}
        ],
        volume_name=f"qm-output-{run_id}",
        volume_labels=volume_labels,
        budget=budget,
    )

    assert not recorded
    assert store.intent is None


def test_container_intent_requires_the_current_engine_request():
    lease, budget, run_id, labels, volume_labels, store = _case(claim_state="finished")

    recorded = record_container_intent(
        lease,
        daemon_id="docker-daemon-1",
        container_name=f"qm-lab-run-{run_id}",
        labels=labels,
        related_containers=[
            {"name": f"qm-output-hold-{run_id}", "labels": volume_labels}
        ],
        volume_name=f"qm-output-{run_id}",
        volume_labels=volume_labels,
        budget=budget,
    )

    assert not recorded
    assert store.intent is None
