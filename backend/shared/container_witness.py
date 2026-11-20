"""Read-only Docker evidence that the exact old PID namespace has ended."""

import json
import re

from backend.shared.process_witness import capture_process

_SERVICES = {
    "engine",
    "quantmind",
    "celery-worker",
    "celery-worker-backtest",
    "celery-worker-heavy",
    "celery-worker-legacy",
    "celery-worker-rdagent",
}
_READ_NAMESPACE = [
    "python",
    "-I",
    "-S",
    "-c",
    "import json,os,pathlib; s=pathlib.Path('/proc/1/stat').read_text(); "
    "print(json.dumps({'pid_namespace':os.readlink('/proc/self/ns/pid'),"
    "'init_start_ticks':s[s.rindex(')')+2:].split()[19],"
    "'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))",
]


def capture_container_identity(client, witness, *, allowed_services=None):
    """Bind a process witness to its exact Docker daemon and Compose service."""
    if not isinstance(witness, dict):
        raise ValueError("Process witness is required")
    host = witness.get("host", "")
    if not isinstance(host, str) or not re.fullmatch(r"[0-9a-f]{12,64}", host):
        raise ValueError("Process witness host is not a container ID")
    container = client.containers.get(host)
    container.reload()
    attrs = getattr(container, "attrs", {}) or {}
    config = attrs.get("Config", {}) or {}
    labels = config.get("Labels", {}) or {}
    service = labels.get("com.docker.compose.service")
    project = labels.get("com.docker.compose.project")
    services = allowed_services or _SERVICES
    container_id = str(getattr(container, "id", ""))
    daemon_id = str(client.info().get("ID") or "")
    state = attrs.get("State", {}) or {}
    if (
        not container_id.startswith(host)
        or config.get("Hostname") != host
        or not state.get("Running")
        or service not in services
        or not isinstance(project, str)
        or not project
        or not daemon_id
    ):
        raise ValueError("Docker container identity could not be verified")
    return {
        **witness,
        "container_id": container_id,
        "docker_daemon_id": daemon_id,
        "compose_project": project,
        "compose_service": service,
    }


def container_namespace_has_stopped(client, witness):
    """Unknown/deleted containers or foreign hosts are not proof of termination.

    A live restarted container has a different PID namespace. The trusted fixed
    command inspects only namespace/boot IDs, never environment or user files.
    """
    if not isinstance(witness, dict):
        return False
    try:
        host = witness.get("host", "")
        if not isinstance(host, str) or not re.fullmatch(r"[0-9a-f]{12,64}", host):
            return False
        if not re.fullmatch(r"pid:\[\d+\]", str(witness.get("pid_namespace", ""))):
            return False
        if (
            not isinstance(witness.get("boot_id"), str)
            or not re.fullmatch(r"[0-9a-f-]{36}", witness["boot_id"])
            or type(witness.get("pid")) is not int
            or witness["pid"] <= 0
            or not re.fullmatch(r"\d+", str(witness.get("start_ticks", "")))
        ):
            return False
        container_id = witness.get("container_id") or host
        if (
            not isinstance(container_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", container_id)
            or not container_id.startswith(host)
        ):
            return False
        expected_daemon_id = witness.get("docker_daemon_id")
        if (
            expected_daemon_id
            and str(client.info().get("ID") or "") != expected_daemon_id
        ):
            return False
        container = client.containers.get(container_id)
        container.reload()
        config = container.attrs.get("Config", {})
        if not container.id.startswith(host) or config.get("Hostname") != host:
            return False
        labels = config.get("Labels", {}) or {}
        service = labels.get("com.docker.compose.service")
        project = labels.get("com.docker.compose.project")
        if service not in _SERVICES:
            return False
        if witness.get("compose_service") and service != witness["compose_service"]:
            return False
        if witness.get("compose_project") and project != witness["compose_project"]:
            return False
        state = container.attrs.get("State", {})
        if state.get("Status") in {"exited", "dead"} and not state.get("Running"):
            return True
        if not state.get("Running"):
            return False
        current_boot_id = capture_process()["boot_id"]
        if witness.get("boot_id") != current_boot_id:
            # The exact container and daemon survived, but the recorded OS boot
            # ended, so its old PID namespace cannot still be executing.
            return bool(expected_daemon_id)
        result = container.exec_run(_READ_NAMESPACE)
        if result.exit_code != 0 or len(result.output) > 4096:
            return False
        observed = json.loads(result.output)
        return (
            observed.get("boot_id") == witness["boot_id"]
            and isinstance(observed.get("pid_namespace"), str)
            and observed["pid_namespace"].startswith("pid:[")
            and (
                observed["pid_namespace"] != witness["pid_namespace"]
                # Linux may reuse a namespace inode immediately after restart.
                # A new init started after the recorded process proves that
                # the recorded execution belonged to the previous incarnation.
                or int(observed["init_start_ticks"]) > int(witness["start_ticks"])
            )
        )
    except Exception:
        return False
