"""Lambda remote session management for capo."""

from capo.remote.config import (
    LambdaSessionConfig,
    LOCAL_TMUX_SESSION,
    LOCAL_WINDOW_REMOTE,
    LOCAL_WINDOW_SYNC,
    LOCAL_WINDOW_LOCAL,
    REMOTE_TMUX_SESSION,
    REMOTE_RUN_ROOT,
    LOCAL_ARTIFACTS_ROOT,
    SSH_READY_TIMEOUT_S,
    SSH_READY_POLL_S,
)
from capo.remote.lambda_session import (
    LambdaSession,
    LambdaInstance,
    parse_instance,
    provision_instance,
    get_instance,
    list_instances,
    wait_for_instance_ip,
    wait_for_ssh_ready,
    ensure_ssh_alias,
    terminate_instance,
    safe_terminate_instance,
)
from capo.remote.lambda_pricing import (
    LambdaCostEstimate,
    estimate_cost,
    get_instance_type_price,
    parse_datetime,
)
from capo.remote.lambda_ssh_keys import (
    find_local_ssh_keys,
    list_remote_ssh_keys,
)
from capo.remote.lambda_instance_types import list_instance_types
from capo.remote.lambda_preflight import run_preflight
from capo.remote.rsync_manager import (
    RsyncManager,
    RsyncResult,
    upload_run_inputs,
    download_run_outputs,
    sync_run_status,
    sync_run_logs,
)
from capo.remote.tmux_manager import (
    TmuxError,
    TmuxManager,
    ensure_local_workspace,
    ensure_local_window,
    send_to_local_window,
    capture_local_window,
    ensure_remote_tmux,
    send_to_remote_tmux,
    capture_remote_tmux,
)
from capo.remote.run_manager import (
    RunSpec,
    RemoteRunPaths,
    RunStatus,
    get_remote_run_paths,
    prepare_remote_run_dir,
    write_remote_spec,
    start_remote_inference,
    start_remote_finetune,
    stop_remote_run,
    read_remote_run_status,
)

__all__ = [
    # Config
    "LambdaSessionConfig",
    "LOCAL_TMUX_SESSION",
    "LOCAL_WINDOW_REMOTE",
    "LOCAL_WINDOW_SYNC",
    "LOCAL_WINDOW_LOCAL",
    "REMOTE_TMUX_SESSION",
    "REMOTE_RUN_ROOT",
    "LOCAL_ARTIFACTS_ROOT",
    "SSH_READY_TIMEOUT_S",
    "SSH_READY_POLL_S",
    # Lambda session
    "LambdaSession",
    "LambdaInstance",
    "parse_instance",
    "provision_instance",
    "get_instance",
    "list_instances",
    "wait_for_instance_ip",
    "wait_for_ssh_ready",
    "ensure_ssh_alias",
    "terminate_instance",
    "safe_terminate_instance",
    # Pricing
    "LambdaCostEstimate",
    "estimate_cost",
    "get_instance_type_price",
    "parse_datetime",
    # SSH keys
    "find_local_ssh_keys",
    "list_remote_ssh_keys",
    # Instance types
    "list_instance_types",
    # Preflight
    "run_preflight",
    # Rsync
    "RsyncManager",
    "RsyncResult",
    "upload_run_inputs",
    "download_run_outputs",
    "sync_run_status",
    "sync_run_logs",
    # Tmux
    "TmuxError",
    "TmuxManager",
    "ensure_local_workspace",
    "ensure_local_window",
    "send_to_local_window",
    "capture_local_window",
    "ensure_remote_tmux",
    "send_to_remote_tmux",
    "capture_remote_tmux",
    # Run manager
    "RunSpec",
    "RemoteRunPaths",
    "RunStatus",
    "get_remote_run_paths",
    "prepare_remote_run_dir",
    "write_remote_spec",
    "start_remote_inference",
    "start_remote_finetune",
    "stop_remote_run",
    "read_remote_run_status",
]
