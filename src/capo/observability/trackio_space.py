"""Seed an HF Space with the trackio dashboard.

Background: when an agent creates a Space (or the user pre-creates one), it
ships with no app.py — so the iframe shows the default Gradio "Get started"
template instead of charts. Trackio's `init()` detects the existing Space but
does NOT auto-bootstrap dashboard files into it, so the dashboard never
materializes.

This helper writes the three files trackio's runtime expects (README.md,
requirements.txt, app.py) into the Space, idempotently, BEFORE the job that
will call `trackio.init()` runs. We deliberately omit `hf_oauth: true` from
the README so the embedded iframe renders without a login click — per-user
privacy is enforced by namespace ownership instead.

Beyond the dashboard files, the helper also creates the metrics bucket and
mounts it on the Space at `/data` (with `TRACKIO_DIR` / `TRACKIO_BUCKET_ID`
Space variables). Without this, the running job writes metrics into a bucket
that the dashboard Space can't read, and the iframe shows "No projects".
"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

# HF Hub imports are deferred to call time. The bucket/volume APIs
# (`create_bucket`, `Volume`, `set_space_volumes`) are only available on
# very recent `huggingface_hub` builds; importing them at module load would
# break the orchestrator on older installs. Callers wrap us in try/except
# and fall back to "no dashboard" on ImportError.


_README = """---
title: Trackio Dashboard
emoji: 📊
colorFrom: pink
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
tags:
  - trackio
---

Embedded trackio dashboard for capo runs.
"""

_REQUIREMENTS = "trackio\n"
_APP_PY = "import trackio\ntrackio.show()\n"

_FILES = {
    "README.md": _README,
    "requirements.txt": _REQUIREMENTS,
    "app.py": _APP_PY,
}


def _already_seeded(api: Any, space_id: str) -> bool:
    """Cheap check: does the Space already have a trackio dashboard app.py?

    Avoids re-uploading the same three files on every job submission. We look
    for the literal `trackio.show` call which is the load-bearing line — any
    other app.py shape (the default gradio shell, a stale custom one) means
    we should re-seed.
    """
    try:
        path = api.hf_hub_download(
            repo_id=space_id, repo_type="space", filename="app.py"
        )
    except Exception:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            return "trackio.show" in f.read()
    except OSError:
        return False


def _get_space_volumes(api: Any, space_id: str) -> list:
    """Return mounted volumes for a Space.

    `get_space_runtime()` doesn't always populate `volumes` even when the
    mount exists; mirror trackio's fallback to `space_info().runtime.volumes`.
    """
    runtime = api.get_space_runtime(space_id)
    if getattr(runtime, "volumes", None):
        return list(runtime.volumes)
    info = api.space_info(space_id)
    if info.runtime and getattr(info.runtime, "volumes", None):
        return list(info.runtime.volumes)
    return []


def _ensure_bucket_mounted(
    api: Any,
    space_id: str,
    bucket_id: str,
    hf_token: str,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Create the bucket if missing, mount it at `/data` on the Space, and
    set the `TRACKIO_DIR` / `TRACKIO_BUCKET_ID` Space variables. Idempotent —
    skips work that has already been done.
    """
    from huggingface_hub import Volume, add_space_variable, create_bucket

    try:
        create_bucket(bucket_id, private=True, exist_ok=True, token=hf_token)
    except Exception as exc:
        if log:
            log(
                f"[trackio] bucket creation skipped (continuing without persistent storage): {exc}"
            )
        return

    existing = _get_space_volumes(api, space_id)
    already_mounted = any(
        getattr(v, "type", None) == "bucket"
        and getattr(v, "source", None) == bucket_id
        and getattr(v, "mount_path", None) == "/data"
        for v in existing
    )
    if not already_mounted:
        preserved = [
            v
            for v in existing
            if not (
                getattr(v, "type", None) == "bucket"
                and (
                    getattr(v, "source", None) == bucket_id
                    or getattr(v, "mount_path", None) == "/data"
                )
            )
        ]
        api.set_space_volumes(
            space_id,
            preserved + [Volume(type="bucket", source=bucket_id, mount_path="/data")],
        )
        if log:
            log(f"[trackio] mounted bucket {bucket_id} at /data on {space_id}")

    variables = api.get_space_variables(space_id)
    desired = {
        "TRACKIO_DIR": "/data/trackio",
        "TRACKIO_BUCKET_ID": bucket_id,
    }
    for key, value in desired.items():
        if getattr(variables.get(key), "value", None) != value:
            add_space_variable(space_id, key, value, token=hf_token)


def ensure_trackio_dashboard(
    space_id: str,
    hf_token: str,
    log: Optional[Callable[[str], None]] = None,
) -> bool:
    """Make sure *space_id* is fully wired for trackio:
    1. Space exists with our dashboard files (README without `hf_oauth`,
       `requirements.txt`, `app.py` calling `trackio.show`).
    2. Bucket `<space_id>-bucket` exists, is mounted at `/data`, and the
       Space has `TRACKIO_DIR` / `TRACKIO_BUCKET_ID` variables set.

    Idempotent — re-running is cheap. Returns True if any seeding happened
    in step (1), False if the dashboard files were already in place. Bucket
    mount is always re-checked.
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)

    create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=hf_token,
    )

    seeded_files = False
    if _already_seeded(api, space_id):
        if log:
            log(f"[trackio] dashboard already seeded on {space_id}")
    else:
        if log:
            log(f"[trackio] seeding dashboard files into {space_id}")
        for path_in_repo, content in _FILES.items():
            api.upload_file(
                path_or_fileobj=io.BytesIO(content.encode("utf-8")),
                path_in_repo=path_in_repo,
                repo_id=space_id,
                repo_type="space",
                commit_message=f"capo: seed trackio dashboard ({path_in_repo})",
            )
        seeded_files = True

    bucket_id = f"{space_id}-bucket"
    _ensure_bucket_mounted(api, space_id, bucket_id, hf_token, log)

    return seeded_files


def resolve_default_space_id(hf_token: str) -> Optional[str]:
    """Resolve the current HF user's default capo trackio Space ID.

    Returns `<username>/capo-trackio` on success, or None if `whoami` fails
    (token rejected, network error, etc.). Returning None lets the caller
    take the warn-and-skip path without raising.

    Bucket IDs must be lowercase and use hyphens; HF usernames are already
    lowercased, but org names may contain uppercase — normalise defensively.
    """
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=hf_token).whoami()
    except Exception:
        return None
    name = (who or {}).get("name")
    if not name:
        return None
    return f"{name.lower()}/capo-trackio"
