"""Regression tests for post-launch failure diagnosis.

The headline regression: ``remote_run_dir`` arrives as ``~/capo_runs/<run_id>``
(a tilde path). The original ``_remote_read`` did ``cat {shlex.quote(path)}``,
which single-quotes the tilde so the remote shell never expands it — every read
returned empty and a perfectly recoverable ``ModuleNotFoundError`` crash was
mis-classified as "unknown / unrecoverable". These tests lock the fix.
"""

from __future__ import annotations

import json
import shlex

import capo.orchestration.post_launch_diagnostics as pld
from capo.orchestration.post_launch_diagnostics import (
    _classify_from_traceback,
    _shell_path_arg,
    diagnose_post_launch_failure,
)

# A real boltz-style traceback: the failure mode this whole report is about.
_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/ubuntu/capo_runs/run/train.py", line 10, in <module>\n'
    "    main()\n"
    '  File "/home/ubuntu/.local/lib/python3.10/site-packages/boltz/model/'
    'layers/triangular_mult.py", line 22, in kernel_triangular_mult\n'
    "    from cuequivariance_torch.primitives.triangle import "
    "triangle_multiplicative_update\n"
    "ModuleNotFoundError: No module named 'cuequivariance_torch'\n"
)


def test_shell_path_arg_expands_tilde():
    arg = _shell_path_arg("~/capo_runs/run/outputs/train_err.log")
    # The original bug: the whole path got single-quoted, tilde and all.
    assert not arg.startswith("'~")
    assert arg.startswith('"$HOME/"')
    # The remainder is preserved verbatim (shlex.quote only quotes when needed).
    assert arg.endswith("capo_runs/run/outputs/train_err.log")


def test_shell_path_arg_expands_home_var():
    arg = _shell_path_arg("$HOME/capo_runs/run/outputs/train.log")
    assert "$HOME" in arg
    assert not arg.startswith("'$HOME")


def test_shell_path_arg_quotes_plain_path():
    # Absolute paths must still be safely quoted, unchanged in meaning.
    assert _shell_path_arg("/abs/path with space/x.log") == shlex.quote(
        "/abs/path with space/x.log"
    )


def test_classify_cuequivariance_module_is_cuda_kernel():
    # A missing cuequivariance kernel is NOT a generic script bug: the fix is to
    # install the GPU kernel set / fall back to --no_kernels, not to patch a file.
    category, _failing, summary = _classify_from_traceback(_TRACEBACK)
    assert category == "cuda_kernel"
    assert "cuequivariance" in summary or "kernel" in summary.lower()


def test_classify_local_module_not_found_is_script_bug():
    # A missing LOCAL module (the run's own src.*) IS a code bug, not a pip gap —
    # it must stay script_bug, never missing_dependency.
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/home/ubuntu/capo_runs/run/train.py", line 5, in <module>\n'
        "    from src.train.main import main\n"
        "ModuleNotFoundError: No module named 'src.train.main'\n"
    )
    category, _failing, _summary = _classify_from_traceback(tb)
    assert category == "script_bug"


def test_diagnose_reads_tilde_remote_dir(tmp_path, monkeypatch):
    """End-to-end: a tilde remote_run_dir must still yield a non-empty
    traceback and a recoverable classification (``cuda_kernel`` for this
    boltz cuequivariance traceback).

    We run the 'remote' command locally via ``sh -c`` with HOME pointed at a
    fake home that mirrors the remote layout — exercising the exact shell
    expansion ``_shell_path_arg`` produces, without touching real SSH.
    """
    fake_home = tmp_path / "home"
    outputs = fake_home / "capo_runs" / "run" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "train_err.log").write_text(_TRACEBACK, encoding="utf-8")

    def fake_ssh_cmd(ssh_alias, key_path, remote_cmd):
        # Drop ssh; run the same command under a shell with our fake HOME.
        return ["sh", "-c", f"HOME={shlex.quote(str(fake_home))}; {remote_cmd}"]

    monkeypatch.setattr(pld, "_ssh_cmd", fake_ssh_cmd)

    local_run_dir = tmp_path / "local"
    (local_run_dir / "reports").mkdir(parents=True)

    failure = diagnose_post_launch_failure(
        ssh_alias="dummy",
        key_path=None,
        remote_run_dir="~/capo_runs/run",
        local_run_dir=local_run_dir,
    )

    assert failure is not None
    assert failure.traceback_tail.strip() != ""           # the bug: was empty
    assert failure.failure_category == "cuda_kernel"       # cuequivariance kernel, not a code bug
    assert failure.recoverable is True                     # the bug: was False
    assert failure.train_err_tail_lines > 0

    persisted = json.loads(
        (local_run_dir / "reports" / "post_launch_failure.json").read_text()
    )
    assert persisted["failure_category"] == "cuda_kernel"


# A loader that can't import the parquet engine observes zero columns and writes the
# WRONG label (data_schema_mismatch). The diagnostic must override to
# missing_dependency — the real fix is `pip install`, not a schema edit.
_PYARROW_LOAD_ERR = {
    "failure_category": "data_schema_mismatch",
    "missing_columns": ["seq", "label"],
    "observed_columns": [],
    "error_message": (
        "All load attempts failed: Unable to find a usable engine; tried using "
        "'pyarrow', 'fastparquet'. Missing optional dependency 'pyarrow'. "
        "datasets=No module named 'datasets'"
    ),
    "hub_lookup_failed": True,
}


def _fake_ssh_to_home(fake_home, monkeypatch):
    def fake_ssh_cmd(ssh_alias, key_path, remote_cmd):
        return ["sh", "-c", f"HOME={shlex.quote(str(fake_home))}; {remote_cmd}"]
    monkeypatch.setattr(pld, "_ssh_cmd", fake_ssh_cmd)


def test_missing_dependency_overrides_schema_mislabel(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    outputs = fake_home / "capo_runs" / "run" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "dataset_load_error.json").write_text(
        json.dumps(_PYARROW_LOAD_ERR), encoding="utf-8"
    )
    (outputs / "train_err.log").write_text(
        "RuntimeError: boltz predict failed with exit code 1\n", encoding="utf-8"
    )
    _fake_ssh_to_home(fake_home, monkeypatch)

    local_run_dir = tmp_path / "local"
    (local_run_dir / "reports").mkdir(parents=True)
    failure = diagnose_post_launch_failure(
        ssh_alias="dummy", key_path=None,
        remote_run_dir="~/capo_runs/run", local_run_dir=local_run_dir,
    )
    assert failure is not None
    assert failure.failure_category == "missing_dependency"   # NOT data_schema_mismatch
    assert "pyarrow" in failure.missing_packages
    assert "datasets" in failure.missing_packages
    assert failure.recoverable is True
    # The remediation is now a GENERIC scaffold (no hard-coded pip command): it
    # must still name the packages, tell the agent to INSTALL them, and explain
    # this is an environment gap that only LOOKS like a schema error.
    rem = failure.remediation.lower()
    assert "install" in rem and "schema" in rem
    assert "pyarrow" in failure.remediation and "datasets" in failure.remediation


# The exact failure this whole change is about: a cosmetic savefig bug in the
# eval-plotting path killed a run that had already trained to a checkpoint. The
# atomic-write temp file ".../loss_curve.png.tmp" made matplotlib infer format
# "tmp". The CSVs + checkpoint were already on disk, so this must classify as a
# RECOVERABLE plotting_bug (disable inline plots + relaunch), never unknown.
_PLOTTING_TRACEBACK = (
    "2026-07-02 07:05:26 ERROR Uncaught exception in training:\n"
    "Traceback (most recent call last):\n"
    '  File "/home/ubuntu/capo_runs/run/src/train/main.py", line 304, in do_eval\n'
    "    made = generate_plots(str(eval_csv), str(results / 'plots'))\n"
    '  File "/home/ubuntu/capo_runs/run/src/eval/plots.py", line 39, in _save\n'
    '    fig.savefig(tmp, dpi=150, bbox_inches="tight")\n'
    '  File "/home/ubuntu/.local/lib/python3.10/site-packages/matplotlib/'
    'figure.py", line 3497, in savefig\n'
    "    self.canvas.print_figure(fname, **kwargs)\n"
    "ValueError: Format 'tmp' is not supported "
    "(supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, "
    "svgz, tif, tiff, webp)\n"
)


def test_classify_plotting_crash_is_plotting_bug():
    category, failing, summary = _classify_from_traceback(_PLOTTING_TRACEBACK)
    assert category == "plotting_bug"
    # points at the user's plot module, not the matplotlib framework frame
    assert failing is not None and failing.endswith("src/eval/plots.py")
    assert "plotting" in summary.lower()


def test_plotting_crash_does_not_steal_genuine_dataset_keyerror():
    # A KeyError in the DATA loader (no plotting frames) must stay
    # data_schema_mismatch — the plotting branch requires a plotting frame.
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/home/ubuntu/capo_runs/run/src/data/dataset.py", line 40, in load\n'
        "    seq = row['sequence']\n"
        "KeyError: 'sequence'\n"
    )
    category, _failing, _summary = _classify_from_traceback(tb)
    assert category == "data_schema_mismatch"


def test_diagnose_prefers_stdout_traceback_over_progress_log(tmp_path, monkeypatch):
    """The headline regression for this change.

    The crash traceback lands in stdout.log (nohup redirect), while train.log
    holds only progress lines and train_err.log is empty. The OLD code stopped
    at the first non-empty log (train.log), saw no traceback, and returned
    "unknown / recoverable=False" — so the recovery loop had nothing to act on.
    The fix scans all logs and prefers the one carrying a traceback.
    """
    fake_home = tmp_path / "home"
    outputs = fake_home / "capo_runs" / "run" / "outputs"
    outputs.mkdir(parents=True)
    # train_err.log empty; train.log = benign progress; stdout.log = the crash.
    (outputs / "train_err.log").write_text("", encoding="utf-8")
    (outputs / "train.log").write_text(
        "epoch 0 step 840 loss 0.13\nepoch 0 step 850 loss 0.12\n", encoding="utf-8"
    )
    (outputs / "stdout.log").write_text(_PLOTTING_TRACEBACK, encoding="utf-8")
    _fake_ssh_to_home(fake_home, monkeypatch)

    local_run_dir = tmp_path / "local"
    (local_run_dir / "reports").mkdir(parents=True)
    failure = diagnose_post_launch_failure(
        ssh_alias="dummy", key_path=None,
        remote_run_dir="~/capo_runs/run", local_run_dir=local_run_dir,
    )
    assert failure is not None
    assert failure.failure_category == "plotting_bug"     # was: unknown
    assert failure.recoverable is True                     # was: False
    # Generic scaffold (no hard-coded flag names): it must convey that the crash
    # is cosmetic, that inline plotting should be disabled, and that the finalizer
    # regenerates the figures — leaving the exact mechanism to the recovery agent.
    rem = failure.remediation.lower()
    assert "plotting" in rem
    assert "disable inline plotting" in rem or "disable_plotting" in rem
    assert "cosmetic" in rem


def test_missing_dependency_from_pure_traceback(tmp_path, monkeypatch):
    # No structured dataset_load_error.json — only a raw traceback in train_err.log.
    fake_home = tmp_path / "home"
    outputs = fake_home / "capo_runs" / "run" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "train_err.log").write_text(
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'pyarrow'\n",
        encoding="utf-8",
    )
    _fake_ssh_to_home(fake_home, monkeypatch)

    local_run_dir = tmp_path / "local"
    (local_run_dir / "reports").mkdir(parents=True)
    failure = diagnose_post_launch_failure(
        ssh_alias="dummy", key_path=None,
        remote_run_dir="~/capo_runs/run", local_run_dir=local_run_dir,
    )
    assert failure is not None
    assert failure.failure_category == "missing_dependency"
    assert "pyarrow" in failure.missing_packages
    assert failure.recoverable is True
