"""
tests/test_cli.py
=================
Tests for the spectralyte CLI (spectralyte/cli.py).

Covers both unit-style tests (direct function calls, no subprocess) and
integration-style tests (subprocess.run against the installed entry point).

Prerequisites: package must be installed in the active environment via
`pip install -e .` for subprocess tests to find the `spectralyte` entry point.

Test groups
-----------
1. _load_embeddings   — valid load, bad extension, missing file, shape errors
2. _parse_config      — empty, typed values, unknown key, invalid value
3. _emit_json         — valid JSON, all keys, schema parity with report.export()
4. audit subcommand   — summary, --json, no ANSI when piped, error paths
5. fix-plan subcommand — each framework, invalid framework, error paths
6. transform subcommand — each strategy, output shape, missing required args
7. version subcommand  — exact output, exit 0
8. Parser unit tests   — build_parser(), subcommand defaults
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from spectralyte import __version__, Spectralyte
from spectralyte.cli import (
    _emit_json,
    _load_embeddings,
    _parse_config,
    build_parser,
    cmd_audit,
    cmd_fix_plan,
    cmd_transform,
    cmd_version,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def npy_file(tmp_path_factory):
    """Write a (100, 64) float64 .npy file for reuse across all tests."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 64).astype(np.float64)
    path = tmp_path_factory.mktemp("data") / "embeddings.npy"
    np.save(str(path), embeddings)
    return str(path)


@pytest.fixture(scope="module")
def report(npy_file):
    """Pre-computed AuditReport for _emit_json tests."""
    embeddings = np.load(npy_file)
    audit = Spectralyte(embeddings, k=5, random_seed=42)
    return audit.run(verbose=False)


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run spectralyte CLI as a subprocess and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "spectralyte.cli"] + args,
        capture_output=True,
        text=True,
        **kwargs,
    )


# ── 1. _load_embeddings ────────────────────────────────────────────────────────


def test_load_npy_returns_ndarray(npy_file):
    """_load_embeddings returns a float64 2D ndarray for a valid .npy file."""
    arr = _load_embeddings(npy_file)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.dtype == np.float64


def test_load_bad_extension_exits():
    """_load_embeddings exits 1 for a non-.npy extension."""
    with pytest.raises(SystemExit) as exc_info:
        _load_embeddings("embeddings.csv")
    assert exc_info.value.code == 1


def test_load_missing_file_exits():
    """_load_embeddings exits 1 for a path that does not exist."""
    with pytest.raises(SystemExit) as exc_info:
        _load_embeddings("/nonexistent/path/embeddings.npy")
    assert exc_info.value.code == 1


def test_load_1d_array_exits(tmp_path):
    """_load_embeddings exits 1 for a 1D array."""
    path = str(tmp_path / "bad.npy")
    np.save(path, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(SystemExit) as exc_info:
        _load_embeddings(path)
    assert exc_info.value.code == 1


def test_load_too_few_rows_exits(tmp_path):
    """_load_embeddings exits 1 when fewer than 3 embeddings are provided."""
    path = str(tmp_path / "tiny.npy")
    np.save(path, np.random.randn(2, 64))
    with pytest.raises(SystemExit) as exc_info:
        _load_embeddings(path)
    assert exc_info.value.code == 1


def test_load_casts_to_float64(tmp_path):
    """_load_embeddings casts any numeric dtype to float64."""
    path = str(tmp_path / "float32.npy")
    np.save(path, np.random.randn(10, 8).astype(np.float32))
    arr = _load_embeddings(path)
    assert arr.dtype == np.float64


# ── 2. _parse_config ──────────────────────────────────────────────────────────


def test_parse_config_empty_list():
    """_parse_config returns {} for an empty list."""
    assert _parse_config([]) == {}


def test_parse_config_none():
    """_parse_config returns {} for None."""
    assert _parse_config(None) == {}


def test_parse_config_int():
    """_parse_config parses k as int."""
    result = _parse_config(["k=5"])
    assert result == {"k": 5}
    assert isinstance(result["k"], int)


def test_parse_config_float():
    """_parse_config parses sensitivity_epsilon as float."""
    result = _parse_config(["sensitivity_epsilon=0.1"])
    assert abs(result["sensitivity_epsilon"] - 0.1) < 1e-9
    assert isinstance(result["sensitivity_epsilon"], float)


def test_parse_config_multiple():
    """_parse_config parses multiple KEY=VALUE pairs."""
    result = _parse_config(["k=7", "random_seed=99"])
    assert result == {"k": 7, "random_seed": 99}


def test_parse_config_sample_size_none():
    """_parse_config maps 'none' to None for sample_size."""
    result = _parse_config(["sample_size=none"])
    assert result["sample_size"] is None


def test_parse_config_invalid_value_exits():
    """_parse_config exits 1 when a value cannot be cast to the expected type."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_config(["k=foo"])
    assert exc_info.value.code == 1


def test_parse_config_unknown_key_warns_and_skips(capsys):
    """_parse_config skips unknown keys with a stderr warning (does not exit)."""
    result = _parse_config(["unknown_param=42"])
    captured = capsys.readouterr()
    assert "unknown_param" in captured.err
    assert "unknown_param" not in result


# ── 3. _emit_json ─────────────────────────────────────────────────────────────


def test_emit_json_is_valid_json(report, capsys):
    """_emit_json writes parseable JSON to stdout."""
    _emit_json(report)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)


def test_emit_json_contains_all_metric_keys(report, capsys):
    """_emit_json output contains all five metric sections."""
    _emit_json(report)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    for key in (
        "anisotropy",
        "dimensionality",
        "density",
        "sensitivity",
        "intrinsic_dim",
    ):
        assert key in data, f"Missing key: {key}"


def test_emit_json_top_level_keys(report, capsys):
    """_emit_json output contains all expected top-level keys."""
    _emit_json(report)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    for key in ("embeddings_shape", "n_issues", "needs_transform", "has_brittle_zones"):
        assert key in data


def test_emit_json_matches_export_schema(report, capsys, tmp_path):
    """_emit_json schema matches report.export() exactly."""
    _emit_json(report)
    captured = capsys.readouterr()
    cli_data = json.loads(captured.out)

    export_path = str(tmp_path / "audit.json")
    report.export(export_path)
    with open(export_path) as f:
        export_data = json.load(f)

    assert set(cli_data.keys()) == set(export_data.keys())
    assert set(cli_data["anisotropy"].keys()) == set(export_data["anisotropy"].keys())
    assert set(cli_data["dimensionality"].keys()) == set(
        export_data["dimensionality"].keys()
    )
    assert set(cli_data["density"].keys()) == set(export_data["density"].keys())
    assert set(cli_data["sensitivity"].keys()) == set(export_data["sensitivity"].keys())
    assert set(cli_data["intrinsic_dim"].keys()) == set(
        export_data["intrinsic_dim"].keys()
    )


# ── 4. audit subcommand ───────────────────────────────────────────────────────


def test_audit_summary_exit_0(npy_file):
    """spectralyte audit exits 0."""
    result = _run(["audit", npy_file])
    assert result.returncode == 0


def test_audit_summary_contains_header(npy_file):
    """spectralyte audit stdout contains the report header."""
    result = _run(["audit", npy_file])
    assert "Spectralyte Audit Report" in result.stdout


def test_audit_json_exit_0(npy_file):
    """spectralyte audit --json exits 0."""
    result = _run(["audit", npy_file, "--json"])
    assert result.returncode == 0


def test_audit_json_is_valid_json(npy_file):
    """spectralyte audit --json stdout is parseable JSON."""
    result = _run(["audit", npy_file, "--json"])
    data = json.loads(result.stdout)
    assert isinstance(data, dict)


def test_audit_json_has_all_metric_keys(npy_file):
    """spectralyte audit --json contains all five metric sections."""
    result = _run(["audit", npy_file, "--json"])
    data = json.loads(result.stdout)
    for key in (
        "anisotropy",
        "dimensionality",
        "density",
        "sensitivity",
        "intrinsic_dim",
    ):
        assert key in data


def test_audit_json_no_ansi_codes(npy_file):
    """spectralyte audit --json stdout contains no ANSI escape codes."""
    result = _run(["audit", npy_file, "--json"])
    assert "\033[" not in result.stdout


def test_audit_summary_no_ansi_when_piped(npy_file):
    """spectralyte audit (no --json) stdout has no ANSI codes when piped."""
    result = _run(["audit", npy_file])
    assert "\033[" not in result.stdout


def test_audit_missing_file_exit_1():
    """spectralyte audit with a nonexistent path exits 1."""
    result = _run(["audit", "/nonexistent/embeddings.npy"])
    assert result.returncode == 1
    assert result.stderr  # error message must be present


def test_audit_config_k(npy_file):
    """spectralyte audit --config k=5 exits 0."""
    result = _run(["audit", npy_file, "--config", "k=5"])
    assert result.returncode == 0


def test_audit_bad_extension_exit_1(tmp_path):
    """spectralyte audit with a .csv file exits 1."""
    path = str(tmp_path / "data.csv")
    open(path, "w").close()
    result = _run(["audit", path])
    assert result.returncode == 1


# ── 5. fix-plan subcommand ────────────────────────────────────────────────────


def test_fix_plan_exit_0(npy_file):
    """spectralyte fix-plan exits 0."""
    result = _run(["fix-plan", npy_file])
    assert result.returncode == 0


def test_fix_plan_contains_header(npy_file):
    """spectralyte fix-plan stdout contains the plan header."""
    result = _run(["fix-plan", npy_file])
    assert "Remediation Plan" in result.stdout


def test_fix_plan_langchain(npy_file):
    """spectralyte fix-plan --framework langchain exits 0 with non-empty output."""
    result = _run(["fix-plan", npy_file, "--framework", "langchain"])
    assert result.returncode == 0
    assert result.stdout.strip()


def test_fix_plan_llamaindex(npy_file):
    """spectralyte fix-plan --framework llamaindex exits 0 with non-empty output."""
    result = _run(["fix-plan", npy_file, "--framework", "llamaindex"])
    assert result.returncode == 0
    assert result.stdout.strip()


def test_fix_plan_generic(npy_file):
    """spectralyte fix-plan --framework generic exits 0 with non-empty output."""
    result = _run(["fix-plan", npy_file, "--framework", "generic"])
    assert result.returncode == 0
    assert result.stdout.strip()


def test_fix_plan_invalid_framework(npy_file):
    """spectralyte fix-plan --framework invalid exits non-zero."""
    result = _run(["fix-plan", npy_file, "--framework", "pytorch"])
    assert result.returncode != 0


def test_fix_plan_missing_file_exit_1():
    """spectralyte fix-plan with nonexistent path exits 1."""
    result = _run(["fix-plan", "/nonexistent/embeddings.npy"])
    assert result.returncode == 1
    assert result.stderr


# ── 6. transform subcommand ───────────────────────────────────────────────────


def test_transform_whiten_creates_file(npy_file, tmp_path):
    """spectralyte transform --strategy whiten creates the output file."""
    out = str(tmp_path / "whitened.npy")
    result = _run(["transform", npy_file, "--strategy", "whiten", "--output", out])
    assert result.returncode == 0
    assert os.path.exists(out)


def test_transform_whiten_output_loadable(npy_file, tmp_path):
    """transform whiten output is a valid loadable .npy file."""
    out = str(tmp_path / "whitened.npy")
    _run(["transform", npy_file, "--strategy", "whiten", "--output", out])
    arr = np.load(out)
    assert arr.ndim == 2


def test_transform_whiten_preserves_shape(npy_file, tmp_path):
    """transform whiten output shape matches input (100, 64)."""
    out = str(tmp_path / "whitened.npy")
    _run(["transform", npy_file, "--strategy", "whiten", "--output", out])
    arr = np.load(out)
    assert arr.shape == (100, 64)


def test_transform_abtt_creates_file(npy_file, tmp_path):
    """spectralyte transform --strategy abtt creates the output file."""
    out = str(tmp_path / "abtt.npy")
    result = _run(["transform", npy_file, "--strategy", "abtt", "--output", out])
    assert result.returncode == 0
    assert os.path.exists(out)


def test_transform_abtt_preserves_shape(npy_file, tmp_path):
    """transform abtt output shape matches input (100, 64)."""
    out = str(tmp_path / "abtt.npy")
    _run(["transform", npy_file, "--strategy", "abtt", "--output", out])
    arr = np.load(out)
    assert arr.shape == (100, 64)


def test_transform_pca_reduce_creates_file(npy_file, tmp_path):
    """spectralyte transform --strategy pca_reduce creates the output file."""
    out = str(tmp_path / "reduced.npy")
    result = _run(["transform", npy_file, "--strategy", "pca_reduce", "--output", out])
    assert result.returncode == 0
    assert os.path.exists(out)


def test_transform_pca_reduce_reduces_dims(npy_file, tmp_path):
    """transform pca_reduce output has fewer dims than the input (64)."""
    out = str(tmp_path / "reduced.npy")
    _run(["transform", npy_file, "--strategy", "pca_reduce", "--output", out])
    arr = np.load(out)
    assert arr.ndim == 2
    assert arr.shape[0] == 100
    assert arr.shape[1] <= 64


def test_transform_confirmation_on_stderr(npy_file, tmp_path):
    """transform writes the confirmation line to stderr, not stdout."""
    out = str(tmp_path / "whitened.npy")
    result = _run(["transform", npy_file, "--strategy", "whiten", "--output", out])
    assert "Saved" in result.stderr
    assert "Saved" not in result.stdout


def test_transform_missing_strategy_exits(npy_file, tmp_path):
    """transform without --strategy exits non-zero."""
    out = str(tmp_path / "out.npy")
    result = _run(["transform", npy_file, "--output", out])
    assert result.returncode != 0


def test_transform_missing_output_exits(npy_file):
    """transform without --output exits non-zero."""
    result = _run(["transform", npy_file, "--strategy", "whiten"])
    assert result.returncode != 0


def test_transform_missing_file_exit_1(tmp_path):
    """transform with nonexistent input exits 1."""
    out = str(tmp_path / "out.npy")
    result = _run(
        [
            "transform",
            "/nonexistent/embeddings.npy",
            "--strategy",
            "whiten",
            "--output",
            out,
        ]
    )
    assert result.returncode == 1


# ── 7. version subcommand ─────────────────────────────────────────────────────


def test_version_exit_0():
    """spectralyte version exits 0."""
    result = _run(["version"])
    assert result.returncode == 0


def test_version_exact_output():
    """spectralyte version stdout is 'spectralyte <version>'."""
    result = _run(["version"])
    assert result.stdout.strip() == f"spectralyte {__version__}"


# ── 8. Parser unit tests ──────────────────────────────────────────────────────


def test_build_parser_returns_parser():
    """build_parser() returns an ArgumentParser instance."""
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_audit_json_default_false():
    """audit subcommand defaults --json to False."""
    parser = build_parser()
    args = parser.parse_args(["audit", "foo.npy"])
    assert args.json is False


def test_parser_audit_config_default_none():
    """audit subcommand defaults --config to None."""
    parser = build_parser()
    args = parser.parse_args(["audit", "foo.npy"])
    assert args.config is None


def test_parser_fix_plan_framework_default():
    """fix-plan subcommand defaults --framework to 'generic'."""
    parser = build_parser()
    args = parser.parse_args(["fix-plan", "foo.npy"])
    assert args.framework == "generic"


def test_parser_version_func():
    """version subcommand sets func to cmd_version."""
    parser = build_parser()
    args = parser.parse_args(["version"])
    assert args.func is cmd_version


def test_parser_audit_func():
    """audit subcommand sets func to cmd_audit."""
    parser = build_parser()
    args = parser.parse_args(["audit", "foo.npy"])
    assert args.func is cmd_audit


def test_parser_fix_plan_func():
    """fix-plan subcommand sets func to cmd_fix_plan."""
    parser = build_parser()
    args = parser.parse_args(["fix-plan", "foo.npy"])
    assert args.func is cmd_fix_plan


def test_parser_transform_func():
    """transform subcommand sets func to cmd_transform."""
    parser = build_parser()
    args = parser.parse_args(
        ["transform", "foo.npy", "--strategy", "whiten", "--output", "out.npy"]
    )
    assert args.func is cmd_transform


# ── Fitted-transform persistence (--save-fit / --apply-fit) ────────────────────
#
# Without these, `spectralyte transform` produced an index that could not be
# queried correctly: the fitted mapping died with the process, so incoming
# queries had no way to reach the same space as the corpus.

def test_transform_save_fit_creates_file(npy_file, tmp_path):
    """--save-fit writes the fitted transform alongside the vectors."""
    out, fit = str(tmp_path / "c.npy"), str(tmp_path / "fit.npz")
    result = _run(["transform", npy_file, "--strategy", "whiten",
                   "--output", out, "--save-fit", fit])
    assert result.returncode == 0
    assert os.path.exists(fit)
    assert "Saved fitted transform" in result.stderr


def test_transform_warns_when_fit_is_not_saved(npy_file, tmp_path):
    """
    Transforming an index without keeping the fit is a footgun; the command
    must say so rather than exiting silently successful.
    """
    out = str(tmp_path / "c.npy")
    result = _run(["transform", npy_file, "--strategy", "whiten", "--output", out])
    assert result.returncode == 0
    assert "--save-fit" in result.stderr
    assert "queries must go through this same transform" in result.stderr


def test_transform_no_warning_when_fit_is_saved(npy_file, tmp_path):
    """The nudge is noise once the fit is being kept."""
    out, fit = str(tmp_path / "c.npy"), str(tmp_path / "fit.npz")
    result = _run(["transform", npy_file, "--strategy", "whiten",
                   "--output", out, "--save-fit", fit])
    assert "queries must go through this same transform" not in result.stderr


def test_apply_fit_reproduces_the_original_mapping(npy_file, tmp_path):
    """
    The point of persistence: vectors put through a saved fit must land
    exactly where they land when transformed in the fitting process.
    """
    corpus_out, fit = str(tmp_path / "c.npy"), str(tmp_path / "fit.npz")
    _run(["transform", npy_file, "--strategy", "whiten",
          "--output", corpus_out, "--save-fit", fit])

    reapplied = str(tmp_path / "again.npy")
    result = _run(["transform", npy_file, "--strategy", "whiten",
                   "--apply-fit", fit, "--output", reapplied])
    assert result.returncode == 0

    assert np.allclose(np.load(corpus_out), np.load(reapplied), atol=1e-6)


def test_apply_fit_does_not_refit(npy_file, tmp_path):
    """
    A saved fit must be applied as-is. Transforming a slice through it has to
    match that slice of the full corpus — if --apply-fit refit on the input,
    the two would differ.
    """
    corpus_out, fit = str(tmp_path / "c.npy"), str(tmp_path / "fit.npz")
    _run(["transform", npy_file, "--strategy", "whiten",
          "--output", corpus_out, "--save-fit", fit])

    full = np.load(npy_file)
    slice_path = str(tmp_path / "slice.npy")
    np.save(slice_path, full[:5])

    slice_out = str(tmp_path / "slice_fixed.npy")
    _run(["transform", slice_path, "--strategy", "whiten",
          "--apply-fit", fit, "--output", slice_out])

    assert np.allclose(np.load(slice_out), np.load(corpus_out)[:5], atol=1e-6)


def test_apply_fit_rejects_mismatched_width(npy_file, tmp_path):
    """A fit from one space must not be applied to another."""
    fit = str(tmp_path / "fit.npz")
    _run(["transform", npy_file, "--strategy", "whiten",
          "--output", str(tmp_path / "c.npy"), "--save-fit", fit])

    wrong = str(tmp_path / "wrong.npy")
    np.save(wrong, np.random.RandomState(0).randn(20, 999))

    result = _run(["transform", wrong, "--strategy", "whiten",
                   "--apply-fit", fit, "--output", str(tmp_path / "o.npy")])
    assert result.returncode == 1
    assert "dimensions" in result.stderr


def test_apply_fit_missing_file_exits_1(npy_file, tmp_path):
    """A missing fit file is an error, not a silent refit."""
    result = _run(["transform", npy_file, "--strategy", "whiten",
                   "--apply-fit", str(tmp_path / "nope.npz"),
                   "--output", str(tmp_path / "o.npy")])
    assert result.returncode == 1
    assert "Error loading transform" in result.stderr
