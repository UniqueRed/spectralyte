"""
spectralyte/cli.py
==================
Command-line interface for Spectralyte.

Entry point: spectralyte = "spectralyte.cli:main"

Subcommands
-----------
  spectralyte audit <path.npy> [--json] [--config KEY=VALUE ...]
  spectralyte fix-plan <path.npy> [--framework langchain|llamaindex|generic] [--config ...]
  spectralyte transform <path.npy> --strategy whiten|abtt|pca_reduce --output <out.npy> [--config ...]
  spectralyte version

stdout / stderr discipline
--------------------------
  stdout  — summary text, JSON payload (--json), fix-plan text, version string
  stderr  — all errors, warnings, transform confirmation line

This keeps `spectralyte audit --json | jq .` and similar pipelines clean.
"""

from __future__ import annotations

import argparse
import json as _json
import sys
import os

import numpy as np

from spectralyte import Spectralyte, __version__
from spectralyte.core.report import AuditReport

# ── Known constructor parameter types ─────────────────────────────────────────

_CONFIG_TYPES: dict[str, type] = {
    "k": int,
    "sensitivity_epsilon": float,
    "sensitivity_m": int,
    "variance_threshold": float,
    "sample_size": int,
    "random_seed": int,
}


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _load_embeddings(path: str) -> np.ndarray:
    """
    Load a .npy file and return a validated float64 embedding matrix.

    Exits with code 1 on any error (bad extension, missing file, wrong shape).
    """
    if not path.endswith(".npy"):
        print(
            f"Error: expected a .npy file, got '{os.path.basename(path)}'.\n"
            f"Only numpy .npy files are supported as input.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        arr = np.load(path)
    except FileNotFoundError:
        print(f"Error: file not found: '{path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading '{path}': {e}", file=sys.stderr)
        sys.exit(1)

    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 2:
        print(
            f"Error: embeddings must be a 2D array of shape (n, d), "
            f"got shape {arr.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    if arr.shape[0] < 3:
        print(
            f"Error: Spectralyte requires at least 3 embeddings, "
            f"got {arr.shape[0]}",
            file=sys.stderr,
        )
        sys.exit(1)

    return arr


def _parse_config(config_args: list[str] | None) -> dict:
    """
    Parse a list of "KEY=VALUE" strings into typed Spectralyte constructor kwargs.

    Unknown keys produce a stderr warning and are skipped.
    Invalid values (wrong type) exit with code 1.
    """
    if not config_args:
        return {}

    result = {}

    for item in config_args:
        if "=" not in item:
            print(
                f"Warning: ignoring malformed config argument '{item}' "
                f"(expected KEY=VALUE)",
                file=sys.stderr,
            )
            continue

        key, _, raw_value = item.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()

        if key not in _CONFIG_TYPES:
            print(
                f"Warning: unknown config key '{key}', ignoring.",
                file=sys.stderr,
            )
            continue

        # Special case: sample_size accepts "none" → None
        if key == "sample_size" and raw_value.lower() == "none":
            result[key] = None
            continue

        target_type = _CONFIG_TYPES[key]
        try:
            result[key] = target_type(raw_value)
        except (ValueError, TypeError):
            print(
                f"Error: invalid value for '{key}': "
                f"'{raw_value}' is not a valid {target_type.__name__}",
                file=sys.stderr,
            )
            sys.exit(1)

    return result


def _emit_json(report: AuditReport) -> None:
    """
    Write the audit report as JSON to stdout.

    Mirrors the schema of report.export() exactly so downstream consumers
    (TUI, MCP server) see identical structure whether reading from a file
    or piping from the CLI.
    """
    data = {
        "embeddings_shape": list(report.embeddings_shape),
        "n_issues": report.n_issues,
        "needs_transform": report.needs_transform,
        "has_brittle_zones": report.has_brittle_zones,
        "anisotropy": {
            "score": report.anisotropy.score,
            "interpretation": report.anisotropy.interpretation,
            "n_vectors": report.anisotropy.n_vectors,
            "n_dims": report.anisotropy.n_dims,
            "sampled": report.anisotropy.sampled,
        },
        "dimensionality": {
            "effective_dims": report.dimensionality.effective_dims,
            "nominal_dims": report.dimensionality.nominal_dims,
            "utilization": report.dimensionality.utilization,
            "participation_ratio": report.dimensionality.participation_ratio,
            "variance_threshold": report.dimensionality.variance_threshold,
            "interpretation": report.dimensionality.interpretation,
        },
        "density": {
            "cv": report.density.cv,
            "mean_knn_distance": report.density.mean_knn_distance,
            "std_knn_distance": report.density.std_knn_distance,
            "n_outliers": report.density.n_outliers,
            "interpretation": report.density.interpretation,
            "k": report.density.k,
        },
        "sensitivity": {
            "mean_stability": report.sensitivity.mean_stability,
            "n_brittle": report.sensitivity.n_brittle,
            "brittle_fraction": report.sensitivity.brittle_fraction,
            "epsilon_used": report.sensitivity.epsilon_used,
            "interpretation": report.sensitivity.interpretation,
            "k": report.sensitivity.k,
            "m": report.sensitivity.m,
        },
        "intrinsic_dim": {
            "d_int": report.intrinsic_dim.d_int,
            "r_squared": report.intrinsic_dim.r_squared,
            "interpretation": report.intrinsic_dim.interpretation,
            "trim_fraction": report.intrinsic_dim.trim_fraction,
            "n_points_used": report.intrinsic_dim.n_points_used,
        },
    }

    _json.dump(data, sys.stdout, indent=2)
    print()  # trailing newline


# ── Command handlers ───────────────────────────────────────────────────────────


def cmd_audit(args: argparse.Namespace) -> None:
    """Run a full geometric audit and print summary or JSON."""
    embeddings = _load_embeddings(args.path)
    kwargs = _parse_config(args.config)

    audit = Spectralyte(embeddings, **kwargs)
    report = audit.run(verbose=False)

    if args.json:
        _emit_json(report)
    else:
        report.summary(use_color=sys.stdout.isatty())


def cmd_fix_plan(args: argparse.Namespace) -> None:
    """Run audit and print a framework-specific remediation plan."""
    embeddings = _load_embeddings(args.path)
    kwargs = _parse_config(args.config)

    audit = Spectralyte(embeddings, **kwargs)
    report = audit.run(verbose=False)

    print(report.fix_plan(framework=args.framework))


def cmd_transform(args: argparse.Namespace) -> None:
    """Apply a geometric correction transform and save the result."""
    embeddings = _load_embeddings(args.path)
    kwargs = _parse_config(args.config)

    audit = Spectralyte(embeddings, **kwargs)
    audit.run(verbose=False)  # required before transform()

    fixed = audit.transform(embeddings, strategy=args.strategy)

    try:
        np.save(args.output, fixed)
    except Exception as e:
        print(f"Error saving to '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

    n, d = fixed.shape
    print(
        f"Saved transformed embeddings to {args.output} ({n} × {d})",
        file=sys.stderr,
    )


def cmd_version(args: argparse.Namespace) -> None:
    """Print the installed package version."""
    print(f"spectralyte {__version__}")


# ── Argument parser ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Kept as a standalone function so tests can import and invoke it
    directly without triggering sys.exit via main().
    """
    parser = argparse.ArgumentParser(
        prog="spectralyte",
        description="Geometric auditing for embedding spaces.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── audit ──────────────────────────────────────────────────────────────────
    p_audit = subparsers.add_parser(
        "audit",
        help="Run a full geometric audit on an embedding matrix.",
    )
    p_audit.add_argument(
        "path",
        help="Path to a .npy embeddings file.",
    )
    p_audit.add_argument(
        "--json",
        action="store_true",
        dest="json",
        default=False,
        help="Emit structured JSON to stdout instead of the human-readable summary.",
    )
    p_audit.add_argument(
        "--config",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "Override Spectralyte constructor kwargs. "
            "E.g. --config k=10 variance_threshold=0.90"
        ),
    )
    p_audit.set_defaults(func=cmd_audit)

    # ── fix-plan ───────────────────────────────────────────────────────────────
    p_fix = subparsers.add_parser(
        "fix-plan",
        help="Run audit and print a remediation plan.",
    )
    p_fix.add_argument(
        "path",
        help="Path to a .npy embeddings file.",
    )
    p_fix.add_argument(
        "--framework",
        choices=["langchain", "llamaindex", "generic"],
        default="generic",
        help="Target framework for generated code snippets (default: generic).",
    )
    p_fix.add_argument(
        "--config",
        nargs="*",
        metavar="KEY=VALUE",
    )
    p_fix.set_defaults(func=cmd_fix_plan)

    # ── transform ──────────────────────────────────────────────────────────────
    p_tx = subparsers.add_parser(
        "transform",
        help="Apply a geometric correction transform and save the result.",
    )
    p_tx.add_argument(
        "path",
        help="Path to a .npy embeddings file.",
    )
    p_tx.add_argument(
        "--strategy",
        choices=["whiten", "abtt", "pca_reduce"],
        required=True,
        help="Transform to apply.",
    )
    p_tx.add_argument(
        "--output",
        required=True,
        metavar="OUT.NPY",
        help="Output path for the transformed embeddings (.npy).",
    )
    p_tx.add_argument(
        "--config",
        nargs="*",
        metavar="KEY=VALUE",
    )
    p_tx.set_defaults(func=cmd_transform)

    # ── version ────────────────────────────────────────────────────────────────
    p_ver = subparsers.add_parser(
        "version",
        help="Print the installed package version.",
    )
    p_ver.set_defaults(func=cmd_version)

    return parser


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
