"""
examples/compare_models.py
============================
Spectralyte — Embedding Model Comparison

Runs Spectralyte on the same corpus embedded with multiple models
and produces a side-by-side geometric health comparison.

This is the "surprising finding" script for the blog post.

Usage:
    python examples/compare_models.py
    python examples/compare_models.py --n-docs 300 --save-plots ./comparison
"""

import argparse
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=300)
    parser.add_argument("--save-plots", type=str, default=None)
    return parser.parse_args()


def load_corpus(n_docs):
    from datasets import load_dataset
    dataset = load_dataset("ag_news", split=f"train[:{n_docs}]")
    return [row["text"].strip() for row in dataset if len(row["text"].strip()) > 50]


def embed_corpus(docs, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(docs, batch_size=64, show_progress_bar=False, convert_to_numpy=True)


def audit_embeddings(embeddings):
    from spectralyte import Spectralyte
    audit = Spectralyte(embeddings, k=10, random_seed=42)
    report = audit.run(verbose=False)
    return {
        "anisotropy":       report.anisotropy.score,
        "aniso_interp":     report.anisotropy.interpretation,
        "effective_dims":   report.dimensionality.effective_dims,
        "nominal_dims":     report.dimensionality.nominal_dims,
        "utilization":      report.dimensionality.utilization,
        "density_cv":       report.density.cv,
        "density_interp":   report.density.interpretation,
        "stability":        report.sensitivity.mean_stability,
        "stability_interp": report.sensitivity.interpretation,
        "intrinsic_dim":    report.intrinsic_dim.d_int,
        "r_squared":        report.intrinsic_dim.r_squared,
        "n_brittle":        report.sensitivity.n_brittle,
        "n_issues":         report.n_issues,
        "report":           report,
    }


def print_table(results):
    models = list(results.keys())
    col_w = 24

    print()
    print("=" * 90)
    print("  Spectralyte -- Embedding Model Geometric Health Comparison")
    print("=" * 90)

    print(f"  {'Metric':<28}", end="")
    for m in models:
        short = m.split("/")[-1][:col_w]
        print(f"  {short:<{col_w}}", end="")
    print()
    print("-" * 90)

    metrics = [
        ("Anisotropy (lower=better)",  "anisotropy",     ".3f",  False),
        ("Effective Dims",             "effective_dims",  "d",    True),
        ("Dim Utilization (higher=better)", "utilization", ".1%", True),
        ("Density CV (lower=better)",  "density_cv",      ".3f",  False),
        ("RSI Stability (higher=better)", "stability",    ".3f",  True),
        ("Intrinsic Dim",              "intrinsic_dim",   ".1f",  None),
        ("Brittle Zones",              "n_brittle",       "d",    False),
        ("Issues Detected",            "n_issues",        "d",    False),
    ]

    for label, key, fmt, higher_better in metrics:
        print(f"  {label:<28}", end="")
        values = [results[m][key] for m in models]
        for val in values:
            if fmt == "d":
                s = str(int(val))
            elif fmt == ".1%":
                s = f"{val:.1%}"
            elif fmt == ".3f":
                s = f"{val:.3f}"
            elif fmt == ".1f":
                s = f"{val:.1f}"
            else:
                s = str(val)
            if higher_better is True:
                marker = " *" if val == max(values) else "  "
            elif higher_better is False:
                marker = " *" if val == min(values) else "  "
            else:
                marker = "  "
            print(f"  {(s + marker):<{col_w}}", end="")
        print()

    print("=" * 90)
    print("  * = best value for that metric")
    print()

    # Ranking
    print("  Overall Geometric Health Ranking:")
    ranked = sorted(
        models,
        key=lambda m: (
            -results[m]["anisotropy"]
            + results[m]["utilization"] * 5
            + results[m]["stability"] * 3
            - results[m]["density_cv"]
            - results[m]["n_issues"]
        ),
        reverse=True,
    )
    for rank, m in enumerate(ranked, 1):
        short = m.split("/")[-1]
        issues = results[m]["n_issues"]
        print(f"    {rank}. {short:<40} ({issues} issue{'s' if issues != 1 else ''})")
    print()


def main():
    args = parse_args()

    models_to_compare = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
    ]

    print("Spectralyte -- Embedding Model Comparison")
    print(f"Corpus: AG News ({args.n_docs} documents)")
    print(f"Models: {', '.join(m.split('/')[-1] for m in models_to_compare)}")

    print("\nLoading corpus...", end=" ", flush=True)
    docs = load_corpus(args.n_docs)
    print(f"{len(docs)} documents")

    results = {}
    embeddings_by_model = {}

    for model_name in models_to_compare:
        short = model_name.split("/")[-1]
        print(f"\nEmbedding + auditing {short}...", end=" ", flush=True)
        t0 = time.time()
        try:
            embeddings = embed_corpus(docs, model_name)
            metrics = audit_embeddings(embeddings)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s, shape={embeddings.shape})")
            results[model_name] = metrics
            embeddings_by_model[model_name] = embeddings
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("No models succeeded.")
        return

    print_table(results)

    if args.save_plots:
        from pathlib import Path
        from spectralyte import Spectralyte
        print(f"Saving per-model plots to {args.save_plots}/")
        for model_name, embeddings in embeddings_by_model.items():
            short = model_name.split("/")[-1]
            model_dir = str(Path(args.save_plots) / short)
            audit = Spectralyte(embeddings, k=10, random_seed=42)
            report = audit.run(verbose=False)
            report.plot(backend="matplotlib", show=False, save_dir=model_dir)
            print(f"  Saved {short} -> {model_dir}/")

    print("Comparison complete.")


if __name__ == "__main__":
    main()