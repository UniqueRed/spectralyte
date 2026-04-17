"""
examples/full_pipeline.py
===========================
Spectralyte — Full Pipeline Example

Demonstrates the complete Spectralyte workflow on real embeddings:
    1. Load AG News corpus (World, Sports, Business, Technology)
    2. Embed with sentence-transformers (all-MiniLM-L6-v2)
    3. Run the full geometric audit
    4. Apply whitening transform if needed
    5. Show before/after comparison
    6. Build and demonstrate the runtime router
    7. Save all six visualizations

Runtime: ~2-3 minutes on a standard laptop.
No API keys required. All computation is local.

Usage:
    python examples/full_pipeline.py
    python examples/full_pipeline.py --n-docs 200 --save-plots ./plots --no-plots
"""

import argparse
import time
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectralyte full pipeline demo on real text embeddings"
    )
    parser.add_argument("--n-docs", type=int, default=500)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--save-plots", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-embeddings", type=str, default=None)
    return parser.parse_args()


def banner(text):
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def step(n, text):
    print(f"\n  [{n}] {text}")


def info(text):
    print(f"      {text}")


def main():
    args = parse_args()

    banner("Spectralyte -- Full Pipeline Demo")
    print(f"  Documents : {args.n_docs}")
    print(f"  Model     : {args.model}")

    # Step 1: Load corpus
    step(1, "Loading AG News corpus...")
    from datasets import load_dataset
    t0 = time.time()
    dataset = load_dataset("ag_news", split=f"train[:{args.n_docs}]")
    docs = [row["text"].strip() for row in dataset if len(row["text"].strip()) > 50]
    info(f"Loaded {len(docs)} documents in {time.time() - t0:.1f}s")
    info(f"Categories: World, Sports, Business, Technology")
    info(f'Sample: "{docs[0][:80]}..."')

    # Step 2: Embed
    step(2, f"Embedding with {args.model}...")
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(args.model)
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    info(f"Embedded {len(docs)} documents in {time.time() - t0:.1f}s")
    info(f"Embedding shape: {embeddings.shape}")

    if args.save_embeddings:
        np.save(args.save_embeddings, embeddings)
        info(f"Saved embeddings to {args.save_embeddings}")

    # Step 3: Run audit
    step(3, "Running Spectralyte geometric audit...")
    from spectralyte import Spectralyte
    t0 = time.time()
    audit = Spectralyte(embeddings, k=10, random_seed=42)
    report = audit.run(verbose=True)
    info(f"Audit completed in {time.time() - t0:.1f}s")
    report.summary()

    # Step 4: Fix plan
    step(4, "Generating remediation plan...")
    print(report.fix_plan(framework="generic"))

    # Step 5: Apply transform
    if report.needs_transform:
        step(5, "Applying whitening transform...")
        t0 = time.time()
        fixed_embeddings = audit.transform(embeddings, strategy="whiten")
        info(f"Transform applied in {time.time() - t0:.1f}s")

        step(5, "Re-auditing fixed embeddings...")
        audit_fixed = Spectralyte(fixed_embeddings, k=10, random_seed=42)
        report_fixed = audit_fixed.run(verbose=True)
        report_fixed._pre_transform_report = report
        print()
        report_fixed.compare()
    else:
        step(5, "No transform needed -- embedding space is geometrically healthy.")
        fixed_embeddings = embeddings
        report_fixed = report
        audit_fixed = audit

    # Step 6: Router
    step(6, "Building runtime router...")
    router = audit_fixed.get_router()
    print(router.summary())

    info("Classifying 5 sample queries:")
    sample_texts = [
        "What are the latest stock market trends?",
        "Who won the championship game last night?",
        "New breakthrough in quantum computing research",
        "Political tensions rise in the Middle East",
        "Tech giant announces new product launch",
    ]
    sample_queries = model.encode(sample_texts, show_progress_bar=False, convert_to_numpy=True)
    zone_icons = {"stable": "OK", "brittle": "BRITTLE", "dense_boundary": "BOUNDARY"}
    for text, emb in zip(sample_texts, sample_queries):
        zone = router.classify(emb)
        info(f"  [{zone_icons.get(zone, zone)}]  '{text[:50]}'")

    # Step 7: Visualize
    if not args.no_plots:
        step(7, "Generating visualizations...")
        save_dir = args.save_plots
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        show_plots = save_dir is None
        report_fixed.plot(backend="matplotlib", show=show_plots, save_dir=save_dir)
        if save_dir:
            info(f"Saved 6 plots to {save_dir}/")
    else:
        step(7, "Skipping visualization (--no-plots)")

    # Step 8: Export
    step(8, "Exporting audit results...")
    report_fixed.export("spectralyte_audit.json")

    # Final summary
    banner("Pipeline Complete")
    print(f"  Documents : {len(docs)}")
    print(f"  Model     : {args.model}")
    print(f"  Shape     : {embeddings.shape}")
    print()
    print(f"  BEFORE transform:")
    print(f"    Anisotropy    : {report.anisotropy.score:.3f}  ({report.anisotropy.interpretation})")
    print(f"    Effective dims: {report.dimensionality.effective_dims}/{report.dimensionality.nominal_dims}")
    print(f"    Density CV    : {report.density.cv:.3f}  ({report.density.interpretation})")
    print(f"    RSI Stability : {report.sensitivity.mean_stability:.3f}  ({report.sensitivity.interpretation})")
    print(f"    Intrinsic dim : {report.intrinsic_dim.d_int:.1f}  (R2={report.intrinsic_dim.r_squared:.3f})")
    if report_fixed is not report:
        print()
        print(f"  AFTER whitening:")
        print(f"    Anisotropy    : {report_fixed.anisotropy.score:.3f}  ({report_fixed.anisotropy.interpretation})")
        print(f"    Effective dims: {report_fixed.dimensionality.effective_dims}/{report_fixed.dimensionality.nominal_dims}")
        print(f"    Density CV    : {report_fixed.density.cv:.3f}  ({report_fixed.density.interpretation})")
        print(f"    RSI Stability : {report_fixed.sensitivity.mean_stability:.3f}  ({report_fixed.sensitivity.interpretation})")
        print(f"    Intrinsic dim : {report_fixed.intrinsic_dim.d_int:.1f}  (R2={report_fixed.intrinsic_dim.r_squared:.3f})")
    print()


if __name__ == "__main__":
    main()