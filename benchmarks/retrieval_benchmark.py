"""
benchmarks/retrieval_benchmark.py
=================================
Does Spectralyte's diagnosis predict whether its remediation helps retrieval?

That is the claim the library rests on, and it is not self-evident: the metrics
measure geometry, while the promise is about search quality. This benchmark
tests the link directly on real embeddings, real queries and human relevance
judgments — two BEIR datasets crossed with three encoders spanning a wide
quality range.

For each (dataset, encoder) pair it reports nDCG@10 before and after each
transform, and checks whether ``report.needs_transform`` agrees with whether a
transform actually helped.

Usage
-----
    pip install "spectralyte[dev]" sentence-transformers datasets torch
    python benchmarks/retrieval_benchmark.py --out results.json

    # Quicker: one dataset, skip the parameter sweeps
    python benchmarks/retrieval_benchmark.py --datasets scifact --no-sweep

Embeddings are cached under --cache so reruns are cheap. First run downloads
the datasets and encoders (~1 GB) and takes roughly 20 minutes on a laptop
CPU, most of it mean-pooling GPT-2.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Encoders, worst to best. GPT-2 mean-pooling is the textbook anisotropic
# space (Ethayarajh 2019); the sentence-transformers models are trained with
# contrastive objectives and are far better behaved.
MODELS = [
    "gpt2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
DATASETS = ["scifact", "nfcorpus"]
RCONDS = [1e-12, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
ABTT_KS = [1, 3, 5, 10, 20, 40]
K = 10


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed(dataset: str, model: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Embed a BEIR dataset's corpus and queries, caching to .npy."""
    from datasets import load_dataset

    tag = f"{dataset}__{model.replace('/', '_')}"
    cf, qf = cache / f"{tag}.corpus.npy", cache / f"{tag}.queries.npy"
    ids = cache / f"{dataset}.corpus_ids.npy"

    if cf.exists() and qf.exists() and ids.exists():
        return np.load(cf), np.load(qf)

    corpus = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    docs = [(t + " " + x).strip() for t, x in zip(corpus["title"], corpus["text"])]
    qs = list(queries["text"])

    if model == "gpt2":
        encode = _gpt2_encoder()
    else:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model)

        def encode(texts):
            return st.encode(texts, batch_size=64, convert_to_numpy=True,
                             show_progress_bar=False).astype(np.float32)

    print(f"  embedding {tag}: {len(docs)} docs, {len(qs)} queries", flush=True)
    D, Q = encode(docs), encode(qs)

    cache.mkdir(parents=True, exist_ok=True)
    np.save(cf, D)
    np.save(qf, Q)
    np.save(ids, np.array(corpus["_id"]))
    np.save(cache / f"{dataset}.query_ids.npy", np.array(queries["_id"]))
    return D, Q


def _gpt2_encoder():
    """Mean-pooled GPT-2 last hidden states."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained("gpt2").eval()

    @torch.no_grad()
    def encode(texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
        return np.vstack(out).astype(np.float32)

    return encode


# ── Evaluation ─────────────────────────────────────────────────────────────────

def load_relevance(dataset: str, cache: Path):
    """Map query row -> set of relevant document rows, from the test qrels."""
    from datasets import load_dataset

    qrels = load_dataset(f"BeIR/{dataset}-qrels", split="test")
    doc_row = {str(c): i for i, c in enumerate(np.load(cache / f"{dataset}.corpus_ids.npy"))}
    q_row = {str(q): i for i, q in enumerate(np.load(cache / f"{dataset}.query_ids.npy"))}

    relevant = defaultdict(set)
    for r in qrels:
        qid, cid = str(r["query-id"]), str(r["corpus-id"])
        if r["score"] > 0 and qid in q_row and cid in doc_row:
            relevant[q_row[qid]].add(doc_row[cid])
    return relevant, sorted(relevant)


def _unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def ndcg_at_k(Q, D, relevant, query_rows, k=K):
    """nDCG@k over the judged queries, by exact cosine search."""
    sims = _unit(Q[query_rows]) @ _unit(D).T
    top = np.argsort(-sims, axis=1)[:, :k]

    total = 0.0
    for row, qi in enumerate(query_rows):
        rel = relevant[qi]
        dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top[row]) if d in rel)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
        total += dcg / idcg if idcg else 0.0
    return float(total / len(query_rows))


# ── Benchmark ──────────────────────────────────────────────────────────────────

def run_pair(dataset, model, cache, sweep=True):
    from spectralyte import Spectralyte
    from spectralyte.core import severity, transform as T

    D, Q = embed(dataset, model, cache)
    relevant, query_rows = load_relevance(dataset, cache)

    audit = Spectralyte(D, k=10, random_seed=42)
    # experimental=True so the record captures every metric for analysis, even
    # though only the core two decide needs_transform.
    report = audit.run(verbose=False, experimental=True)

    baseline = ndcg_at_k(Q, D, relevant, query_rows)
    row = {
        "dataset": dataset,
        "model": model,
        "dims": int(D.shape[1]),
        "baseline_ndcg@10": baseline,
        "n_issues": report.n_issues,
        "needs_transform": bool(report.needs_transform),
        "diagnosis": {m: getattr(report, m).interpretation
                      for m in report.measured_metrics},
        "severity": {m: severity.severity(getattr(report, m))
                     for m in report.measured_metrics},
        "core_metrics": list(severity.CORE_METRICS),
        "d_int": float(report.intrinsic_dim.d_int),
        "condition_number": float(report.condition_number),
        "defaults": {},
    }

    # Retrieval under each strategy at library defaults — what a user gets.
    for strategy in ("whiten", "abtt", "pca_reduce"):
        score = ndcg_at_k(audit.transform(Q, strategy=strategy),
                          audit.transform(D, strategy=strategy),
                          relevant, query_rows)
        row["defaults"][strategy] = score

    best = max(row["defaults"].values())
    row["transform_helps"] = bool(best > baseline)
    row["advice_correct"] = bool(row["needs_transform"] == row["transform_helps"])

    if sweep:
        # Fitting is an eigendecomposition plus an SVD; it does not need the
        # five metrics recomputed per parameter value.
        eff = report.dimensionality.effective_dims
        row["whiten_sweep"] = {}
        for rc in RCONDS:
            fit = T.fit(D, effective_dims=eff, whiten_rcond=rc)
            row["whiten_sweep"][f"{rc:.0e}"] = ndcg_at_k(
                fit.apply(Q, "whiten"), fit.apply(D, "whiten"), relevant, query_rows)

        fit = T.fit(D, effective_dims=eff)
        row["abtt_sweep"] = {
            str(k): ndcg_at_k(fit.apply(Q, "abtt", abtt_k=k),
                              fit.apply(D, "abtt", abtt_k=k), relevant, query_rows)
            for k in ABTT_KS
        }

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--cache", default=".benchmark_cache", type=Path)
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--no-sweep", action="store_true",
                    help="Skip the whiten_rcond / abtt_k parameter sweeps.")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for model in args.models:
            print(f"{dataset} x {model}", flush=True)
            results.append(run_pair(dataset, model, args.cache, sweep=not args.no_sweep))

    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\n{'dataset':<10}{'model':<22}{'nDCG@10':>9}{'needsT':>8}"
          f"{'best transform':>16}{'advice':>9}")
    for r in results:
        best_name = max(r["defaults"], key=r["defaults"].get)
        best = r["defaults"][best_name]
        print(f"{r['dataset']:<10}{r['model'].split('/')[-1]:<22}"
              f"{r['baseline_ndcg@10']:>9.4f}{str(r['needs_transform']):>8}"
              f"{best:>9.4f} ({best - r['baseline_ndcg@10']:+.3f})"
              f"{'ok' if r['advice_correct'] else 'WRONG':>9}")

    n_ok = sum(r["advice_correct"] for r in results)
    print(f"\nneeds_transform agreed with measured outcome on "
          f"{n_ok}/{len(results)} pairs")
    print(f"full results written to {args.out}")


if __name__ == "__main__":
    main()
