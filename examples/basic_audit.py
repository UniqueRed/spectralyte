"""
examples/basic_audit.py
=========================
Spectralyte — Basic Audit Example

The simplest possible Spectralyte usage. Shows how to:
    - Run a full audit on any numpy embedding matrix
    - Read the summary report
    - Apply a transform if issues are detected
    - Export results to JSON

This is the script referenced in the README quick start.

Usage:
    python examples/basic_audit.py
"""

import numpy as np
from spectralyte import Spectralyte

# ── Option A: Use your own embeddings ─────────────────────────────────────────
# embeddings = np.load("my_embeddings.npy")   # shape (n, d)

# ── Option B: Generate synthetic embeddings for demonstration ─────────────────
# This creates a mildly anisotropic space to demonstrate the audit
rng = np.random.RandomState(42)
n, d = 1000, 384

# Create anisotropic embeddings — vectors biased toward a dominant direction
dominant_direction = rng.randn(d)
dominant_direction /= np.linalg.norm(dominant_direction)
embeddings = rng.randn(n, d) + 1.5 * dominant_direction

print("Spectralyte — Basic Audit Example")
print(f"Embedding matrix: {n} vectors × {d} dims\n")

# ── Run the audit ──────────────────────────────────────────────────────────────
audit = Spectralyte(embeddings, k=10, random_seed=42)
report = audit.run()

# ── Print the summary ──────────────────────────────────────────────────────────
report.summary()

# ── Fix detected issues ────────────────────────────────────────────────────────
if report.needs_transform:
    print("Applying whitening transform...\n")
    fixed_embeddings = audit.transform(embeddings, strategy="whiten")

    # Re-audit to verify improvement
    audit_fixed = Spectralyte(fixed_embeddings, k=10, random_seed=42)
    report_fixed = audit_fixed.run(verbose=False)
    report_fixed._pre_transform_report = report
    report_fixed.compare()

# ── Get remediation plan ───────────────────────────────────────────────────────
print(report.fix_plan())

# ── Build the router ──────────────────────────────────────────────────────────
router = audit.get_router()
print(router.summary())

# Classify a sample query
sample_query = rng.randn(d)
zone = router.classify(sample_query)
print(f"Sample query zone: {zone}\n")

# ── Export results ─────────────────────────────────────────────────────────────
report.export("spectralyte_audit.json")
print("Done. Results saved to spectralyte_audit.json")