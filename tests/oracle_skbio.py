#!/usr/bin/env python3
"""scikit-bio PCA oracle for rsomics-pca compat tests.

Reads a feature/sample table TSV (empty top-left cell, feature IDs in the
header, sample ID + tab-separated values per row) on argv[1], runs
skbio.stats.ordination.pca with the method on argv[2] (default eigh), and dumps
eigenvalues, proportion_explained, per-sample scores, and per-feature loadings
as tab-separated rows the Rust compat harness parses for a value-level diff.
"""

import sys

import numpy as np
from skbio.stats.ordination import pca


def main():
    path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "eigh"
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.startswith("#")]
    feature_ids = lines[0].split("\t")[1:]
    sample_ids = []
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        sample_ids.append(parts[0])
        rows.append([float(v) for v in parts[1:]])
    X = np.array(rows, dtype=float)
    res = pca(X, method=method)

    def row(label, values):
        return label + "\t" + "\t".join(repr(float(v)) for v in values)

    out = [row("eigvals", res.eigvals.values),
           row("proportion_explained", res.proportion_explained.values)]
    samples = res.samples.values
    for i, sid in enumerate(sample_ids):
        out.append(row("S:" + sid, samples[i]))
    # res.features is (n_components, n_features); emit one row per feature so the
    # Rust side (feature rows, PC cols) can align by feature id.
    feats = res.features.values
    for j, fid in enumerate(feature_ids):
        out.append(row("F:" + fid, feats[:, j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
