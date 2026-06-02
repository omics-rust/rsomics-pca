# rsomics-pca

**Principal Component Analysis (PCA)** of a feature/sample table — the classical
ordination that finds the orthogonal directions of maximum variance among
samples.

Reads a numeric table TSV (an empty top-left cell, feature IDs in the header
row, then one row per sample: sample ID followed by tab-separated values) and
writes the eigenvalues (variances), proportion of variance explained, sample
scores, and feature loadings for every principal component.

```
rsomics-pca table.tsv
rsomics-pca table.tsv --method svd --dimensions 2 -o pca.tsv
```

## Method

Matches `skbio.stats.ordination.pca`:

1. Center the data matrix by feature (subtract each column's mean).
2. Decompose, either:
   - `eigh` (default): eigendecomposition of the covariance matrix
     `Σ = Xcᵀ·Xc / (n-1)`, eigenvalues sorted descending. Total variance is the
     trace of `Σ`.
   - `svd`: singular value decomposition of the centered matrix; variances are
     `σ² / (n-1)` and total variance is `‖Xc‖²_F / (n-1)`. Avoids forming the
     covariance matrix and is steadier for very small variances.
3. Sample scores are the centered samples projected onto the components,
   `Xc · componentsᵀ`. Loadings are the components (one per PC).
4. `proportion_explained = variance / total_variance`.

`--dimensions N` keeps the first `N` components (default: all,
`min(n_samples, n_features)`).

### Eigenvector sign

The sign of a component — and therefore of a score/loading axis — is arbitrary;
flipping a whole axis is an equally valid PCA solution. Eigenvalues and
proportion explained are sign-independent. The compat differential compares
those directly and compares scores and loadings up to a per-axis sign flip
(orienting each axis by its largest-magnitude entry on both sides).

### Output

A flat TSV: an `# eigenvalues` block (eigenvalue + proportion_explained rows
over `PC1 … PCk`), then a `# samples` block (one row per sample, ID followed by
its per-PC score), then a `# features` block (one row per feature, ID followed
by its per-PC loading). Floats use Python's shortest round-trip `repr`.

## Origin

This crate is an independent Rust reimplementation of the PCA operation provided
by `scikit-bio` (`skbio.stats.ordination.pca`, methods `eigh` and `svd`, which
delegate to `scipy.linalg.eigh` / `scipy.linalg.svd`), based on:

- Pearson, K. (1901), *On lines and planes of closest fit to systems of points
  in space*, Philosophical Magazine 2(11):559-572,
  <https://doi.org/10.1080/14786440109462720>.
- Hotelling, H. (1933), *Analysis of a complex of statistical variables into
  principal components*, Journal of Educational Psychology 24(6):417-441,
  498-520, <https://doi.org/10.1037/h0071325>.
- The black-box behaviour of `skbio.stats.ordination.pca`: column centering,
  the covariance / SVD variance scaling by `n-1`, descending order, total
  variance as the covariance trace, and `proportion_explained` over total
  variance.

scikit-bio is BSD-3-Clause and was read and cited. The eigendecomposition and
SVD use [`faer`](https://crates.io/crates/faer) (pure Rust, SIMD + rayon —
external-dependency quadrant ①). Test fixtures are deterministically generated.

License: MIT OR Apache-2.0.
Upstream credit: scikit-bio <https://scikit-bio.org> (BSD-3-Clause).

## Compatibility & performance

`tests/compat.rs` checks this binary against a committed scikit-bio-captured
golden (runs in CI with no oracle install) and, when scikit-bio is importable,
runs a live differential via `tests/oracle_skbio.py`. Eigenvalues and proportion
explained match directly; sample scores and loadings match up to a per-axis sign
flip (epsilon `1e-6`).

The PCA hot path is the `O(min(n,p)³)` decomposition plus the `O(n·p·k)`
projection; `faer` carries the decomposition single-threaded and scales across
cores.
