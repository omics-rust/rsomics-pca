use std::io::{BufRead, Write};

use faer::Mat;
use faer::linalg::solvers::{SelfAdjointEigen, Svd};
use rsomics_common::{Result, RsomicsError};

mod fmt;
use fmt::push_pyrepr;

/// A numeric data table: samples are rows, features are columns. TSV form is an
/// empty top-left cell then feature IDs as the header, then one row per sample
/// (sample ID + tab-separated values).
pub struct FeatureTable {
    pub sample_ids: Vec<String>,
    pub feature_ids: Vec<String>,
    /// Row-major `n_samples × n_features`.
    pub data: Vec<f64>,
}

impl FeatureTable {
    /// # Errors
    /// Errors on a missing header, a ragged body, or a non-numeric or
    /// non-finite (NaN/inf) cell.
    pub fn parse<R: BufRead>(reader: R, delim: char) -> Result<FeatureTable> {
        let mut lines = reader.lines();
        let header = loop {
            match lines.next() {
                Some(line) => {
                    let line = line.map_err(RsomicsError::Io)?;
                    if line.trim().is_empty() || line.starts_with('#') {
                        continue;
                    }
                    break line;
                }
                None => return Err(RsomicsError::InvalidInput("empty feature table".into())),
            }
        };
        let feature_ids: Vec<String> = header
            .split(delim)
            .skip(1)
            .map(|s| s.trim().to_string())
            .collect();
        let p = feature_ids.len();
        if p == 0 {
            return Err(RsomicsError::InvalidInput(
                "header has no feature columns (need an empty top-left cell + ≥1 feature)".into(),
            ));
        }

        let mut sample_ids = Vec::new();
        let mut data = Vec::new();
        for line in lines {
            let line = line.map_err(RsomicsError::Io)?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split(delim);
            let label = fields.next().unwrap_or("").trim().to_string();
            let row_start = data.len();
            for field in fields {
                let col = data.len() - row_start + 1;
                let v: f64 = field.trim().parse().map_err(|_| {
                    RsomicsError::InvalidInput(format!(
                        "sample '{label}', column {col}: '{}' is not numeric",
                        field.trim()
                    ))
                })?;
                // skbio delegates to numpy's asarray_chkfinite, which rejects any
                // non-finite cell; a NaN/inf otherwise silently poisons the whole
                // decomposition into an all-nan table (eigh) or a solver panic (svd).
                if !v.is_finite() {
                    return Err(RsomicsError::InvalidInput(format!(
                        "sample '{label}', column {col}: '{}' is not finite (PCA input must not contain NaN or inf)",
                        field.trim()
                    )));
                }
                data.push(v);
            }
            let got = data.len() - row_start;
            if got != p {
                return Err(RsomicsError::InvalidInput(format!(
                    "sample '{label}' has {got} values, expected {p}"
                )));
            }
            sample_ids.push(label);
        }
        if sample_ids.is_empty() {
            return Err(RsomicsError::InvalidInput("no data rows".into()));
        }
        Ok(FeatureTable {
            sample_ids,
            feature_ids,
            data,
        })
    }

    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.sample_ids.len()
    }

    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_ids.len()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Eigh,
    Svd,
}

/// Result of a PCA: eigenvalues (variances along each principal axis),
/// proportion of total variance explained, sample scores (projection of the
/// centered samples onto the components), and loadings (the components
/// themselves). Matches `skbio.stats.ordination.pca`.
pub struct Ordination {
    pub sample_ids: Vec<String>,
    pub feature_ids: Vec<String>,
    pub eigvals: Vec<f64>,
    pub proportion_explained: Vec<f64>,
    /// Row-major `n_samples × n_axes`.
    pub sample_scores: Vec<f64>,
    /// Row-major `n_axes × n_features` — one loading row per principal component.
    pub loadings: Vec<f64>,
}

impl Ordination {
    /// # Errors
    /// Errors when there are fewer than 2 samples, when `dimensions` is not in
    /// `1..=min(n_samples, n_features)`, or when the decomposition fails.
    pub fn compute(
        table: &FeatureTable,
        method: Method,
        dimensions: Option<usize>,
    ) -> Result<Ordination> {
        let n = table.n_samples();
        let p = table.n_features();
        // Variance divides by n-1; a single sample makes that 1/0 = inf and the
        // covariance 0*inf = nan, so skbio raises here rather than emit garbage.
        if n < 2 {
            return Err(RsomicsError::InvalidInput(format!(
                "PCA needs at least 2 samples, got {n}"
            )));
        }
        if let Some(d) = dimensions
            && (d == 0 || d > n.min(p))
        {
            return Err(RsomicsError::InvalidInput(
                "dimensions must be a positive integer ≤ min(n_samples, n_features)".into(),
            ));
        }

        let col_means: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| table.data[i * p + j]).sum::<f64>() / n as f64)
            .collect();
        let xc = Mat::from_fn(n, p, |i, j| table.data[i * p + j] - col_means[j]);

        let (mut variances, components, total_variance) = match method {
            Method::Eigh => eigh_decompose(&xc, n, p)?,
            Method::Svd => svd_decompose(&xc, n, p)?,
        };

        if let Some(d) = dimensions {
            variances.truncate(d);
        }
        let k = variances.len();

        // sample scores = X_c · componentsᵀ
        let mut sample_scores = vec![0.0_f64; n * k];
        for i in 0..n {
            for a in 0..k {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += xc[(i, j)] * components[a * p + j];
                }
                sample_scores[i * k + a] = acc;
            }
        }

        let mut loadings = vec![0.0_f64; k * p];
        loadings[..k * p].copy_from_slice(&components[..k * p]);

        let proportion_explained: Vec<f64> =
            variances.iter().map(|&v| v / total_variance).collect();

        Ok(Ordination {
            sample_ids: table.sample_ids.clone(),
            feature_ids: table.feature_ids.clone(),
            eigvals: variances,
            proportion_explained,
            sample_scores,
            loadings,
        })
    }

    /// Write the flat ordination TSV the rsomics ordination family shares: an
    /// `# eigenvalues` block, a sample-score table, then a loadings table, axes
    /// labelled `PC1..PCk`.
    ///
    /// # Errors
    /// Propagates write errors.
    pub fn write_tsv<W: Write>(&self, mut out: W) -> Result<()> {
        let k = self.eigvals.len();
        let mut line = String::new();

        writeln!(out, "# eigenvalues").map_err(RsomicsError::Io)?;
        write_axis_header(&mut out, k)?;
        line.push_str("eigval");
        for &v in &self.eigvals {
            line.push('\t');
            push_pyrepr(&mut line, v);
        }
        writeln!(out, "{line}").map_err(RsomicsError::Io)?;

        line.clear();
        line.push_str("proportion_explained");
        for &v in &self.proportion_explained {
            line.push('\t');
            push_pyrepr(&mut line, v);
        }
        writeln!(out, "{line}").map_err(RsomicsError::Io)?;

        writeln!(out, "# samples").map_err(RsomicsError::Io)?;
        write_axis_header(&mut out, k)?;
        for (i, id) in self.sample_ids.iter().enumerate() {
            line.clear();
            line.push_str(id);
            for a in 0..k {
                line.push('\t');
                push_pyrepr(&mut line, self.sample_scores[i * k + a]);
            }
            writeln!(out, "{line}").map_err(RsomicsError::Io)?;
        }

        writeln!(out, "# features").map_err(RsomicsError::Io)?;
        write_axis_header(&mut out, k)?;
        for (j, id) in self.feature_ids.iter().enumerate() {
            line.clear();
            line.push_str(id);
            for a in 0..k {
                line.push('\t');
                push_pyrepr(&mut line, self.loadings[a * self.feature_ids.len() + j]);
            }
            writeln!(out, "{line}").map_err(RsomicsError::Io)?;
        }
        Ok(())
    }
}

/// eigh of the covariance Xcᵀ·Xc/(n-1); returns descending variances, the
/// component rows (`k × p`, row-major), and the total variance (trace of cov).
fn eigh_decompose(xc: &Mat<f64>, n: usize, p: usize) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let inv_dof = 1.0 / (n as f64 - 1.0);
    let cov = Mat::from_fn(p, p, |i, j| {
        let mut acc = 0.0;
        for s in 0..n {
            acc += xc[(s, i)] * xc[(s, j)];
        }
        acc * inv_dof
    });

    let eig: SelfAdjointEigen<f64> = cov
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| RsomicsError::UpstreamError(format!("eigendecomposition failed: {e:?}")))?;
    let s = eig.S();
    let u = eig.U();

    // faer (LAPACK) returns ascending; skbio reverses to descending.
    let mut variances = Vec::with_capacity(p);
    let mut components = vec![0.0_f64; p * p];
    for a in 0..p {
        let col = p - 1 - a;
        variances.push(s[col]);
        for j in 0..p {
            components[a * p + j] = u[(j, col)];
        }
    }
    let total_variance: f64 = (0..p).map(|i| cov[(i, i)]).sum();
    Ok((variances, components, total_variance))
}

/// SVD of the centered matrix; variances are σ²/(n-1), components are the right
/// singular vectors (`min(n,p) × p`), total variance is ‖Xc‖²_F/(n-1).
fn svd_decompose(xc: &Mat<f64>, n: usize, p: usize) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let inv_dof = 1.0 / (n as f64 - 1.0);
    let svd: Svd<f64> = xc
        .svd()
        .map_err(|e| RsomicsError::UpstreamError(format!("SVD failed: {e:?}")))?;
    let sv = svd.S().column_vector();
    let v = svd.V();
    let k = sv.nrows();

    let variances: Vec<f64> = (0..k).map(|a| sv[a] * sv[a] * inv_dof).collect();
    let mut components = vec![0.0_f64; k * p];
    for a in 0..k {
        for j in 0..p {
            components[a * p + j] = v[(j, a)];
        }
    }
    let mut frob = 0.0;
    for s in 0..n {
        for j in 0..p {
            frob += xc[(s, j)] * xc[(s, j)];
        }
    }
    Ok((variances, components, frob * inv_dof))
}

fn write_axis_header<W: Write>(out: &mut W, k: usize) -> Result<()> {
    let mut header = String::new();
    for a in 1..=k {
        header.push('\t');
        header.push_str("PC");
        header.push_str(&a.to_string());
    }
    writeln!(out, "{header}").map_err(RsomicsError::Io)
}

/// # Errors
/// Propagates parse, compute, and write errors.
pub fn run<R: BufRead, W: Write>(
    reader: R,
    out: W,
    delim: char,
    method: Method,
    dimensions: Option<usize>,
) -> Result<()> {
    let table = FeatureTable::parse(reader, delim)?;
    let ord = Ordination::compute(&table, method, dimensions)?;
    ord.write_tsv(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> &'static str {
        "\tF1\tF2\tF3\tF4\n\
         S1\t0.417022\t0.720324\t0.000114\t0.302333\n\
         S2\t0.146756\t0.092339\t0.186260\t0.345561\n\
         S3\t0.396767\t0.538817\t0.419195\t0.685220\n\
         S4\t0.204452\t0.878117\t0.027388\t0.670468\n\
         S5\t0.417305\t0.558690\t0.140387\t0.198101\n\
         S6\t0.800745\t0.968262\t0.313424\t0.692323\n"
    }

    #[test]
    fn parses_table() {
        let t = FeatureTable::parse(table().as_bytes(), '\t').unwrap();
        assert_eq!(t.sample_ids.len(), 6);
        assert_eq!(t.feature_ids, ["F1", "F2", "F3", "F4"]);
    }

    #[test]
    fn eigh_and_svd_agree() {
        let t = FeatureTable::parse(table().as_bytes(), '\t').unwrap();
        let e = Ordination::compute(&t, Method::Eigh, None).unwrap();
        let s = Ordination::compute(&t, Method::Svd, None).unwrap();
        assert_eq!(e.eigvals.len(), s.eigvals.len());
        for (a, &ev) in e.eigvals.iter().enumerate() {
            assert!(
                (ev - s.eigvals[a]).abs() < 1e-10,
                "axis {a}: {ev} vs {}",
                s.eigvals[a]
            );
        }
    }

    #[test]
    fn proportions_sum_to_one() {
        let t = FeatureTable::parse(table().as_bytes(), '\t').unwrap();
        let o = Ordination::compute(&t, Method::Eigh, None).unwrap();
        let p: f64 = o.proportion_explained.iter().sum();
        assert!((p - 1.0).abs() < 1e-9, "sum {p}");
        assert!(o.eigvals[0] >= o.eigvals[1]);
    }

    #[test]
    fn dimensions_caps_axes() {
        let t = FeatureTable::parse(table().as_bytes(), '\t').unwrap();
        let o = Ordination::compute(&t, Method::Eigh, Some(2)).unwrap();
        assert_eq!(o.eigvals.len(), 2);
        assert_eq!(o.sample_scores.len(), 6 * 2);
        assert_eq!(o.loadings.len(), 2 * 4);
    }

    #[test]
    fn dimensions_out_of_range_errors() {
        let t = FeatureTable::parse(table().as_bytes(), '\t').unwrap();
        assert!(Ordination::compute(&t, Method::Eigh, Some(0)).is_err());
        assert!(Ordination::compute(&t, Method::Eigh, Some(5)).is_err());
    }

    #[test]
    fn ragged_row_errors() {
        let bad = "\tA\tB\nS1\t1\nS2\t1\t2\n";
        assert!(FeatureTable::parse(bad.as_bytes(), '\t').is_err());
    }

    fn parse_err(text: &str) -> String {
        match FeatureTable::parse(text.as_bytes(), '\t') {
            Err(e) => format!("{e}"),
            Ok(_) => panic!("expected parse to fail on: {text:?}"),
        }
    }

    #[test]
    fn nan_cell_rejected() {
        assert!(parse_err("\tA\tB\nS1\t1\tnan\nS2\t3\t4\n").contains("finite"));
    }

    #[test]
    fn inf_cell_rejected() {
        for tok in ["inf", "-inf", "infinity"] {
            let msg = parse_err(&format!("\tA\tB\nS1\t1\t{tok}\nS2\t3\t4\n"));
            assert!(msg.contains("finite"), "{tok}: {msg}");
        }
    }

    #[test]
    fn single_sample_errors() {
        let one = "\tA\tB\nS1\t1\t2\n";
        let t = FeatureTable::parse(one.as_bytes(), '\t').unwrap();
        assert_eq!(t.n_samples(), 1);
        assert!(Ordination::compute(&t, Method::Eigh, None).is_err());
        assert!(Ordination::compute(&t, Method::Svd, None).is_err());
    }
}
