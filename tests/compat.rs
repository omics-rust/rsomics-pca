use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

const EPS: f64 = 1e-6;

fn ours_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-pca"))
}

fn golden(name: &str) -> String {
    format!("{}/tests/golden/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn oracle_script() -> String {
    format!("{}/tests/oracle_skbio.py", env!("CARGO_MANIFEST_DIR"))
}

/// scikit-bio is the named oracle; skip loudly if it (or python) is unavailable.
/// `RSOMICS_SKBIO_PYTHON` overrides the interpreter (e.g. an isolated venv).
fn skbio_python() -> Option<String> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("RSOMICS_SKBIO_PYTHON") {
        candidates.push(p);
    }
    candidates.push("python3".into());
    candidates.push("python".into());
    for py in candidates {
        let probe = Command::new(&py)
            .args(["-c", "import skbio.stats.ordination"])
            .output();
        if let Ok(out) = probe
            && out.status.success()
        {
            return Some(py);
        }
    }
    eprintln!(
        "SKIP: scikit-bio not importable — install `scikit-bio` to run the live differential"
    );
    None
}

/// eigenvalues, proportions, sample scores (S:id), loadings (F:id).
struct Pca {
    eigvals: Vec<f64>,
    proportion: Vec<f64>,
    samples: HashMap<String, Vec<f64>>,
    features: HashMap<String, Vec<f64>>,
}

fn parse_ours(text: &str) -> Pca {
    let mut eigvals = Vec::new();
    let mut proportion = Vec::new();
    let mut samples = HashMap::new();
    let mut features = HashMap::new();
    let mut block = "";
    for line in text.lines() {
        match line {
            "# eigenvalues" => {
                block = "eig";
                continue;
            }
            "# samples" => {
                block = "samples";
                continue;
            }
            "# features" => {
                block = "features";
                continue;
            }
            _ => {}
        }
        if line.starts_with('\t') {
            continue;
        }
        let mut it = line.split('\t');
        let label = it.next().unwrap();
        let vals: Vec<f64> = it.map(|s| s.parse().unwrap()).collect();
        match (block, label) {
            ("eig", "eigval") => eigvals = vals,
            ("eig", "proportion_explained") => proportion = vals,
            ("samples", _) => {
                samples.insert(label.to_string(), vals);
            }
            ("features", _) => {
                features.insert(label.to_string(), vals);
            }
            _ => {}
        }
    }
    Pca {
        eigvals,
        proportion,
        samples,
        features,
    }
}

fn parse_oracle(text: &str) -> Pca {
    let mut eigvals = Vec::new();
    let mut proportion = Vec::new();
    let mut samples = HashMap::new();
    let mut features = HashMap::new();
    for line in text.lines() {
        let mut it = line.split('\t');
        let label = it.next().unwrap();
        let vals: Vec<f64> = it.map(|s| s.parse().unwrap()).collect();
        if label == "eigvals" {
            eigvals = vals;
        } else if label == "proportion_explained" {
            proportion = vals;
        } else if let Some(id) = label.strip_prefix("S:") {
            samples.insert(id.to_string(), vals);
        } else if let Some(id) = label.strip_prefix("F:") {
            features.insert(id.to_string(), vals);
        }
    }
    Pca {
        eigvals,
        proportion,
        samples,
        features,
    }
}

fn ours_output(table: &str, method: &str) -> String {
    let out = Command::new(ours_bin())
        .arg(golden(table))
        .args(["--method", method])
        .output()
        .expect("run rsomics-pca");
    assert!(
        out.status.success(),
        "ours failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn oracle_output(py: &str, table: &str, method: &str) -> String {
    let out = Command::new(py)
        .arg(oracle_script())
        .arg(golden(table))
        .arg(method)
        .output()
        .expect("run scikit-bio oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS + EPS * b.abs()
}

/// The sign of the largest-magnitude value on axis `a` across all ids, giving a
/// stable per-axis orientation independent of the eigensolver's arbitrary sign.
fn axis_sign(scores: &HashMap<String, Vec<f64>>, a: usize) -> f64 {
    let mut best = 0.0_f64;
    let mut sign = 1.0_f64;
    for v in scores.values() {
        if v[a].abs() > best {
            best = v[a].abs();
            sign = if v[a] < 0.0 { -1.0 } else { 1.0 };
        }
    }
    sign
}

fn check(ours: &Pca, theirs: &Pca, ctx: &str) {
    assert_eq!(ours.eigvals.len(), theirs.eigvals.len(), "{ctx} axis count");
    for (a, &o) in ours.eigvals.iter().enumerate() {
        assert!(
            approx(o, theirs.eigvals[a]),
            "{ctx} eigval PC{} {o} vs {}",
            a + 1,
            theirs.eigvals[a]
        );
    }
    for (a, &o) in ours.proportion.iter().enumerate() {
        assert!(
            approx(o, theirs.proportion[a]),
            "{ctx} prop PC{} {o} vs {}",
            a + 1,
            theirs.proportion[a]
        );
    }

    // Sign of each axis is arbitrary; orient samples and loadings by the same
    // per-axis sign before diffing.
    let k = ours.eigvals.len();
    for a in 0..k {
        let so = axis_sign(&ours.samples, a);
        let st = axis_sign(&theirs.samples, a);
        for (id, v) in &ours.samples {
            let o = v[a] * so;
            let t = theirs.samples[id][a] * st;
            assert!(approx(o, t), "{ctx} sample {id} PC{} {o} vs {t}", a + 1);
        }
        for (id, v) in &ours.features {
            let o = v[a] * so;
            let t = theirs.features[id][a] * st;
            assert!(approx(o, t), "{ctx} loading {id} PC{} {o} vs {t}", a + 1);
        }
    }
}

/// Always-on check against the committed skbio-captured golden (runs in CI with
/// no oracle install).
fn against_golden(table: &str, method: &str, golden_out: &str) {
    let ours = parse_ours(&ours_output(table, method));
    let theirs = parse_oracle(&std::fs::read_to_string(golden(golden_out)).unwrap());
    check(&ours, &theirs, &format!("{table}/{method} golden"));
}

/// Live differential against scikit-bio when importable.
fn against_oracle(table: &str, method: &str) {
    let Some(py) = skbio_python() else { return };
    let ours = parse_ours(&ours_output(table, method));
    let theirs = parse_oracle(&oracle_output(&py, table, method));
    check(&ours, &theirs, &format!("{table}/{method} live"));
}

#[test]
fn golden_eigh() {
    against_golden("rand_6x4.tsv", "eigh", "rand_6x4.eigh.golden");
}

#[test]
fn golden_svd() {
    against_golden("rand_6x4.tsv", "svd", "rand_6x4.svd.golden");
}

#[test]
fn live_eigh() {
    against_oracle("rand_6x4.tsv", "eigh");
    against_oracle("iris_like.tsv", "eigh");
}

#[test]
fn live_svd() {
    against_oracle("rand_6x4.tsv", "svd");
    against_oracle("iris_like.tsv", "svd");
}
