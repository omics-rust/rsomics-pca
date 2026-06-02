use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_pca(c: &mut Criterion) {
    let bin = env!("CARGO_BIN_EXE_rsomics-pca");
    let table = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/rand_6x4.tsv");
    c.bench_function("rsomics-pca rand_6x4 eigh", |b| {
        b.iter(|| {
            let out = Command::new(black_box(bin))
                .arg(&table)
                .args(["-t", "1"])
                .output()
                .unwrap();
            assert!(out.status.success());
        });
    });
}

criterion_group!(benches, bench_pca);
criterion_main!(benches);
