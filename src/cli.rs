use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, Result, RsomicsError, Tool, ToolMeta};
use rsomics_help::{Example, FlagSpec, HelpSpec, Origin, Section};

use rsomics_pca::{Method, run};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MethodArg {
    Eigh,
    Svd,
}

impl From<MethodArg> for Method {
    fn from(m: MethodArg) -> Method {
        match m {
            MethodArg::Eigh => Method::Eigh,
            MethodArg::Svd => Method::Svd,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "rsomics-pca", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// Feature/sample table TSV (samples × features); reads stdin when "-" or omitted.
    #[arg(default_value = "-")]
    input: PathBuf,

    /// Parse the input as comma-separated instead of tab-separated.
    #[arg(long, default_value_t = false)]
    csv: bool,

    /// Decomposition method.
    #[arg(long, value_enum, default_value_t = MethodArg::Eigh)]
    method: MethodArg,

    /// Keep only the first N principal components (default: all, min(rows,cols)).
    #[arg(long, value_name = "N")]
    dimensions: Option<usize>,

    /// Output path; writes stdout when "-".
    #[arg(short = 'o', long, default_value = "-")]
    output: String,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Tool for Cli {
    fn meta() -> ToolMeta {
        META
    }
    fn common(&self) -> &CommonFlags {
        &self.common
    }

    fn execute(self) -> Result<()> {
        self.common.install_rayon_pool()?;

        let delim = if self.csv { ',' } else { '\t' };

        let reader: Box<dyn std::io::BufRead> = if self.input.as_os_str() == "-" {
            Box::new(BufReader::new(std::io::stdin().lock()))
        } else {
            Box::new(BufReader::new(File::open(&self.input).map_err(|e| {
                RsomicsError::InvalidInput(format!("{}: {e}", self.input.display()))
            })?))
        };
        let mut out: Box<dyn Write> = if self.output == "-" && self.common.json {
            Box::new(std::io::sink())
        } else if self.output == "-" {
            Box::new(BufWriter::new(std::io::stdout().lock()))
        } else {
            Box::new(BufWriter::new(
                File::create(&self.output).map_err(RsomicsError::Io)?,
            ))
        };
        run(reader, &mut out, delim, self.method.into(), self.dimensions)?;
        out.flush().map_err(RsomicsError::Io)
    }
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "Principal Component Analysis of a feature/sample table.",
    origin: Some(Origin {
        upstream: "scikit-bio skbio.stats.ordination.pca",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("10.1080/14786440109462720"),
    }),
    usage_lines: &["[table.tsv] [--method eigh|svd] [--dimensions N] [-o ordination.tsv]"],
    sections: &[Section {
        title: "OPTIONS",
        flags: &[
            FlagSpec {
                short: None,
                long: "csv",
                aliases: &[],
                value: None,
                type_hint: None,
                required: false,
                default: Some("false"),
                description: "Parse the table as comma-separated.",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "method",
                aliases: &[],
                value: Some("<eigh|svd>"),
                type_hint: None,
                required: false,
                default: Some("eigh"),
                description: "Eigendecomposition of the covariance, or SVD of the centered matrix.",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "dimensions",
                aliases: &[],
                value: Some("<N>"),
                type_hint: Some("usize"),
                required: false,
                default: None,
                description: "Keep only the first N principal components.",
                why_default: None,
            },
            FlagSpec {
                short: Some('o'),
                long: "output",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: Some("-"),
                description: "Output path (- for stdout).",
                why_default: None,
            },
        ],
    }],
    examples: &[
        Example {
            description: "PCA of a feature table",
            command: "rsomics-pca table.tsv",
        },
        Example {
            description: "Keep the first two PCs via SVD",
            command: "rsomics-pca table.tsv --method svd --dimensions 2 -o pca.tsv",
        },
    ],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
