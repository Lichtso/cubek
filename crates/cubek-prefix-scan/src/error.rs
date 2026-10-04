use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PrefixScanError {
    #[error("Unsupported rank {rank}, expected 1")]
    UnsupportedRank { rank: usize },

    #[error("Input / output shape mismatch: input {input:?} output {output:?}")]
    InputOutputShapeMismatch {
        input: Vec<usize>,
        output: Vec<usize>,
    },
}
