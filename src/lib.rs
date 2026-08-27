//! Native half of `bpe-continue`.
//!
//! Everything crosses the Python boundary as a serialized `tokenizer.json` string, which keeps
//! this module completely independent of the `tokenizers` extension module the caller has
//! installed: the two never share Rust types, only JSON. See `python/bpe_continue/__init__.py`
//! for the layer that turns those strings back into `tokenizers.Tokenizer` objects.
use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use tokenizers::models::bpe::BpeTrainer;
use tokenizers::models::{ModelWrapper, TrainerWrapper};
use tokenizers::{AddedToken, Tokenizer};

/// Build the trainer used for every entry point below.
#[allow(clippy::too_many_arguments)]
fn build_trainer(
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    max_token_length: Option<usize>,
    show_progress: bool,
) -> TrainerWrapper {
    BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(min_frequency)
        .max_token_length(max_token_length)
        .show_progress(show_progress)
        .special_tokens(
            special_tokens
                .into_iter()
                .map(|token| AddedToken::from(token, true))
                .collect(),
        )
        .continue_from_model(true)
        .build()
        .into()
}

fn load(tokenizer_json: &str) -> PyResult<Tokenizer> {
    let tokenizer = Tokenizer::from_str(tokenizer_json)
        .map_err(|e| PyValueError::new_err(format!("Could not parse the tokenizer: {}", e)))?;
    match tokenizer.get_model() {
        ModelWrapper::BPE(_) => Ok(tokenizer),
        other => Err(PyValueError::new_err(format!(
            "Continuing training means keeping the existing merges, which only a BPE model has, \
             but this tokenizer uses {}.",
            match other {
                ModelWrapper::WordPiece(_) => "WordPiece",
                ModelWrapper::WordLevel(_) => "WordLevel",
                ModelWrapper::Unigram(_) => "Unigram",
                ModelWrapper::BPE(_) => unreachable!(),
            }
        ))),
    }
}

fn dump(tokenizer: &Tokenizer) -> PyResult<String> {
    tokenizer
        .to_string(false)
        .map_err(|e| PyValueError::new_err(format!("Could not serialize the tokenizer: {}", e)))
}

/// Continue training `tokenizer_json` on the given text files, returning the extended
/// tokenizer as JSON.
///
/// The files are streamed, so the corpus never has to fit in memory.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    tokenizer_json,
    files,
    vocab_size,
    min_frequency = 0,
    special_tokens = Vec::new(),
    max_token_length = None,
    show_progress = true,
))]
fn continue_from_files(
    py: Python<'_>,
    tokenizer_json: &str,
    files: Vec<String>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    max_token_length: Option<usize>,
    show_progress: bool,
) -> PyResult<String> {
    let mut tokenizer = load(tokenizer_json)?;
    let mut trainer = build_trainer(
        vocab_size,
        min_frequency,
        special_tokens,
        max_token_length,
        show_progress,
    );
    // Training is long-running and touches no Python objects, so let other threads run.
    py.allow_threads(|| {
        tokenizer
            .train_from_files(&mut trainer, files)
            .map_err(|e| PyOSError::new_err(e.to_string()))
    })?;
    dump(&tokenizer)
}

/// Continue training `tokenizer_json` on the given sequences, returning the extended tokenizer
/// as JSON. Each sequence is treated as one document.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    tokenizer_json,
    texts,
    vocab_size,
    min_frequency = 0,
    special_tokens = Vec::new(),
    max_token_length = None,
    show_progress = true,
))]
fn continue_from_texts(
    py: Python<'_>,
    tokenizer_json: &str,
    texts: Vec<String>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    max_token_length: Option<usize>,
    show_progress: bool,
) -> PyResult<String> {
    let mut tokenizer = load(tokenizer_json)?;
    let mut trainer = build_trainer(
        vocab_size,
        min_frequency,
        special_tokens,
        max_token_length,
        show_progress,
    );
    py.allow_threads(|| {
        tokenizer
            .train(&mut trainer, texts.into_iter())
            .map_err(|e| PyOSError::new_err(e.to_string()))
    })?;
    dump(&tokenizer)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(continue_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(continue_from_texts, m)?)?;
    Ok(())
}
