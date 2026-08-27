//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
use std::{iter, mem};

mod model;
mod serialization;
pub mod trainer;
mod word;

type Pair = (u32, u32);

/// Errors that can be encountered while using or constructing a `BPE` model.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An error encountered while reading files mainly.
    #[error("IoError: {0}")]
    Io(#[from] std::io::Error),
    /// An error forwarded from Serde, while parsing JSON
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
    /// When the vocab.json file is in the wrong format
    #[error("Bad vocabulary json file")]
    BadVocabulary,
    /// When the merges.txt file is in the wrong format. This error holds the line
    /// number of the line that caused the error.
    #[error("Merges text file invalid at line {0}")]
    BadMerges(usize),
    /// If a token found in merges, is not in the vocab
    #[error("Token `{0}` out of vocabulary")]
    MergeTokenOutOfVocabulary(String),
    /// If the provided unk token is out of vocabulary
    #[error("Unk token `{0}` not found in the vocabulary")]
    UnkTokenOutOfVocabulary(String),
    /// Dropout not between 0 and 1.
    #[error("Dropout should be between 0 and 1")]
    InvalidDropout,
    /// `continue_from_model` was requested but the model has nothing to continue from.
    #[error(
        "`continue_from_model` requires a BPE model that is already trained, but the model \
         being trained has an empty vocabulary. Load an existing tokenizer (for example with \
         `Tokenizer::from_file`) and train that one, instead of training a fresh `BPE::default()`."
    )]
    EmptyBaseModel,
    /// `vocab_size` is smaller than the vocabulary we are continuing from.
    #[error(
        "`vocab_size` is {requested} but the model being continued from already has {base} \
         tokens. When continuing training, `vocab_size` is the size of the *final* vocabulary, \
         so it must be greater than or equal to the size of the base vocabulary."
    )]
    VocabSizeTooSmall { requested: usize, base: usize },
    /// The base vocabulary does not use a dense `0..n` id range.
    #[error(
        "The base vocabulary must assign a dense range of ids `0..{expected}`, but {missing} \
         id(s) in that range are unused (first missing id: {first_missing}). Continuing training \
         from a sparse vocabulary would silently renumber tokens."
    )]
    SparseBaseVocabulary {
        expected: usize,
        missing: usize,
        first_missing: u32,
    },
    /// A word from the training corpus could not be segmented with the base model.
    #[error("Could not segment {word:?} with the base model: {source}")]
    SegmentationFailed {
        word: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The base model quietly dropped part of a word it could not represent.
    #[error(
        "The base model cannot represent {word:?}: it segments to {segmented:?}, silently \
         dropping the rest. Every character of the training corpus has to be reachable from \
         the base vocabulary — a byte-level base covers this by construction. Give the base \
         model an `unk_token` if dropping unknown characters is what you want."
    )]
    UnrepresentableWord { word: String, segmented: String },
}

/// Provides access to the `FirstLastIterator` to any Iterator
pub(crate) trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

impl<I> WithFirstLastIterator for I
where
    I: Iterator,
{
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,
            iter: self.peekable(),
        }
    }
}

/// Provides information about whether an item is the first and/or the last of the iterator
pub(crate) struct FirstLastIterator<I>
where
    I: Iterator,
{
    first: bool,
    iter: iter::Peekable<I>,
}

impl<I> Iterator for FirstLastIterator<I>
where
    I: Iterator,
{
    /// (is_first, is_last, item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first = mem::replace(&mut self.first, false);
        self.iter
            .next()
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}

// Re-export
pub use model::*;
pub use trainer::*;
use word::*;
