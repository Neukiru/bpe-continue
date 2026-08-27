#![allow(clippy::map_entry)]

use super::{Error, Pair, WithFirstLastIterator, Word, BPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Model, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: HashSet<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
    continue_from_model: bool,
    reserved_tokens: HashMap<String, u32>,
}

/// Where the atomic units handed to the BPE merge loop come from.
enum Atoms<'a> {
    /// Split each word into its characters. This is the regular, from-scratch behavior.
    Chars,
    /// Split each word using an already-trained BPE model, so that its tokens — rather than
    /// single characters — are the smallest units the merge loop can work with.
    Model(&'a BPE),
}

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,
                vocab_size: 30000,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
                continue_from_model: false,
                reserved_tokens: HashMap::new(),
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.config.initial_alphabet = alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }
    /// Set max_token_length
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    /// Continue training from the vocabulary and merges the model already holds, instead of
    /// starting from a fresh character alphabet. See [`BpeTrainer::continue_from_model`].
    #[must_use]
    pub fn continue_from_model(mut self, continue_from_model: bool) -> Self {
        self.config.continue_from_model = continue_from_model;
        self
    }

    /// Set the tokens the tokenizer holds outside of the model, and the ids it gave them.
    /// See [`BpeTrainer::reserved_tokens`].
    #[must_use]
    pub fn reserved_tokens(mut self, tokens: HashMap<String, u32>) -> Self {
        self.config.reserved_tokens = tokens;
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            continue_from_model: self.config.continue_from_model,
            reserved_tokens: self.config.reserved_tokens,
            words: HashMap::new(),
        }
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::{BPE, BpeTrainer};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = BPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
/// ```
///
/// # Continuing the training of an existing tokenizer
///
/// With [`continue_from_model`](BpeTrainer::continue_from_model) the trainer does not start
/// from a character alphabet. It starts from the vocabulary and merges the model already
/// holds, and treats that model's *tokens* as the atomic units the merge loop may combine.
/// The result is a strict superset of the base tokenizer: every base token keeps its id, every
/// base merge keeps its rank, and the newly learned merges are appended after them.
///
/// Because the new merges rank after every base merge, they can only glue whole base tokens
/// together — so any text tokenizes to a *coarsening* of what the base produced, never to a
/// different split. Text the new corpus covers well gets fewer tokens; everything else is
/// left exactly as it was. Old token ids therefore stay valid, which is what makes it safe to
/// keep using an embedding matrix trained against the base.
///
/// Every character of the corpus has to be reachable from the base vocabulary. A byte-level
/// base (GPT-2, Whisper, Pythia, …) satisfies this for any input; a base that would silently
/// drop characters is rejected with an error naming the offending word.
///
/// Because the base is the tokenizer being trained, its normalizer and pre-tokenizer are
/// applied exactly once — by the tokenizer itself — so there is no risk of double-normalizing
/// the corpus.
///
/// ```no_run
/// use tokenizers::models::bpe::BpeTrainer;
/// use tokenizers::Tokenizer;
///
/// let mut tokenizer = Tokenizer::from_file("gpt2/tokenizer.json").unwrap();
///
/// let mut trainer = BpeTrainer::builder()
///     .vocab_size(60_000)          // size of the *final* vocabulary
///     .continue_from_model(true)
///     .build();
///
/// tokenizer
///     .train_from_files(&mut trainer, vec!["corpus.txt".to_string()]).unwrap()
///     .save("extended.json", true).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    pub min_frequency: u64,
    /// The target vocabulary size
    pub vocab_size: usize,
    /// Whether to show progress while training
    pub show_progress: bool,
    /// A list of special tokens that the model should know of
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    pub initial_alphabet: HashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to caracterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An optional parameter to limit the max length of any single token
    pub max_token_length: Option<usize>,
    /// Continue training from the vocabulary and merges the model already holds rather than
    /// from a character alphabet, using its tokens as the atomic units of the merge loop.
    ///
    /// The model being trained must already be trained; `vocab_size` is then the size of the
    /// *final* vocabulary, so it must be at least as large as the base vocabulary.
    /// See the type-level documentation for details.
    #[serde(default)]
    pub continue_from_model: bool,
    /// The tokens the tokenizer holds outside of its model, and the ids it gave them.
    ///
    /// A tokenizer's added tokens live outside its model but share its id space: Whisper, for
    /// instance, has a dense model vocabulary over `0..=50256` and about 1600 added tokens
    /// directly above it. When continuing training, those are folded into the vocabulary at
    /// the ids they already have, so that the tokens learned here land above them and nothing
    /// is renumbered.
    ///
    /// The tokenizer fills this in itself just before training, so it is rarely set by hand.
    #[serde(default)]
    pub reserved_tokens: HashMap<String, u32>,

    words: HashMap<String, u64>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(&self, w2id: &mut HashMap<String, u32>, id2w: &mut Vec<String>) {
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                w2id.insert(token.content.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        // Compute the alphabet from seen words
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                alphabet
                    .entry(c)
                    .and_modify(|cnt| *cnt += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            alphabet
                .entry(*c)
                .and_modify(|cnt| *cnt = std::usize::MAX)
                .or_insert(std::usize::MAX);
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| {
                if alphabet.len() > limit {
                    alphabet.len() - limit
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| (*k.0) as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
        });
    }

    /// Seed the vocabulary from an already-trained model, so that its tokens — rather than
    /// single characters — become the atomic units of the merge loop.
    ///
    /// Every base token keeps the id it already has, so the resulting vocabulary is a strict
    /// superset of the base one. Returns the base merges, in rank order, so that they can be
    /// re-emitted ahead of the ones we are about to learn.
    fn seed_from_model(
        &self,
        model: &BPE,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) -> Result<Vec<(Pair, u32)>> {
        if model.vocab.is_empty() {
            return Err(Error::EmptyBaseModel.into());
        }

        // `id2w` is indexed positionally, so the base ids have to form a dense `0..n` range
        // for the base tokens to keep the ids they already have.
        let mut by_id: Vec<Option<&str>> = vec![None; model.vocab.len()];
        for (token, &id) in &model.vocab {
            if let Some(slot) = by_id.get_mut(id as usize) {
                *slot = Some(token.as_str());
            }
        }
        let missing: Vec<u32> = by_id
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_none())
            .map(|(id, _)| id as u32)
            .collect();
        if let Some(&first_missing) = missing.first() {
            return Err(Error::SparseBaseVocabulary {
                expected: model.vocab.len(),
                missing: missing.len(),
                first_missing,
            }
            .into());
        }

        w2id.reserve(model.vocab.len() + self.reserved_tokens.len());
        id2w.reserve(model.vocab.len() + self.reserved_tokens.len());
        for token in by_id {
            let token = token
                .expect("all ids are filled in, checked above")
                .to_owned();
            w2id.insert(token.clone(), id2w.len() as u32);
            id2w.push(token);
        }

        // Fold in the tokens the tokenizer keeps outside the model — its added tokens — at the
        // ids it already gave them. They have to be part of the vocabulary for those ids to
        // survive: a reloaded tokenizer re-derives the id of any added token it cannot find in
        // the model, so leaving them out would renumber every one of them as soon as the model
        // grows. Sitting in the vocabulary also keeps them from being handed to a new token.
        let mut reserved: Vec<(&String, u32)> = self
            .reserved_tokens
            .iter()
            .map(|(token, id)| (token, *id))
            .filter(|(token, id)| *id >= id2w.len() as u32 && !w2id.contains_key(*token))
            .collect();
        reserved.sort_unstable_by_key(|(_, id)| *id);
        for (token, id) in reserved {
            // Only a token that lands exactly on the next free id can be placed. Anything past
            // a gap would leave a hole in the vocabulary, and the tokenizer would renumber it
            // on reload regardless, so it is left to be renumbered.
            if id != id2w.len() as u32 {
                break;
            }
            w2id.insert(token.clone(), id);
            id2w.push(token.clone());
        }

        if self.vocab_size < id2w.len() {
            return Err(Error::VocabSizeTooSmall {
                requested: self.vocab_size,
                base: id2w.len(),
            }
            .into());
        }

        // Re-emit the base merges, in their original order, ahead of the ones we are about to
        // learn. Without them the resulting model would know the base tokens but have no rule
        // to build them, and could not reproduce the base tokenization.
        let mut ranked: Vec<(u32, Pair, u32)> = model
            .merges
            .iter()
            .map(|(pair, (rank, new_id))| (*rank, *pair, *new_id))
            .collect();
        ranked.sort_unstable_by_key(|(rank, _, _)| *rank);
        Ok(ranked
            .into_iter()
            .map(|(_, pair, new_id)| (pair, new_id))
            .collect())
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
        atoms: Atoms<'_>,
        p: &Option<ProgressBar>,
    ) -> Result<(Vec<Word>, Vec<u64>)> {
        // `wc` is a HashMap, so iterating it directly would let a run-to-run random order
        // decide which ids the tokens below are given. Sort first, so training is reproducible.
        let mut entries: Vec<(&String, &u64)> = wc.iter().collect();
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        // Splitting words into atoms needs no shared state, and with `Atoms::Model` it is by
        // far the most expensive part of this step, so do it up front and in parallel.
        let segmented: Vec<Vec<String>> = match atoms {
            Atoms::Chars => entries
                .iter()
                .map(|(word, _)| word.chars().map(|c| c.to_string()).collect())
                .collect(),
            Atoms::Model(model) => {
                // A BPE model with no `unk_token` drops characters it does not know instead of
                // failing, which would quietly corrupt the counts we are training on. We can
                // catch that by checking the atoms still spell the word — but only when the
                // base attaches no affixes to them, which is the case for byte-level bases.
                let exact = model.unk_token.is_none()
                    && model.continuing_subword_prefix.is_none()
                    && model.end_of_word_suffix.is_none();
                entries
                    .maybe_par_iter()
                    .map(|(word, _)| -> Result<Vec<String>> {
                        let atoms: Vec<String> = model
                            .tokenize(word)
                            .map_err(|source| -> crate::Error {
                                Error::SegmentationFailed {
                                    word: (*word).to_owned(),
                                    source,
                                }
                                .into()
                            })?
                            .into_iter()
                            .map(|token| token.value)
                            .collect();
                        if exact && atoms.concat() != **word {
                            return Err(Error::UnrepresentableWord {
                                word: (*word).to_owned(),
                                segmented: atoms.concat(),
                            }
                            .into());
                        }
                        Ok(atoms)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
            }
        };

        let mut words: Vec<Word> = Vec::with_capacity(entries.len());
        let mut counts: Vec<u64> = Vec::with_capacity(entries.len());

        for ((_, count), atoms) in entries.iter().zip(segmented) {
            let mut current_word = Word::new();
            counts.push(**count);

            for (is_first, is_last, atom) in atoms.into_iter().with_first_and_last() {
                // `max_token_length` is expressed in characters, so measure the atom before
                // any affix is attached — the same thing the char path does by passing `1`.
                let len = atom.chars().count();
                let mut s = atom;
                if w2id.contains_key(&s) {
                    // Found the initial atom in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s = format!("{}{}", prefix, s);
                        }
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s = format!("{}{}", s, suffix);
                        }
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&s) {
                        id2w.push(s.clone());
                        w2id.insert(s.clone(), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&s], len);
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        Ok((words, counts))
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        max_token_length: usize,
        p: &Option<ProgressBar>,
    ) -> (HashMap<Pair, i32>, HashMap<Pair, HashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = HashMap::new();
                let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for (cur_pair, merged_len) in word.get_pairs_iter() {
                    // Merging this pair would already overshoot the limit. With single
                    // characters as atoms this can never happen, but when we continue from a
                    // model the atoms are whole tokens and can be long on their own.
                    if merged_len > max_token_length {
                        continue;
                    }

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    if !pair_counts.contains_key(&cur_pair) {
                        pair_counts.insert(cur_pair, 0);
                    }

                    // Then update counts
                    let count = counts[i];
                    where_to_update
                        .entry(cur_pair)
                        .and_modify(|h| {
                            h.insert(i);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(i);
                            h
                        });
                    *pair_counts.get_mut(&cur_pair).unwrap() += count as i32;
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        pair_counts.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    for (k, v) in wtu {
                        where_to_update
                            .entry(k)
                            .and_modify(|set| *set = set.union(&v).copied().collect())
                            .or_insert(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        // The model is only cloned when we continue from it, and only so that we can keep
        // reading the base vocabulary and merges while `model` is overwritten further down.
        let base_model = if self.continue_from_model {
            Some(model.clone())
        } else {
            None
        };

        //
        // 1. Build the initial vocabulary
        //
        let base_merges = match &base_model {
            Some(base_model) => {
                // Continue from what the model already knows: its tokens become the atomic
                // units of the merge loop, and every one of them keeps its current id.
                let merges = self.seed_from_model(base_model, &mut word_to_id, &mut id_to_word)?;
                // Special tokens the base does not already have go after everything else, so
                // that no existing id is shifted.
                self.add_special_tokens(&mut word_to_id, &mut id_to_word);
                merges
            }
            None => {
                self.add_special_tokens(&mut word_to_id, &mut id_to_word);
                self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
                vec![]
            }
        };

        //
        // 2. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let atoms = match &base_model {
            Some(base_model) => Atoms::Model(base_model),
            None => Atoms::Chars,
        };
        let (words, counts) = self.tokenize_words(
            word_counts,
            &mut word_to_id,
            &mut id_to_word,
            atoms,
            &progress,
        )?;
        self.finalize_progress(&progress, words.len());

        //
        // 3. Count pairs in words
        //
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) =
            self.count_pairs(&words, &counts, max_token_length, &progress);

        // Insert them in the queue
        let mut queue = BinaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });
        self.finalize_progress(&progress, words.len());

        //
        // 4. Do merges
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        // When continuing, the base merges come first and keep their original ranks, so that
        // the new ones can only ever glue whole base tokens together.
        let mut merges: Vec<(Pair, u32)> = base_merges;
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= self.vocab_size {
                break;
            }

            if queue.is_empty() {
                break;
            }

            let mut top = queue.pop().unwrap();
            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].to_owned();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix {
                if part_b.starts_with(prefix) {
                    let prefix_byte_len = prefix.chars().map(|c| c.len_utf8()).sum();
                    part_b = part_b[prefix_byte_len..].to_string();
                }
            }
            let new_token = format!("{}{}", part_a, part_b);
            // implement sentencepiece-like merge.
            // if this code were to be merged, integrate a way in the python bindings to communicate this variable
            // default should be 0/None to maintain previous behavior. 16 is the spm default.

            // Insert new token if it does not already exist
            let new_token_id = word_to_id
                .get(&new_token)
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if word_to_id.get(&new_token).is_none() {
                id_to_word.push(new_token.clone());
                word_to_id.insert(new_token.clone(), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // Merge the new pair in every words
            let changes = top
                .pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    let word = &words[i] as *const _ as *mut Word;
                    // We can merge each of these words in parallel here because each position
                    // can be there only once (HashSet). So this is safe.
                    unsafe {
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;
                pair_counts
                    .entry(pair)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
                if change > 0 {
                    where_to_update
                        .entry(pair)
                        .and_modify(|h| {
                            h.insert(iw);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(iw);
                            h
                        });
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, merges.len());

        // Transfer new vocab & options to model
        model.vocab = word_to_id;
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();

        if let Some(prefix) = &self.continuing_subword_prefix {
            model.continuing_subword_prefix = Some(prefix.to_owned());
        } else {
            model.continuing_subword_prefix = None;
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            model.end_of_word_suffix = Some(suffix.to_owned());
        } else {
            model.end_of_word_suffix = None;
        }

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for BpeTrainer {
    type Model = BPE;

    /// Train a BPE model
    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }

    fn reserve_tokens(&mut self, tokens: &HashMap<String, u32>) {
        // Only relevant when we keep the model's existing ids; a from-scratch run numbers the
        // vocabulary itself, and the tokenizer renumbers its added tokens afterwards.
        if self.continue_from_model {
            self.reserved_tokens
                .extend(tokens.iter().map(|(t, id)| (t.clone(), *id)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BpeTrainer, Pair, BPE};
    use crate::tokenizer::Model;
    use std::collections::HashMap;

    /// Train a small BPE model from scratch, to be used as the base of a continued training.
    fn base_model(word_counts: &HashMap<String, u64>, vocab_size: usize) -> BPE {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(0)
            .vocab_size(vocab_size)
            .build();
        let mut model = BPE::default();
        trainer.do_train(word_counts, &mut model).unwrap();
        model
    }

    fn counts(words: &[(&str, u64)]) -> HashMap<String, u64> {
        words
            .iter()
            .map(|(word, count)| ((*word).to_string(), *count))
            .collect()
    }

    fn tokens(model: &BPE, word: &str) -> Vec<String> {
        model
            .tokenize(word)
            .unwrap()
            .into_iter()
            .map(|token| token.value)
            .collect()
    }

    /// Whether `coarse` is `fine` with some runs of adjacent tokens glued together — that is,
    /// whether it introduces no token boundary that `fine` did not already have.
    fn is_coarsening_of(coarse: &[String], fine: &[String]) -> bool {
        let mut fine = fine.iter();
        for token in coarse {
            let mut acc = String::new();
            while acc.len() < token.len() {
                match fine.next() {
                    Some(next) => acc.push_str(next),
                    None => return false,
                }
            }
            if &acc != token {
                return false;
            }
        }
        fine.next().is_none()
    }

    #[test]
    fn test_train() {
        let word_counts: HashMap<String, u64> = [
            ("roses".into(), 1),
            ("are".into(), 2),
            ("red".into(), 1),
            ("voilets".into(), 1),
            ("blue".into(), 1),
            ("BERT".into(), 1),
            ("is".into(), 2),
            ("big".into(), 1),
            ("and".into(), 1),
            ("so".into(), 1),
            ("GPT-2".into(), 1),
        ]
        .iter()
        .cloned()
        .collect();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(2)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&word_counts, &mut model).unwrap();

        // Vocab should contain all of the characters from the `word_counts` mapping
        // as well as three merges: 're', 'are', and 'is'.
        let expected_vocab: HashMap<String, u32> = [
            ("-".into(), 0),
            ("2".into(), 1),
            ("B".into(), 2),
            ("E".into(), 3),
            ("G".into(), 4),
            ("P".into(), 5),
            ("R".into(), 6),
            ("T".into(), 7),
            ("a".into(), 8),
            ("b".into(), 9),
            ("d".into(), 10),
            ("e".into(), 11),
            ("g".into(), 12),
            ("i".into(), 13),
            ("l".into(), 14),
            ("n".into(), 15),
            ("o".into(), 16),
            ("r".into(), 17),
            ("s".into(), 18),
            ("t".into(), 19),
            ("u".into(), 20),
            ("v".into(), 21),
            ("re".into(), 22),
            ("are".into(), 23),
            ("is".into(), 24),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.vocab, expected_vocab);

        // The keys in `merges` are pairs of symbols, the values are tuples of (rank, id),
        // where 'rank' determines the order in which this merge will be applied during
        // tokenization, and 'id' is the vocab id of the symbol resulting from merging
        // the pair of symbols in the corresponding key.
        let expected_merges: HashMap<Pair, (u32, u32)> = [
            ((17, 11), (0, 22)), // 'r' + 'e'  -> 're'
            ((8, 22), (1, 23)),  // 'a' + 're' -> 'are'
            ((13, 18), (2, 24)), // 'i' + 's'  -> 'is'
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.merges, expected_merges);
    }
    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length series of tests test the max_token_length flag of bpetrainer
        // this is the more robust version that only tests max length of learned tokens
        // (pre) tokenizer settings or vocab can be easily modified when necessary
         */

        let max_token_length = 16;
        let long_word_counts: HashMap<String, u64> = [
            ("singlelongtokenwithoutcasechange", 2),
            ("singleLongTokenWithCamelCaseChange", 2),
            ("Longsingletokenwithpunctu@t!onwithin", 2),
            ("Anotherlongsingletokenwithnumberw1th1n", 2),
            ("짧은한글문자열짧은한", 2),             // korean 10 char
            ("긴한글문자열긴한글문자열긴한글문", 2), // korean 16 char
            ("短字符串短字符串短字", 2),             //simplified chinese 10 char
            ("长字符串长字符串长字符串长字符串", 2), // simp. chinese 16 char
            ("短い文字列短い文字列", 2),             // japanese 10 char
            ("長い文字列長い文字列長い文字列長", 2), // japanese 16 char
            ("so", 2),
            ("GPT-2", 2),
        ]
        .iter()
        .map(|(key, value)| (key.to_string(), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let vocab = model.get_vocab();
        for token in vocab.keys() {
            assert!(
                token.chars().count() <= max_token_length,
                "token too long : {} , chars().count() = {}",
                token,
                token.chars().count()
            )
        }
    }
    #[test]
    fn bpe_test_max_token_length_direct_assert() {
        /* more direct version of bpe_test_max_token_length test
        // directly compares tokens with known expected values.
        // maybe unstable depending on specific settings or changes.
         */
        let long_word_counts: HashMap<String, u64> = [
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]
        .iter()
        .map(|(key, value)| (key.to_string(), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(2))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let trained_vocab: HashMap<String, u32> = model.get_vocab();
        let expected_vocab: HashMap<String, u32> = [
            ("短", 12),
            ("n", 6),
            ("i", 5),
            ("s", 8),
            ("字符", 23),
            ("長", 14),
            ("긴", 17),
            ("い文", 22),
            ("L", 2),
            ("in", 21),
            ("o", 7),
            ("은한", 29),
            ("S", 4),
            ("P", 3),
            ("so", 27),
            ("符", 13),
            ("文", 11),
            ("字", 10),
            ("짧", 19),
            ("GP", 25),
            ("글", 16),
            ("G", 1),
            ("An", 24),
            ("长", 15),
            ("A", 0),
            ("Lo", 26),
            ("긴한", 28),
            ("い", 9),
            ("한", 20),
            ("은", 18),
        ]
        .iter()
        .cloned()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(trained_vocab, expected_vocab)
    }

    /// Words for the base corpus. Between them they cover every character the "new" corpora
    /// below use, which is what lets the base segment those corpora at all.
    const BASE_WORDS: &[(&str, u64)] = &[
        ("roses", 1),
        ("are", 2),
        ("red", 1),
        ("is", 2),
        ("so", 1),
        ("abcdefghijklmnopqrstuvwxyz", 1),
    ];

    #[test]
    fn continue_from_model_preserves_the_base() {
        let base = base_model(&counts(BASE_WORDS), 40);
        let base_vocab = base.get_vocab();
        let base_merges = base.merges.clone();
        assert!(
            !base_merges.is_empty(),
            "the base should have merges to keep"
        );

        // Something the base has never seen, alongside the words it was trained on.
        let new_counts = counts(&[
            ("banana", 20),
            ("bananas", 12),
            ("bandana", 8),
            ("are", 2),
            ("red", 1),
        ]);
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(0)
            .vocab_size(base_vocab.len() + 8)
            .continue_from_model(true)
            .build();

        let mut model = base.clone();
        trainer.do_train(&new_counts, &mut model).unwrap();
        let vocab = model.get_vocab();

        // Every base token keeps the exact id it had.
        for (token, id) in &base_vocab {
            assert_eq!(
                vocab.get(token),
                Some(id),
                "base token {:?} did not keep its id",
                token
            );
        }
        assert_eq!(vocab.len(), base_vocab.len() + 8);

        // Every base merge keeps its rank, and the new merges come after them.
        for (pair, (rank, new_id)) in &base_merges {
            assert_eq!(
                model.merges.get(pair),
                Some(&(*rank, *new_id)),
                "base merge {:?} was not preserved",
                pair
            );
        }
        assert!(model.merges.len() > base_merges.len());

        // And every base token boundary survives: because the new merges rank after all the
        // base ones, they can only glue whole base tokens together, never re-split them.
        for word in ["roses", "are", "red", "is", "so", "banana"] {
            let before = tokens(&base, word);
            let after = tokens(&model, word);
            assert!(
                is_coarsening_of(&after, &before),
                "{:?}: {:?} is not a coarsening of {:?}",
                word,
                after,
                before
            );
        }
    }

    #[test]
    fn continue_from_model_learns_the_new_corpus() {
        let base = base_model(&counts(BASE_WORDS), 40);

        let new_counts = counts(&[("banana", 20), ("bananas", 12), ("bandana", 8)]);
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(0)
            .vocab_size(base.get_vocab().len() + 6)
            .continue_from_model(true)
            .build();

        let mut model = base.clone();
        trainer.do_train(&new_counts, &mut model).unwrap();

        let before = tokens(&base, "banana");
        let after = tokens(&model, "banana");
        assert!(
            after.len() < before.len(),
            "expected fewer tokens after continuing: {:?} -> {:?}",
            before,
            after
        );
        assert_eq!(
            after.concat(),
            "banana",
            "the segmentation must still round-trip"
        );
    }

    #[test]
    fn continue_from_model_folds_in_reserved_tokens_at_their_ids() {
        let base = base_model(&counts(BASE_WORDS), 40);
        let base_size = base.get_vocab().len() as u32;

        // Stand in for a tokenizer's added tokens, which live outside the model but share its
        // id space, sitting directly above the model vocabulary.
        let reserved: HashMap<String, u32> = (0..5)
            .map(|i| (format!("<|special-{}|>", i), base_size + i))
            .collect();

        let new_counts = counts(&[("banana", 20), ("bananas", 12), ("bandana", 8)]);
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(0)
            .vocab_size(base_size as usize + 5 + 4)
            .continue_from_model(true)
            .reserved_tokens(reserved.clone())
            .build();

        let mut model = base.clone();
        trainer.do_train(&new_counts, &mut model).unwrap();
        let vocab = model.get_vocab();

        // Each reserved token is now in the vocabulary, at exactly the id it came with.
        for (token, id) in &reserved {
            assert_eq!(vocab.get(token), Some(id), "reserved token {:?}", token);
        }
        // So the tokens learned here had to go above them.
        for (token, id) in &vocab {
            assert!(
                *id < base_size || reserved.values().any(|r| r == id) || *id >= base_size + 5,
                "token {:?} took reserved id {}",
                token,
                id
            );
        }
        // And the vocabulary is dense, with no hole where a reserved id used to be.
        let mut ids: Vec<u32> = vocab.values().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..vocab.len() as u32).collect::<Vec<_>>());
        assert_eq!(vocab.len(), base_size as usize + 5 + 4);
    }

    #[test]
    fn continue_from_model_measures_max_token_length_in_characters() {
        let base_counts = counts(&[("aaaa", 10), ("bbbb", 10)]);
        let base = base_model(&base_counts, 40);
        // The base already learned multi-character tokens, so the atoms below are not chars.
        assert!(base.get_vocab().keys().any(|t| t.chars().count() > 1));

        let new_counts = counts(&[("aaaabbbb", 20), ("aaaa", 10), ("bbbb", 10)]);
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(0)
            .vocab_size(base.get_vocab().len() + 10)
            .max_token_length(Some(5))
            .continue_from_model(true)
            .build();

        let mut model = base.clone();
        trainer.do_train(&new_counts, &mut model).unwrap();

        for token in model.get_vocab().keys() {
            assert!(
                token.chars().count() <= 5,
                "token {:?} is longer than max_token_length",
                token
            );
        }
    }

    #[test]
    fn continue_from_model_rejects_an_untrained_model() {
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .continue_from_model(true)
            .build();
        let err = trainer
            .do_train(&counts(&[("banana", 2)]), &mut BPE::default())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("empty vocabulary"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn continue_from_model_rejects_a_shrinking_vocab_size() {
        let base = base_model(&counts(BASE_WORDS), 40);
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(3)
            .continue_from_model(true)
            .build();
        let err = trainer
            .do_train(&counts(&[("bananas", 2)]), &mut base.clone())
            .unwrap_err()
            .to_string();
        assert!(err.contains("final"), "unexpected error: {}", err);
    }

    #[test]
    fn continue_from_model_reports_words_it_cannot_represent() {
        // A base with no `unk_token` drops characters it has never seen instead of failing.
        let base = base_model(&counts(&[("abc", 2)]), 40);
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(100)
            .continue_from_model(true)
            .build();
        let err = trainer
            .do_train(&counts(&[("zzz", 2)]), &mut base.clone())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("zzz"),
            "the offending word should be named: {}",
            err
        );
    }
}
