# Vendored dependencies

## `tokenizers`

A modified copy of the [`tokenizers`](https://github.com/huggingface/tokenizers) Rust crate by
Hugging Face, Apache-2.0. It is vendored rather than depended on from crates.io because the
published crate does not have the feature this package is built around.

**Base:** `tokenizers` 0.19.1 (upstream commit `71c2a8d0`).

**Modifications:**

- `BpeTrainer::continue_from_model` — seeds the vocabulary and merges from the model being
  trained rather than from a character alphabet, and uses that model's tokens as the atomic
  units of the merge loop. The base merges are re-emitted at their original ranks so newly
  learned merges can only combine whole existing tokens.
- `BpeTrainer::reserved_tokens` and the `Trainer::reserve_tokens` hook — lets a tokenizer tell
  the trainer about the tokens it holds outside the model, so its added tokens keep their ids
  when the vocabulary grows.
- BPE deserialization accepts merges in both formats: the space-joined strings written up to
  `tokenizers` 0.19, and the token pairs written from 0.20 on.
- `max_token_length` is enforced on the initial pairs too, which matters once the atoms are
  whole tokens rather than single characters.

**Not vendored:** `benches/`, `tests/`, `examples/` and the crate's dev-dependencies, none of
which are needed to build this package. The corresponding `[[bench]]` targets and
`[dev-dependencies]` section were removed from the vendored `Cargo.toml`; nothing else in it
was changed.

**Upgrading:** re-copy `src/`, `Cargo.toml`, `LICENSE` and `README.md` from the fork, then
re-apply the two manifest edits above. The modifications live in
`src/models/bpe/{trainer,serialization,word,mod}.rs`, `src/models/mod.rs`,
`src/models/wordpiece/trainer.rs` and `src/tokenizer/{mod,added_vocabulary}.rs`.
