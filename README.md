# bpe-continue

[![PyPI](https://img.shields.io/pypi/v/bpe-continue.svg)](https://pypi.org/project/bpe-continue/)
[![Python versions](https://img.shields.io/pypi/pyversions/bpe-continue.svg)](https://pypi.org/project/bpe-continue/)
[![License](https://img.shields.io/pypi/l/bpe-continue.svg)](https://github.com/Neukiru/bpe-continue/blob/main/LICENSE)

Extend a trained BPE tokenizer with new vocabulary, without invalidating the old one.

Ordinary BPE training starts from single characters. `bpe-continue` starts from a tokenizer that
is already trained and treats *its tokens* as the smallest units new merges may be built from.
Nothing is discarded, so the result is a strict superset of the tokenizer you started with.

```python
from tokenizers import Tokenizer
from bpe_continue import continue_training

base = Tokenizer.from_pretrained("gpt2")
extended = continue_training(base, files=["corpus.txt"], vocab_size=60000)
extended.save("extended.json")
```

The output is an ordinary `tokenizer.json`. Load it with `tokenizers` or `transformers` like any
other tokenizer; this package is only needed to produce it.

## The problem it solves

A pretrained tokenizer over-segments text that was not well represented in its training data —
an unfamiliar language, a technical domain, a markup or code dialect. Sequences get longer, the
context window holds less, and throughput drops.

Training a replacement tokenizer fixes the segmentation but reassigns every token id, so a
pretrained embedding matrix no longer means anything and the model has to be retrained from
scratch. Continued training avoids that trade-off: existing ids keep their meaning, and new
vocabulary is appended.

## Guarantees

- **Every existing token keeps its id.** Rows of a pretrained embedding matrix stay valid. To
  adopt an extended tokenizer you grow the matrix by the number of tokens learned and
  initialise only the new rows.
- **Every existing merge keeps its rank**, and new merges rank after all of them. New merges can
  therefore only glue whole existing tokens *together*: any text tokenizes to a coarsening of
  what the base produced, never to a different split. Text your corpus covers becomes shorter;
  everything else is byte-for-byte unchanged.
- **Added tokens keep their ids.** Special and control tokens that live above the model
  vocabulary are folded into it at the ids they already hold, so they survive a save/load cycle
  rather than being renumbered when the vocabulary grows.
- **The input tokenizer is never modified.** A new tokenizer is returned.

## Installation

```sh
pip install bpe-continue
```

It installs alongside `tokenizers` rather than replacing it, so the rest of your environment is
unaffected.

## Usage

### From files

Files are streamed, so the corpus does not have to fit in memory.

```python
extended = continue_training(base, files=["a.txt", "b.txt"], vocab_size=60000)
```

`base` may be a `Tokenizer`, a path to a `tokenizer.json`, or the contents of one:

```python
extended = continue_training("base/tokenizer.json", files=["corpus.txt"], vocab_size=60000)
```

### From memory

```python
extended = continue_training(base, texts=["a document", "another"], vocab_size=60000)
```

### Command line

```sh
bpe-continue base.json extended.json corpus/*.txt --vocab-size 60000
```

Add `--sample held-out.txt` to report how much shorter that file's tokenization became.

## Parameters

| Parameter | Description |
| --- | --- |
| `base` | Tokenizer to extend: a `Tokenizer`, a path to a `tokenizer.json`, or its contents. |
| `vocab_size` | Size of the **final** vocabulary. Nothing is discarded, so this must exceed the base's size; the difference is how many tokens are learned. |
| `files` / `texts` | The training corpus. Exactly one of the two. |
| `min_frequency` | Minimum occurrences before a pair may be merged. Default `0`. |
| `special_tokens` | Special tokens to add. Ones the base already has stay where they are; new ones are appended after everything else. |
| `max_token_length` | Refuse to create tokens longer than this many characters. |
| `show_progress` | Whether to draw progress bars. Default `True`. |

## Requirements for the base tokenizer

- **It must use a BPE model.** Continued training means keeping the existing merges, and only
  BPE has them. WordPiece, WordLevel and Unigram are rejected with an explicit error.
- **Every character in the corpus must be reachable from its vocabulary.** Byte-level
  tokenizers — which covers most modern ones — can represent any input, so this is automatic.
  A tokenizer that would silently drop characters instead raises an error naming the word,
  rather than training on a quietly corrupted corpus.

Both `tokenizer.json` merge formats are accepted: the space-joined strings written before
`tokenizers` 0.20, and the token pairs written since.

## How it works

1. The corpus is split into words by the base tokenizer's own normalizer and pre-tokenizer, so
   it is processed exactly as the base would process it.
2. Each word is split into atoms by the base *model* — its tokens, not characters.
3. The vocabulary is seeded with the base's tokens at their existing ids, and its added tokens
   are folded in at theirs.
4. The standard BPE merge loop runs on top, appending merges after the base's.

Because step 4 only ever combines units produced by step 2, no existing token boundary can be
split differently. That is where the guarantees above come from.

## Notes

- `vocab_size` counts the whole final vocabulary, including any added tokens folded in at
  step 3. A tokenizer with 50,000 model tokens and 256 added tokens starts from 50,256.
- Continued training can only make the tokenization of a given text shorter or leave it
  unchanged. It cannot repair a base tokenizer whose *pre-tokenizer* splits your text badly,
  since word boundaries are fixed before BPE runs.

## Licence

Apache-2.0. Built on a modified copy of Hugging Face's
[`tokenizers`](https://github.com/huggingface/tokenizers) Rust crate; see [`NOTICE`](https://github.com/Neukiru/bpe-continue/blob/main/NOTICE) for
details. This is an independent project, not affiliated with or endorsed by Hugging Face.
