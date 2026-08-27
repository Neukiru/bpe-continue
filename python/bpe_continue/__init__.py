"""Continue the training of an existing BPE tokenizer on a new corpus.

Standard BPE training starts from single characters. This starts from a tokenizer that is
already trained, and treats *its tokens* as the smallest units new merges can be built from.
Nothing is discarded, so the result is a strict superset of what you started with:

- every existing token keeps its id, and every existing merge keeps its rank;
- the new merges rank after all of them, so they can only glue whole existing tokens together.
  Text is tokenized into *fewer* tokens than before, never into different ones, which is what
  keeps an embedding matrix trained against the original valid;
- added tokens keep their ids too.

The result is an ordinary ``tokenizer.json``: save it and load it with stock ``tokenizers`` or
``transformers``. This package is only needed to produce it.

    >>> from tokenizers import Tokenizer
    >>> from bpe_continue import continue_training
    >>> base = Tokenizer.from_pretrained("gpt2")
    >>> extended = continue_training(base, files=["corpus.txt"], vocab_size=60000)
    >>> extended.save("extended.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from tokenizers import Tokenizer

from . import _native

__version__ = _native.__version__
__all__ = ["continue_training"]

#: A tokenizer to continue from: a ``Tokenizer``, a path to a ``tokenizer.json``, or the
#: contents of one.
TokenizerLike = Union[Tokenizer, str, bytes, "os.PathLike[str]"]


def _as_json(tokenizer: TokenizerLike) -> str:
    """Serialize whatever the caller passed as a base into ``tokenizer.json`` text.

    A ``Tokenizer`` is accepted by duck typing rather than by class, so a tokenizer coming from
    a different build of ``tokenizers`` than the one imported here still works.
    """
    to_str = getattr(tokenizer, "to_str", None)
    if callable(to_str):
        return to_str()

    if isinstance(tokenizer, bytes):
        tokenizer = tokenizer.decode("utf-8")

    if isinstance(tokenizer, (str, os.PathLike)):
        text = str(tokenizer)
        # A path and the contents of a file are both plain strings, so tell them apart by
        # looking at the value rather than by trying to open it and hoping for the best.
        if text.lstrip().startswith("{"):
            return text
        path = Path(tokenizer)
        if not path.is_file():
            raise FileNotFoundError(
                f"No tokenizer at {path}. Pass a Tokenizer, a path to a tokenizer.json, "
                "or the contents of one."
            )
        return path.read_text(encoding="utf-8")

    raise TypeError(
        "base must be a Tokenizer, a path to a tokenizer.json, or the contents of one, "
        f"not {type(tokenizer).__name__}."
    )


def _vocab_size_of(tokenizer_json: str) -> int:
    """How many tokens the base already has, counting added tokens only once.

    They may also appear in the model vocabulary, which is exactly what continuing training
    arranges, so a naive sum would count them twice.
    """
    parsed = json.loads(tokenizer_json)
    vocab = parsed.get("model", {}).get("vocab", {})
    ids = set(vocab.values())
    ids.update(token["id"] for token in parsed.get("added_tokens", []))
    return len(ids)


def continue_training(
    base: TokenizerLike,
    *,
    vocab_size: int,
    files: Optional[Union[str, "os.PathLike[str]", Sequence[Union[str, "os.PathLike[str]"]]]] = None,
    texts: Optional[Iterable[str]] = None,
    min_frequency: int = 0,
    special_tokens: Optional[Sequence[str]] = None,
    max_token_length: Optional[int] = None,
    show_progress: bool = True,
) -> Tokenizer:
    """Extend a trained BPE tokenizer by training it further on a new corpus.

    Args:
        base: The tokenizer to extend, as a :class:`~tokenizers.Tokenizer`, a path to a
            ``tokenizer.json``, or the contents of one. It is never modified.
        vocab_size: Size of the *final* vocabulary. Because nothing is discarded this must be
            larger than what ``base`` already has; the difference is how many tokens are learned.
        files: Path, or paths, of UTF-8 text files to train on. These are streamed, so the
            corpus does not have to fit in memory. Mutually exclusive with ``texts``.
        texts: An iterable of strings to train on, each treated as one document. Held in memory,
            so prefer ``files`` for a large corpus. Mutually exclusive with ``files``.
        min_frequency: Minimum number of times a pair must occur before it can be merged.
        special_tokens: Special tokens to add. Any the base already has stay where they are;
            new ones are appended after everything else.
        max_token_length: Refuse to create tokens longer than this many characters.
        show_progress: Whether to draw progress bars.

    Returns:
        A new :class:`~tokenizers.Tokenizer`. Save it with ``.save(path)`` and it can be loaded
        by stock ``tokenizers`` and ``transformers`` like any other tokenizer.

    Raises:
        ValueError: If the base is not a BPE tokenizer, if neither or both of ``files`` and
            ``texts`` are given, if ``vocab_size`` does not grow the vocabulary, or if the
            corpus contains characters the base cannot represent (a byte-level base — GPT-2,
            Whisper, Pythia, … — can represent anything).
    """
    if (files is None) == (texts is None):
        raise ValueError("Pass exactly one of `files` or `texts`.")

    tokenizer_json = _as_json(base)

    current = _vocab_size_of(tokenizer_json)
    if vocab_size <= current:
        raise ValueError(
            f"vocab_size must be larger than the {current} tokens the base already has, since "
            f"none of them are discarded, but it is {vocab_size}. Pass the size you want the "
            "final vocabulary to be."
        )

    shared = dict(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens or []),
        max_token_length=max_token_length,
        show_progress=show_progress,
    )

    if files is not None:
        paths = _normalize_files(files)
        extended = _native.continue_from_files(tokenizer_json, paths, **shared)
    else:
        extended = _native.continue_from_texts(tokenizer_json, list(texts), **shared)

    return Tokenizer.from_str(extended)


def _normalize_files(files) -> List[str]:
    """Accept a single path or a collection of them, and check they all exist up front."""
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    paths = [str(f) for f in files]
    if not paths:
        raise ValueError("`files` is empty, so there is nothing to train on.")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("No such training file(s): " + ", ".join(missing))
    return paths
