"""Command line entry point: ``bpe-continue``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from tokenizers import Tokenizer

from . import __version__, continue_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpe-continue",
        description=(
            "Continue the training of an existing BPE tokenizer on a new corpus. Every token "
            "keeps its id and every merge keeps its rank, so the result tokenizes text into "
            "fewer tokens than the original, never into different ones."
        ),
        epilog=(
            "The output is an ordinary tokenizer.json: load it with stock tokenizers or "
            "transformers like any other."
        ),
    )
    parser.add_argument("base", help="tokenizer.json to continue training from")
    parser.add_argument("output", help="where to write the extended tokenizer.json")
    parser.add_argument("corpus", nargs="+", help="UTF-8 text file(s) to train on")
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        metavar="N",
        help="size of the final vocabulary; must be larger than the base's",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=0,
        metavar="N",
        help="minimum number of occurrences before a pair may be merged (default: 0)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        dest="special_tokens",
        metavar="TOKEN",
        help="add a special token; repeat for more than one",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=None,
        metavar="N",
        help="refuse to create tokens longer than N characters",
    )
    parser.add_argument("--quiet", action="store_true", help="do not draw progress bars")
    parser.add_argument(
        "--sample",
        metavar="FILE",
        help="after training, report how much shorter this file's tokenization became",
    )
    parser.add_argument("--version", action="version", version=f"bpe-continue {__version__}")
    return parser


def _report(base: Tokenizer, extended: Tokenizer, sample: Path) -> None:
    text = sample.read_text(encoding="utf-8")
    before = len(base.encode(text, add_special_tokens=False).ids)
    after = len(extended.encode(text, add_special_tokens=False).ids)
    if before == 0:
        print(f"{sample} is empty, nothing to compare.")
        return
    print(f"{sample}: {before} -> {after} tokens ({100 * (1 - after / before):.1f}% fewer)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    base_path = Path(args.base)
    corpus: List[str] = list(args.corpus)

    try:
        base = Tokenizer.from_file(str(base_path))
        before = len(base.get_vocab(with_added_tokens=True))
        if not args.quiet:
            print(f"{base_path}: {before} tokens")

        extended = continue_training(
            base,
            files=corpus,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens,
            max_token_length=args.max_token_length,
            show_progress=not args.quiet,
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        extended.save(args.output)

        after = len(extended.get_vocab(with_added_tokens=True))
        print(f"{args.output}: {after} tokens ({after - before} learned)")

        if args.sample:
            _report(base, extended, Path(args.sample))
    except Exception as exc:  # noqa: BLE001 - the CLI reports failures, it does not raise them
        print(f"bpe-continue: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
