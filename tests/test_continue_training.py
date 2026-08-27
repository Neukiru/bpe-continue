import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from bpe_continue import continue_training

OLD_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks and the fox runs away",
    "quick thinking saves the lazy dog again",
]
NEW_CORPUS = [
    "tokenization tokenizer tokenizing tokenized",
    "tokenization is tokenization and tokenizer is tokenizer",
    "the tokenizer tokenizes tokenization tokens",
]


@pytest.fixture(scope="module")
def base():
    """A byte-level BPE tokenizer, the kind continued training is meant to start from.

    The full byte alphabet is seeded in, as a real byte-level tokenizer does, so it can
    represent any input at all.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        OLD_CORPUS,
        trainer=trainers.BpeTrainer(
            vocab_size=300,
            min_frequency=0,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tokenizer


def extend(base, n=20, vocab_size=None, **kwargs):
    """Continue training `base`, targeting `n` more tokens than it already has.

    `vocab_size` has to be given when `base` is not a Tokenizer, since the size cannot be read
    off a path or a JSON string.
    """
    kwargs.setdefault("texts", NEW_CORPUS)
    if vocab_size is None:
        vocab_size = len(base.get_vocab(with_added_tokens=True)) + n
    return continue_training(base, vocab_size=vocab_size, show_progress=False, **kwargs)


class TestGuarantees:
    def test_every_existing_id_is_kept(self, base):
        before = base.get_vocab(with_added_tokens=False)
        vocab = extend(base).get_vocab(with_added_tokens=False)

        assert len(vocab) == len(before) + 20
        for token, id in before.items():
            assert vocab[token] == id, f"{token!r} was renumbered"

    def test_new_merges_only_coarsen(self, base):
        extended = extend(base)
        text = "the tokenizer tokenizes tokenization"
        old, new = base.encode(text).tokens, extended.encode(text).tokens

        # Strictly fewer tokens, and every new one is a run of old ones glued together, so no
        # boundary the base produced is split differently.
        assert len(new) < len(old)
        assert "".join(new) == "".join(old)
        it = iter(old)
        for token in new:
            acc = ""
            while len(acc) < len(token):
                acc += next(it)
            assert acc == token
        assert next(it, None) is None

    def test_output_round_trips(self, base):
        extended = extend(base)
        text = "the tokenizer tokenizes tokenization"
        assert extended.decode(extended.encode(text).ids) == text

    def test_added_tokens_keep_their_ids(self, base):
        tokenizer = Tokenizer.from_str(base.to_str())
        tokenizer.add_special_tokens(["<|endoftext|>", "<|startoftranscript|>"])
        added = {t.content: i for i, t in tokenizer.get_added_tokens_decoder().items()}
        assert len(added) == 2

        extended = extend(tokenizer)
        for token, id in added.items():
            assert extended.token_to_id(token) == id, f"{token!r} was renumbered"

        # Dense vocabulary, so it saves and reloads without holes or renumbering.
        vocab = extended.get_vocab(with_added_tokens=True)
        assert sorted(vocab.values()) == list(range(len(vocab)))
        reloaded = Tokenizer.from_str(extended.to_str())
        for token, id in added.items():
            assert reloaded.token_to_id(token) == id

    def test_the_base_is_never_modified(self, base):
        before = base.to_str()
        extend(base)
        assert base.to_str() == before


class TestInputs:
    def test_accepts_a_tokenizer(self, base):
        assert extend(base) is not None

    def test_accepts_a_path(self, base, tmp_path):
        size = len(base.get_vocab(with_added_tokens=True)) + 20
        path = tmp_path / "base.json"
        base.save(str(path))
        assert extend(str(path), vocab_size=size) is not None
        assert extend(path, vocab_size=size) is not None

    def test_accepts_tokenizer_json(self, base):
        size = len(base.get_vocab(with_added_tokens=True)) + 20
        assert extend(base.to_str(), vocab_size=size) is not None

    def test_trains_from_files(self, base, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("\n".join(NEW_CORPUS), encoding="utf-8")
        extended = extend(base, files=str(corpus), texts=None)
        assert len(extended.get_vocab(True)) == len(base.get_vocab(True)) + 20

    def test_reads_the_modern_merges_format(self, base):
        """`tokenizers` 0.20+ writes merges as pairs rather than space-joined strings."""
        parsed = json.loads(base.to_str())
        parsed["model"]["merges"] = [
            m if isinstance(m, list) else m.split(" ", 1) for m in parsed["model"]["merges"]
        ]
        assert isinstance(parsed["model"]["merges"][0], list)
        size = len(base.get_vocab(with_added_tokens=True)) + 20
        assert extend(json.dumps(parsed), vocab_size=size) is not None


class TestErrors:
    def test_rejects_a_non_bpe_tokenizer(self):
        tokenizer = Tokenizer(models.WordLevel({"a": 0}, unk_token="a"))
        with pytest.raises(ValueError, match="BPE"):
            continue_training(tokenizer, texts=NEW_CORPUS, vocab_size=1000)

    def test_rejects_a_vocab_size_that_does_not_grow(self, base):
        with pytest.raises(ValueError, match="larger"):
            continue_training(base, texts=NEW_CORPUS, vocab_size=10)

    def test_requires_exactly_one_corpus(self, base):
        with pytest.raises(ValueError, match="exactly one"):
            continue_training(base, vocab_size=100000)
        with pytest.raises(ValueError, match="exactly one"):
            continue_training(base, files=["a.txt"], texts=NEW_CORPUS, vocab_size=100000)

    def test_reports_a_missing_training_file(self, base):
        with pytest.raises(FileNotFoundError, match="nope.txt"):
            extend(base, files=["nope.txt"], texts=None)

    def test_reports_a_missing_base(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            continue_training(tmp_path / "nope.json", texts=NEW_CORPUS, vocab_size=100000)

    def test_rejects_an_unusable_base_type(self):
        with pytest.raises(TypeError, match="Tokenizer"):
            continue_training(42, texts=NEW_CORPUS, vocab_size=100000)

    def test_reports_characters_the_base_cannot_represent(self):
        # Not byte-level, so it only knows the characters it was trained on and would
        # otherwise drop the rest of the corpus without saying anything.
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.train_from_iterator(
            ["aaa bbb ccc"],
            trainer=trainers.BpeTrainer(vocab_size=30, min_frequency=0, show_progress=False),
        )
        with pytest.raises(Exception, match="zzz"):
            continue_training(tokenizer, texts=["zzz zzz"], vocab_size=100, show_progress=False)


class TestCli:
    def test_end_to_end(self, base, tmp_path, capsys):
        from bpe_continue._cli import main

        base_path, out_path = tmp_path / "base.json", tmp_path / "out.json"
        corpus, sample = tmp_path / "corpus.txt", tmp_path / "sample.txt"
        base.save(str(base_path))
        corpus.write_text("\n".join(NEW_CORPUS), encoding="utf-8")
        sample.write_text("the tokenizer tokenizes tokenization", encoding="utf-8")

        target = len(base.get_vocab(True)) + 20
        code = main(
            [
                str(base_path),
                str(out_path),
                str(corpus),
                "--vocab-size",
                str(target),
                "--quiet",
                "--sample",
                str(sample),
            ]
        )

        assert code == 0
        assert len(Tokenizer.from_file(str(out_path)).get_vocab(True)) == target
        assert "fewer" in capsys.readouterr().out

    def test_reports_failure_without_a_traceback(self, base, tmp_path, capsys):
        from bpe_continue._cli import main

        base_path, corpus = tmp_path / "base.json", tmp_path / "corpus.txt"
        base.save(str(base_path))
        corpus.write_text("hello", encoding="utf-8")

        code = main([str(base_path), str(tmp_path / "o.json"), str(corpus), "--vocab-size", "5"])
        assert code == 1
        assert "bpe-continue:" in capsys.readouterr().err
