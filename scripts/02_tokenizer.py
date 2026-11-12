#!/usr/bin/env python3
"""Train the SentencePiece tokenizer in the layout FUTO's native code requires.

Requirements (see docs/FUTO_FORMAT.md):
- Unigram, whitespace as suffix (`inverted_space`)
- ids: <unk>=0 <s>=1 </s>=2 <pad>=3 (FUTO prepends BOS id 1)
- user-defined symbols <XBU> <XBC> <XEC> <CHAR_A>..<CHAR_Z>; CHAR_* ids must be consecutive (given by order)
"""
import argparse, random, string
from pathlib import Path
import sentencepiece as spm

CONTROL = ["<XBU>", "<XBC>", "<XEC>"] + [f"<CHAR_{c}>" for c in string.ascii_uppercase]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="models/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--per-source", type=int, default=1_500_000, help="max sentences taken from each source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.seed)
    sample = out / "tok_sample.txt"
    total = 0
    with open(sample, "w", encoding="utf-8") as f:
        for p in sorted(Path(args.raw_dir).glob("*.txt")):
            lines = p.read_text(encoding="utf-8").splitlines()
            if len(lines) > args.per_source:
                lines = rnd.sample(lines, args.per_source)
            f.write("\n".join(lines) + "\n")
            total += len(lines)
            print(f"{p.name}: using {len(lines)} lines")

    spm.SentencePieceTrainer.train(
        input=str(sample),
        model_prefix=str(out / "tokenizer"),
        model_type="unigram",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        treat_whitespace_as_suffix=True,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        user_defined_symbols=CONTROL,
        input_sentence_size=total,
        shuffle_input_sentence=True,
        num_threads=16,
    )

    sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
    ids = [sp.piece_to_id(f"<CHAR_{c}>") for c in string.ascii_uppercase]
    assert ids == list(range(ids[0], ids[0] + 26)), f"CHAR tokens not consecutive: {ids}"
    for t in ["<XBU>", "<XBC>", "<XEC>", "▁"]:
        assert sp.piece_to_id(t) != 0, f"missing {t}"
    print("vocab", sp.get_piece_size(), "CHAR_A id", ids[0], "XBU/XBC/XEC", [sp.piece_to_id(t) for t in CONTROL[:3]])
    for s in ["Das ist ein Test für Übergrößen.", "Donaudampfschifffahrtsgesellschaft", "<XBU><CHAR_H><CHAR_A><XBC>Haus ▁<XEC>"]:
        print(sp.encode(s, out_type=str))


if __name__ == "__main__":
    main()
