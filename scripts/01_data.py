#!/usr/bin/env python3
"""Stream German text from HF datasets, split into sentences, clean, dedupe, write data/raw/<source>.txt.

Usage: uv run scripts/01_data.py --source wiki|fineweb|opus100 --max-lines N
One sentence per line. Cleaning is deliberately conservative: keep umlauts, ß, case and punctuation
(a keyboard model must learn capitalization and German orthography).
"""
import argparse, hashlib, re, sys
from pathlib import Path
from datasets import load_dataset

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"»(])")
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
WS_RE = re.compile(r"\s+")
# allowed chars: letters (incl. German), digits, common punctuation and quotes
BAD_CHARS = re.compile(r"[^\w\s.,;:!?'\"„“”‚‘’»«()\-–—/%€$&+*#@=<>…]")
GERMAN_LETTERS = set("abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ")
COMMON_DE = {"der", "die", "das", "und", "ist", "nicht", "ich", "ein", "eine", "zu", "mit", "den", "von", "auf",
             "für", "es", "sich", "dem", "wir", "sie", "auch", "im", "des", "als", "wie", "ja", "du", "was"}
COMMON_EN = {"the", "and", "is", "you", "that", "it", "of", "to", "in", "for", "with", "this", "are", "was"}


ABBREV = {"dr", "prof", "nr", "str", "ca", "bzw", "usw", "z", "b", "d", "h", "u", "a", "o", "vgl", "evtl", "inkl", "ggf",
          "geb", "gest", "hl", "hrsg", "jh", "mio", "mrd", "st", "tel", "dipl", "ing", "sog", "etc", "ebd", "engl", "frz", "lat"}


def split_sentences(para: str):
    parts, cur = [], ""
    for piece in SENT_SPLIT.split(para):
        cur = f"{cur} {piece}" if cur else piece
        last = cur.rstrip().rsplit(" ", 1)[-1].rstrip(".").lower()
        if cur.endswith(".") and (last in ABBREV or re.fullmatch(r"[A-ZÄÖÜ]", cur.rstrip(".")[-1:] or "") and len(last) == 1):
            continue
        parts.append(cur)
        cur = ""
    if cur:
        parts.append(cur)
    return parts


def balanced(s: str) -> bool:
    for a, b in ("()", "[]"):
        if s.count(a) != s.count(b):
            return False
    return "()" not in s and "( )" not in s and s.count("\u201e") == s.count("\u201c")


def clean_sentence(s: str, min_words=3, max_words=40, need_end=True):
    s = WS_RE.sub(" ", URL_RE.sub(" ", s)).strip()
    if not s:
        return None
    w = s.split(" ")
    if not (min_words <= len(w) <= max_words):
        return None
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 8 or sum(c in GERMAN_LETTERS for c in letters) / len(letters) < 0.97:
        return None
    if BAD_CHARS.search(s) or not balanced(s):
        return None
    if need_end and s[-1] not in '.!?"“«\'':
        return None
    if not s[0].isupper() and not s[0] in '„"»(\'' and not s[0].isdigit():
        return None
    if sum(c.isdigit() for c in s) / len(s) > 0.2:
        return None
    lw = {x.lower().strip(".,;:!?\"'()") for x in w}
    if len(lw & COMMON_EN) > len(lw & COMMON_DE):  # drop English leakage
        return None
    return s


def iter_texts(source: str):
    if source == "wiki":
        ds = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
        for ex in ds:
            yield ex["text"]
    elif source == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-2", "deu_Latn", split="train", streaming=True)
        for ex in ds:
            yield ex["text"]
    elif source == "opus100":
        ds = load_dataset("Helsinki-NLP/opus-100", "de-en", split="train", streaming=True)
        for ex in ds:
            yield ex["translation"]["de"]
    else:
        sys.exit(f"unknown source {source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["wiki", "fineweb", "opus100"])
    ap.add_argument("--max-lines", type=int, default=1_000_000)
    ap.add_argument("--out-dir", default="data/raw")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen, n, docs = set(), 0, 0
    with open(out / f"{args.source}.txt", "w", encoding="utf-8") as f:
        for text in iter_texts(args.source):
            docs += 1
            for para in text.split("\n"):
                para = para.strip()
                if not para:
                    continue
                # Wikipedia headings / list fragments: require sentence-final punctuation for paragraphs
                for sent in split_sentences(para):
                    c = clean_sentence(sent, need_end=args.source != "opus100")
                    if c is None:
                        continue
                    h = hashlib.blake2b(c.encode(), digest_size=8).digest()
                    if h in seen:
                        continue
                    seen.add(h)
                    f.write(c + "\n")
                    n += 1
            if docs % 20000 == 0:
                print(f"{args.source}: {docs} docs, {n} lines", flush=True)
            if n >= args.max_lines:
                break
    print(f"DONE {args.source}: {docs} docs, {n} lines -> {out / (args.source + '.txt')}")


if __name__ == "__main__":
    main()
