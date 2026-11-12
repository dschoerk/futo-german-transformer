#!/usr/bin/env python3
"""Build packed, pre-tokenized training data with FUTO-format autocorrect examples.

Port of FUTO's TrainingDataGenerator.kt (xlm/) adapted to QWERTZ + German:
  * ~1/3 of eligible words per selected sentence become  `<XBU><CHAR_x>..<XBC>truth <XEC>`
  * typo model: transpositions, deletion, gaussian key-position noise on QWERTZ, prefix truncation
  * German: typed form maps umlauts/ß like a real user would; truth keeps case+umlauts. Umlaut modes:
      long-press on QWERTZ (ü->u), FUTO "Deutsch" layout key (app zeroes non a-z keys -> nearest letter, ü->p, ö/ä->l),
      digraph (ue), dropped (app's non-mixing fallback skips non a-z)
  * sequences start with BOS + often a mid-sentence cut, like the app's safeguardContext (clause / last words)
  * text is encoded as `line + " "` with a whitespace-preserving tokenizer, matching FUTO's `trim(ctx) + " "`
Output: data/tok/{train,val}_s<seed>.npy  uint16 [N, seq_len]; data/test.txt raw held-out sentences.
Targets <XBU>,<XBC>,<CHAR_*> are masked in the training loop (04_train.py), they are forced / noise at inference.
"""
import argparse, hashlib, math, random, re, string
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import sentencepiece as spm

# QWERTZ: FUTO's QWERTY coordinates with y<->z swapped
KEYS = {'q': (75, 106), 'w': (214, 106), 'e': (363, 106), 'r': (499, 106), 't': (645, 106), 'z': (789, 106),
        'u': (928, 106), 'i': (1073, 106), 'o': (1216, 106), 'p': (1357, 106),
        'a': (150, 312), 's': (291, 312), 'd': (434, 312), 'f': (574, 312), 'g': (717, 312), 'h': (859, 312),
        'j': (1005, 312), 'k': (1140, 312), 'l': (1288, 312),
        'SHIFT': (113, 515), 'y': (287, 515), 'x': (434, 515), 'c': (576, 515), 'v': (718, 515), 'b': (860, 515),
        'n': (1003, 515), 'm': (1145, 515), 'BACK': (1329, 515)}
KEY_LIST = list(KEYS.items())
TAP = 80.0
ELIGIBLE = re.compile(r"[A-Za-zÄÖÜäöüß'\-]+")
SP = None
UMLAUT_BASE = {"ä": "a", "ö": "o", "ü": "u"}
UMLAUT_DIGRAPH = {"ä": "ae", "ö": "oe", "ü": "ue"}
UMLAUT_KEYNBR = {"ä": "l", "ö": "l", "ü": "p"}  # FUTO "Deutsch" layout: q..p ü / a..l ö ä


def closest_key(x, y):
    return min(KEY_LIST, key=lambda kv: (kv[1][0] - x) ** 2 + (kv[1][1] - y) ** 2)[0]


def substitute_keys(word, temp, rnd):
    out = []
    for ch in word:
        if ch not in KEYS:
            continue
        x, y = KEYS[ch]
        k = closest_key(rnd.gauss(x, temp * TAP), rnd.gauss(y, temp * TAP))
        if k == 'SHIFT':
            pass
        elif k == 'BACK':
            if out:
                out.pop()
        else:
            out.append(k)
    return "".join(out)


def typed_form(word, rnd, correctness=0.8):
    """How a user might have typed `word` (lowercase a-z only)."""
    w = word.lower().replace("'", "")
    # German-specific input of umlauts / eszett
    r = rnd.random()
    out = []
    for ch in w:
        if ch in UMLAUT_BASE:
            out.append(UMLAUT_BASE[ch] if r < 0.35 else UMLAUT_KEYNBR[ch] if r < 0.6 else
                       UMLAUT_DIGRAPH[ch] if r < 0.75 else "")
        elif ch == "ß":
            out.append("s" if r < 0.5 else "ss" if r < 0.8 else "")
        else:
            out.append(ch)
    w = "".join(out)
    g = lambda: rnd.random() ** correctness
    if len(w) >= 2 and g() > 0.5:
        i, j = rnd.sample(range(len(w)), 2)
        l = list(w); l[i], l[j] = l[j], l[i]; w = "".join(l)
    if len(w) >= 2 and g() > 0.5:
        i = rnd.randrange(len(w) - 1)
        l = list(w); l[i], l[i + 1] = l[i + 1], l[i]; w = "".join(l)
    if w and g() > 0.5:
        i = rnd.randrange(len(w)); w = w[:i] + w[i + 1:]
    w = substitute_keys(w, 1.0 * g(), rnd)
    if rnd.random() > 0.33 and len(w) >= 2:  # unfinished word -> teaches completion
        n = max(2, math.ceil((1.0 - g() * g()) * len(w)))
        w = w[:n]
    return "".join(c for c in w if c in string.ascii_lowercase)


def corrupt_sentence(s, rnd, proportion=0.333):
    words = s.split(" ")
    idx = [i for i, w in enumerate(words) if ELIGIBLE.fullmatch(w)]
    rnd.shuffle(idx)
    for i in idx[: int(len(words) * proportion)]:
        t = typed_form(words[i], rnd)
        if t:
            words[i] = f"<XBU>{''.join(f'<CHAR_{c.upper()}>' for c in t)}<XBC>{words[i]} <XEC>"
    return " ".join(words).replace("<XBC> ", "<XBC>").replace("<XEC> ", "<XEC>")


def init_worker(tok_path):
    global SP
    SP = spm.SentencePieceProcessor(model_file=tok_path)
    assert SP.encode("ein ", out_type=str)[-1].endswith("\u2581"), "tokenizer strips trailing space; retrain 02"


def cut_start(line, rnd):
    """Mimic LanguageModel.kt safeguardContext: context often starts after a comma or is just the last few words."""
    words = line.split(" ")
    if len(words) < 4:
        return line
    commas = [i for i, w in enumerate(words[:-2]) if w.endswith(",")]
    if commas and rnd.random() < 0.5:
        return " ".join(words[rnd.choice(commas) + 1:])
    return " ".join(words[rnd.randint(1, len(words) - 3):])


def process(args):
    lines, seed, p_auto, p_cut, seq_len = args
    rnd = random.Random(seed)
    bos, pad = 1, 3
    seqs, cur, n_corrupt = [], [bos], 0
    for line in lines:
        if len(cur) == 1 and rnd.random() < p_cut:  # first line of a sequence: start mid-sentence
            line = cut_start(line, rnd)
        if rnd.random() < p_auto:
            line = corrupt_sentence(line, rnd); n_corrupt += 1
        ids = SP.encode(line + " ")
        if len(ids) > seq_len - 1:
            continue
        if len(cur) + len(ids) > seq_len:
            seqs.append(cur + [pad] * (seq_len - len(cur)))
            cur = [bos]
        cur += ids
    if len(cur) > 1:
        seqs.append(cur + [pad] * (seq_len - len(cur)))
    return np.asarray(seqs, dtype=np.uint16), n_corrupt, len(lines)


def split_of(line):
    h = int.from_bytes(hashlib.blake2b(line.encode(), digest_size=4).digest(), "little") % 1000
    return "val" if h == 0 else "test" if h == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--tok", default="models/tokenizer/tokenizer.model")
    ap.add_argument("--out-dir", default="data/tok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=128, help="app context is <=70 chars + partial word (<=40 chars)")
    ap.add_argument("--p-cut", type=float, default=0.6, help="prob. a sequence starts mid-sentence")
    ap.add_argument("--p-auto", type=float, default=0.5, help="fraction of sentences that get autocorrect examples")
    ap.add_argument("--opus-repeat", type=int, default=2, help="upweight conversational source")
    ap.add_argument("--max-per-source", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data = {"train": [], "val": [], "test": []}
    for p in sorted(Path(args.raw_dir).glob("*.txt")):
        rep = args.opus_repeat if p.stem == "opus100" else 1
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                sp = split_of(line)
                data[sp].append((line, rep if sp == "train" else 1))
                n += 1
                if args.max_per_source and n >= args.max_per_source:
                    break
        print(f"{p.name}: {n} lines")
    Path("data/test.txt").write_text("\n".join(l for l, _ in data["test"]) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in data.items()})

    with Pool(args.workers, initializer=init_worker, initargs=(args.tok,)) as pool:
        for split in ("val", "train"):
            rnd = random.Random(args.seed)
            # keep document order (consecutive sentences) inside chunks; shuffle chunk order only.
            # Upweighted sources are repeated as separate passes (never back-to-back duplicates).
            max_rep = max((r for _, r in data[split]), default=1)
            chunks = []
            for k in range(max_rep):
                lines = [l for l, r in data[split] if r > k]
                chunks += [lines[i:i + args.chunk] for i in range(0, len(lines), args.chunk)]
            rnd.shuffle(chunks)
            jobs = [(c, args.seed * 1_000_003 + i, args.p_auto, args.p_cut, args.seq_len) for i, c in enumerate(chunks)]
            arrs, nc, nl = [], 0, 0
            for k, (a, c, l) in enumerate(pool.imap(process, jobs)):
                arrs.append(a); nc += c; nl += l
                if k % 20 == 0:
                    print(f"{split}: chunk {k}/{len(jobs)}", flush=True)
            arr = np.concatenate(arrs)
            # shuffle sequences so batches mix sources
            arr = arr[np.random.default_rng(args.seed).permutation(len(arr))]
            np.save(out / f"{split}_s{args.seed}.npy", arr)
            real = int((arr != 3).sum())
            print(f"{split}: {arr.shape[0]} seqs, {real/1e6:.1f}M real tokens, {nc}/{nl} sentences with autocorrect")

    sp = spm.SentencePieceProcessor(model_file=args.tok)
    ex = np.load(out / f"val_s{args.seed}.npy")
    for row in ex[:3]:
        print(repr(sp.decode([int(t) for t in row if t != 3])[:300]))


if __name__ == "__main__":
    main()
