#!/usr/bin/env python3
"""Evaluate a checkpoint the way FUTO Keyboard queries it (held-out sentences from data/test.txt).

* next word:   BOS + tokenize(safeguard(ctx) + " ")  -> app sampler (3-wide beam, banned tokens) to a word end
* autocorrect: BOS + tokenize(ctx + " ") + <XBU> + key-mixed CHAR embeddings of a noisy QWERTZ typing + <XBC>
               -> app sampler until <XEC>   (typed form from 03_build_dataset.typed_form, incl. prefixes)
Reports top-1 / top-3 accuracy (case-sensitive, as shown to the user) and the share where the typed letters
already equal the target (a trivial-copy baseline).
"""
import argparse, importlib.util, json, random, sys
from pathlib import Path
import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM

ROOT = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


build = load("build", "03_build_dataset.py")
train = load("train", "04_train.py")
XBU, XBC, XEC, CHAR_LO = 4, 5, 6, 7


def safeguard(ctx):
    """LanguageModel.kt safeguardContext (without the cross-call matching)."""
    need = lambda c: len(c) > 70 or c.count(" ") > 16
    if need(ctx):
        v = max(ctx.rfind(ch) for ch in ".?!")
        if v != -1:
            ctx = ctx[v + 1:].strip()
    while need(ctx) and "," in ctx:
        ctx = ctx[ctx.index(",") + 1:].strip()
    if need(ctx) and " " in ctx:
        ctx = " ".join(ctx.split(" ")[-5:])
    return "" if len(ctx) > 144 else ctx


BLACKLIST = set(".!@#$%^&*()_=?/,\\][{};:\"><+`~|\r\n\t\x0b\x0c")


class AppSampler:
    """Port of the app's Sample()/transform_logits() (org_futo_inputmethod_latin_xlm_LanguageModel.cpp):
    3-wide beam over <=10 tokens, probabilities multiplied, punctuation tokens banned with their mass moved to the
    space token, space and '/- banned as first token, stop at a word-final piece or (correction mode) <XEC>."""

    def __init__(self, model, sp, n_results=3):
        self.model, self.sp, self.n = model, sp, n_results
        V = model.get_input_embeddings().weight.shape[0]
        pieces = [sp.id_to_piece(i) for i in range(V)]
        self.space = sp.piece_to_id("\u2581")
        self.dash = sp.piece_to_id("-")
        self.sep = torch.tensor([i for i, p in enumerate(pieces) if any(c in BLACKLIST for c in p)])
        self.start_ban = torch.tensor([i for i, p in enumerate(pieces) if p[:1] in ("'", "-")] + [self.space])
        self.general_ban = [sp.piece_to_id("-\u2581")]
        self.word_end = [p.endswith("\u2581") for p in pieces]

    def transform(self, logits, first, correction, prev):
        p = torch.softmax(logits.float(), -1)
        sep = self.sep.to(p.device)
        if correction:
            sep = sep[sep != XEC]
        p[self.space] += p[sep].clamp_min(0).sum()
        p[sep] = -999.0
        if first:
            p[self.start_ban.to(p.device)] = -999.0
        for i in self.general_ban:
            p[i] = -999.0
        if prev == self.dash:
            p[self.dash] = -999.0
        return p

    @torch.no_grad()
    def __call__(self, prefix, correction):
        m, dev = self.model, self.model.get_input_embeddings().weight.device
        E = m.get_input_embeddings().weight
        base = train.embed_with_mixing(m, torch.tensor([prefix], device=dev), p_mix=1.0)[0]
        logits = m(inputs_embeds=base[None]).logits[0, -1]
        p = self.transform(logits, True, correction, 0)
        v, i = p.topk(self.n)
        seqs = [(float(a), [int(b)]) for a, b in zip(v, i)]
        outs = []
        for _ in range(10):
            live = []
            for sc, t in seqs:
                if t[-1] == XEC:
                    outs.append((sc, t[:-1]))
                elif self.word_end[t[-1]]:
                    outs.append((sc, t))
                else:
                    live.append((sc, t))
            rem = self.n - len(outs)
            if rem <= 0 or not live:
                break
            embs = torch.stack([torch.cat([base, E[torch.tensor(t, device=dev)]]) for _, t in live])
            lg = m(inputs_embeds=embs).logits[:, -1]
            cand = []
            for (sc, t), l in zip(live, lg):
                pp = self.transform(l, False, correction, t[-1])
                v, i = pp.topk(rem)
                cand += [(float(a) * sc, t + [int(b)]) for a, b in zip(v, i)]
            cand.sort(key=lambda c: -c[0])
            seqs = cand[:rem]
        res, seen = [], set()
        for sc, t in sorted(outs, key=lambda c: -c[0]):
            w = self.sp.decode(t).strip()
            if w and w not in seen:
                seen.add(w); res.append(w)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/ckpt/hf_latest")
    ap.add_argument("--tokenizer", default="models/tokenizer/tokenizer.model")
    ap.add_argument("--test", default="data/test.txt")
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = LlamaForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32).to(args.device).eval()
    rnd = random.Random(args.seed); torch.manual_seed(args.seed)
    lines = [l for l in Path(args.test).read_text(encoding="utf-8").splitlines() if len(l.split()) >= 4]
    rnd.shuffle(lines)
    sampler = AppSampler(model, sp)
    stats = {k: [0, 0, 0] for k in ("next_word", "autocorrect", "autocorrect_full_word")}
    copy_ok, examples = 0, []
    for line in lines[: args.n]:
        words = line.split(" ")
        i = rnd.randrange(1, len(words))
        target = words[i]
        if not build.ELIGIBLE.fullmatch(target):
            continue
        ctx = safeguard(" ".join(words[:i]))
        prefix = [1] + sp.encode(ctx + " ") if ctx else [1]
        # next word
        res = sampler(prefix, correction=False)
        st = stats["next_word"]; st[0] += 1; st[1] += res[:1] == [target]; st[2] += target in res
        # autocorrect / completion from a noisy typing
        typed = build.typed_form(target, rnd)
        if not typed:
            continue
        ac_prefix = prefix + [XBU] + [CHAR_LO + ord(c) - 97 for c in typed] + [XBC]
        res2 = sampler(ac_prefix, correction=True)
        for key in ("autocorrect",) + (("autocorrect_full_word",) if len(typed) >= len(target) - 1 else ()):
            st = stats[key]; st[0] += 1; st[1] += res2[:1] == [target]; st[2] += target in res2
        copy_ok += typed == target.lower()
        if len(examples) < 12:
            examples.append({"ctx": ctx, "target": target, "next": res, "typed": typed, "corrected": res2})
    report = {k: {"n": v[0], "top1": round(v[1] / max(1, v[0]), 4), "top3": round(v[2] / max(1, v[0]), 4)}
              for k, v in stats.items()}
    report["typed_equals_target_share"] = round(copy_ok / max(1, stats["autocorrect"][0]), 4)
    print(json.dumps(report, indent=1))
    for e in examples:
        print(f"  [{e['ctx'][-40:]!s:>40}] → {e['target']!r:16} next={e['next']}  typed={e['typed']!r} → {e['corrected']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"report": report, "examples": examples}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
