#!/usr/bin/env python3
"""Evaluate a checkpoint the way FUTO Keyboard queries it (held-out sentences from data/test.txt).

* next word:   BOS + tokenize(safeguard(ctx) + " ")            -> beam search to the first word-final piece
* autocorrect: BOS + tokenize(ctx + " ") + <XBU> + key-mixed CHAR embeddings of a noisy QWERTZ typing + <XBC>
               -> beam search until <XEC>   (typed form from 03_build_dataset.typed_form, incl. prefixes)
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


@torch.no_grad()
def beam(model, prefix, sp, end, beam_w=8, max_tok=8, k=3):
    """prefix: list of token ids. end(tok_id)->bool marks a finished word. Returns top-k (word, logprob)."""
    E = model.get_input_embeddings().weight
    dev = E.device
    base = train.embed_with_mixing(model, torch.tensor([prefix], device=dev), p_mix=1.0)  # mixes CHAR positions
    beams, done = [([], 0.0)], []
    for _ in range(max_tok):
        embs = torch.stack([torch.cat([base[0], E[torch.tensor(t, dtype=torch.long, device=dev)]]) if t else base[0]
                            for t, _ in beams])
        logp = torch.log_softmax(model(inputs_embeds=embs).logits[:, -1].float(), -1)
        cand = []
        for (t, s), lp in zip(beams, logp):
            v, i = lp.topk(beam_w)
            cand += [(t + [int(j)], s + float(x)) for x, j in zip(v, i)]
        cand.sort(key=lambda c: -c[1])
        beams = []
        for t, s in cand:
            (done if end(t[-1]) else beams).append((t, s))
            if len(beams) >= beam_w:
                break
        if len(done) >= beam_w or not beams:
            break
    out, seen = [], set()
    for t, s in sorted(done, key=lambda c: -c[1]):
        w = sp.decode([x for x in t if x != XEC]).strip()
        if w and w not in seen:
            seen.add(w); out.append((w, s))
        if len(out) == k:
            break
    return out


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
    word_end = lambda t: sp.id_to_piece(t).endswith("▁")
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
        res = [w for w, _ in beam(model, prefix, sp, word_end)]
        st = stats["next_word"]; st[0] += 1; st[1] += res[:1] == [target]; st[2] += target in res
        # autocorrect / completion from a noisy typing
        typed = build.typed_form(target, rnd)
        if not typed:
            continue
        ac_prefix = prefix + [XBU] + [CHAR_LO + ord(c) - 97 for c in typed] + [XBC]
        res2 = [w for w, _ in beam(model, ac_prefix, sp, lambda t: t == XEC, max_tok=10)]
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
