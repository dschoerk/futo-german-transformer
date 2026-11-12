#!/usr/bin/env python3
"""Validate an exported GGUF: FUTO metadata/tokenizer contract + logit parity with the PyTorch checkpoint.

Parity runs the GGUF through llama.cpp (llama-cpp-python) and the HF model through PyTorch on identical token ids
(tokenized by our SentencePiece model, as the app does), comparing next-token distributions.
"""
import argparse, string, sys
import numpy as np
import gguf, torch
import sentencepiece as spm
from transformers import LlamaForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("gguf")
ap.add_argument("--model-dir", default="models/ckpt/hf_latest")
args = ap.parse_args()
ok = True

def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)

r = gguf.GGUFReader(args.gguf)
f = {k: v for k, v in r.fields.items()}
val = lambda k: f[k].contents() if k in f else None
check(val("general.architecture") == "llama", "arch llama")
feats = (val("keyboardlm.features") or "").split()
check({"base_v1", "inverted_space", "xbu_char_autocorrect_v1"} <= set(feats), f"features {feats}")
check(val("keyboardlm.languages") == "de", f"languages {val('keyboardlm.languages')}")
check(val("keyboardlm.ext_tokenizer_type") == "sentencepiece", "ext tokenizer sentencepiece")
names = {t.name for t in r.tensors}
check({"token_embd.weight", "output.weight", "output_norm.weight"} <= names, "embedding/output tensors present")
sp = spm.SentencePieceProcessor(); sp.LoadFromSerializedProto(bytes(val("keyboardlm.ext_tokenizer_data")))
check(sp.encode("Das ist ein ", out_type=str)[-1].endswith("▁"), "embedded tokenizer keeps trailing space")
ids = [sp.piece_to_id(f"<CHAR_{c}>") for c in string.ascii_uppercase]
check(ids == list(range(ids[0], ids[0] + 26)) and ids[0] != sp.unk_id(), "CHAR_A..Z consecutive")
check(all(sp.piece_to_id(t) != sp.unk_id() for t in ["<XBU>", "<XBC>", "<XEC>", "▁"]), "XBU/XBC/XEC/space present")
check(sp.bos_id() == 1, "BOS id 1 (app hard-codes it)")

from llama_cpp import Llama
llm = Llama(model_path=args.gguf, n_ctx=256, logits_all=True, verbose=False, n_threads=4)
hf = LlamaForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32).eval()
texts = ["Ich habe heute keine Zeit, weil ich ", "Kannst du mir bitte ",
         "Das <XBU><CHAR_H><CHAR_A><CHAR_U><XBC>"]
agree, tot, kls = 0, 0, []
for t in texts:
    tok = [1] + sp.encode(t)
    llm.reset(); llm.eval(tok)
    a = np.array(llm.scores[: len(tok)], dtype=np.float64)
    with torch.no_grad():
        b = hf(input_ids=torch.tensor([tok])).logits[0].double().numpy()
    pa = np.exp(a - a.max(-1, keepdims=True)); pa /= pa.sum(-1, keepdims=True)
    pb = np.exp(b - b.max(-1, keepdims=True)); pb /= pb.sum(-1, keepdims=True)
    kls += list((pb * (np.log(pb + 1e-12) - np.log(pa + 1e-12))).sum(-1))
    agree += int((a.argmax(-1) == b.argmax(-1)).sum()); tot += len(tok)
    print(f"  {t!r}: gguf next={sp.id_to_piece(int(a[-1].argmax()))!r} torch next={sp.id_to_piece(int(b[-1].argmax()))!r}")
check(agree / tot >= 0.9, f"top-1 agreement {agree}/{tot}")
check(float(np.mean(kls)) < 0.05, f"mean KL(torch||gguf) {np.mean(kls):.4f}")
llm.close()
sys.exit(0 if ok else 1)
