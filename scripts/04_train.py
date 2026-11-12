#!/usr/bin/env python3
"""Train the German keyboard LM (Llama architecture, ~42M params) on packed shards from 03_build_dataset.py.

* epoch e uses data/tok/train_s{e}.npy (a fresh typo/shuffle sample per epoch)
* loss masked on targets <XBU>,<XBC>,<CHAR_*> and padding (forced / noise at inference time)
* <CHAR_*> inputs are fed as proximity-weighted mixes of up to 4 key embeddings, like FUTO's
  DecodePromptAndMixes (the app never feeds one-hot letters when tap coordinates exist)
* resumable: checkpoints to models/ckpt/last.pt every --save-every steps
* thermal throttle: pauses when GPU/CPU too hot
"""
import argparse, json, math, os, subprocess, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

PAD, XBU, XBC, XEC = 3, 4, 5, 6
CHAR_LO, CHAR_HI = 7, 32  # <CHAR_A>..<CHAR_Z>
# QWERTZ key centres (FUTO's eyeballed coordinates, y<->z swapped) in units of one key pitch (~143px)
_KEYS = {'q': (75, 106), 'w': (214, 106), 'e': (363, 106), 'r': (499, 106), 't': (645, 106), 'z': (789, 106),
         'u': (928, 106), 'i': (1073, 106), 'o': (1216, 106), 'p': (1357, 106), 'a': (150, 312), 's': (291, 312),
         'd': (434, 312), 'f': (574, 312), 'g': (717, 312), 'h': (859, 312), 'j': (1005, 312), 'k': (1140, 312),
         'l': (1288, 312), 'y': (287, 515), 'x': (434, 515), 'c': (576, 515), 'v': (718, 515), 'b': (860, 515),
         'n': (1003, 515), 'm': (1145, 515)}
KEYXY = torch.tensor([_KEYS[c] for c in "abcdefghijklmnopqrstuvwxyz"], dtype=torch.float32) / 143.0


def embed_with_mixing(model, x, p_mix=0.85, sigma=0.3, tau=0.45):
    """Token embeddings, with <CHAR_*> positions replaced by FUTO-style key mixes around a simulated tap."""
    E = model.get_input_embeddings().weight
    emb = E[x]
    is_char = (x >= CHAR_LO) & (x <= CHAR_HI)
    n = int(is_char.sum())
    if n == 0 or p_mix <= 0:
        return emb
    keyxy = KEYXY.to(x.device)
    letters = x[is_char] - CHAR_LO
    touch = keyxy[letters] + torch.randn(n, 2, device=x.device) * sigma
    w = torch.softmax(-((touch[:, None, :] - keyxy[None]) ** 2).sum(-1) / (2 * tau ** 2), -1)
    w = torch.where(w < 0.05, torch.zeros_like(w), w)              # app: proportions < 0.05 -> 0
    top_w, top_i = w.topk(4, -1)
    top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-6)
    onehot = torch.rand(n, device=x.device) >= p_mix                # app fallback without coordinates: one-hot
    top_i[onehot, 0] = letters[onehot]
    top_w[onehot] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x.device)
    mixed = (top_w[..., None] * E[top_i + CHAR_LO]).sum(1)
    emb = emb.clone()
    emb[is_char] = mixed.to(emb.dtype)
    return emb


def hot(max_gpu, max_cpu):
    try:
        g = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]).split()[0])
        s = subprocess.run(["sensors"], capture_output=True, text=True).stdout
        c = next((int(float(l.split()[1].strip("+°C"))) for l in s.splitlines() if "Tctl" in l), 0)
        return g >= max_gpu or c >= max_cpu, g, c
    except Exception:
        return False, -1, -1


def masked_targets(x):
    """Next-token targets with masked positions set to -100 (fixed shapes -> no allocator fragmentation)."""
    t = x[:, 1:].clone()
    m = (t == PAD) | (t == XBU) | (t == XBC) | ((t >= CHAR_LO) & (t <= CHAR_HI))
    t[m] = -100
    return t


def evaluate(model, val, bs, device, max_batches=100):
    model.eval(); tot, cnt, correct = 0.0, 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, min(len(val), bs * max_batches), bs):
            x = torch.from_numpy(val[i:i + bs].astype(np.int64)).to(device)
            logits = model(inputs_embeds=embed_with_mixing(model, x)).logits[:, :-1]
            t = masked_targets(x); m = t != -100
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), t.reshape(-1), ignore_index=-100, reduction="sum")
            tot += loss.item(); cnt += m.sum().item(); correct += ((logits.argmax(-1) == t) & m).sum().item()
    model.train()
    return tot / cnt, correct / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/tok")
    ap.add_argument("--out-dir", default="models/ckpt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=8, help="English FUTO model: 8 layers, 512 hidden")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--inter", type=int, default=1280, help="multiple of 256 so every tensor can be Q6_K")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--p-mix", type=float, default=0.85, help="fraction of CHAR inputs fed as key-embedding mixes")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-gpu-temp", type=int, default=83)
    ap.add_argument("--max-cpu-temp", type=int, default=85)
    args = ap.parse_args()

    device = "cuda"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = LlamaConfig(vocab_size=args.vocab, hidden_size=args.hidden, intermediate_size=args.inter,
                      num_hidden_layers=args.layers, num_attention_heads=args.heads, num_key_value_heads=args.heads,
                      max_position_embeddings=args.seq_len, rms_norm_eps=1e-5, rope_theta=10000.0,
                      tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=3,
                      attn_implementation="sdpa")
    model = LlamaForCausalLM(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)

    shards = [np.load(f"{args.data_dir}/train_s{e}.npy", mmap_mode="r") for e in range(args.epochs)]
    val = np.load(f"{args.data_dir}/val_s0.npy")
    eff = args.batch * args.accum
    steps_per = [len(s) // eff for s in shards]
    total = sum(steps_per)
    print(f"steps per epoch {steps_per}, total {total}, tokens/step {eff*args.seq_len}", flush=True)

    step = 0
    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
        print(f"resumed from step {step}", flush=True)

    fwd = torch.compile(model) if args.compile else model

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, total - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    log = open("logs/train.jsonl", "a")
    model.train(); t0 = time.time(); tok_seen = 0; run_loss = 0.0; run_n = 0
    while step < total:
        ep, s_in = 0, step
        while s_in >= steps_per[ep]:
            s_in -= steps_per[ep]; ep += 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        base = s_in * eff
        for a in range(args.accum):
            x = torch.from_numpy(shards[ep][base + a * args.batch: base + (a + 1) * args.batch].astype(np.int64)).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(inputs_embeds=embed_with_mixing(model, x, p_mix=args.p_mix)).logits[:, :-1]
            t = masked_targets(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), t.reshape(-1), ignore_index=-100) / args.accum
            loss.backward()
            run_loss += loss.item() * args.accum; run_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1; tok_seen += eff * args.seq_len

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step}/{total} ep {ep} loss {run_loss/run_n:.4f} lr {lr_at(step):.2e} {tok_seen/dt/1e3:.0f}k tok/s", flush=True)
            log.write(json.dumps({"step": step, "loss": run_loss / run_n, "lr": lr_at(step)}) + "\n"); log.flush()
            run_loss, run_n = 0.0, 0
            h, g, c = hot(args.max_gpu_temp, args.max_cpu_temp)
            while h:
                print(f"THROTTLE: gpu={g}C cpu={c}C, sleeping 30s", flush=True); time.sleep(30)
                h, g, c = hot(args.max_gpu_temp - 5, args.max_cpu_temp - 5)
        if step % args.eval_every == 0:
            vl, va = evaluate(model, val, args.batch, device)
            print(f"== step {step} val_loss {vl:.4f} ppl {math.exp(vl):.1f} acc {va:.4f}", flush=True)
            log.write(json.dumps({"step": step, "val_loss": vl, "val_acc": va}) + "\n"); log.flush()
        if step % args.save_every == 0 or step == total or (args.max_steps and step >= args.max_steps):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
            model.save_pretrained(out / "hf_latest")
        if args.max_steps and step >= args.max_steps:
            break
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
