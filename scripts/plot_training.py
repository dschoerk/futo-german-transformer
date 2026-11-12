#!/usr/bin/env python3
"""Plot training progress from logs/train.jsonl -> results/training_progress.png

Three panels on a shared step axis (no dual axes): loss (train vs val), val token accuracy, learning rate
(with the remaining cosine schedule dashed, showing how far the run is).
"""
import argparse, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train.jsonl")
    ap.add_argument("--out", default="results/training_progress.png")
    ap.add_argument("--total", type=int, default=52960)
    ap.add_argument("--epoch-steps", type=int, default=17654)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    args = ap.parse_args()

    tr, va = [], []
    for line in Path(args.log).read_text().splitlines():
        r = json.loads(line)
        (va if "val_loss" in r else tr).append(r)
    ts, tl, tlr = [r["step"] for r in tr], [r["loss"] for r in tr], [r["lr"] for r in tr]
    vs, vl, vacc = [r["step"] for r in va], [r["val_loss"] for r in va], [100 * r["val_acc"] for r in va]
    now = ts[-1]

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2, 1.2], "hspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    for a in (a1, a2, a3):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for e in range(1, math.ceil(args.total / args.epoch_steps)):
            a.axvline(e * args.epoch_steps, color=INK2, linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.6)
        a.axvline(now, color=INK, linewidth=1.0, alpha=0.35)

    # 1) loss
    a1.plot(ts, tl, color=S1, linewidth=1.2, alpha=0.85, label="train loss (50-step mean)")
    a1.plot(vs, vl, color=S2, linewidth=2, marker="o", markersize=3.5, label="validation loss")
    lo = min(min(tl[10:] or tl), min(vl)) - 0.1
    a1.set_ylim(lo, 4.2)
    a1.set_ylabel("loss (nats/token)")
    a1.legend(loc="upper right", frameon=False)
    a1.annotate(f"val {vl[-1]:.2f}", (vs[-1], vl[-1]), xytext=(6, 8), textcoords="offset points", color=INK, fontsize=9)
    a1.text(0.06, 0.90, "first ~500 steps off-scale (loss 9.0 → 4.8)", transform=a1.transAxes, va="top",
            fontsize=8, color=INK2)
    for e in range(1, math.ceil(args.total / args.epoch_steps)):
        a1.text(e * args.epoch_steps, 4.2, f" epoch {e + 1}", va="top", fontsize=8, color=INK2)

    # 2) val accuracy
    a2.plot(vs, vacc, color=S1, linewidth=2, marker="o", markersize=3.5)
    a2.set_ylim(min(vacc[1:] or vacc) - 1, max(vacc) + 1.5)
    a2.set_ylabel("val token acc. (%)")
    a2.annotate(f"{vacc[-1]:.1f}%", (vs[-1], vacc[-1]), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=9)

    # 3) learning rate: done (solid) + remaining schedule (dashed)
    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.total - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))
    fut = list(range(now, args.total + 1, 200))
    a3.plot(ts, [x * 1e3 for x in tlr], color=S1, linewidth=2)
    a3.plot(fut, [lr_at(s) * 1e3 for s in fut], color=S1, linewidth=1.5, linestyle=(0, (4, 3)), alpha=0.6)
    a3.set_ylabel("LR (×1e-3)")
    a3.set_xlabel("training step")
    a3.set_xlim(0, args.total)
    a3.text(now, a3.get_ylim()[1] * 0.95, f"  now: step {now:,} ({100 * now / args.total:.0f}%)",
            va="top", fontsize=8.5, color=INK)

    fig.suptitle(f"German FUTO keyboard LM (34.4M params) — validation loss {vl[0]:.2f} → {vl[-1]:.2f}, "
                 f"token accuracy {vacc[0]:.1f}% → {vacc[-1]:.1f}%", x=0.07, ha="left", fontsize=12, color=INK)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out} (step {now})")


if __name__ == "__main__":
    main()
