# Progress log

Goal: a German (`de`) transformer LM for FUTO Keyboard (next-word, completion, autocorrect), as GGUF.
Related: futo-org/android-keyboard#2400 (Hungarian request; same need).

## Milestones
- [x] M0 Research: FUTO model format (`docs/FUTO_FORMAT.md`), community models (`docs/PRIOR_ART.md`)
- [x] M1 Environment: uv-managed (`uv sync`, `uv run ...`), torch 2.14+cu130 sees the RTX 3090
- [x] M2 Data: 10.67M cleaned sentences (wiki 5.0M, fineweb-2 5.0M, opus-100 0.67M), `scripts/01_data.py`
- [x] M3 Tokenizer: 20k unigram, suffix whitespace, CHAR_A..Z consecutive from id 7 (`scripts/02_tokenizer.py`)
- [x] M4 Training data: 3 epoch shards × 4.52M seqs (128 tok), 457M real tokens each, ~50% sentences with autocorrect examples
- [ ] M5 Train 34.4M Llama (8L/512/FFN1280) on RTX 3090 — running since 11:13, 52,960 steps, ~107k tok/s, ETA ~4.5h
- [ ] M6 GGUF export + validation (`scripts/05_export_gguf.py`)
- [ ] M7 Eval (top-k next word, autocorrect) and on-device test in FUTO
- [ ] M8 Publish model (HF) + comment on upstream issue

## Log
- 2026-11-12: Read issue #2400, Greek repo, FUTO source. Key finding: layout gate rejects QWERTZ unless
  dev toggle enabled; Greek model lacks real autocorrect protocol. Repo initialised.
- 2026-11-12: Switched to uv (pyproject.toml, uv.lock). Local git repo only; GitHub remote pending (`gh` not installed).
- 2026-11-12: Data + tokenizer done. First tokenizer run was OOM-killed (VS Code froze); reran detached (setsid, nice, 4 threads) with `scripts/thermal_guard.sh`. Tokenizer/data steps are CPU-only; GPU first used in M5.
- 2026-11-12: Prior-art survey (`docs/PRIOR_ART.md`). Two German FUTO models already exist (jblechert, VerbalJungle).
  FUTO says the transformer code will be replaced (#2371, 2026-10-27). Found that stock FUTO garbles keystrokes after
  umlauts (UTF-8 byte vs keystroke index mismatch; `docs/FUTO_FORMAT.md`).
- 2026-11-12: Code review against FUTO's real English GGUF + app source (docs/CODE_REVIEW.md). Critical tokenizer whitespace bug fixed (tokenizer retrained), dataset/training aligned with app inference (short contexts, key-embedding mixing, umlaut layouts). `scripts/run_all.sh` builds shards + trains detached.
- 2026-11-12: Training started. Found + fixed critical export bug (tokenizer blob INT32 vs UINT8 expected by app).
- 2026-11-12 12:20: Interim checkpoint 10k/52,960 steps (val loss 3.29). App-faithful eval (06_eval.py ports the app's Sample/transform_logits; 420 held-out cases): autocorrect top-1/top-3 59.5%/75.0% (full words 67.0%/83.3%), next word 14.3%/25.0%. Interim GGUF de_keyboard_10k_q6_k.gguf (37.4 MB, all Q6_K) passes check_gguf incl. app-style tokenizer load + parity. GPU power-capped to 280 W by user: no more thermal pauses, 183 steps/min.
