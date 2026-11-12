# Progress log

Goal: a German (`de`) transformer LM for FUTO Keyboard (next-word, completion, autocorrect), as GGUF.
Related: futo-org/android-keyboard#2400 (Hungarian request; same need).

## Milestones
- [x] M0 Research: FUTO model format (`docs/FUTO_FORMAT.md`), community models (`docs/PRIOR_ART.md`)
- [x] M1 Environment: uv-managed (`uv sync`, `uv run ...`), torch 2.14+cu130 sees the RTX 3090
- [x] M2 Data: 10.67M cleaned sentences (wiki 5.0M, fineweb-2 5.0M, opus-100 0.67M), `scripts/01_data.py`
- [x] M3 Tokenizer: 20k unigram, suffix whitespace, CHAR_A..Z consecutive from id 7 (`scripts/02_tokenizer.py`)
- [ ] M4 Training data with QWERTZ-aware corruptions (`scripts/03_build_dataset.py`) — building (3 epoch shards)
- [ ] M5 Train ~35M Llama model (8 layers, like FUTO's English model) on RTX 3090 (`scripts/04_train.py`, ~100k tok/s)
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
