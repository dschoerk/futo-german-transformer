# Progress log

Goal: a German (`de`) transformer LM for FUTO Keyboard (next-word, completion, autocorrect), as GGUF.
Related: futo-org/android-keyboard#2400 (Hungarian request; same need).

## Milestones
- [x] M0 Research: FUTO model format, Greek community precedent (`docs/FUTO_FORMAT.md`)
- [x] M1 Environment: uv-managed (`uv sync`, `uv run ...`), torch 2.14+cu130 sees the RTX 3090
- [ ] M2 Data: download + clean German corpora (`scripts/01_data.py`)
- [ ] M3 Tokenizer with CHAR/XBU/XBC/XEC tokens (`scripts/02_tokenizer.py`)
- [ ] M4 Training data with QWERTZ-aware corruptions (`scripts/03_build_dataset.py`)
- [ ] M5 Train ~40M Llama model on RTX 3090 (`scripts/04_train.py`)
- [ ] M6 GGUF export + validation (`scripts/05_export_gguf.py`)
- [ ] M7 Eval (top-k next word, autocorrect) and on-device test in FUTO
- [ ] M8 Publish model (HF) + comment on upstream issue

## Log
- 2026-11-12: Read issue #2400, Greek repo, FUTO source. Key finding: layout gate rejects QWERTZ unless
  dev toggle enabled; Greek model lacks real autocorrect protocol. Repo initialised.
- 2026-11-12: Switched to uv (pyproject.toml, uv.lock). Local git repo only; GitHub remote pending (`gh` not installed).
