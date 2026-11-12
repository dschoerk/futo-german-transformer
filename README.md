# FUTO Keyboard German transformer

Training pipeline for a German language model usable by [FUTO Keyboard](https://github.com/futo-org/android-keyboard).
See `PROGRESS.md` for status and `docs/FUTO_FORMAT.md` for the format contract.

Pipeline (run in order, from repo root, via `uv run`; set up with `uv sync`):
`scripts/01_data.py` → `02_tokenizer.py` → `03_build_dataset.py` → `04_train.py` → `05_export_gguf.py` → `check_gguf.py` / `06_eval.py`

Shards + training run detached: `setsid nohup scripts/run_all.sh > logs/run_all.log 2>&1 &` (thermally guarded, resumable).
