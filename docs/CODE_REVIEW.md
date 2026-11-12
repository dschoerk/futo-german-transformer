# Code review (2026-11-12) — findings and fixes

| # | Sev. | Finding | Status |
|---|---|---|---|
| 1 | Critical | Tokenizer trained with `remove_extra_whitespaces=True`: the trailing space of FUTO's `trim(ctx)+" "` was stripped, so the context ended mid-word at inference while training had `word▁` | Fixed: retrained with `False`; dataset encodes `line + " "`; assertion in 02/03 |
| 2 | High | `--opus-repeat` duplicated each line back-to-back → copy bias, autocorrect answers leaked from previous sentence | Fixed: repeats are separate shuffled passes |
| 3 | High | Train-time `<CHAR_*>` inputs were one-hot; app always feeds 4-key proximity mixes | Fixed: `embed_with_mixing` in 04_train.py |
| 4 | Medium | Umlaut typing ignored FUTO's "Deutsch" layout (ü/ö/ä keys arrive as neighbouring letters) | Fixed: added mode |
| 5 | Medium | 10 layers / 42M params vs English 8 layers / ~30M → slower on device | Fixed: 8 layers (~35M) |
| 6 | Medium | All sequences began at sentence starts; app sends BOS + clause / last words | Fixed: `cut_start`, seq_len 128 |
| 7 | Medium | Thermal guard killed jobs (lost work); `pgrep -f` matched its own and waiting shells → never exited | Fixed: pause/resume, python-only targets |
| 8 | Medium | Variable-shape boolean-mask loss fragmented CUDA memory (OOM warning) | Fixed: `ignore_index=-100` |
| 9 | High | (export) HF q/k weights need llama.cpp RoPE permutation; the Greek exporter skips it | Done in 05_export_gguf.py; to verify by logit parity |
| 11 | Critical | (export) tokenizer blob written as INT32 array; the app reads `gguf_get_arr_n` *bytes* → SentencePiece load would fail on device | Fixed: explicit UINT8; `check_gguf.py` emulates the app's read and checks all metadata types (old export FAILS, new PASSES); full key/type/tensor diff vs English model is identical |
| 10 | Low | val/test split per sentence leaks document neighbours; FineWeb sample is mostly the 2013 crawl; no emoji | Open |
