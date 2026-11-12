# v2 plan: chat-focused German model

v1, the current run, is trained mostly on Wikipedia and web prose. What people type on a phone is mostly **chat messages**: short, often lowercase, often without final punctuation, written with "du" and full of questions, abbreviations and emoji. v2 targets that.

Items are ordered by expected impact relative to effort.

## 1. Chat-domain data (highest impact)
| Source | Status | Use |
|---|---|---|
| OpenSubtitles v2024 German (OPUS mono, ≈150M lines) | downloaded: `data/download/opensubtitles_v2024_de.txt.gz` | Main proxy for chat (dialogue). Clean out speaker tags `(Off)` / `(Löwe)`, sound cues `* … *`, broken hyphenation `unter- liegt`, `- ` dialogue dashes. Deduplicate (subtitle lines repeat a lot). |
| Tatoeba German (786k, CC BY 2.0 FR) | downloaded: `data/download/tatoeba_deu_sentences.tsv.bz2` | Short everyday sentences; upweight. Needs attribution in the model card. |
| German Reddit (GeRedE / Pushshift extraction) | not started | The only large source of genuinely *typed* informal text. Check the licence and Reddit's terms first. |
| Real chat corpora (MoCoDa 2, WhatsApp corpora) | research licences | Probably unusable for a public model. Ask the owners if needed. |
| Machine-translated chat sets (OASST-DE, ultra-chat_de) | rejected | Translated assistant dialogue: the wrong register. |

Proposed mix: about 50% subtitles, 10% Tatoeba (upweighted), 40% v1 data as replay.

## 2. Chat-style preprocessing and augmentation
- Separate cleaning profile for chat sources: allow lowercase starts, no final punctuation, and 1–2 word messages ("ok", "bin gleich da").
- Augmentation on dialogue lines:
  - lowercase the first letter (~50%), and sometimes the nouns too, because people type lowercase and the model must still offer capitalized corrections;
  - drop the final period (~70%);
  - sometimes swap in common chat abbreviations (vllt, lg, hdl, gn8, bzw, iwie, ne/nee, glei) and colloquial forms (hab/habe, is/ist, nich/nicht).
- Packing: treat each subtitle "scene" (a block of consecutive lines) like a chat thread, so context comes from the previous lines.

## 3. Tokenizer v2 (forces a full retrain, so do it together with 1–2)
- **Emoji:** add about 300 emoji as user-defined symbols, as FUTO's English tokenizer does. Today emoji fall back to byte tokens like `<0xF0>`, which the app bans, so they can never be suggested. Training data needs emoji, which subtitles lack: options are Reddit, or a keyword→emoji insertion heuristic as a stopgap.
- **Contractions:** add German contractions as whole pieces (`geht's▁`, `gibt's▁`, `hab's▁`, `wie's▁`, `ist's▁`), like English `I'm▁`.
- Re-check vocabulary size (16k vs 20k): smaller makes the output softmax cheaper on the phone.

## 4. Evaluation that reflects real use
- Chat-style held-out set: subtitle and Tatoeba lines, not seen in training, run through the same chat augmentation.
- **Keystroke savings**: the share of keystrokes saved when the user always taps a correct suggestion. This is more meaningful than top-k accuracy alone.
- Capitalization case (`haus` → `Haus`) and umlaut restoration as separate metrics.
- Split train/test by document, not by sentence (v1 leaks neighbouring sentences into test).

## 5. Data hygiene (v1 leftovers)
- FineWeb-2: sample across crawl dumps. v1 took the first shard, which is mostly the 2013 crawl.
- Deduplicate across sources and filter more boilerplate.

## 6. Device validation
- Test import and suggestions on a real phone with both German layouts, QWERTZ and "Deutsch".
- Benchmark latency on ARM (llama.cpp on device or via adb) and compare with English ml4 (8 layers, FFN 1024). Shrink if needed.
- Check what the app actually feeds for umlaut-key taps (`decomposeTapPosition`) and calibrate `typed_form` against it.

## 7. Upstream and community
- Open a German-model issue (or comment on futo-org/android-keyboard#2400, the Hungarian request) with results and the release link.
- PR: per-model layout list (`keyboardlm.layouts`) instead of the hard-coded QWERTY gate (`docs/upstream_patch.md`).
- Ask FUTO whether community models can go into the in-app model catalogue.

## 8. Optional, private: personal fine-tune
On-device fine-tuning is currently disabled in the app (the setting is commented out in `PredictiveText.kt`). A user can export their own chats (e.g. WhatsApp → Export chat, .txt) and fine-tune a copy on their PC. Such a model must **never be published**, because it can reproduce private messages. Script idea: `scripts/personal_finetune.py`, which strips sender and timestamps and trains briefly at a low learning rate with replay.

## Deferred / probably not worth it
- Swipe typing (`xc0_swipe_typing_v1`): FUTO now ships separate swipe models, and training it needs swipe trajectories.
- Longer contexts: the app trims context to ≤70 characters anyway.
