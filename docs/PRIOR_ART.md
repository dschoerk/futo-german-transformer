# Prior art: community FUTO Keyboard transformer models

Survey date: 2026-11-12. The `keyboardlm.features`, shape and vocab columns come from each GGUF header, read over
HTTP (metadata only, no weights downloaded). Everything else comes from READMEs, release pages and FUTO issues.

Feature abbreviations: **base** = `base_v1`, **inv** = `inverted_space`, **xbu** = `xbu_char_autocorrect_v1`,
**xc0** = `xc0_swipe_typing_v1`, **cem** = `char_embed_mixing_v1`, **lora** = `lora_finetunable_v1`.

## Upstream status
- The only official model is `English v1` (`ml4_q6_k.gguf`): 8×512, ctx 2048, 15,008 tokens,
  features base inv xbu xc0 cem. Reference copy: `breadlicker45/futo-keyboard-lm` (`ml4_1_f16_meta_fixed.gguf`).
- **2026-10-27, FUTO dev abb128 on #2371:** the prediction system is being reworked, "most of the transformer code
  is expected to be removed and replaced", and custom models "will most likely stop working in the future". The
  hope is official support for other languages in the new system. Same warning on #1212 (2026-01-14).
- #2371 (Unicode keystrokes in `LanguageModel.cpp`) was closed as *not planned*. See `FUTO_FORMAT.md`,
  "Non-ASCII keystrokes".
- Central discussion: futo-org/android-keyboard#1212 "Transformer Models for More Languages".

## Models, grouped by declared features

### Full autocorrect protocol (base + inv + xbu)
| Lang | Project | Shape, vocab | Features | Notes |
|---|---|---|---|---|
| pt-BR | GitHub `danmaxis/futo-portuguese` v8.2 (2026-05) | 8×512, 15,008 | base inv xbu xc0 cem | Copies the English layout exactly. `GUIDE.md` is the best write-up of the format. |
| pt-BR | HF `barddoo/futo-keyboard-ptbr` (2026-09) | 8×512, 16,000 | base inv xbu xc0 cem | |
| eu | GitHub `itzune/futo-basque` v2.0.0 (2026-07) | 8×512, 4,096 | base inv xbu cem | Latxa corpus, 3B tokens. 4k unigram for agglutinative morphology. Has `RESEARCH.md`. |
| el, sq | HF `mandododo/greek-typing-lm-futo`, `albanian-typing-lm-futo` (2026-09) | 8×512, 16,000 | base inv xbu cem lora | Needs the polyglot APK (below). |
| de | GitHub `VerbalJungle/FUTO-Transformer-DE` v0.9.1 (2026-10) | 8×512, 33M params | base inv xbu xc0 **`char_embedding_mixing_v1`** | The last flag is misspelled, so the app probably ignores it (untested). |

### xbu + cem only, no base / inv (jblechert pipeline and its descendants)
| Lang | Project | Shape, vocab | Notes |
|---|---|---|---|
| de | GitHub `jblechert/keyboard-lm-de` v0.5 / v0.5.1 (2026-06) | 10×512 / 12×768 (~136M), 14,979 | See below. |
| pl | GitHub `areksh/keyboard-lm-pl` v0.1.0 (2026-07), Codeberg upstream | 10×512, 14,979 | Inspired by jblechert. User report: next-word good, autocorrect to diacritic forms poor. |
| da+en | HF `MrBob1337/keyboard-lm-daen` (2026-09) | 12×512, 15,008 | Ships jblechert's MIT license for its scripts. Bilingual `languages = "da en"`. |
| ru | HF `mrfukk7/ruto-keyboard-ru-65m` (2026-09) | 12×512, 24,000 | Ships its own `main_ru.dict`. |
| ru | HF `ushastoek/russian-keyboard-learning` (2026-06) | 10×512, 14,979 | Same vocab size as jblechert, probably the same pipeline. |

Not checked: what the app does when `base_v1` / `inverted_space` are missing.

### Next-word / completion only (base + inv, no xbu)
| Lang | Project | Shape, vocab | Notes |
|---|---|---|---|
| el | HF `stamchry/greek-keyboard-lm` (2026-08) | 9×512, 15,008 | README claims autocorrect; the GGUF doesn't declare it. Needs the `stamchry/android-keyboard` fork. |
| cs, sk | HF `Padakovec/futolm-cs-base-gguf` (2026-10) | 8×512, 8,192 | base / chat / synth variants for each language, ~1B tokens seen. |

### Unverified or not released
- sr: GitHub `markomi1/FUTO-Keyboard-Serbian-Model`. The models ship as zips, so their headers weren't read. The README
  follows the xbu format.
- fi: GitHub `juhovan/futo-lm-fi`. Port of the futo-portuguese pipeline; no release as of 2026-11-12.
- en: HF `trevon/futo-english-tap`. An experimental English retrain; the GGUF says "not phone tested".

### Won't load in FUTO (no `keyboardlm.*` metadata, so the app shows "unsupported")
- HF `sisiXenos/keyboard-lm-pl`: gpt2 architecture, byte-level (259 tokens).
- HF `jan3com/echos-keyboard-lm`: gptneox, 30M params.

## German in detail

### jblechert/keyboard-lm-de
- Releases: v0.4 (2026-05-29), v0.5 (06-03, 10×512, Q8 58 MB), v0.5.1-pre (06-09, 12×768, ~136M, Q8 139 MB).
- Tokenizer: SentencePiece **BPE**, `treat_whitespace_as_suffix`, 14,979 tokens, CHAR_A..Z autocorrect tokens.
- Data: Tatoeba, mC4-de, FineWeb2-HQ-de, synthetic (Qwen via Ollama), podcast transcripts (partly CC BY-NC-SA).
- License: code MIT, **weights CC BY-NC 4.0**.
- Recommends QWERTZ with umlauts on long-press only, not as main-layer keys.
- Extras: LoRA / distillation scripts, an Android "keyboard-collector" app to gather misrecognitions, and planned
  device-tiered model sizes.
- Feedback in #1212: "the solution from jblechert works really good" (2026-06-24).

### VerbalJungle/FUTO-Transformer-DE
- ~33M params, 15k SentencePiece with suffix whitespace, trained on AMD ROCm.
- Releases: v0.9 (2026-08-24), v0.9.1 (10-18/19; q4km 18 MB, q5km, q8 33 MB, unquantized, training zip).
- Corpus published on HF: `VerbalJungle/German-KeyboardLM-Corpus`.
- Ships a llama.cpp patch for the FUTO header (`patches/futo_metadata.patch`).
- Misspelled `char_embedding_mixing_v1` (see table).

## Patched app builds
- GitHub `stamchry/android-keyboard`: Greek fork. Commit `5795368` adds Unicode keystrokes.
- GitHub `armandodollia/futo-keyboard-polyglot`: APK `0.1.30-polyglot.12` (2026-10-13). Patch 0001 (based on stamchry)
  sends any Unicode letter to the model, maps keys to the model's single-letter tokens when there are no CHAR tokens,
  and no longer crashes on unknown letters. English models behave as before. Also adds dictation cleanup, model
  downloads and a space-bar touch-area patch. Still needs the dev toggle "Allow transformer models on non QWERTY
  layouts".

## Not transformer LMs (for completeness)
- GitHub `tribixbite/CleverKeys`: a separate keyboard with its own swipe engine, unrelated to FUTO models.
- Egyptian Arabic (GitHub `ahmedgomaaa/futo-keyboard-egyptian-arabic`) and Hungarian (GitHub `ZRktty/futo-keyboard-magyar`):
  dictionary and voice models only.
- HF `futo-org/futo-swipe`: official swipe models, English only.

## Requested, no model
Hungarian #2400 (no replies). Polish #1916 is now covered by areksh. Norwegian, French, Swedish, Turkish and Romanian
come up in #1212.

## Takeaways for this repo
- This isn't the first German model. Evaluate against jblechert v0.5/v0.5.1 and VerbalJungle v0.9.1 on the same
  next-word and autocorrect test sets.
- Copy the feature string exactly from the stock English model. Two community models ship flags that are missing
  or misspelled.
- Umlauts: stock FUTO garbles keystrokes after any non-ASCII letter (see `FUTO_FORMAT.md`). That limits
  autocorrect on umlaut words, whatever the model does.
- FUTO's announced rework may retire this model format.
