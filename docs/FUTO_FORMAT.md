# FUTO Keyboard transformer: what a model must satisfy

Verified against `futo-org/android-keyboard` (HEAD, Nov 2026). Paths are in that repo.

## GGUF container
- Arch `llama`; tensors must include `output.weight` (duplicate `token_embd.weight` if tied).
- Metadata read by `xlm/ModelPaths.kt` (`ModelInfo`): `keyboardlm.languages`, `keyboardlm.features`,
  `keyboardlm.ext_tokenizer_type` (= `sentencepiece`), `keyboardlm.ext_tokenizer_data`,
  `keyboardlm.finetuning_count`. A model is "unsupported" if features, tokenizer type or languages are empty.
- Supported features: `base_v1`, `inverted_space`, `xbu_char_autocorrect_v1`, `lora_finetunable_v1`,
  `xc0_swipe_typing_v1`, `char_embed_mixing_v1`.
- Import requires file extension `.gguf` and `GGUF` magic.

## Tokenizer
- SentencePiece Unigram, whitespace as suffix (`inverted_space`): words end in `▁`.
- With `xbu_char_autocorrect_v1` the native code asserts tokens `<XBU>`, `<XBC>`, `<XEC>`, `<CHAR_A>` exist,
  and **`<CHAR_A>`..`<CHAR_Z>` must have consecutive ids** (`LETTERS_TO_IDS[i] = CHAR_A + i`).
- Also looks up `▁`, `-`, `*`, `-▁`.

## Autocorrect training format (`xlm/TrainingDataGenerator.kt`)
```
<context words> <XBU><CHAR_t><CHAR_e><CHAR_s><CHAR_t><XBC>truth ▁<XEC>
```
- Only a-z letters of the typed (misspelled) word are mapped to CHAR tokens (others dropped).
- No space after `<XBC>` / `<XEC>`. About 1/3 of words corrupted in FUTO's own generator;
  corrupted words may be truncated (prefix) to teach completion.
- Inference (`PredictCorrection`): `BOS + ctx + <XBU>` then CHAR token mixes, then forced `<XBC>`, then sample until `<XEC>`.

## Layout gate (important for German)
`LanguageModelFacilitator.makePredictionInputValues`: while composing a word the transformer is disabled unless
layout set name is `qwerty` or the letters spell `qwertyuiopasdfghjklzxcvbnm`, **or** the Developer setting
"Allow transformer models on non QWERTY layouts" is on. German QWERTZ fails this gate.
=> Users need the dev toggle, or FUTO needs a patch (see `docs/upstream_patch.md` once written).

## Non-ASCII keystrokes at inference (important for German)
`native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp` (last changed 2026-07-27, `f7749cfcc0`), mix building
around L1052–1150:
- `partialWordString` comes from `jstring2string`, so it is UTF-8 **bytes**. `inComposeX/Y` hold one coordinate per
  keystroke, and `inputSize = min(#coords, #bytes)`.
- The loop indexes both by the same `i`. Bytes outside a–z / A–Z / 0–9 are skipped (L1056). ä/ö/ü/ß are 2 bytes each,
  so after an umlaut each byte gets the coordinate of a later keystroke, and the final keystrokes fall off the end.
- Mix tokens come from the keys nearest each tap coordinate (L1143), not from the typed character. Only a–z keys
  get weight.
- Example: for "über" the model sees taps for `e r` only; for "schön", `s c h` only.
- From reading the code; not tested on a device. The polyglot fork (patch 0001) fixes it, and upstream declined to
  (#2371).

## Community models
See `docs/PRIOR_ART.md`. Two German models already exist (jblechert, VerbalJungle). The Greek model
(stamchry/greek-keyboard-lm) exports `base_v1 inverted_space lora_finetunable_v1`: no `xbu_char_autocorrect_v1` and no
CHAR tokens, despite its README, so next-word and completion only. We aim for the full protocol.

## Reference: FUTO's English model (`ml4_q6_k.gguf`, from gitlab.futo.org/keyboard/keyboard-large-resources)
- llama, 8 layers, 512 hidden, FFN 1024, 8 heads, rms eps 1e-6, rope base 10000, `context_length` 2048, Q6_K.
- Separate `token_embd.weight` and `output.weight` tensors.
- features: `base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embed_mixing_v1`.
- SentencePiece unigram, vocab 15008, suffix whitespace, byte fallback, split digits,
  **`remove_extra_whitespaces: false`**, `add_dummy_prefix: false`; ids pad=0 bos=1 eos=2 unk=3.
  `'This is a '` → `This▁ is▁ a▁` (the trailing space survives; required because the app tokenizes `trim(ctx) + " "`).
- 300 user-defined symbols: `<FUTO0..23>`, English contractions (`I'm▁`), `<XBU>`=174 … `<CHAR_A>`=182, emoji.

## Inference details that shape training (from the app source)
- Context (`LanguageModel.kt safeguardContext`): only the current line, trimmed to ≤70 chars / ≤16 spaces by cutting
  at the last sentence end, then at commas, else the last 5 words. Prompt = `BOS + tokenize(trim(ctx) + " ")`.
  ⇒ train with short contexts that often start mid-sentence right after BOS.
- Typed letters are **always** fed as proximity-weighted mixes of up to 4 nearest *letter* keys' `<CHAR_*>`
  embeddings (`DecodePromptAndMixes`; proportions < 0.05 dropped, non a-z keys zeroed). One-hot only when
  the app has no tap coordinates. ⇒ train with embedding mixing.
- Capitalization: lowercase typing → `IgnoredCapitals` (model may output "Haus" for "haus"); a typed capital
  bans lowercase-initial first tokens.
- FUTO German layouts: `german.yaml` "Deutsch" (`q…p ü / a…l ö ä / y…m`) and `qwertz.yaml`. Umlaut keys are not
  a-z, so a tap on ü reaches the model as its neighbouring letters (≈ `p`).
