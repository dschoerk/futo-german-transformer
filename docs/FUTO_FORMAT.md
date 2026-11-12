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

## Differences from the community Greek model (stamchry/greek-keyboard-lm)
It exports features `base_v1 inverted_space lora_finetunable_v1` (no `xbu_char_autocorrect_v1`, no CHAR tokens
despite its README), i.e. next-word / completion only. We aim for the full protocol.
