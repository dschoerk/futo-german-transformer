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
