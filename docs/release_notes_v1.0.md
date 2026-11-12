German (`de`) transformer language model for [FUTO Keyboard](https://github.com/futo-org/android-keyboard): next-word prediction, in-word completion and autocorrect. This is the final checkpoint of the v1 training run (52,960 steps, 3 epochs, about 1.7B tokens).

## What it does
- Next-word prediction
- In-word completion and autocorrect, using FUTO's `<XBU>/<XBC>/<XEC>` protocol (`xbu_char_autocorrect_v1`). It was trained on QWERTZ typing noise and restores umlauts and ß (e.g. `schln` → *schön*/*schon*, `morgn` → *morgen*, `spatee` → *später*, `deztdcl` → *Deutschland*)

## Model
- Llama architecture: 8 layers, 512 hidden size, FFN 1280, 34.4M parameters. A 20k-token SentencePiece vocabulary is embedded in the file. Q6_K quantization, 37.4 MB
- Same GGUF layout and metadata types as FUTO's English model (`ml4_q6_k.gguf`)
- Validated with [`scripts/check_gguf.py`](https://github.com/dschoerk/futo-german-transformer/blob/main/scripts/check_gguf.py):
  - app-style metadata and tokenizer loading;
  - llama.cpp vs PyTorch parity. The f16 export matches PyTorch exactly. Q6_K has a mean KL of 0.0056, and every top-1 difference is a near-tie.

## Evaluation
Measured on held-out sentences, with the app's sampling logic ported from its native code (3-wide beam, banned punctuation tokens):

| Task | Top-1 | Top-3 | Cases |
|---|---|---|---|
| Autocorrect/completion from noisy typing | 65.5% | 78.7% | 1,913 |
| Full-word autocorrect | 74.0% | 86.8% | 1,435 |
| Next word | 19.8% | 33.5% | 1,919 |

Compared with the [v0.1-10k preview](https://github.com/dschoerk/futo-german-transformer/releases/tag/v0.1-10k) on the same 420 cases, top-1 improved for every task:

| Task | v0.1 (step 10k) | v1.0 |
|---|---|---|
| Autocorrect | 59.5% | 66.9% |
| Full-word autocorrect | 67.0% | 73.9% |
| Next word | 14.3% | 20.2% |

The test sentences come from the same prose sources as the training data. Chat messages are likely to score lower on next-word prediction.

Validation loss 2.98 and token accuracy 46.7% (10k checkpoint: 3.36 and 41.7%). The training curve is in [`results/training_progress.png`](https://github.com/dschoerk/futo-german-transformer/blob/main/results/training_progress.png).

## Install
1. Download `de_keyboard_v1.0_q6_k.gguf` to your phone.
2. FUTO Keyboard → Settings → Help & About → tap **Version code** 8 times to unlock Developer settings.
3. Developer settings → enable **Allow transformer models on non QWERTY layouts**. German QWERTZ and "Deutsch" layouts fail the app's hard-coded QWERTY check. Without this toggle you only get next-word suggestions.
4. Predictive text → Transformer models → **Import from file**. The model should be listed as the default for `de`.
5. Switch to a German layout.

## Known limitations
- **Trained on written prose** (German Wikipedia, FineWeb-2 `deu_Latn`, OPUS-100). Chat-style text works, but colloquial forms are weak:
  - contractions such as *geht's*, *hab's*, *wird's* are rarely suggested (*gibt's* works);
  - typing `gehts` keeps *gehts*.
  
  A chat-focused v2 is planned ([`docs/V2_PLAN.md`](https://github.com/dschoerk/futo-german-transformer/blob/main/docs/V2_PLAN.md)).
- No emoji predictions.
- Not tested on a real device yet. Feedback via issues is welcome.
- On the "Deutsch" layout, stock FUTO may misalign keystrokes after a directly typed umlaut key. This comes from the app's input handling ([`docs/FUTO_FORMAT.md`](https://github.com/dschoerk/futo-german-transformer/blob/main/docs/FUTO_FORMAT.md)), not from the model.

SHA-256: `6f7117754c1338d57d7ca67a82b3bd04e239696053e4070145294b074f208e29`
