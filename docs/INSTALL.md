# Using the German model in FUTO Keyboard

1. Download `de_keyboard_q6_k.gguf` (see the GitHub release) to your phone.
2. FUTO Keyboard settings → Help & About → tap **Version code** 8 times to unlock Developer settings.
3. Developer settings → enable **Allow transformer models on non QWERTY layouts**.
   German layouts (QWERTZ / "Deutsch") fail the app's hard-coded QWERTY check; without this toggle you only get
   next-word suggestions, not in-word completion/autocorrect. See `upstream_patch.md`.
4. Settings → Predictive text (Text prediction) → Transformer models → **Import from file** → pick the `.gguf`.
   It should report it as the default for `de`.
5. Make sure German is an active language and **Transformer LM** prediction is enabled, then switch to a German layout.

Notes
- The app selects the model by the language of the active input (`keyboardlm.languages = de`).
- Umlaut keys (ü ö ä on the "Deutsch" layout) are not passed to the model as letters by the app; the model was
  trained to restore umlauts from what the app does pass (neighbouring letters, `u`, `ue`, or nothing).
