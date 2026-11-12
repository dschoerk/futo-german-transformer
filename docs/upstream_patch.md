# Proposed upstream change: allow transformer typing on QWERTZ / German layouts

## Problem
`LanguageModelFacilitator.makePredictionInputValues` (java/src/org/futo/inputmethod/latin/xlm/) skips the transformer
while a word is being composed unless the layout set is `qwerty` or its letter keys spell exactly
`qwertyuiopasdfghjklzxcvbnm`. Both German layouts fail the check:

| layout file | letters (sorted by position) |
|---|---|
| `Default/qwertz.yaml` | `qwertzuiopasdfghjklyxcvbnm` |
| `LatinScript/german.yaml` ("Deutsch") | `qwertzuiopüasdfghjklöäyxcvbnm` |

So a German model gets next-word prediction but no in-word completion/autocorrect unless the user enables
Developer settings → "Allow transformer models on non QWERTY layouts".

## Why the gate exists
Typed letters reach the model as `<CHAR_x>` embeddings mixed by tap proximity. A model trained on QWERTY typo
statistics would mis-correct on other layouts. A German model trained on QWERTZ statistics (this repo) has the
opposite problem, so the check should depend on the model, not be hard-coded.

## Suggested change
Let the model declare the layouts it was trained for, defaulting to today's behaviour:

```kotlin
// ModelInfo / GGUF metadata: keyboardlm.layouts = "qwertz" (optional, space separated)
val accepted = model.layouts.ifEmpty { listOf("qwerty") }
val letters = keyboard.sortedKeys.filter { isWordCodePoint(it.code) && it.code != '\''.code }
    .sortedBy { it.x + 10000 * it.y }.joinToString("") { it.label.lowercase() }
    .filter { it in 'a'..'z' }                       // umlaut keys are not a-z; the app already zeroes them
prevAcceptedLayoutResult = accepted.any { LAYOUT_LETTERS[it] == letters }
// LAYOUT_LETTERS = mapOf("qwerty" to "qwertyuiopasdfghjklzxcvbnm", "qwertz" to "qwertzuiopasdfghjklyxcvbnm", ...)
```

Our GGUF can carry `keyboardlm.layouts = "qwertz"`. Unknown metadata keys are ignored by current app versions.

## Workaround until then
Developer settings (Help & About → tap "Version code" 8×) → enable "Allow transformer models on non QWERTY layouts".
