#!/usr/bin/env python3
"""Export a trained HF Llama checkpoint to a FUTO Keyboard GGUF (mirrors FUTO's English ml4_q6_k.gguf layout).

* arch `llama`, tensors named like llama.cpp; q/k weights permuted for llama.cpp RoPE (as convert_hf_to_gguf.py does)
* tied embeddings duplicated as `output.weight` (the app needs both tensors)
* `keyboardlm.*` metadata; the SentencePiece model is embedded in `keyboardlm.ext_tokenizer_data`
* quantization through llama.cpp's C API (llama-cpp-python), Q6_K like the English model
"""
import argparse, json
from datetime import date
from pathlib import Path
import numpy as np
import gguf
import sentencepiece as spm
from safetensors.numpy import load_file

FEATURES = "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1"
LAYER_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight", "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def permute(w, n_head):
    """HF rotate-half q/k layout -> llama.cpp interleaved RoPE layout (from llama.cpp convert_hf_to_gguf.py)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/ckpt/hf_latest")
    ap.add_argument("--tokenizer", default="models/tokenizer/tokenizer.model")
    ap.add_argument("--out", default="models/gguf/de_keyboard_f16.gguf")
    ap.add_argument("--name", default="German v1")
    ap.add_argument("--languages", default="de")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--quantize", default="Q6_K", help="Q6_K, Q8_0 or 'none'")
    args = ap.parse_args()

    md = Path(args.model_dir)
    cfg = json.loads((md / "config.json").read_text())
    sd = load_file(str(md / "model.safetensors"))
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    tok_bytes = Path(args.tokenizer).read_bytes()
    n_head, n_kv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    assert sp.get_piece_size() == cfg["vocab_size"], "tokenizer/model vocab mismatch"

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_name(args.name)
    w.add_string("keyboardlm.languages", args.languages)
    w.add_uint32("keyboardlm.finetuning_count", 0)
    w.add_string("keyboardlm.history", f"{date.today().isoformat()}: Model created (futo-german-transformer)")
    w.add_string("keyboardlm.features", args.features)
    w.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    w.add_string("keyboardlm.layouts", "qwertz")  # informational, see docs/upstream_patch.md (ignored by the app today)
    w.add_array("keyboardlm.ext_tokenizer_data", np.frombuffer(tok_bytes, dtype=np.uint8).tolist())

    w.add_context_length(2048)  # app's LLAMA_CONTEXT_SIZE; English model declares the same
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_rope_dimension_count(cfg["hidden_size"] // n_head)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_rope_freq_base(cfg.get("rope_theta", (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0)))
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    tokens, scores, types = [], [], []
    for i in range(sp.get_piece_size()):
        tokens.append(sp.id_to_piece(i).encode("utf-8")); scores.append(sp.get_score(i))
        types.append(gguf.TokenType.UNKNOWN if sp.is_unknown(i) else gguf.TokenType.CONTROL if sp.is_control(i)
                     else gguf.TokenType.BYTE if sp.is_byte(i) else gguf.TokenType.UNUSED if sp.is_unused(i)
                     else gguf.TokenType.USER_DEFINED if sp.id_to_piece(i).startswith("<") and sp.id_to_piece(i).endswith(">") and i < 64
                     else gguf.TokenType.NORMAL)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens); w.add_token_scores(scores); w.add_token_types(types)
    w.add_bos_token_id(sp.bos_id()); w.add_eos_token_id(sp.eos_id()); w.add_unk_token_id(sp.unk_id())
    w.add_pad_token_id(sp.pad_id())

    def put(name, arr):
        arr = arr.astype(np.float32 if arr.ndim == 1 else np.float16)
        w.add_tensor(name, arr)

    emb = sd["model.embed_tokens.weight"]
    put("token_embd.weight", emb)
    put("output_norm.weight", sd["model.norm.weight"])
    put("output.weight", sd.get("lm_head.weight", emb))  # tied -> duplicate
    for l in range(cfg["num_hidden_layers"]):
        for hf, gg in LAYER_MAP.items():
            t = sd[f"model.layers.{l}.{hf}"]
            if hf.startswith("self_attn.q_proj"):
                t = permute(t, n_head)
            elif hf.startswith("self_attn.k_proj"):
                t = permute(t, n_kv)
            put(f"blk.{l}.{gg}", t)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    if args.quantize.lower() != "none":
        import ctypes, llama_cpp
        q = out.with_name(out.name.replace("_f16", f"_{args.quantize.lower()}"))
        params = llama_cpp.llama_model_quantize_default_params()
        params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{args.quantize.upper()}")
        rc = llama_cpp.llama_model_quantize(str(out).encode(), str(q).encode(), ctypes.byref(params))
        assert rc == 0, f"quantize failed ({rc})"
        print(f"wrote {q} ({q.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
