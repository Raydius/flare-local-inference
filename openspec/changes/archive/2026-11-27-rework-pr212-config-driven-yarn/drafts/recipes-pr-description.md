Title: qwen38: serve the YaRN context a checkpoint's rotary configuration declares

Depends on NVIDIA/hyperflare-inference#212, which adds `Checkpoint.context` and `Checkpoint.rope_parameters`. `uv.lock` is not updated here; it needs the merged core commit.

**Why.** Qwen3.8 Flash Next trains 262,144 positions, and its [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) extends the same weights to 1,048,576 tokens by setting `text_config.rope_parameters` to `rope_type: "yarn"` with `factor` 4.0 and `original_max_position_embeddings` 262144. The recipe could not serve that configuration: its rotary tables implemented only unscaled frequencies, and preparation rejected a context above 262,144. Reading the scaling from the checkpoint's own configuration, as vLLM and SGLang do for the same block, needs no alias and no flag, and one validator decides which rotary configurations the recipe implements. Frequencies and the attention factor come from the installed Transformers `ROPE_INIT_FUNCTIONS["yarn"]`, which `kimi31/dflash.py` already uses, so the recipe carries no copy of the formula.

**What changed.**

- `RotaryTables.allocate` (`qwen38/attention.py`): a `default` configuration takes the existing code path unchanged. A `yarn` configuration takes the reference inverse frequencies and multiplies the cos and sin tables by the reference attention factor, `0.1 ln(factor) + 1` unless the configuration gives `attention_factor`, or both `mscale` and `mscale_all_dim`. Any other `rope_type` is refused.
- `serving_context` (`qwen38/checkpoint.py`) validates the rotary block: the fixed MRoPE geometry, an `original_max_position_embeddings` of 262,144, and a `factor` whose context is 64-aligned, above 262,144, and at most 1,048,576. Other keys are refused by name.
- The recipe's `inspect_replacement` hook returns a resolved checkpoint directory with `context` and `rope_parameters` set from that block, and preparation rejects a context above what the loaded configuration serves.
- `Qwen38Parameters.validate` rejects a scaled context on SM103.
- `docs/model-preparation.md` describes preparing a YaRN checkpoint directory; `docs/TODO.txt` names SM103 at a scaled context as unqualified, with its removal criterion.

The fused Q/K rotation kernels, the b12x sparse-attention plan inputs, the SM103 score-workspace sizing, and the preparation identity are unchanged. The identity already hashes `config.json`, so native and scaled preparations, and two scaled ceilings, never share one.

**Result.** The pinned `qwen3.8-flash` checkpoint plans and prepares as before. A checkpoint directory with the model card's YaRN block plans a 1,048,576-token context, builds tables equal to the Transformers `Qwen4ExpTextRotaryEmbedding`, and exports and serves on SM120.

**Evidence.** One commit, `671b6609`, on `main` `8df08c21`, with Ignition at `3c73cd1d` and Transformers 5.23.0.

CPU and unit level:

- `tests/recipes/test_qwen38_rope_scaling.py`, 26 cases: gathered tables equal `Qwen4ExpTextRotaryEmbedding` at positions through 1,048,575 with zero tolerance; one case pins ramp bounds 14 and 22 and factor 1.1386; native tables are checked bit for bit; accepted and refused configurations (including `factor: 1000` and `rope_type: "longrope"`), the resolution hook, the SM103 rejection, and identity separation are covered.
- `test_qk_head_norm_mrope_matches_canonical` gains a YaRN case over a 1,048,576-row table: 60 passed on one SM120 GPU (RTX 5060 Ti).
- The Qwen checkpoint, attention, bootstrap, declaration, and `tests/cli` suites show 87 failures and 61 errors on `671b6609` and the same 148 on `8df08c21` with Ignition `74618652`; this host lacks the `flare_heliopause` build they need.

Export and serving, one RTX PRO 6000 (SM120, 96 GB), one image built from `3c73cd1d` and `671b6609`, `--max-concurrency 8`, MTP with 5 draft tokens:

| package | checkpoint config | `max_model_len` | export stages |
|---|---|---|---|
| `gb202-tp1-202611280514` | model card YaRN block, factor 4.0 | 1048576 | catalog 112 s, tune 490 s, compile 71 forwards 1,299 s, warm 260 s, capture 101 graphs 206 s |
| `gb202-tp1-202611280608` | pinned native | 262144 | same image, warm kernel cache |

Probe: `qualify/long_context_passkey.py` from the fork (not part of this pull request), `top_k=1`, 32 reply tokens, thinking off, one request per cell, longest first, depths 0.1, 0.5 and 0.9. Passkey splices one five-digit key sentence into a synthetic catalog. Tracking is RULER's multi-hop variable tracking ([NVIDIA/RULER](https://github.com/NVIDIA/RULER)): a four-hop chain of variable assignments spread through the prompt, answered only by following the chain to the first value.

| package | task | 8k | 64k | 200k | 320k | 520k | 1,000k (1,009,7xx tokens) |
|---|---|---|---|---|---|---|---|
| YaRN | passkey | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| YaRN | tracking | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| native | passkey | 3/3 | 3/3 | 3/3 | — | — | — |
| native | tracking | 2/3 | 3/3 | 3/3 | — | — | — |

Short-context comparison, the 18 cells both packages ran: the YaRN package passed 18, the native package 17, and 17 greedy replies were identical. The one difference is tracking at 8k, depth 0.5: the YaRN package replied with the bare value, and the native package began a step-by-step explanation that the 32-token limit cut before the value. A 1,000k request took 87 to 90 s.

Not measured: free-form generation quality, any benchmark with graded answers, SM103, and tensor-parallel execution. Thirty-six short-reply cells do not show that static YaRN leaves short-prompt quality unchanged; they show no degradation on these two tasks.
