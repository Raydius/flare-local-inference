"""Pure-PyTorch DFlash2 block-diffusion draft reference for teacher-forced acceptance measurement."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PINNED = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "hidden_size": 2560,
    "intermediate_size": 7680,
    "num_hidden_layers": 5,
    "layer_types": ["sliding_attention"] * 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "sliding_window": 2048,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "is_causal": False,
    "tie_word_embeddings": False,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248077,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 15, 25, 35, 45],
    },
    "num_target_layers": 48,
}
# Structural fields checked even when the numeric geometry is relaxed for small fabricated tests.
STRUCTURAL = ("architectures", "model_type", "hidden_act", "attention_bias", "is_causal",
              "tie_word_embeddings", "rope_parameters")


def _check(config, field, expected, prefix=""):
    if field not in config:
        raise ValueError(f"config.json missing {prefix}{field}")
    if config[field] != expected:
        raise ValueError(f"config.json {prefix}{field}: expected {expected!r}, got {config[field]!r}")


def validate_config(config: dict, strict: bool = True) -> None:
    """Reject a config that is not the pinned DFlash2 draft; strict=False keeps only the structural checks."""
    if "dflash_config" not in config:
        raise ValueError("config.json missing dflash_config")
    if strict:
        for field, expected in PINNED.items():
            if field != "dflash_config":
                _check(config, field, expected)
        for key, value in PINNED["dflash_config"].items():
            _check(config["dflash_config"], key, value, "dflash_config.")
        return
    for field in STRUCTURAL:
        _check(config, field, PINNED[field])
    _check(config["dflash_config"], "conv_kernel_size", 2, "dflash_config.")
    _check(config, "layer_types", ["sliding_attention"] * config["num_hidden_layers"])
    if config["hidden_size"] % config["dflash_config"]["conv_group_size"]:
        raise ValueError("config.json dflash_config.conv_group_size must divide hidden_size")


def rms_norm(x, weight, eps=1e-6):
    x32 = x.float()
    return (x32 * (x32.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()).bfloat16()


def rotary(x, positions, theta=10000.):
    """Neox rotate-half over the full head on x [n, heads, head_dim] at positions [n]."""
    dim = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = positions.float()[:, None] * frequencies
    cosine, sine = angles.cos().bfloat16().float()[:, None], angles.sin().bfloat16().float()[:, None]
    left, right = x.float().chunk(2, -1)
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).bfloat16()


def grouped_convolve(x, coefficients, base, side, block_size=8):
    """Two-tap block-local grouped convolution of x [n, H] with coefficients [n, 2*taps*H/group]."""
    n, hidden = x.shape
    taps = base.shape[1]
    group = hidden // (coefficients.shape[-1] // (2 * taps))
    x = x.float().view(-1, block_size, hidden)
    coeff = coefficients.float().view(*x.shape[:2], 2, taps, -1)[:, :, side].repeat_interleave(group, -1)
    output = torch.zeros_like(x)
    for tap in range(taps):
        output[:, tap:] += (base[side, tap].float() + coeff[:, tap:, tap]) * x[:, :block_size - tap]
    return output.reshape(n, hidden).bfloat16()


def select_step(unary, projected, predecessor_code, successor_codes):
    """FP32 selector scores for one position: unary [k] + <A[pred] * P, B[cand]> over [k, rank] successors."""
    coded = (predecessor_code * projected).float()
    return unary.float() + successor_codes.float() @ coded


class DFlash2Reference:
    def __init__(self, config: dict, tensors: dict[str, torch.Tensor], strict: bool = True):
        validate_config(config, strict)
        self.config = config
        dflash = config["dflash_config"]
        self.hidden, self.layers = config["hidden_size"], config["num_hidden_layers"]
        self.heads, self.kv_heads, self.head_dim = (config["num_attention_heads"],
                                                    config["num_key_value_heads"], config["head_dim"])
        self.window, self.eps = config["sliding_window"], config["rms_norm_eps"]
        self.theta = float(config["rope_parameters"]["rope_theta"])
        self.block_size, self.mask_id = dflash["block_size"], dflash["mask_token_id"]
        self.top_k = dflash["selector_top_k"]
        expected = self.expected_shapes(config)
        for name in sorted(set(expected) | set(tensors)):
            if name not in tensors or name not in expected:
                raise ValueError(f"{'missing' if name not in tensors else 'unexpected'} draft tensor {name}")
            if tuple(tensors[name].shape) != expected[name] or tensors[name].dtype != torch.bfloat16:
                raise ValueError(f"draft tensor {name}: expected BF16 {expected[name]}, "
                                 f"got {tensors[name].dtype} {tuple(tensors[name].shape)}")
        self.w = tensors
        self.device = tensors["fc.weight"].device

    @classmethod
    def expected_shapes(cls, config: dict) -> dict[str, tuple[int, ...]]:
        h, inter, dflash = config["hidden_size"], config["intermediate_size"], config["dflash_config"]
        heads, kv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        vocab, rank = config["vocab_size"], dflash["selector_rank"]
        taps, channels = dflash["conv_kernel_size"], 2 * dflash["conv_kernel_size"] * h // dflash["conv_group_size"]
        shapes = {
            "fc.weight": (h, h * len(dflash["target_layer_ids"])),
            "hidden_norm.weight": (h,), "norm.weight": (h,),
            "candidate_selector.hidden_projection.weight": (rank, h),
            "candidate_selector.predecessor_codebook": (vocab, rank),
            "candidate_selector.successor_codebook": (vocab, rank),
        }
        for i in range(config["num_hidden_layers"]):
            p = f"layers.{i}."
            shapes.update({
                p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,),
                p + "self_attn.q_proj.weight": (heads * d, h), p + "self_attn.k_proj.weight": (kv * d, h),
                p + "self_attn.v_proj.weight": (kv * d, h), p + "self_attn.o_proj.weight": (h, heads * d),
                p + "self_attn.q_norm.weight": (d,), p + "self_attn.k_norm.weight": (d,),
                p + "mlp.gate_proj.weight": (inter, h), p + "mlp.up_proj.weight": (inter, h),
                p + "mlp.down_proj.weight": (h, inter),
                p + "attention_conv.base_kernel": (2, taps, h), p + "attention_conv.kernel_projection.weight": (channels, h),
                p + "mlp_conv.base_kernel": (2, taps, h), p + "mlp_conv.kernel_projection.weight": (channels, h),
            })
        return shapes

    @classmethod
    def load(cls, snapshot: Path, device) -> "DFlash2Reference":
        from safetensors import safe_open

        config = json.loads((Path(snapshot) / "config.json").read_text())
        validate_config(config, strict=True)
        with safe_open(str(Path(snapshot) / "model.safetensors"), "pt", device=str(device)) as file:
            tensors = {name: file.get_tensor(name) for name in file.keys()}
        return cls(config, tensors, strict=True)

    def _kv(self, i, h, positions):
        p = f"layers.{i}.self_attn."
        k = F.linear(h, self.w[p + "k_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
        k = rotary(rms_norm(k, self.w[p + "k_norm.weight"], self.eps), positions, self.theta)
        v = F.linear(h, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
        return k, v

    @torch.no_grad()
    def context(self, features: torch.Tensor, positions: torch.Tensor) -> SimpleNamespace:
        """Per-layer normed and rotated (k, v) for the prefix; block() keeps the last `sliding_window` rows."""
        ctx = rms_norm(F.linear(features.to(self.device, torch.bfloat16), self.w["fc.weight"]),
                       self.w["hidden_norm.weight"], self.eps)
        positions = positions.to(self.device)
        return SimpleNamespace(layers=[self._kv(i, ctx, positions) for i in range(self.layers)],
                               positions=positions)

    def block_inputs(self, pending: int, embedding: torch.Tensor) -> torch.Tensor:
        ids = torch.tensor([pending] + [self.mask_id] * (self.block_size - 1), device=embedding.device)
        return embedding[ids].to(self.device, torch.bfloat16)

    def _attention(self, i, h, positions, ctx_k, ctx_v):
        p = f"layers.{i}.self_attn."
        q = F.linear(h, self.w[p + "q_proj.weight"]).view(-1, self.heads, self.head_dim)
        q = rotary(rms_norm(q, self.w[p + "q_norm.weight"], self.eps), positions, self.theta)
        k, v = self._kv(i, h, positions)
        k, v = torch.cat((ctx_k, k)), torch.cat((ctx_v, v))
        share = self.heads // self.kv_heads
        k, v = k.repeat_interleave(share, 1).float(), v.repeat_interleave(share, 1).float()
        scores = torch.einsum("qhd,khd->hqk", q.float(), k) * self.head_dim ** -0.5
        attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).bfloat16()
        return F.linear(attended.reshape(h.shape[0], -1), self.w[p + "o_proj.weight"])

    def _prepare(self, prefix, h):
        coefficients = F.linear(h, self.w[prefix + "kernel_projection.weight"])
        return grouped_convolve(h, coefficients, self.w[prefix + "base_kernel"], 0, self.block_size), coefficients

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, position: int, context) -> torch.Tensor:
        """Run the draft layers over one block of input embeddings [block_size, hidden]."""
        positions = position + torch.arange(self.block_size, device=self.device)
        residual = inputs
        for i in range(self.layers):
            p = f"layers.{i}."
            ctx_k, ctx_v = (t[-self.window:] for t in context.layers[i])
            h = rms_norm(residual, self.w[p + "input_layernorm.weight"], self.eps)
            h, coefficients = self._prepare(p + "attention_conv.", h)
            a = self._attention(i, h, positions, ctx_k, ctx_v)
            residual = residual + grouped_convolve(a, coefficients, self.w[p + "attention_conv.base_kernel"], 1, self.block_size)
            h = rms_norm(residual, self.w[p + "post_attention_layernorm.weight"], self.eps)
            h, coefficients = self._prepare(p + "mlp_conv.", h)
            m = F.linear(F.silu(F.linear(h, self.w[p + "mlp.gate_proj.weight"])) * F.linear(h, self.w[p + "mlp.up_proj.weight"]),
                         self.w[p + "mlp.down_proj.weight"])
            residual = residual + grouped_convolve(m, coefficients, self.w[p + "mlp_conv.base_kernel"], 1, self.block_size)
        return rms_norm(residual, self.w["norm.weight"], self.eps)

    def block(self, pending: int, position: int, context, embedding: torch.Tensor) -> torch.Tensor:
        return self.forward(self.block_inputs(pending, embedding), position, context)

    @torch.no_grad()
    def candidates(self, hidden: torch.Tensor, lm_head: torch.Tensor):
        """Top-k ids [block_size-1, k] and FP32 unary logits for block positions 1.. ."""
        logits = F.linear(hidden[1:], lm_head.to(self.device, torch.bfloat16)).float()
        unary, ids = logits.topk(self.top_k, dim=-1)
        return ids, unary

    @torch.no_grad()
    def walk(self, hidden: torch.Tensor, pending: int, ids: torch.Tensor, unary: torch.Tensor) -> torch.Tensor:
        """Greedy lattice walk: each position picks the candidate best scored against the previous choice."""
        projected = F.linear(hidden[1:], self.w["candidate_selector.hidden_projection.weight"])
        a, b = self.w["candidate_selector.predecessor_codebook"], self.w["candidate_selector.successor_codebook"]
        chosen, predecessor = [], pending
        for p in range(ids.shape[0]):
            scores = select_step(unary[p], projected[p], a[predecessor], b[ids[p]])
            # Lowest id among tied best scores: the component scatters the shortlist into the full
            # vocabulary and takes argmax there, so a tie resolves to the lowest token id.
            predecessor = int(ids[p][scores == scores.max()].min())
            chosen.append(predecessor)
        return torch.tensor(chosen, dtype=torch.int64, device=self.device)

    def propose(self, pending: int, position: int, context, embedding, lm_head) -> torch.Tensor:
        hidden = self.block(pending, position, context, embedding)
        return self.walk(hidden, pending, *self.candidates(hidden, lm_head))
