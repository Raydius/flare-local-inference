"""DFlash2 draft checkpoint tensors loaded as BF16 projections, norms, convolutions, and selector codebooks."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.qwen38_dflash2 import DFlash2Reference


def check_tensors(tensors, text):
    """Reject a checkpoint whose tensor names, shapes, or dtypes differ from the reference's expectation."""
    expected = DFlash2Reference.expected_shapes(text)
    for name in sorted(expected.keys() | tensors.keys()):
        if name not in tensors or name not in expected:
            raise ValueError(f"{'missing' if name not in tensors else 'unexpected'} draft tensor {name}")
        tensor = tensors[name]
        if tuple(tensor.shape) != expected[name] or tensor.dtype != "BF16":
            raise ValueError(f"draft tensor {name}: expected BF16 {expected[name]}, "
                             f"got {tensor.dtype} {tuple(tensor.shape)}")


class DraftWeightLoader(DirectWeightLoader):
    def tensor(self, name):
        value = self.load_tensor(name)
        if value.dtype != torch.bfloat16:
            raise ValueError(f"DFlash2 requires BF16 parameters: {name}")
        return value


@dataclass(frozen=True)
class DraftWeights:
    projection: torch.Tensor
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    context_kv: torch.Tensor
    layers: tuple[dict, ...]
    selector: dict

    @classmethod
    def load(cls, snapshot, config, device, *, progress=None):
        manifest = inspect_indexed_checkpoint(snapshot)
        check_tensors(manifest.tensors, config.text)
        with DraftWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            return cls.assemble(config, loader.tensor)

    @classmethod
    def assemble(cls, config, tensor):
        """Build the weights from ``tensor(name)``, concatenating each layer's Q/K/V and every layer's context K/V."""
        projection = tensor("fc.weight")
        hidden_norm, norm = tensor("hidden_norm.weight"), tensor("norm.weight")
        layers = []
        for index in range(config.layers):
            p = f"layers.{index}."
            qkv = torch.cat([tensor(p + "self_attn." + name + ".weight") for name in ("q_proj", "k_proj", "v_proj")])
            layers.append(dict(qkv=qkv, context=qkv[config.heads * config.head_dim:],
                output=tensor(p + "self_attn.o_proj.weight"),
                q_norm=tensor(p + "self_attn.q_norm.weight"), k_norm=tensor(p + "self_attn.k_norm.weight"),
                input_norm=tensor(p + "input_layernorm.weight"),
                post_norm=tensor(p + "post_attention_layernorm.weight"),
                mlp=tuple(tensor(p + "mlp." + name + ".weight") for name in ("gate_proj", "up_proj", "down_proj")),
                **{kind: dict(base=tensor(p + kind + ".base_kernel"),
                    projection=tensor(p + kind + ".kernel_projection.weight"))
                   for kind in ("attention_conv", "mlp_conv")}))
        selector = dict(projection=tensor("candidate_selector.hidden_projection.weight"),
            predecessor=tensor("candidate_selector.predecessor_codebook"),
            successor=tensor("candidate_selector.successor_codebook"))
        context_kv = torch.cat([layer["context"] for layer in layers])
        return cls(projection, hidden_norm, norm, context_kv, tuple(layers), selector)
