"""Sliding-window DFlash draft block over per-row E4M3 K/V rings.

Recipes adapt their batch metadata into per-token positions, state rows, and a
live mask, pass their rings as ``[rows, window, kv_heads, head_dim]`` tensors,
and append their own candidate selector after the block.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .embedding import PreparedEmbedding
from .grouped_convolution import PreparedGroupedConvolution
from .linear import prepare_projection
from .paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from .projection_overlap import ConcurrentProjectionChain
from .rms_norm import PreparedRMSNorm
from .vocabulary import PreparedVocabularyProjection
from .window_attention import stage_windows


def _norm(source, weight, epsilon):
    values = source.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(source.dtype)


def _out_features(weight):
    return weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features


def ring_slots(live, rows, positions, window):
    """Flat ring slot per token: ``row * window + position % window`` for live tokens, ``-1`` otherwise.

    ``live`` is decided by the caller before indexing so dead tokens (request id
    ``-1``) never address another row's ring.
    """
    return torch.where(live, rows * window + positions % window, -1)


@dataclass(frozen=True)
class DraftScratch:
    activations: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, hidden, vocabulary, buffers):
        def zeros(shape):
            return buffers.zeros(shape, torch.bfloat16)
        return cls(zeros((tokens, hidden)), zeros((tokens, hidden)),
            zeros((rows, hidden)), zeros((rows, vocabulary)), zeros((rows, vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.normalized[:tokens], self.hidden[:rows],
                          self.local_logits[:rows], self.logits[:rows])


class Rotary:
    def __init__(self, config, device):
        c = config.text
        self.inverse = 1 / c["rope_theta"] ** (torch.arange(0, c["head_dim"], 2,
            dtype=torch.float32, device=device) / c["head_dim"])

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.inverse[None]
        cosine, sine = angles.cos().to(values.dtype).float()[:, None], angles.sin().to(values.dtype).float()[:, None]
        left, right = values.float().chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


class PreparedContext:
    """Project target features into every draft layer's context K/V and write them into the rings.

    ``draft_weights`` carries ``projection`` (fc), ``hidden_norm``, ``context_kv``
    (every layer's K/V rows concatenated), and ``layers[i]["k_norm"]``; ``rings``
    is one ``(key, value)`` pair per layer.
    """

    def __init__(self, draft_weights, auxiliary, rings, *, window, epsilon, rotary, session, name, workspace, buffers):
        draft = draft_weights
        tokens, hidden = auxiliary.shape[0], draft.hidden_norm.shape[0]
        self.window, self.epsilon = window, epsilon
        self.projected = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.normalized = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.project = prepare_projection(auxiliary, draft.projection, self.projected,
            session=session, name=name + ".project", workspace=workspace, activation_mode="auto")
        self.norm = PreparedRMSNorm(self.projected, draft.hidden_norm, self.normalized,
            eps=self.epsilon, session=session, name=name + ".norm")
        self.kv = buffers.zeros((tokens, draft.context_kv.shape[0]), torch.bfloat16)
        self.project_kv = prepare_projection(self.normalized, draft.context_kv, self.kv,
            session=session, name=name + ".kv", workspace=workspace)
        shape = (tokens, *rings[0][0].shape[2:])
        self.key, self.value = (buffers.zeros(shape, torch.bfloat16) for _ in range(2))
        self.slots = buffers.zeros((tokens,), torch.int64)
        self.rotary = rotary
        self.stages = []
        for index, (layer, (key_ring, value_ring)) in enumerate(zip(draft.layers, rings, strict=True)):
            writer = PreparedPagedKVWriter(self.key, self.value, key_ring, value_ring, self.slots,
                session=session, name=f"{name}.{index}.write")
            self.stages.append((layer["k_norm"], writer))

    def __call__(self, auxiliary, positions, rows, live):
        self.slots.copy_(ring_slots(live, rows, positions, self.window))
        self.norm(self.project(auxiliary))
        self.project_kv(self.normalized)
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), 2, *self.key.shape[1:])
        for index, (norm, writer) in enumerate(self.stages):
            self.key.copy_(self.rotary(_norm(chunks[:, index, 0], norm, self.epsilon), positions))
            self.value.copy_(chunks[:, index, 1])
            writer()


class DraftAttention:
    def __init__(self, weights, source, metadata, rings, config, *, sink=None, page_tokens,
                 session, name, workspace, buffers):
        self.weights, self.metadata = weights, metadata
        c, tokens = config.text, source.shape[0]
        self.window, self.epsilon = c["sliding_window"], c["rms_norm_eps"]
        qh, kh, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.projected = buffers.zeros((tokens, (qh + 2 * kh) * dim), torch.bfloat16)
        self.query = buffers.zeros((tokens, qh, dim), torch.bfloat16)
        self.key, self.value = (buffers.zeros((tokens, kh, dim), torch.bfloat16) for _ in range(2))
        self.attended = buffers.zeros(self.query.shape, torch.bfloat16)
        self.project = prepare_projection(source, weights["qkv"], self.projected,
            session=session, name=name + ".qkv", workspace=workspace)
        self.output = prepare_projection(self.attended.flatten(1), weights["output"], buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".output", workspace=workspace)
        self.rotary = Rotary(config, source.device)
        self.history = tuple(rings)
        rows, pages = metadata.lengths.numel(), (self.window + config.block_size + page_tokens - 1) // page_tokens
        self.staged = tuple(buffers.zeros((rows * pages, page_tokens, kh, dim), history.dtype)
                            for history in self.history)
        self.pages = torch.arange(rows * pages, device=source.device, dtype=torch.int32).view(rows, pages)
        self.lengths = buffers.zeros((rows,), torch.int32)
        self.attention = PreparedPagedAttention(self.query, *self.staged, self.pages, self.lengths,
            metadata.query_start_loc, self.attended, sink=sink, mode="verify", width=config.block_size,
            window=self.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention")

    def __call__(self, source):
        m = self.metadata
        q, k, v = self.project(source).split((self.query[0].numel(), self.key[0].numel(), self.value[0].numel()), -1)
        self.query.copy_(self.rotary(_norm(q.view_as(self.query), self.weights["q_norm"], self.epsilon), m.positions))
        self.key.copy_(self.rotary(_norm(k.view_as(self.key), self.weights["k_norm"], self.epsilon), m.positions))
        self.value.copy_(v.view_as(self.value))
        stage_windows(self.history[0], self.key, self.staged[0], m)
        stage_windows(self.history[1], self.value, self.staged[1], m)
        self.lengths.copy_(m.starts.clamp(max=self.window).int() + m.lengths)
        self.attention()
        return self.output(self.attended.flatten(1))


class DraftMLP:
    def __init__(self, weights, source, *, session, name, workspace, buffers):
        gate, up, down = weights
        shape = (source.shape[0], _out_features(gate))
        self.intermediate = buffers.zeros(shape, torch.bfloat16)
        self.gate, self.up = (prepare_projection(source, weight, buffers.zeros(shape, torch.bfloat16),
            session=session, name=name + suffix, workspace=workspace, activation_mode="auto")
            for weight, suffix in ((gate, ".gate"), (up, ".up")))
        self.down = prepare_projection(self.intermediate, down, buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".down", workspace=workspace, activation_mode="auto")
        self.concurrent_up = (ConcurrentProjectionChain((self.up,), parallel_plans=(self.gate.plan,),
            session=session, workspace=workspace) if workspace.overlap else None)

    def __call__(self, source):
        if self.concurrent_up is not None:
            self.concurrent_up.start(source)
        gate = self.gate(source)
        up = (self.up(source) if self.concurrent_up is None else self.concurrent_up.finish(source, gate))
        self.intermediate.copy_((F.silu(gate.float()) * up.float()).bfloat16())
        return self.down(self.intermediate)


class PreparedDraftBlock:
    """Embedding, convolved sliding-window layers, final norm, and head over a draft block.

    ``weights`` carries ``embedding``, ``head``, ``config.draft`` (``text``,
    ``block_size``), ``config.hidden``, and ``draft.layers``/``draft.norm``;
    ``rings`` is one ``(key, value)`` pair per layer. A layer's optional
    ``"sink"`` entry is passed to its attention.
    """

    def __init__(self, weights, scratch, metadata, rings, *, page_tokens, vocabulary_runtime, coordinator,
                 session, name, workspace, layer_buffers):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c, s = weights.config.draft, scratch
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, s.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers, previous = [], None
        for index, (w, ring) in enumerate(zip(weights.draft.layers, rings, strict=True)):
            prefix, storage = f"{name}.layer{index}", layer_buffers.cursor()
            norm = PreparedRMSNorm(s.activations, w["input_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=previous, session=session, name=prefix + ".norm")
            convs = []
            for kind in ("attention_conv", "mlp_conv"):
                coefficients = 2 * c.text["dflash_config"]["conv_kernel_size"] * weights.config.hidden // c.text["dflash_config"]["conv_group_size"]
                convs.append(PreparedGroupedConvolution(s.normalized, w[kind], block_size=c.block_size,
                    group_size=c.text["dflash_config"]["conv_group_size"], session=session, name=prefix + "." + kind,
                    buffers={key: storage.zeros((s.activations.shape[0], width), torch.bfloat16)
                        for key, width in (("coefficients", coefficients), ("before", weights.config.hidden), ("after", weights.config.hidden))}))
            attention = DraftAttention(w, s.normalized, metadata, ring, c, sink=w.get("sink"), page_tokens=page_tokens,
                session=session, name=prefix + ".attention", workspace=workspace, buffers=storage)
            post = PreparedRMSNorm(s.activations, w["post_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=s.normalized, session=session, name=prefix + ".post_norm")
            mlp = DraftMLP(w["mlp"], s.normalized, session=session, name=prefix + ".mlp", workspace=workspace, buffers=storage)
            self.layers.append((norm, convs[0], attention, post, convs[1], mlp))
            previous = s.normalized
        self.norm = PreparedRMSNorm(s.activations, weights.draft.norm, s.normalized,
            eps=c.text["rms_norm_eps"], residual=previous, session=session, name=name + ".norm")
        self.head = PreparedVocabularyProjection(s.hidden, weights.head, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto")

    def __call__(self):
        residual, output = self.embedding(), None
        for norm, conv, attention, post, mlp_conv, mlp in self.layers:
            output = conv.finish(attention(conv.prepare(norm(residual, output))))
            output = mlp_conv.finish(mlp(mlp_conv.prepare(post(residual, output))))
        self.scratch.hidden.copy_(self.norm(residual, output)[self.metadata.sample_indices])
        self.head(self.scratch.hidden)


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None
