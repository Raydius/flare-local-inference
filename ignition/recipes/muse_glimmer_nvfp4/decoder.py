# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Muse Glimmer text layers with dense NVFP4 SwiGLU and a softcapped vocabulary head."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor
from ignition.recipes.muse_spark.decoder import PreparedResume
from .attention import AttentionScratch, PreparedAttention
from .model import decoder_layer, norm


@dataclass(frozen=True)
class Scratch:
    """Operand declarations for one text forward; the forward allocates its own values."""
    embedding: torch.Tensor
    local_embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    mlp: MLPScratch
    output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device, *, examples=None):
        def zeros(role, shape):
            return declaration_tensor(examples, "muse_glimmer.target." + role, shape, torch.bfloat16, device)
        # Verify staging must hold the drafter's whole block; without a drafter nothing is verified.
        width = 1 if config.draft is None else config.draft.block_size
        attention = AttentionScratch.allocate(config, shard, tokens, device, requests=samples, examples=examples,
                                              max_verify_width=width)
        intermediate = config.intermediate // shard.world_size
        mlp = MLPScratch(zeros("mlp.gate_up", (tokens, 2 * intermediate)),
                         zeros("mlp.intermediate", (tokens, intermediate)), zeros("mlp.output", (tokens, config.hidden)))
        return cls(zeros("embedding", (tokens, config.hidden)),
            zeros("local_embedding", (tokens, config.hidden // shard.world_size)),
            zeros("normalized", (tokens, config.hidden)), attention, mlp, zeros("output", (tokens, config.hidden)),
            zeros("hidden", (samples, config.hidden)), zeros("local_logits", (samples, config.vocabulary_partition(shard)[2])),
            zeros("logits", (samples, config.vocabulary)))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, field).rows(tokens) if field in ("attention", "mlp")
            else getattr(self, field)[:samples if field in ("hidden", "local_logits", "logits") else tokens]
            for field in self.__dataclass_fields__))


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        self.mlp = PreparedMLP(scratch.normalized, weights.mlp, scratch.mlp.gate_up, scratch.mlp.intermediate,
            scratch.mlp.output, session=session, name=name, workspace=workspace).functional(session=session)
        self.reduce = PreparedAllReduce(scratch.mlp.output, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session)

    def forward(self, hidden_states):
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        return self.reduce(self.mlp(rows)).view_as(hidden_states)


def softcap_logits(logits, multiplier, softcap):
    logits.copy_(torch.tanh(logits * multiplier / softcap) * softcap)


class PreparedDecoder:
    """Text layers whose forward takes its packed metadata and returns local values.

    ``scratch`` and ``metadata`` declare operand shapes for preparation; ``release_outputs`` drops
    them once the enclosing forward has prepared the operators that read the decoder's outputs.
    """

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, examples=None, scoring=None, vision_state=None):
        self.weights, self.scratch, self.pool = weights, scratch, pool
        # Prefill with images: ``<|patch|>`` rows take published features after the embedding norm, where
        # the canonical ``MuseGlimmerModel.forward`` scatters them into the normed ``inputs_embeds``.
        self.vision_state = vision_state
        c = weights.config.text
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator, session=session,
            name=name + ".embedding", num_rows=metadata.live_counts[:1]).functional(session=session, dynamic_inputs=True)
        self.embedding_norm = norm("rms", dimension=c["hidden_size"], eps=c["rms_norm_eps"], with_scale=False)
        self.layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            attention = PreparedAttention(index, w, weights.config, weights.shard, scratch.normalized,
                scratch.attention, metadata, pool, mode=mode, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".attention", workspace=workspace)
            feedforward = PreparedFeedForward(w, scratch, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp", workspace=workspace)
            self.layers.append(decoder_layer(weights.config, w, attention, feedforward))
        self.norm = norm("rms", weights.norm, dimension=c["hidden_size"], eps=c["rms_norm_eps"])
        self.target_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.verify = mode == "verify"
        # The draft context reads these rows; a forward without a drafter returns none.
        self.auxiliary = None if not self.target_layers else declaration_tensor(examples,
            "muse_glimmer.target.auxiliary", (scratch.embedding.shape[0], len(self.target_layers) * c["hidden_size"]),
            torch.bfloat16, scratch.embedding.device)
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.vocabulary, session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session,
            name=name + ".head").functional(session=session)
        self.output_multiplier, self.softcap = c["output_multiplier"], c["final_logit_softcapping"]

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def __call__(self, metadata):
        m = metadata
        hidden = self.embedding_norm(self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1]))
        if self.vision_state is not None:
            self.vision_state.inject(m, hidden)
        records = [] if self.verify else None
        taps = [None] * len(self.target_layers)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, metadata=m, records=records)
            if index in self.target_layers:
                taps[self.target_layers.index(index)] = hidden
        auxiliary = torch.cat(taps, -1) if taps else None
        if self.scoring is not None:
            normalized = self.norm(hidden)
            for start, count, projection, reduction in self.scoring.parts:
                scores = projection(normalized[start:start + count])
                softcap_logits(scores, self.output_multiplier, self.softcap)
                reduction(scores)
            sampled = normalized[m.sample_indices]
        else:
            sampled = self.norm(hidden[m.sample_indices])
        logits = self.head(sampled)
        softcap_logits(logits, self.output_multiplier, self.softcap)
        return sampled, logits, auxiliary, records

    def commit(self, counts, records, metadata):
        for layer, record in zip(self.layers, records, strict=True):
            layer.self_attn.commit(counts, record, metadata)
