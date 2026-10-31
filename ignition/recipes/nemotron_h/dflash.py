"""LiLiCorr context injection and bidirectional draft blocks."""

import torch

from ignition.components.dflash_draft import DraftForwards, PreparedContext as PreparedDraftContext, PreparedDraftBlock, Rotary
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state import save_hidden

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder, Scratch
from .lilicorr import PreparedLatticeSelector

def _rings(weights, pool):
    return tuple((pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value"))
                 for index in range(len(weights.draft.layers)))


class PreparedContext(PreparedDraftContext):
    """Paged adapter: derive token positions, rows, and liveness from `Metadata`, then save the anchor."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, buffers):
        self.metadata, self.pool = metadata, pool
        c = weights.config.draft
        super().__init__(weights.draft, auxiliary, _rings(weights, pool), window=c.text["sliding_window"],
            epsilon=c.text["rms_norm_eps"], rotary=Rotary(c, auxiliary.device),
            session=session, name=name, workspace=workspace, buffers=buffers)
        self.anchor = pool.reserved_region("draft_anchor")

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        super().__call__(auxiliary, m.positions, m.state_rows[m.requests], live)
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, self.normalized.shape[0] - 1)
        save_hidden(self.normalized[last], self.anchor, m.state_rows, counts > 0)


class PreparedDraft(PreparedDraftBlock):
    """Draft block followed by the anchor copy and the LiLiCorr lattice selector."""

    def __init__(self, weights, scratch, metadata, pool, *, vocabulary_runtime, coordinator,
                 session, name, workspace, buffers, layer_buffers, runtime=None):
        super().__init__(weights, scratch, metadata, _rings(weights, pool), page_tokens=PAGE_TOKENS,
            vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=name, workspace=workspace, layer_buffers=layer_buffers)
        c = weights.config.draft
        self.anchor = buffers.zeros((metadata.lengths.numel(), weights.config.hidden), torch.bfloat16)
        self.saved_anchor = pool.reserved_region("draft_anchor")
        self.selector = PreparedLatticeSelector(weights.draft.lattice, c, weights.embedding,
            scratch.hidden.view(-1, c.block_size, weights.config.hidden), scratch.logits, self.anchor,
            session=session, name=name + ".selector", workspace=workspace, buffers=buffers, layer_buffers=layer_buffers)

    def __call__(self):
        super().__call__()
        c = self.weights.config.draft
        self.anchor.copy_(self.saved_anchor[self.metadata.state_rows])
        self.selector.prepare(self.scratch.hidden.view(-1, c.block_size, self.weights.config.hidden),
            self.scratch.logits.view(-1, c.block_size, self.weights.config.vocabulary), self.anchor,
            (self.metadata.starts > 0) & (self.metadata.lengths > 0))


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, *, completion, frame, sequence,
                 session, workspace, coordinator, buffers, layer_buffers):
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, None
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=coordinator)
        self.markov_runtime = None
        self.buffers, self.layer_buffers = buffers, layer_buffers
        self.cursor = None
        self.forwards = DraftForwards(self.decoder, self.context, self.draft)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = None

    def begin(self, batch, width, target_token_capacity):
        self.cursor = self.buffers.cursor()
        self.scratch = Scratch.allocate(target_token_capacity, batch * width, self.weights.config, self.cursor)
        self.draft_scratch = Scratch.allocate(batch * self.block_size,
            batch * self.block_size, self.weights.config, self.cursor)

    def decoder(self, weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
        return PreparedDecoder(weights, scratch, metadata, pool, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)

    def context(self, *args, **kwargs):
        return PreparedContext(*args, buffers=self.cursor, **kwargs)

    def draft(self, *args, **kwargs):
        return PreparedDraft(*args, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)
