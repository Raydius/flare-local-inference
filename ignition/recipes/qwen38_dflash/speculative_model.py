"""DFlash2 block drafts through the shared proposal-only round, verified by Qwen's hybrid-state verifier.

The shared ``PreparedParallelDraftRound`` owns the drafter path (block forward,
selector, proposal sampling, history copies, draft RNG); this module owns the
target side the paged round cannot: the ``PreparedVerificationState`` metadata,
the Qwen decoder with its feature taps, acceptance, commit, GDN rollback, and
the ring writes of accepted tokens' features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.candidate_selector import PreparedCandidateSelector
from ignition.components.dflash_draft import (
    DraftForwards, DraftScratch, PreparedContext as PreparedDraftContext, PreparedDraftBlock, Rotary,
)
from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit
from ignition.components.parallel_draft import (
    PreparedParallelDraftRound, allocate_copy_buffers, prepare_sampling_broadcasts, prepare_sampling_scratch,
)
from ignition.components.speculation import (
    PreparedSpeculativeCommit, PreparedSpeculativeDistributions, PreparedStochasticVerification,
    SpeculativeRngState,
)
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.qwen38.decoder import PreparedDecoder
from ignition.recipes.qwen38.speculative_state import PreparedVerificationState

PAGE_TOKENS = 64
# The target's per-stream width: the draft embeds, taps, and projects at this width.
HIDDEN = 2560
VOCABULARY = 248320


@dataclass(frozen=True)
class SpeculativeConfig:
    """What the draft block reads beside the draft's own configuration."""

    draft: object
    hidden: int = HIDDEN
    vocabulary: int = VOCABULARY


@dataclass(frozen=True)
class SpeculativeWeights:
    """The target's embedding and head, the DFlash2 draft, and both configurations.

    The shared round reads ``config.draft.noise_token`` and ``draft``; the draft
    block reads ``embedding``, ``head``, ``config.hidden``, ``draft.layers`` and
    ``draft.norm``; ``shard`` is the target's tensor-parallel shard.
    """

    embedding: torch.Tensor
    head: torch.Tensor
    draft: object
    config: SpeculativeConfig
    shard: object


def rings(config, pool):
    """One ``(key, value)`` E4M3 ring pair per draft layer, viewed over every state row."""
    return tuple((pool.region(f"dflash.{index}.key"), pool.region(f"dflash.{index}.value"))
                 for index in range(config.layers))


class PreparedAdmission:
    """The owner-supplied admission of the proposal-only round: validate, then the MTP round's active mask.

    A module-level object rather than a closure: compiled forwards are serialized to
    compiler workers, where a local function would be an unavailable process-local resource.
    """

    def __init__(self, owner, batch, validate):
        self.owner, self.batch, self.validate = owner, batch, validate
        self.rows = torch.arange(batch, device=owner.inputs.row_slot_ids.device)

    def __call__(self):
        self.validate()
        o, b = self.owner, self.batch
        slots = o.inputs.row_slot_ids[:b]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        return ((self.rows < o.inputs.live_rows[0]) & (slots >= 0) & (slots < o.requests.capacity)
                & (o.requests.generation[safe] == o.inputs.row_generations[:b]) & (o.requests.status[safe] == 1))


def prefill_feature_rows(owner, qsa, window):
    """Positions, state rows, and liveness of a prefill chunk's tapped features.

    Liveness is decided before any row index is taken, so dead tokens (request id
    ``-1``) never select a row; only the chunk's last ``window`` positions stay
    live, since earlier positions of one chunk would share their ring slots.
    """
    positions = qsa.query_positions
    live = positions >= 0
    live &= positions > positions.max() - window
    slots = owner.inputs.row_slot_ids[qsa.request_ids.clamp_min(0)].long().clamp(0, owner.requests.capacity - 1)
    rows = torch.where(live, owner.inputs.request_state_rows[slots, 0], -1)
    return positions, rows, live


def verification_feature_rows(owner, qsa, batch, width, counts):
    """Positions, state rows, and liveness of the accepted columns of a verification round."""
    columns = torch.arange(width, device=counts.device)
    positions = qsa.query_positions
    live = (columns[None, :] < counts[:, None]).reshape(-1) & (positions >= 0)
    safe = owner.inputs.row_slot_ids[:batch].long().clamp(0, owner.requests.capacity - 1)
    rows = owner.inputs.request_state_rows[safe, 0][:, None].expand(batch, width).reshape(-1)
    return positions, torch.where(live, rows, -1), live


class PreparedDraft(PreparedDraftBlock):
    """The component block followed by the predecessor-conditioned candidate selector."""

    def __init__(self, weights, scratch, metadata, pool, *, session, name, workspace, layer_buffers,
                 runtime=None, vocabulary_runtime=None, coordinator=None):
        c = weights.config.draft
        super().__init__(weights, scratch, metadata, rings(c, pool), page_tokens=PAGE_TOKENS,
            vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=name, workspace=workspace, layer_buffers=layer_buffers)
        self.selector = PreparedCandidateSelector(scratch.hidden.view(-1, c.block_size, weights.config.hidden),
            weights.draft.selector, top_k=c.top_k, session=session, name=name + ".selector")

    def __call__(self):
        super().__call__()
        c = self.weights.config.draft
        self.selector.prepare(self.scratch.hidden.view(-1, c.block_size, self.weights.config.hidden))


class SpeculativePreparation:
    """Shared persistent owners for every prepared DFlash2 graph capacity."""

    def __init__(self, *, target_weights, draft_weights, draft_config, pool, requests, inputs, scratch,
                 workspace, rotary, cache, table, session, completion, frame, sequence, max_cache_pages,
                 runtime=None, vocabulary_runtime=None, coordinator=None, verification_width=8):
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.target_weights, self.pool, self.requests, self.inputs = target_weights, pool, requests, inputs
        self.scratch, self.workspace, self.rotary = scratch, workspace, rotary
        self.cache, self.table, self.session = cache, table, session
        self.completion, self.frame, self.sequence, self.max_cache_pages = completion, frame, sequence, max_cache_pages
        device = pool.backing.device
        if type(verification_width) is not int or not 1 <= verification_width <= 8:
            raise ValueError("verification width must be between one and eight")
        self.verification_width = verification_width
        language = target_weights.tree.model.language_model
        # The draft head is the target's: it projects into the request vocabulary the round samples from.
        self.weights = SpeculativeWeights(language.embed_tokens.weight, target_weights.tree.lm_head.weight,
            draft_weights, SpeculativeConfig(draft_config, vocabulary=requests.vocab_size), target_weights.shard)
        self.feature_taps, self.window = draft_config.target_layers, draft_config.window
        self.feature_width = len(self.feature_taps) * HIDDEN
        self.block_size, self.logit_offset, self.page_tokens = draft_config.block_size, 1, PAGE_TOKENS
        # The shared round reads both; DFlash2 proposes without Markov drafts or PLE tables.
        self.markov_runtime = self.tables = None
        self.draft_rotary = Rotary(draft_config, device)
        # One cursor per graph: a prefill's tapped features and context scratch, or a round's draft
        # scratch, sampling scratch, and context scratch. Separate from the decoder scratch's pool,
        # so nothing here aliases storage the decoder writes inside the same graph.
        self.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        self.layer_buffers = GraphBufferPool(device)
        # Each round allocates its draft scratch from its own cursor before building the shared round.
        self.draft_scratch = None
        self.forwards = DraftForwards(None, self.context, self.draft)
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.copy_buffers = allocate_copy_buffers(requests.capacity, device)
        self.previous_accepted = torch.ones(requests.capacity, dtype=torch.int32, device=device)
        self.candidates = torch.zeros((requests.capacity, 8), dtype=torch.int64, device=device)
        # The verifier's feedback stores land here instead of MTP regions; nothing reads them back.
        # Rows are state rows, which a request's live row indexes beyond request capacity.
        rows = pool.backing.shape[0]
        self.feedback = (torch.zeros((rows, 10240), dtype=torch.bfloat16, device=device),
                         torch.zeros((rows, 3), dtype=torch.int64, device=device))
        heads = next(iter(pool.gdn.values())).recurrent.shape[1] if pool.gdn else 0
        self.gdn_records = (GDNLayerRecords.allocate(pool.gdn, requests.capacity, device,
                                                     key_heads=heads // 3, value_heads=heads)
                            if pool.gdn else None)
        self.owners, self.decoders = [], []

    def buffers(self):
        return dict(self.copy_buffers)

    def cursor(self):
        return self.sampling_buffers.cursor()

    def auxiliary(self, tokens, buffers):
        return buffers.zeros((tokens, self.feature_width), torch.bfloat16)

    def begin(self, batch, buffers):
        c = self.weights.config
        self.draft_scratch = DraftScratch.allocate(batch * self.block_size, batch * self.block_size,
                                                   c.hidden, c.vocabulary, buffers)

    def context(self, auxiliary, name, buffers):
        c = self.weights.config.draft
        return PreparedDraftContext(self.weights.draft, auxiliary, rings(c, self.pool), window=c.window,
            epsilon=c.eps, rotary=self.draft_rotary, session=self.session, name=name,
            workspace=self.workspace, buffers=buffers)

    def draft(self, weights, scratch, metadata, pool, *, session, name, workspace, **collectives):
        return PreparedDraft(weights, scratch, metadata, pool, session=session, name=name, workspace=workspace,
                             layer_buffers=self.layer_buffers, **collectives)

    def prefill(self, decoder, name, *, buffers=None):
        operation = PreparedPrefillContext(self, decoder, name, buffers=buffers)
        self.owners.append(operation)
        return operation

    def round(self, batch, width, name):
        operation = PreparedDFlashRound(self, batch, width, name)
        self.owners.append(operation)
        return operation

    def rebind_pages_for_preparation(self, cache):
        self.cache = cache


class PreparedPrefillContext:
    """Prefill through the tapped decoder, then write the chunk's features into the request's rings."""

    def __init__(self, owner, decoder, name, *, buffers=None):
        if decoder.auxiliary is None or decoder.feature_taps != owner.feature_taps:
            raise ValueError("the speculative prefill decoder must tap the draft's target layers")
        self.owner, self.decoder = owner, decoder
        buffers = owner.cursor() if buffers is None else buffers
        self.context = owner.context(decoder.auxiliary, name + ".context", buffers)

    def __call__(self):
        self.decoder()
        positions, rows, live = prefill_feature_rows(self.owner, self.decoder.state.qsa, self.owner.window)
        self.context(self.decoder.auxiliary, positions, rows, live)


class PreparedDFlashRound:
    """One graph: the shared proposal-only round, the Qwen verification decoder, acceptance, and commits."""

    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width, self.name = owner, batch, width, name
        if not 1 <= width <= owner.verification_width:
            raise ValueError("round exceeds prepared verification scratch capacity")
        o, b, w = owner, batch, width
        device = o.pool.backing.device
        self.token_limits = o.inputs.query_lengths[:b]
        self.state = PreparedVerificationState(o.inputs, o.requests, o.pool, o.candidates, o.previous_accepted,
                                               width=w, rows=b, token_limits=self.token_limits)
        buffers = o.cursor()
        self.auxiliary = o.auxiliary(b * w, buffers)
        self.decoder = PreparedDecoder(o.target_weights, o.scratch.rows(b * w, b * w), self.state, o.cache,
            o.rotary, o.table, o.workspace, o.completion, o.frame, o.sequence,
            session=o.session, name=f"{name}.target", max_cache_pages=o.max_cache_pages, verification=True,
            gdn_records=o.gdn_records if w > 1 else None, gdn_in_place=w == 1 and o.gdn_records is not None,
            feature_taps=o.feature_taps, auxiliary=self.auxiliary, **o.collectives)
        o.decoders.append(self.decoder)
        o.begin(b, buffers)
        self.round = PreparedParallelDraftRound(o, b, w, name, buffers=buffers, proposal_only=True,
            admission=PreparedAdmission(o, b, self.state._validate), draft_page_blocks=1)
        self.context = o.context(self.auxiliary, f"{name}.context", buffers)
        self.sampling_scratch = prepare_sampling_scratch(b, w, o.requests.vocab_size, device, buffers=buffers,
                                                         proposal=False, verification=True)
        self.buffers = buffers
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w, r = self.owner, self.batch, self.width, self.round
        slots, epochs = o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b]
        r.prepare()
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1),
            o.requests, slots, epochs, o.inputs.live_rows, r.drafts, workspace=self.sampling_scratch["target"])
        target.prepare()
        verify = PreparedStochasticVerification(r.probabilities, r.drafts, target.probabilities,
            o.requests, slots, epochs, o.inputs.live_rows, rng=o.rng,
            draft_valid=r.valid, target_valid=target.valid, token_limits=self.token_limits,
            workspace=self.sampling_scratch["verify"])
        verify.prepare()
        prepare_sampling_broadcasts((), verify, collectives=o.collectives, session=o.session,
                                    name=self.name, buffers=self.buffers)
        commit = PreparedSpeculativeCommit(o.requests, verify.output,
            self.state.token_ids.view(b, w), slots, epochs, o.inputs.live_rows,
            o.completion, o.frame, o.sequence, logits=self.decoder.scratch.logits.view(b, w, -1), rng=o.rng,
            copy_offered_mask=r.copy_mask, draft_context_lag=0, token_limits=self.token_limits)
        commit.prepare()
        # The feedback residual is the decoder's own; its stores land in the owner's private pair.
        self.state.prepare_commit(commit.emitted_count, commit.consumed_before,
                                  self.decoder.scratch.residual, self.decoder.scratch.sample_hidden,
                                  feedback=o.feedback)
        model_commit = tuple(self.state._commit)
        if o.gdn_records is not None and w > 1:
            heads = o.gdn_records.delta.shape[2]
            model_commit += (PreparedGDNCommit(o.pool.backing,
                {layer: state.recurrent for layer, state in o.pool.gdn.items()}, o.gdn_records, self.state.gdn,
                commit.emitted_count, key_heads=heads // 3, value_heads=heads),)
        self.operations = (target, verify, commit, model_commit)

    def __call__(self):
        o, b, w, r = self.owner, self.batch, self.width, self.round
        if self.operations is None:
            raise RuntimeError("speculative round prepare() must complete before capture")
        target, verify, commit, model_commit = self.operations
        r.propose()
        o.candidates[:b, :w].copy_(r.candidates)
        self.decoder()
        target()
        verify()
        commit()
        for operation in model_commit:
            operation()
        positions, rows, live = verification_feature_rows(o, self.state.qsa, b, w, commit.emitted_count)
        self.context(self.auxiliary, positions, rows, live)
