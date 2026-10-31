"""Proposal-only shared round: owner-supplied admission and draft page-table width."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import parallel_draft
from ignition.components.paged_batch import BatchInputs, PreparedBatch
from ignition.components.parallel import TensorParallelShard
from ignition.components.parallel_draft import PreparedParallelDraftRound, allocate_copy_buffers
from ignition.components.sampling import DeviceRequestState, RequestAdmission

BATCH, WIDTH, VOCAB, BLOCK, BLOCKS, CAPACITY = 2, 3, 64, 4, 8, 4
DRAFT_TOKEN = 7


class ArgmaxProposal:
    """CPU stand-in for the CUDA draft sampler: greedy over the bound (else the given) logits, exact one-hot."""

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return {}

    def __init__(self, logits, requests, slots, generations, live_rows, drafts, *, prefix_offset, seed, offset,
                 sampled, workspace=None):
        self.sampled, self.source_logits, self.source_bias = sampled, None, None
        self.probabilities = torch.zeros((logits.shape[0], 1, logits.shape[-1]), dtype=torch.float32)
        self.valid = torch.zeros((logits.shape[0], 1), dtype=torch.bool)
        self.logits = logits

    def bind_logits(self, logits, bias=None):
        self.source_logits, self.source_bias = logits, bias

    def prepare(self):
        pass

    def prepare_broadcast(self, *, runtime, session, coordinator, name, buffers=None):
        pass

    def __call__(self):
        source = self.logits[:, 0] if self.source_logits is None else self.source_logits
        self.sampled.copy_(source.argmax(1))
        self.probabilities.zero_()
        self.probabilities[:, 0].scatter_(1, self.sampled[:, None], 1.0)
        self.valid.fill_(True)


class _Scratch:
    def rows(self, tokens, samples):
        return SimpleNamespace(hidden=torch.zeros((samples, 4)), logits=torch.zeros((samples, VOCAB)))


class _Draft:
    """Stub drafter: every block position proposes ``DRAFT_TOKEN``."""

    def __init__(self, scratch, metadata):
        self.scratch, self.metadata = scratch, metadata

    def __call__(self):
        self.scratch.logits.fill_(-1e4)
        self.scratch.logits[:, DRAFT_TOKEN] = 10.0


@pytest.fixture(autouse=True)
def cpu_proposals(monkeypatch):
    monkeypatch.setattr(parallel_draft, "PreparedDraftDistribution", ArgmaxProposal)


def owner(*, page_capacity):
    state = DeviceRequestState.allocate("cpu", vocab_size=VOCAB, capacity=CAPACITY)
    for slot in range(BATCH):
        state.admit(slot, RequestAdmission(generation=11 + slot, seed=41800 + slot, max_new_tokens=100, top_k=-1,
                                           top_p=1.0), [4, 5, 4])
    state.pending_token.fill_(4)
    state.consumed_tokens.fill_(10)
    inputs = BatchInputs.allocate(BATCH, BLOCKS, "cpu", capacity=CAPACITY)
    inputs.row_slot_ids.copy_(torch.tensor([1, 0, -1, -1], dtype=torch.int32))
    inputs.row_generations.copy_(state.generation[inputs.row_slot_ids.long().clamp_min(0)])
    inputs.live_rows.fill_(BATCH)
    inputs.request_state_rows[:, 0].copy_(torch.arange(1, CAPACITY + 1))
    inputs.page_tables.copy_(torch.arange(CAPACITY * BLOCKS, dtype=torch.int32).view(CAPACITY, BLOCKS))
    inputs.query_lengths.fill_(WIDTH)
    return SimpleNamespace(
        requests=state, inputs=inputs, offset=torch.zeros(CAPACITY, dtype=torch.int64),
        generation=state.generation.clone(),
        weights=SimpleNamespace(shard=TensorParallelShard(0, 1), draft=None,
                                config=SimpleNamespace(draft=SimpleNamespace(noise_token=0))),
        pool=SimpleNamespace(memory_capacity=torch.tensor([16, page_capacity]), backing=torch.empty(1)),
        collectives=dict(runtime=None, vocabulary_runtime=None, coordinator=None), session=None, workspace=None,
        forwards=SimpleNamespace(decoder=None, context=None, draft=lambda w, s, m, p, **k: _Draft(s, m), markov=None),
        draft_scratch=_Scratch(), block_size=BLOCK, logit_offset=1, page_tokens=64, markov_runtime=None,
        tables=None, rng=None, copy_buffers=allocate_copy_buffers(CAPACITY, "cpu"))


def test_admission_replaces_the_paged_validator():
    # Page capacity 0 rejects every row under the paged validator; the owner's admission decides instead.
    o = owner(page_capacity=0)
    calls = []

    def admission():
        calls.append(len(calls))
        return torch.tensor([True, False])

    # Row 0 (slot 1) is offered one copy for consumed count 10.
    c = o.copy_buffers
    c["copy_token_ids"][1, 0] = 9
    c["copy_token_count"][1] = 1
    c["copy_expected_consumed"][1] = 10
    c["copy_generation"][1] = o.requests.generation[1]
    round = PreparedParallelDraftRound(o, BATCH, WIDTH, "test.admission", proposal_only=True, admission=admission,
                                       draft_page_blocks=1)
    assert round.validator is None and round.validation is None and round.metadata is None
    round.prepare()
    round.propose()
    assert calls == [0]
    assert round.active.tolist() == [True, False]
    assert round.drafts.tolist() == [[9, DRAFT_TOKEN], [DRAFT_TOKEN, DRAFT_TOKEN]]
    assert round.valid.tolist() == [[True, True], [False, False]]
    assert round.copy_mask.tolist() == [0b1, 0]
    assert round.candidates.tolist() == [[4, 9, DRAFT_TOKEN], [4, DRAFT_TOKEN, DRAFT_TOKEN]]
    assert round.probabilities[0, 0].eq(1).sum() == 1 and round.probabilities[0, 0, 9] == 1
    assert round.draft_metadata.page_table.shape == (BATCH, 1)
    assert round.draft_metadata.page_table.tolist() == [[8], [0]]
    assert round.draft_metadata.lengths.tolist() == [BLOCK, 0]
    assert round.draft_metadata.starts.tolist() == [10, 10]
    assert o.requests.status.tolist()[:BATCH] == [1, 1]


class _RecordingBatch:
    """Stands in for the CUDA-only paged validator: records its geometry and admits rows 0 and 1."""

    instances = []

    def __init__(self, inputs, requests, metadata, *, mode, shard, pool, verification_width=1,
                 target_token_capacity=None, page_tokens=None, buffers=None):
        self.metadata, self.mode, self.verification_width = metadata, mode, verification_width
        self.target_token_capacity, self.page_tokens = target_token_capacity, page_tokens
        self.validation_input = torch.zeros((1, 2 * requests.capacity), dtype=torch.int32)
        self.validation_sum = torch.zeros_like(self.validation_input)
        self.token_state = SimpleNamespace(active=torch.zeros(metadata.starts.numel(), dtype=torch.int32))
        self.calls = []
        _RecordingBatch.instances.append(self)

    def validate(self):
        self.calls.append("validate")
        return self.validation_input

    def begin(self, totals):
        self.calls.append("begin")
        self.token_state.active.fill_(1)


def test_defaults_keep_the_validator_and_full_page_tables(monkeypatch):
    monkeypatch.setattr(parallel_draft, "PreparedBatch", _RecordingBatch)
    _RecordingBatch.instances.clear()
    o = owner(page_capacity=CAPACITY * BLOCKS)
    for arguments in ({}, dict(admission=None, draft_page_blocks=None)):
        round = PreparedParallelDraftRound(o, BATCH, WIDTH, "test.default", proposal_only=True, **arguments)
        validator = _RecordingBatch.instances[-1]
        assert round.validator is validator and round.validation is not None and round.validation.single
        assert validator.metadata.page_table.shape == (BATCH, BLOCKS)
        assert validator.metadata.token_ids.numel() == BATCH and validator.metadata.starts.numel() == BATCH
        assert (validator.mode, validator.verification_width, validator.target_token_capacity, validator.page_tokens
                ) == ("decode", WIDTH, BATCH * WIDTH, o.page_tokens)
        assert round.draft_page_blocks == BLOCKS
        assert round.draft_metadata.page_table.shape == (BATCH, BLOCKS)
        assert round.metadata is None
        round.prepare()
        round.propose()
        assert validator.calls == ["validate", "begin"]
        assert round.active.tolist() == [True, True]
        assert round.drafts.tolist() == [[DRAFT_TOKEN] * (WIDTH - 1)] * BATCH
        assert round.valid.tolist() == [[True, True]] * BATCH
        assert round.draft_metadata.page_table.tolist() == o.inputs.page_tables[[1, 0]].tolist()


def test_target_round_keeps_its_metadata(monkeypatch):
    monkeypatch.setattr(parallel_draft, "PreparedBatch", _RecordingBatch)
    o = owner(page_capacity=CAPACITY * BLOCKS)
    o.scratch = _Scratch()
    o.forwards = SimpleNamespace(
        decoder=lambda w, t, s, m, p, **k: SimpleNamespace(auxiliary=None, scratch=s, metadata=m),
        context=lambda *a, **k: None, draft=lambda w, s, m, p, **k: _Draft(s, m), markov=None)
    round = PreparedParallelDraftRound(o, BATCH, WIDTH, "test.target")
    assert round.metadata is round.decoder.metadata
    assert round.metadata.page_table.shape == (BATCH, BLOCKS)
    assert round.metadata.token_ids.numel() == BATCH * WIDTH and round.metadata.max_query_tokens == WIDTH
    assert round.draft_metadata.page_table.shape == (BATCH, BLOCKS)
