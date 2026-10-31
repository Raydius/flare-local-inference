"""The DFlash2 owner drives the shared proposal-only round with Qwen's admission, on CPU.

``PreparedDraftDistribution.prepare()`` and the paged validator need CUDA, so
the proposal sampler is the greedy stand-in of the component test and the
admission is the owner's; the round code itself runs unchanged. The target
decoder and verification state need CUDA and are not built here.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ignition.components.dflash_draft import DraftForwards, ring_slots
from ignition.components.parallel_draft import PreparedParallelDraftRound
from ignition.reference.qwen38_dflash2 import PINNED
from ignition.recipes.qwen38.decoder_state import DecoderInputs
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.recipes.qwen38.state import HybridStatePool
from ignition.recipes.qwen38_dflash.config import DraftConfig
from ignition.recipes.qwen38_dflash.recipe import DRAFT_WINDOWS
from ignition.recipes.qwen38_dflash.speculative_model import (
    PreparedAdmission, SpeculativePreparation, prefill_feature_rows, verification_feature_rows,
)
from tests.components.test_parallel_draft_admission import cpu_proposals  # noqa: F401 (autouse fixture)

BATCH, WIDTH, VOCAB, CAPACITY, BLOCK = 2, 4, 64, 2, 8
DRAFT_TOKEN = 7


class _Draft:
    """Stub drafter with a selector hook: every block position proposes ``DRAFT_TOKEN``."""

    def __init__(self, scratch, metadata):
        self.scratch, self.metadata = scratch, metadata
        self.selector = self.select
        self.selected = []

    def select(self, logits, predecessor, position):
        self.selected.append((predecessor.clone(), position))

    def __call__(self):
        self.scratch.logits.fill_(-1e4)
        self.scratch.logits[:, DRAFT_TOKEN] = 10.0


@pytest.fixture(scope="module")
def pool():
    return HybridStatePool.allocate(CAPACITY + 1, "cpu", speculative_tokens=7, mtp=False, draft_windows=DRAFT_WINDOWS)


def owner(pool):
    requests = DeviceRequestState.allocate("cpu", vocab_size=VOCAB, capacity=CAPACITY)
    for slot in range(CAPACITY):
        requests.admit(slot, RequestAdmission(generation=11 + slot, seed=41800 + slot, max_new_tokens=100, top_k=-1,
                                              top_p=1.0, eos_token_ids=()), [4, 5, 4])
    requests.pending_token.fill_(4)
    requests.consumed_tokens.fill_(10)
    inputs = DecoderInputs.allocate("cpu", context=256, tokens=32, capacity=CAPACITY)
    inputs.row_slot_ids.copy_(torch.tensor([1, 0], dtype=torch.int32))
    inputs.row_generations.copy_(requests.generation[inputs.row_slot_ids.long()])
    inputs.live_rows.fill_(BATCH)
    inputs.request_state_rows[:, 0].copy_(torch.arange(1, CAPACITY + 1))
    inputs.query_lengths.fill_(WIDTH)
    target = SimpleNamespace(shard=None, tree=SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(VOCAB, 2560)))),
        lm_head=SimpleNamespace(weight=torch.zeros(VOCAB, 2560))))
    return SpeculativePreparation(target_weights=target, draft_weights=None, draft_config=DraftConfig.from_dict(deepcopy(PINNED)),
        pool=pool, requests=requests, inputs=inputs, scratch=None, workspace=None, rotary=None, cache=None,
        table=None, session=None, completion=None, frame=None, sequence=None, max_cache_pages=None)


def proposal_round(o, width, validate, drafts):
    """The shared round exactly as ``PreparedDFlashRound`` builds it, with a stub drafter."""
    o.forwards = DraftForwards(None, o.context, lambda w, s, m, p, **k: drafts.append(_Draft(s, m)) or drafts[-1])
    cursor = o.cursor()
    o.begin(BATCH, cursor)
    return PreparedParallelDraftRound(o, BATCH, width, "test.round", buffers=cursor, proposal_only=True,
                                      admission=PreparedAdmission(o, BATCH, validate), draft_page_blocks=1)


def test_owner_round_substitutes_copies_under_the_qwen_admission(pool):
    o = owner(pool)
    assert o.block_size == BLOCK and o.logit_offset == 1 and o.page_tokens == 64 and o.markov_runtime is None
    assert o.weights.config.draft.noise_token == PINNED["dflash_config"]["mask_token_id"]
    assert o.feedback[0].shape == (CAPACITY + 1, 10240) and o.feedback[1].shape == (CAPACITY + 1, 3)
    assert o.candidates.shape == (CAPACITY, 8) and o.previous_accepted.eq(1).all()
    assert set(o.buffers()) == {"copy_token_ids", "copy_token_count", "copy_expected_consumed", "copy_generation"}
    # Slot 0 (row 1) is finished; row 0 (slot 1) is a short lane offered three copies for consumed count 10.
    o.requests.status[0] = 2
    o.inputs.query_lengths[0] = 2
    c = o.copy_buffers
    c["copy_token_ids"][1, :3] = torch.tensor([9, 8, 6])
    c["copy_token_count"][1] = 3
    c["copy_expected_consumed"][1] = 10
    c["copy_generation"][1] = o.requests.generation[1]
    validations, drafts = [], []
    round = proposal_round(o, WIDTH, lambda: validations.append(True), drafts)
    assert round.validator is None and round.validation is None and round.metadata is None
    assert round.draft is drafts[0] and round.selector == drafts[0].select
    assert o.draft_scratch.logits.shape == (BATCH * BLOCK, VOCAB)
    round.prepare()
    round.propose()
    assert validations == [True]
    assert round.active.tolist() == [True, False]
    # The short lane offers one draft: the first copy is substituted and the mask stays below 1 << offered.
    assert round.copy_mask.tolist() == [0b1, 0] and round.copy_mask[0] < 1 << (o.inputs.query_lengths[0] - 1)
    assert round.drafts.tolist() == [[9, DRAFT_TOKEN, DRAFT_TOKEN], [DRAFT_TOKEN] * 3]
    assert round.valid.tolist() == [[True, False, False], [False, False, False]]
    assert round.candidates.tolist() == [[4, 9, DRAFT_TOKEN, DRAFT_TOKEN], [4, DRAFT_TOKEN, DRAFT_TOKEN, DRAFT_TOKEN]]
    assert round.probabilities[0, 0].eq(1).sum() == 1 and round.probabilities[0, 0, 9] == 1
    # The selector saw the predecessor of every position after the pending token.
    assert [position for _, position in drafts[0].selected] == [1, 2, 3]
    assert drafts[0].selected[0][0].tolist() == [4, 0]
    # The drafter's ring window sits at each request's consumed count over one page-table column.
    assert round.draft_metadata.starts.tolist() == [10, 10] and round.draft_metadata.lengths.tolist() == [BLOCK, 0]
    assert round.draft_metadata.page_table.shape == (BATCH, 1)
    # The inactive row's request state is untouched.
    assert o.requests.status.tolist() == [2, 1] and o.offset.tolist() == [0, 4 * (WIDTH - 1)]


def test_width_one_round_proposes_nothing(pool):
    o = owner(pool)
    drafts = []
    round = proposal_round(o, 1, lambda: None, drafts)
    assert round.draft is None and drafts == []
    round.prepare()
    round.propose()
    assert round.active.tolist() == [True, True]
    assert round.candidates.tolist() == [[4], [4]] and round.drafts.shape == (BATCH, 0)
    assert round.copy_mask.tolist() == [0, 0]


def test_prefill_features_keep_the_window_and_leave_dead_tokens_unslotted(pool):
    o = owner(pool)
    o.inputs.row_slot_ids.copy_(torch.tensor([1, -1], dtype=torch.int32))
    window = 4
    qsa = SimpleNamespace(request_ids=torch.tensor([0, 0, 0, 0, 0, 0, -1, -1]),
                          query_positions=torch.tensor([0, 1, 2, 3, 4, 5, -1, -1]))
    positions, rows, live = prefill_feature_rows(o, qsa, window)
    assert live.tolist() == [False, False, True, True, True, True, False, False]
    assert rows.tolist() == [-1, -1, 2, 2, 2, 2, -1, -1]
    slots = ring_slots(live, rows, positions, window)
    assert slots.tolist() == [-1, -1, 2 * window + 2, 2 * window + 3, 2 * window + 0, 2 * window + 1, -1, -1]
    assert len(set(slots[live].tolist())) == window


def test_verification_features_cover_the_emitted_columns_only(pool):
    o = owner(pool)
    counts = torch.tensor([2, 0], dtype=torch.int32)
    qsa = SimpleNamespace(query_positions=torch.tensor([10, 11, 12, 13, -1, -1, -1, -1]))
    positions, rows, live = verification_feature_rows(o, qsa, BATCH, WIDTH, counts)
    assert live.tolist() == [True, True, False, False, False, False, False, False]
    assert rows.tolist() == [2, 2, -1, -1, -1, -1, -1, -1]
    assert ring_slots(live, rows, positions, 2048).tolist() == [2 * 2048 + 10, 2 * 2048 + 11, -1, -1, -1, -1, -1, -1]
