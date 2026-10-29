"""Shared-round history copies: substitution gating and the commit's offered-draft bound."""

import pytest
import torch

from tests.components.completion_support import RecordOnlyLogprobs, completion_outputs

from ignition.components.parallel_draft import allocate_copy_buffers, substitute_copies
from ignition.components.speculation import PreparedSpeculativeCommit, SpeculativeAcceptance
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission

VOCAB = 32


@pytest.fixture(autouse=True)
def records_only_cpu_commit(monkeypatch):
    # CPU record tests; the logprob reduction has independent CUDA coverage.
    monkeypatch.setattr("ignition.components.speculation.PreparedLogprobs", RecordOnlyLogprobs)


def round_tensors(batch, width, token=21):
    """Drafter output: every position proposes ``token`` with a smeared distribution."""
    drafts = torch.full((batch, width - 1), token, dtype=torch.int64)
    probabilities = torch.full((batch, width - 1, VOCAB), 1 / VOCAB, dtype=torch.float32)
    valid = torch.ones((batch, width - 1), dtype=torch.bool)
    return drafts, probabilities, valid, torch.full((batch,), -1, dtype=torch.int64)


def substitute(width, *, buffers, slots, active, before, limits, generation):
    drafts, probabilities, valid, mask = round_tensors(slots.numel(), width)
    reference = tuple(tensor.clone() for tensor in (drafts, probabilities, valid))
    substitute_copies(drafts, probabilities, valid, mask, active=active, before=before, limits=limits,
        slots=slots, generation=generation, copy_ids=buffers["copy_token_ids"],
        copy_count=buffers["copy_token_count"], copy_expected=buffers["copy_expected_consumed"],
        copy_generation=buffers["copy_generation"], vocab_ids=torch.arange(VOCAB))
    return drafts, probabilities, valid, mask, reference


def test_allocated_copy_buffers_are_inert():
    buffers = allocate_copy_buffers(4, "cpu")
    assert buffers["copy_token_ids"].shape == (4, 7) and buffers["copy_token_ids"].eq(-1).all()
    assert buffers["copy_token_count"].dtype == torch.int32 and not buffers["copy_token_count"].any()
    assert buffers["copy_expected_consumed"].eq(-1).all() and buffers["copy_generation"].dtype == torch.int64
    drafts, probabilities, valid, mask, reference = substitute(8, buffers=buffers,
        slots=torch.tensor([0, 1, 2, 3]), active=torch.ones(4, dtype=torch.bool), before=torch.zeros(4, dtype=torch.int64),
        limits=torch.full((4,), 8, dtype=torch.int32), generation=torch.zeros(4, dtype=torch.int64))
    assert mask.tolist() == [0] * 4
    for tensor, expected in zip((drafts, probabilities, valid), reference):
        assert tensor.equal(expected)


def test_substitution_gates_each_row_and_keeps_later_proposals():
    buffers = allocate_copy_buffers(8, "cpu")
    generation = torch.full((8,), 5, dtype=torch.int64)
    buffers["copy_generation"].fill_(5)
    buffers["copy_expected_consumed"].fill_(64)
    buffers["copy_token_ids"][:, :3] = torch.tensor([7, 8, 9])
    buffers["copy_token_count"].fill_(3)
    # Row 0 copies three; row 1 copies two (count); row 2 stale epoch; row 3 wrong consumed count;
    # row 4 out-of-vocabulary second copy; row 5 inactive; row 6 count zero; row 7 width budget 3.
    buffers["copy_token_count"][1] = 2
    buffers["copy_generation"][2] = 4
    buffers["copy_expected_consumed"][3] = 63
    buffers["copy_token_ids"][4, 1] = VOCAB
    buffers["copy_token_count"][6] = 0
    buffers["copy_token_count"][7] = 7
    buffers["copy_token_ids"][7] = torch.arange(1, 8)
    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    active = torch.tensor([True, True, True, True, True, False, True, True])
    limits = torch.tensor([8, 8, 8, 8, 8, 8, 8, 3], dtype=torch.int32)
    drafts, probabilities, valid, mask, (ref_drafts, ref_probs, ref_valid) = substitute(8, buffers=buffers,
        slots=slots, active=active, before=torch.full((8,), 64, dtype=torch.int64), limits=limits,
        generation=generation)
    assert mask.tolist() == [0b111, 0b11, 0, 0, 0b101, 0, 0, 0b11]
    assert drafts[0].tolist() == [7, 8, 9, 21, 21, 21, 21]
    assert drafts[1].tolist() == [7, 8, 21, 21, 21, 21, 21]
    assert drafts[4].tolist() == [7, 21, 9, 21, 21, 21, 21]
    assert drafts[7].tolist() == [1, 2, 21, 21, 21, 21, 21]
    for row in (2, 3, 5, 6):
        assert drafts[row].equal(ref_drafts[row]) and probabilities[row].equal(ref_probs[row])
        assert valid[row].equal(ref_valid[row])
    for row in range(8):
        for i in range(7):
            if mask[row] >> i & 1:
                assert probabilities[row, i].dtype == torch.float32
                assert probabilities[row, i].eq(1).sum() == 1 and probabilities[row, i, drafts[row, i]] == 1
                assert probabilities[row, i].sum() == 1 and valid[row, i]
            else:
                assert probabilities[row, i].equal(ref_probs[row, i]) and drafts[row, i] == ref_drafts[row, i]
                assert valid[row, i] == ref_valid[row, i]


def test_substitution_never_invalidates_a_drafter_position():
    buffers = allocate_copy_buffers(1, "cpu")
    buffers["copy_token_count"].fill_(2)
    buffers["copy_token_ids"][0, :2] = torch.tensor([3, 4])
    buffers["copy_expected_consumed"].fill_(10)
    drafts, probabilities, valid, mask = round_tensors(1, 4)
    valid[0, 2] = False
    substitute_copies(drafts, probabilities, valid, mask, active=torch.tensor([True]),
        before=torch.tensor([10]), limits=torch.tensor([4], dtype=torch.int32), slots=torch.tensor([0]),
        generation=torch.zeros(1, dtype=torch.int64), copy_ids=buffers["copy_token_ids"],
        copy_count=buffers["copy_token_count"], copy_expected=buffers["copy_expected_consumed"],
        copy_generation=buffers["copy_generation"], vocab_ids=torch.arange(VOCAB))
    assert mask.tolist() == [0b11] and valid[0].tolist() == [True, True, False]


def test_width_one_round_is_a_no_op():
    buffers = allocate_copy_buffers(2, "cpu")
    buffers["copy_token_count"].fill_(7)
    buffers["copy_token_ids"].fill_(1)
    buffers["copy_expected_consumed"].zero_()
    drafts, probabilities, valid, mask = round_tensors(2, 1)
    substitute_copies(drafts, probabilities, valid, mask, active=torch.ones(2, dtype=torch.bool),
        before=torch.zeros(2, dtype=torch.int64), limits=torch.ones(2, dtype=torch.int32),
        slots=torch.tensor([0, 1]), generation=torch.zeros(2, dtype=torch.int64), copy_ids=buffers["copy_token_ids"],
        copy_count=buffers["copy_token_count"], copy_expected=buffers["copy_expected_consumed"],
        copy_generation=buffers["copy_generation"], vocab_ids=torch.arange(VOCAB))
    assert drafts.shape == (2, 0) and probabilities.shape == (2, 0, VOCAB) and mask.tolist() == [0, 0]


@pytest.mark.parametrize("offered,expected", [(2, 0b11), (0, 0), (7, 0b1111111)])
def test_commit_bounds_copy_mask_by_offered_drafts(offered, expected):
    state = DeviceRequestState.allocate("cpu", vocab_size=16)
    state.admit(2, RequestAdmission(generation=7, seed=1, max_new_tokens=20, temperature=1, top_k=0, top_p=1,
        eos_token_ids=()), [1, 2], consumed_tokens=2)
    state.pending_token[2] = 3
    slots, epochs = torch.tensor([2], dtype=torch.int32), torch.tensor([7], dtype=torch.int64)
    live = torch.tensor([1], dtype=torch.int32)
    inputs = torch.arange(3, 11)[None, :].clone()
    acceptance = SpeculativeAcceptance.allocate(1, 8, "cpu")
    acceptance.token_ids[:, :1] = 4
    acceptance.proposed_count.fill_(1)
    acceptance.accepted_drafts.fill_(0)
    acceptance.offered_drafts.fill_(offered)
    acceptance.valid.fill_(True)
    records = torch.full((2, 4, 24), -99, dtype=torch.int64)
    frame, sequence = torch.tensor([1], dtype=torch.int32), torch.tensor([37], dtype=torch.int64)
    commit = PreparedSpeculativeCommit(state, acceptance, inputs, slots, epochs, live, completion_outputs(records),
        frame, sequence, logits=torch.zeros((1, 8, 16)), copy_offered_mask=torch.tensor([0b1111111]))
    commit()
    assert commit.emitted_count.item() == 1
    assert records[1, 2, 9].item() == offered and records[1, 2, 11].item() == expected
