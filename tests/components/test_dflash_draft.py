"""Ring slot rule of the shared DFlash draft context."""

import torch

from ignition.components.dflash_draft import ring_slots


def test_ring_slots_wrap_live_tokens_and_blank_dead_ones():
    window = 4
    positions = torch.tensor([0, 5, 9, -1, 7])
    # Dead tokens carry request id -1, which Python indexing wraps to the last row.
    rows = torch.tensor([2, 2, 0, 3, 1])
    live = torch.tensor([True, True, True, False, False])
    slots = ring_slots(live, rows, positions, window)
    assert slots.tolist() == [2 * 4 + 0, 2 * 4 + 1, 0 * 4 + 1, -1, -1]
