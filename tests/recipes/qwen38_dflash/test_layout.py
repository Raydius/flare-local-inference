"""DFlash2 draft-window regions in the Qwen hybrid row, and the MTP layouts kept byte-identical."""

import hashlib

import pytest
import torch

from ignition.recipes.qwen38.decoder import validate_feature_taps
from ignition.recipes.qwen38.state import HybridStateLayout, HybridStatePool

DRAFT_WINDOWS = (5, 2048, 8, 128)

# (speculative_tokens, world_size) -> (row_bytes, regions, draft_ring_capacity, digest of every region's
# name/offset/shape/dtype) captured from the layout before `mtp`/`draft_windows` existed.
FROZEN_MTP_LAYOUTS = {
    (0, 1): (116505344, 161, 0, "9ce057027f6bb76fa0e9b38fb3ae65e84cff0d52a2120907a66fde8261203e72"),
    (0, 2): (58383104, 161, 0, "ec4f329c41460c6f372850c0b21b135d3782b44a90c21cfbb65c600107ce5aee"),
    (0, 4): (29518592, 161, 0, "b2462f039d498777a1931a3932acd7792af4abfeacda0953bac312d049cb20fb"),
    (7, 1): (122014720, 173, 16, "0486cf6dcabe6b07333647380153531bc80e30c973053fac04ae093cb12b35f8"),
    (7, 2): (61246464, 173, 16, "4cf62089dc30f2d69ce97e1681bc7cf02c927204ec5b7a093051fccaa7ebf937"),
    (7, 4): (31091712, 173, 16, "39234ac93ff6c9d15ab5e0040fe768eeb0c35221a62f07abd9a9599b92c653b1"),
}


def digest(layout):
    regions = tuple((r.name, r.offset, r.shape, str(r.dtype)) for r in layout.regions)
    return hashlib.sha256(repr(regions).encode()).hexdigest()


def test_dflash_layout_carries_rings_and_no_mtp_regions():
    layout = HybridStateLayout.build(speculative_tokens=7, mtp=False, draft_windows=DRAFT_WINDOWS)
    assert layout.speculative_tokens == 7
    assert layout.draft_ring_capacity == 0
    assert layout.row_bytes % 256 == 0
    for layer in range(5):
        for field in ("key", "value"):
            region = layout.region(f"dflash.{layer}.{field}")
            assert region.shape == (2048, 8, 128)
            assert region.dtype == torch.float8_e4m3fn
    names = [r.name for r in layout.regions]
    assert names.index("dflash.0.key") > names.index("qsa.47.boundary_selector")
    assert not any(name.startswith("draft_") or name.startswith("target_residual") for name in names)
    with pytest.raises(KeyError):
        layout.region("target_residual")
    # The speculative widening of the histories the verifier needs stays.
    assert layout.region("gdn.0.convolution").shape == (10240, 10)
    assert layout.region("ple_convolution").shape == (10240, 16)
    # Ten E4M3 [2048, 8, 128] rings: exactly 20 MiB per row over the ringless speculative layout.
    assert layout.row_bytes == 142821120
    assert layout.row_bytes == HybridStateLayout.build(speculative_tokens=7, mtp=False).row_bytes + 20 * (1 << 20)


def test_pool_forwards_draft_windows_and_views_rings():
    pool = HybridStatePool.allocate(2, "cpu", speculative_tokens=7, mtp=False, draft_windows=DRAFT_WINDOWS)
    assert pool.layout.draft_ring_capacity == 0
    ring = pool.region("dflash.4.value")
    assert ring.shape == (2, 2048, 8, 128) and ring.dtype == torch.float8_e4m3fn
    assert ring.stride(0) == pool.layout.row_bytes
    with pytest.raises(KeyError):
        pool.region("draft_raw_keys")


@pytest.mark.parametrize("windows", [(5, 2048, 8), (0, 2048, 8, 128), (5, 2048.0, 8, 128)])
def test_invalid_draft_windows_reject(windows):
    with pytest.raises(ValueError):
        HybridStateLayout.build(speculative_tokens=7, mtp=False, draft_windows=windows)


@pytest.mark.parametrize(("speculative_tokens", "world_size"), sorted(FROZEN_MTP_LAYOUTS))
def test_default_layouts_are_byte_identical(speculative_tokens, world_size):
    layout = HybridStateLayout.build(speculative_tokens=speculative_tokens, world_size=world_size)
    row_bytes, regions, ring, frozen = FROZEN_MTP_LAYOUTS[speculative_tokens, world_size]
    assert (layout.row_bytes, len(layout.regions), layout.draft_ring_capacity) == (row_bytes, regions, ring)
    assert digest(layout) == frozen
    explicit = HybridStateLayout.build(speculative_tokens=speculative_tokens, world_size=world_size,
                                       mtp=True, draft_windows=None)
    assert explicit == layout


def test_mtp_false_without_speculation_matches_default():
    assert HybridStateLayout.build(mtp=False) == HybridStateLayout.build()


def test_feature_tap_auxiliary_validation():
    taps = (5, 15, 25, 35, 45)
    device = torch.device("cpu")
    validate_feature_taps((), None, 16, device)
    validate_feature_taps(taps, torch.zeros(16, 12800, dtype=torch.bfloat16), 16, device)
    for auxiliary in (torch.zeros(16, 12800), torch.zeros(8, 12800, dtype=torch.bfloat16),
                      torch.zeros(16, 2560, dtype=torch.bfloat16), torch.zeros(16, 12800, dtype=torch.bfloat16, device="meta")):
        with pytest.raises(ValueError):
            validate_feature_taps(taps, auxiliary, 16, device)
    with pytest.raises(ValueError):
        validate_feature_taps(taps, None, 16, device)
    with pytest.raises(ValueError):
        validate_feature_taps((), torch.zeros(16, 0, dtype=torch.bfloat16), 16, device)
    for bad in ((5, 5), (48,), (-1,), (5.0,)):
        with pytest.raises(ValueError):
            validate_feature_taps(bad, torch.zeros(16, 2560 * len(bad), dtype=torch.bfloat16), 16, device)
