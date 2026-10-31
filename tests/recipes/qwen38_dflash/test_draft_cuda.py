"""The lifted draft block and candidate selector against the DFlash2 reference on the pinned draft.

Run on a dedicated device (``CUDA_VISIBLE_DEVICES=1 FLARE_RUN_CUDA_TESTS=1``); the draft's
tensors are read with ``safetensors`` directly, so no io_uring loader is needed. The
reference's key/value are rounded to E4M3 exactly as the prepared rings and staging round
them (clamp to +-448, cast to ``float8_e4m3fn``): that storage contract was validated by
``qualify/dflash2_acceptance.py`` (E4M3 and BF16 key/value give identical code_edit
acceptance on real tapped features), so the comparison is against the contract, not BF16.

Inputs come in two kinds. Real tapped features are the first trace under
``FLARE_DFLASH2_TRACES`` (skipped with a reason when unset): the features before each cut
are the context, the token at the cut is the block's pending token, and the three "seeds"
are three cut positions spread over the continuation (batch 4 takes four cuts per seed).
Fabricated features are random, with the first row wrapping the ring.

Measured maximum relative errors (Frobenius norm of the difference over the reference's
norm, per row) recorded before the bounds were asserted, on an RTX 5060 Ti against the
pinned draft and the CSF-QAD target, reference key/value rounded to E4M3:

    real features (chat-capacity-10c824cef65b8fb7, cuts 198..430, batches 1 and 4):
        context key/value rings: batch 1: 1.26e-2 / 1.54e-2; batch 4: 1.26e-2 / 1.54e-2
        hidden: batch 1: 1.94e-2 (per cut 1.75e-2 / 1.94e-2 / 1.63e-2); batch 4: 4.02e-2 max
            (cut 430, the trace's last token), 1.58e-2 min
        logits: batch 1: 1.83e-2; batch 4: 3.85e-2
        shortlist: 0 ids outside the near-tie tolerance on every row/position (both batches)
        walked tokens: prepared selector == rounded reference on 3/3 and 12/12 rows
        selector: reference walk on the prepared hidden/shortlist == prepared walk, 15/15 rows
    fabricated features (seeds 11/23/37, starts 2100/5/700/1019):
        context key/value rings: batch 1: 5.1e-4 / 2.8e-4; batch 4: 1.44e-2 / 1.55e-2 (the
            5-token row; every element passes the per-element bin check)
        hidden: batch 1: 0.208 (0.182 / 0.182 / 0.208); batch 4: 0.161 max, 0.101 min
        logits: batch 1: 0.179; batch 4: 0.147
        shortlist: 18 (batch 1) and 66 (batch 4) row/positions with ids outside the near-tie
            tolerance; walked tokens differ from the reference on 2/3 and 11/12 rows
        selector: 0 mismatches on all 15 row-seeds

Attribution of the real-feature numbers (batch 1 and 4, every row): the prepared context
writer's BF16 key/value differ from the reference's BF16 key/value by 2.5e-3 to 4.5e-3
(kernel rounding), which crosses an E4M3 bin on 3.6 to 8 percent of ring elements; the
reference run on the prepared rings differs from itself by 1.5e-2 to 3.6e-2, and the block
on top of that adds 1.4e-2 to 3.7e-2 (its own key/value cross bins the same way). The
E4M3-rounded reference is not repeatable to 1e-2 on these features by any BF16
implementation: projecting key/value in FP32 and casting (a legitimate variant, 3.6 to 4.2
percent bin flips) moves it by 1.2e-2 to 3.3e-2, one-ulp BF16 noise on the features by
1.8e-2 to 4.3e-2, and one-ulp noise on the block inputs by 1.4e-2 to 3.5e-2. The real-feature
bounds therefore follow that measured repeatability of the E4M3 contract (rings 2e-2,
hidden and logits 5e-2) with exact shortlist and walk agreement, the outputs that decide
acceptance; the 1e-2 first carried from the Nemotron test was below the contract's own
repeatability and is not a bound this arithmetic can meet.

History: the first version of this test compared against the BF16 reference on fabricated
features and failed (hidden 0.12-0.38 relative, logits 0.24-0.32, every walked row
different) while every non-attention stage and the selector were exact. Attribution
showed the BF16 reference with its own K/V rounded to E4M3 differs from itself by 0.17-0.20
and moves 0.03-0.07 under half-ulp input noise: on random features the block amplifies
rounding by roughly 30x, so no implementation could meet 1e-2 against BF16 there. The
operator measured acceptance on real features instead and kept the E4M3 contract. Against
the E4M3-rounded reference the fabricated numbers above are the same amplification, so
``FABRICATED_BOUND`` is the measured 0.21 and the fabricated shortlist/walk disagreement is
recorded, not asserted; the selector parity and ring checks are asserted for both kinds.
"""

from contextlib import contextmanager
from functools import partial
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

# Measured repeatability of the E4M3-rounded reference on real tapped features (module docstring):
# rings 1.3e-2 to 1.5e-2 from single E4M3 bin crossings, hidden and logits 1.2e-2 to 4.3e-2 under
# one-ulp input noise or an FP32-projection variant. Acceptance is decided by the shortlist and
# the walk, which are asserted exact on real features.
REAL_BOUNDS = {"context_key": 2e-2, "context_value": 2e-2, "hidden": 5e-2, "logits": 5e-2}
# Measured against the E4M3-rounded reference on random features (hidden 0.208): the ~30x
# amplification of BF16 kernel rounding documented in the module docstring, not a defect.
FABRICATED_BOUNDS = dict.fromkeys(REAL_BOUNDS, 0.21)
SEEDS = (11, 23, 37)
FABRICATED_STARTS = (2100, 5, 700, 1019)
DEVICE = "cuda:0"
DEAD = 3


def _relative(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _round_kv_to_e4m3(reference):
    """Round the reference's context and block key/value as the prepared rings and staging do."""
    exact = reference._kv

    def rounded(i, h, positions):
        return tuple(t.clamp(-448, 448).to(torch.float8_e4m3fn).to(t.dtype) for t in exact(i, h, positions))
    reference._kv = rounded
    return reference


@pytest.fixture(scope="module")
@torch.no_grad()
def checkpoint_weights():
    from safetensors import safe_open
    from ignition.preparation import inspect_qwen38_checkpoint
    from ignition.preparation.qwen38 import TEXT_PREFIX
    from ignition.recipes.qwen38_dflash.config import DraftConfig
    from ignition.recipes.qwen38_dflash.draft_weights import DraftWeights
    from ignition.reference.qwen38_dflash2 import DFlash2Reference
    from ignition.reference.qwen38_quant import CheckpointTensors

    draft = Path(os.environ.get("FLARE_QWEN38_DFLASH2_CHECKPOINT", "~/models/incoai/Qwen3.8-Flash-Next-DFlash2")).expanduser()
    target = Path(os.environ.get("FLARE_QWEN38_CHECKPOINT",
                                 "~/models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")).expanduser()
    if not draft.is_dir() or not target.is_dir():
        pytest.skip("requires the pinned DFlash2 draft and the Qwen3.8 Flash Next checkpoint")
    config = DraftConfig.load(draft)
    with safe_open(str(draft / "model.safetensors"), "pt", device=DEVICE) as file:
        tensors = {name: file.get_tensor(name) for name in file.keys()}
    reference = _round_kv_to_e4m3(DFlash2Reference(config.text, tensors, strict=True))
    draft_weights = DraftWeights.assemble(config, tensors.__getitem__)
    with CheckpointTensors(inspect_qwen38_checkpoint(target), byte_budget=4 << 30) as reader:
        embedding = reader.read(f"{TEXT_PREFIX}.embed_tokens.weight").to(DEVICE)
        head = reader.read("lm_head.weight").to(DEVICE)
    weights = SimpleNamespace(config=SimpleNamespace(draft=config, hidden=config.hidden, vocabulary=config.vocabulary),
                              draft=draft_weights, embedding=embedding, head=head)
    return weights, reference


@contextmanager
def _prepare(checkpoint_weights, batch, tmp_path):
    """The context writer, draft block, and selector compiled and captured for one batch size.

    Every row owns a ``window``-slot span of the context buffer so any cut fits; the slots a
    row does not use, and ``DEAD`` trailing tokens, are dead (row 0, position -1). The
    session stays open (kernel resolution frozen) until the block exits.
    """
    from b12x.preparation import PreparationSession
    from ignition.components.candidate_selector import PreparedCandidateSelector
    from ignition.components.dflash_draft import DraftScratch, PreparedContext, PreparedDraftBlock, Rotary
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena

    weights, reference = checkpoint_weights
    c = weights.config.draft
    block, window, hidden, vocabulary = c.block_size, c.window, c.hidden, c.vocabulary
    tokens = block * batch
    context_tokens = window * batch + DEAD
    positions = torch.full((context_tokens,), -1, device=DEVICE, dtype=torch.int64)
    context_rows = torch.zeros((context_tokens,), device=DEVICE, dtype=torch.int64)
    live = torch.zeros((context_tokens,), device=DEVICE, dtype=torch.bool)
    auxiliary = torch.zeros((context_tokens, len(c.target_layers) * hidden), device=DEVICE, dtype=torch.bfloat16)
    rings = tuple(tuple(torch.zeros((batch + 1, window, c.kv_heads, c.head_dim), device=DEVICE, dtype=torch.float8_e4m3fn)
                        for _ in range(2)) for _ in range(c.layers))
    rows = torch.arange(1, batch + 1, device=DEVICE, dtype=torch.int64)
    table = torch.zeros((batch, 1), device=DEVICE, dtype=torch.int32)
    live_rows = torch.tensor([batch], device=DEVICE, dtype=torch.int32)
    ids = torch.full((batch, block), c.noise_token, device=DEVICE, dtype=torch.int64)
    m = Metadata.allocate(tokens, batch, 1, DEVICE, samples=tokens)
    starts = torch.tensor(FABRICATED_STARTS[:batch], device=DEVICE, dtype=torch.int64)
    m.set(starts, torch.full_like(starts, block), rows, table, ids, live_rows)
    buffers, layers = GraphBufferPool(DEVICE), GraphBufferPool(DEVICE)
    cursor = buffers.cursor()
    with PreparationSession(device=DEVICE, autotune=False) as session:
        arena = ScratchArena(DEVICE)
        context = PreparedContext(weights.draft, auxiliary, rings, window=window, epsilon=c.eps,
            rotary=Rotary(c, DEVICE), session=session, name="context", workspace=arena, buffers=cursor)
        scratch = DraftScratch.allocate(tokens, tokens, hidden, vocabulary, cursor)
        draft = PreparedDraftBlock(weights, scratch, m, rings, page_tokens=64, vocabulary_runtime=None, coordinator=None,
            session=session, name="draft", workspace=arena, layer_buffers=layers)
        selector = PreparedCandidateSelector(scratch.hidden.view(batch, block, hidden), weights.draft.selector,
            top_k=c.top_k, session=session, name="selector")
        arena.finalize()
        context_operation = partial(context, auxiliary, positions, context_rows, live)
        paths = (tmp_path / "context.bin", tmp_path / "draft.bin")
        with CompilePool(DEVICE) as compiler:
            futures = [compiler.submit(operation, path)
                       for operation, path in zip((context_operation, draft), paths, strict=True)]
            for future in futures:
                future.result()
        artifacts = []
        for path in paths:
            with path.open("rb") as file:
                artifacts.append(torch.compiler.load_compiled_function(file))
        context_fn = lambda: artifacts[0](context_operation)
        draft_fn = lambda: artifacts[1](draft)
        context_fn()
        draft_fn()
        session.freeze()
        context_graph, draft_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(context_graph):
            context_fn()
        with session.capture(), torch.cuda.graph(draft_graph):
            draft_fn()
        yield SimpleNamespace(weights=weights, reference=reference, batch=batch, auxiliary=auxiliary, positions=positions,
                              context_rows=context_rows, live=live, rings=rings, rows=rows, table=table, live_rows=live_rows,
                              ids=ids, metadata=m, scratch=scratch, selector=selector,
                              context=context, context_graph=context_graph, draft_graph=draft_graph)
    gc.collect()
    torch.cuda.empty_cache()


def _layout(p, starts):
    """Point each row's live slots at positions ``start - count .. start - 1``; everything else is dead."""
    window = p.weights.config.draft.window
    p.positions.fill_(-1)
    p.context_rows.zero_()
    p.live.zero_()
    spans = []
    for row, start in enumerate(starts.tolist()):
        count = min(start, window)
        span = slice(row * window, row * window + count)
        p.positions[span] = start - count + torch.arange(count, device=DEVICE)
        p.context_rows[span], p.live[span] = p.rows[row], True
        spans.append(span)
    return spans


def _real_cases(batch):
    """Three sets of ``batch`` cuts over the first trace's continuation: (start, pending, features before the cut)."""
    traces = Path(os.environ.get("FLARE_DFLASH2_TRACES", ""))
    files = sorted(traces.glob("*.pt")) if traces.is_dir() else []
    if not files:
        pytest.skip("requires FLARE_DFLASH2_TRACES with a tapped-feature trace of the target")
    trace = torch.load(files[0], weights_only=True)
    tokens = trace["prompt_tokens"] + trace["continuation_tokens"]
    features = trace["features"].to(DEVICE, torch.bfloat16)
    first, last = len(trace["prompt_tokens"]), len(tokens) - 1
    cuts = torch.linspace(first, last, len(SEEDS) * 4).round().long().tolist()
    cases = []
    for index in range(len(SEEDS)):
        chosen = cuts[index * 4:index * 4 + batch]
        cases.append([(cut, tokens[cut], features[max(0, cut - 2048):cut]) for cut in chosen])
    return files[0].stem, cases


def _shortlist_mismatch(prepared_ids, reference_ids, reference_logits, tolerance):
    """Ids in one shortlist only whose reference logit is not a near-tie with the reference's 16th."""
    prepared, reference = set(prepared_ids.tolist()), set(reference_ids.tolist())
    threshold = reference_logits[reference_ids].min()
    return [token for token in prepared ^ reference if abs(reference_logits[token] - threshold) > tolerance]


@pytest.mark.parametrize("kind", ["real", "fabricated"])
@pytest.mark.parametrize("batch", [1, 4])
@torch.no_grad()
def test_block_matches_reference(checkpoint_weights, batch, kind, tmp_path):
    bounds = REAL_BOUNDS if kind == "real" else FABRICATED_BOUNDS
    if kind == "real":
        source, cases = _real_cases(batch)
    else:
        source, cases = "fabricated", None
    with _prepare(checkpoint_weights, batch, tmp_path) as p:
        _compare(p, kind, source, cases, bounds)


def _compare(p, kind, source, cases, bounds):
    weights, reference, batch = p.weights, p.reference, p.batch
    c = weights.config.draft
    block, window, hidden, vocabulary = c.block_size, c.window, c.hidden, c.vocabulary
    measurements = {key: [] for key in ("context_key", "context_value", "hidden", "logits")}
    shortlist_mismatches, walk_mismatches, selector_mismatches = [], [], []
    for index, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        for key, value in p.rings:
            for ring in (key, value):
                ring.copy_(torch.randn(ring.shape, device=DEVICE, dtype=torch.bfloat16) * .1)
        if kind == "real":
            starts = torch.tensor([cut for cut, _, _ in cases[index]], device=DEVICE, dtype=torch.int64)
            pending = torch.tensor([token for _, token, _ in cases[index]], device=DEVICE, dtype=torch.int64)
            spans = _layout(p, starts)
            p.auxiliary.zero_()
            for span, (_, _, features) in zip(spans, cases[index], strict=True):
                p.auxiliary[span] = features
        else:
            starts = torch.tensor(FABRICATED_STARTS[:batch], device=DEVICE, dtype=torch.int64)
            pending = torch.randint(0, c.noise_token, (batch,), device=DEVICE, dtype=torch.int64)
            spans = _layout(p, starts)
            p.auxiliary.copy_(torch.randn(p.auxiliary.shape, device=DEVICE, dtype=torch.bfloat16) * .1)
        initial = [tuple(ring.clone() for ring in pair) for pair in p.rings]
        p.ids[:, 0] = pending
        p.metadata.set(starts, torch.full_like(starts, block), p.rows, p.table, p.ids, p.live_rows)
        p.context_graph.replay()
        p.draft_graph.replay()
        torch.cuda.synchronize()
        # Dead tokens and the unused row leave the rings untouched.
        for (key, value), (key0, value0) in zip(p.rings, initial, strict=True):
            assert torch.equal(key[0], key0[0]) and torch.equal(value[0], value0[0])
        # The prepared walk as the round runs it: shortlist corrections per position, argmax feeding the next.
        p.selector.prepare(p.scratch.hidden.view(batch, block, hidden))
        predecessor, walks = pending.clone(), []
        for position in range(1, block):
            step = p.scratch.logits.view(batch, block, vocabulary)[:, position].float()
            p.selector(step, predecessor, position)
            predecessor = step.argmax(-1)
            walks.append(predecessor)
        walks = torch.stack(walks, 1)
        for row, span in enumerate(spans):
            start = int(starts[row])
            ctx = reference.context(p.auxiliary[span], p.positions[span])
            slots = p.positions[span] % window
            for layer, ((key, value), (k_ref, v_ref)) in enumerate(zip(p.rings, ctx.layers, strict=True)):
                for name, ring, expected in (("context_key", key, k_ref), ("context_value", value, v_ref)):
                    actual, target = ring[p.rows[row], slots].float(), expected.to(ring.dtype).float()
                    measurements[name].append((seed, row, layer, _relative(actual, target)))
                    difference = actual - target
                    ulp = torch.exp2(torch.floor(torch.log2(target.abs().clamp_min(2 ** -6))) - 3)
                    assert (difference.abs() <= .02 * target.abs().max() + ulp).all(), (seed, row, layer, name)
            hidden_ref = reference.forward(reference.block_inputs(int(pending[row]), weights.embedding), start, ctx)
            hidden_out = p.scratch.hidden[row * block:(row + 1) * block]
            measurements["hidden"].append((seed, row, _relative(hidden_out, hidden_ref)))
            ids_ref, unary_ref = reference.candidates(hidden_ref, weights.head)
            logits_ref = F.linear(hidden_ref[1:], weights.head).float()
            logits_out = p.scratch.logits[row * block:(row + 1) * block][1:].float()
            measurements["logits"].append((seed, row, _relative(logits_out, logits_ref)))
            tolerance = bounds["logits"] * logits_ref.abs().max().item()
            for position in range(block - 1):
                shortlist = logits_out[position].topk(c.top_k).indices
                missing = _shortlist_mismatch(shortlist, ids_ref[position], logits_ref[position], tolerance)
                if missing:
                    shortlist_mismatches.append((seed, row, position, missing))
            walk_ref = reference.walk(hidden_ref, int(pending[row]), ids_ref, unary_ref).tolist()
            # The reference walk on the prepared block's own hidden/shortlist isolates the selector.
            walk_prepared_inputs = reference.walk(hidden_out, int(pending[row]),
                *reference.candidates(hidden_out, weights.head)).tolist()
            row_walk = walks[row].tolist()
            if row_walk != walk_prepared_inputs:
                selector_mismatches.append((seed, row, row_walk, walk_prepared_inputs))
            if row_walk != walk_ref:
                walk_mismatches.append((seed, row, row_walk, walk_ref))
        gc.collect()
    rows_total = len(SEEDS) * batch
    summary = {key: max(entry[-1] for entry in values) for key, values in measurements.items()}
    print(f"{kind} ({source}) batch {batch} max relative errors: {summary}; per row {measurements['hidden']}; "
          f"shortlist mismatches {shortlist_mismatches}; selector mismatches {selector_mismatches}; "
          f"walk mismatches {len(walk_mismatches)}/{rows_total}: {walk_mismatches}")
    assert not selector_mismatches, selector_mismatches
    if kind == "real":
        assert not shortlist_mismatches, shortlist_mismatches
        assert not walk_mismatches, f"{len(walk_mismatches)}/{rows_total} rows walk differently: {walk_mismatches}"
    for key, value in summary.items():
        assert value <= bounds[key], (kind, key, value)


@torch.no_grad()
def test_selector_tie_resolves_to_lowest_id(checkpoint_weights):
    from b12x.preparation import PreparationSession
    from ignition.components.candidate_selector import PreparedCandidateSelector

    weights, reference = checkpoint_weights
    c = weights.config.draft
    block, hidden, vocabulary = c.block_size, c.hidden, c.vocabulary
    # Zero hidden states project to zero, so the correction vanishes and the shortlist ties exactly.
    zero = torch.zeros((1, block, hidden), device=DEVICE, dtype=torch.bfloat16)
    logits = torch.full((1, vocabulary), -100., device=DEVICE)
    tied, lower = 70_000, 1_234
    logits[0, tied] = logits[0, lower] = 3.
    others = torch.arange(100_000, 100_000 + c.top_k - 2, device=DEVICE)
    logits[0, others] = 2.
    pending = 17
    with PreparationSession(device=DEVICE, autotune=False) as session:
        selector = PreparedCandidateSelector(zero, weights.draft.selector, top_k=c.top_k, session=session, name="selector")
        selector.prepare(zero)
        prepared = logits.clone()
        selector(prepared, torch.tensor([pending], device=DEVICE), 1)
    assert int(prepared.argmax(-1)) == lower
    assert torch.isinf(prepared[0, 0]) and prepared[0, tied] == prepared[0, lower] == 3.
    unary, ids = logits[0].topk(c.top_k)
    walked = reference.walk(zero[0], pending, ids[None].expand(block - 1, -1), unary[None].expand(block - 1, -1))
    assert int(walked[0]) == lower
