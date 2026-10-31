"""Disposable DFlash2 storage cleared before publishing a runtime package."""

from ignition.preparation.disposable import storage_ranges
from ignition.recipes.qwen38.export import disposable_ranges as qwen_disposable_ranges


def disposable_ranges(*, cache, scratch, workspaces, vision_scratch, decoders, owners, speculation):
    ranges = qwen_disposable_ranges(cache=cache, scratch=scratch, workspaces=workspaces,
        vision_scratch=vision_scratch, decoders=decoders, owners=owners, speculation=None)
    if speculation is None:
        return ranges
    from .speculative_model import PreparedDFlashRound

    # Per-graph cursors (tapped features, context scratch, draft scratch, sampling scratch), the draft
    # layers' intermediates, and the verifier's unread feedback stores.
    values = [speculation.sampling_buffers.chunks, speculation.layer_buffers.chunks, speculation.feedback]
    for operation in speculation.owners:
        if not isinstance(operation, PreparedDFlashRound):
            continue
        shared = operation.round
        # ``draft_inputs`` keeps the noise token in its mask columns and stays.
        values.extend(getattr(shared, name) for name in (
            "before", "active", "candidates", "drafts", "copy_mask", "seed", "offset", "base_offset", "sampled",
            "valid", "markov_token", "logits", "probabilities"))
        if shared.selector is not None:
            values.append(shared.selector.projected)
        _, proposals, _, _ = shared.operations
        target, verify, _, _ = operation.operations
        for distribution in (target, *proposals):
            values.extend(getattr(distribution, name) for name in (
                "scores", "probabilities", "maximum", "valid", "packed_top_k", "packed_top_p"))
        for proposal in proposals:
            values.extend((proposal.proposal.candidates, proposal.proposal.selected))
        values.extend(getattr(verify, name) for name in (
            "safe_draft_probs", "safe_target_probs", "safe_ids", "seed_bits", "offset",
            "independent_accepted", "emitted_drafts", "output"))
    return (*ranges, *storage_ranges(values))
