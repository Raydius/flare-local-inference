"""Compare the prepared DFlash2 feature tap against a canonical trace on one prompt.

Prepares the qwen38_dflash catalog with the compiled decoder, prefills one traced prompt
through the speculative prefill graph, and compares the tapped ``[tokens, 12800]`` feature
buffer (the four-stream residual mean after layers 5, 15, 25, 35, 45) against the trace
written by ``qualify/dflash2_acceptance.py trace``. Also checks the ring liveness the
prefill context computed for the prompt's rows.
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from qualify.mtp import device
from qualify.provenance import plain, write_json

SNAPSHOT = Path('/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD')
HUB_CACHE = Path('~/.cache/huggingface/hub').expanduser()
TAPS = (5, 15, 25, 35, 45)
WIDTH = 2560
WINDOW = 2048
PAGE_TOKENS = 64


# pure parts: column slicing, per-layer errors, ring liveness, JSON assembly

def tap_columns(layer):
    """The feature-buffer column block of one tapped layer, in trace column order."""
    if layer not in TAPS:
        raise ValueError(f'layer {layer} is not a tapped layer')
    tap = TAPS.index(layer)
    return slice(tap * WIDTH, (tap + 1) * WIDTH)


def layer_errors(auxiliary, features):
    """Per tapped layer: Frobenius relative error over all rows and the max absolute element difference."""
    if auxiliary.shape != features.shape or auxiliary.ndim != 2 or auxiliary.shape[1] != len(TAPS) * WIDTH:
        raise ValueError(f'expected two [tokens, {len(TAPS) * WIDTH}] feature tensors, '
                         f'got {list(auxiliary.shape)} and {list(features.shape)}')
    result = {}
    for layer in TAPS:
        columns = tap_columns(layer)
        actual, reference = auxiliary[:, columns].double().cpu(), features[:, columns].double().cpu()
        difference = actual - reference
        norm = reference.norm().item()
        result[str(layer)] = {'max_relative_error': difference.norm().item() / norm if norm > 0 else 0.0,
                              'max_abs_difference': difference.abs().max().item()}
    return result


def ring_liveness(slots, prompt_tokens, window=WINDOW):
    """Check the context's ring slots for a prompt: live (>= 0) exactly on its last ``window`` positions."""
    slots = slots[:prompt_tokens].cpu()
    live = slots >= 0
    expected_live = min(prompt_tokens, window)
    first_expected = prompt_tokens - expected_live
    expected = torch.arange(prompt_tokens) >= first_expected
    live_positions = live.nonzero().flatten().tolist()
    return {'live': int(live.sum()), 'expected_live': expected_live,
            'first_live_position': live_positions[0] if live_positions else None,
            'first_expected_position': first_expected,
            'matches': bool(torch.equal(live, expected))}


def tap_report(trace, errors, ring, *, prepared_model_id, prepared_revision):
    """Assemble the JSON document the tool writes and prints."""
    return {'trace': f"{trace['case']}-{trace['digest']}", 'prompt_tokens': len(trace['prompt_tokens']),
            'layers': errors, 'ring_slots': ring,
            'model_id': prepared_model_id, 'revision': prepared_revision,
            'trace_model_id': trace['model_id'], 'trace_revision': trace['revision']}


def prefill_graph(capacities, prompt_tokens):
    """The speculative prefill graph that fits the prompt: ``prefill{capacity}`` without the ``-no-mtp`` suffix."""
    fitting = [capacity for capacity in capacities if capacity >= prompt_tokens]
    if not fitting:
        raise ValueError(f'no prepared prefill capacity holds {prompt_tokens} prompt tokens')
    return f'prefill{min(fitting)}'


# GPU-only: preparation, prefill replay, readback

def draft_snapshot(args):
    """The pinned DFlash2 draft, resolved as the recipe's ``prepare`` resolves it."""
    if args.draft is not None:
        return args.draft
    from huggingface_hub import snapshot_download

    from ignition.recipes.qwen38_dflash.config import DRAFT_REPOSITORY, DRAFT_REVISION

    return Path(snapshot_download(DRAFT_REPOSITORY, revision=DRAFT_REVISION, cache_dir=args.hub_cache,
                                  allow_patterns=('config.json', 'model.safetensors')))


def stage_ple_rows(table):
    """Stage the compact PLE rows native serving delivers, as the prompt-logprobs harness does."""
    from ignition.recipes.qwen38.ple import ROWS_PER_PLANE

    ids = table.hashes.cpu().flatten().tolist()
    with ExitStack() as files:
        opened = {}
        for destination, sources, width in ((table.weights, table.sources[0], 80),
                                            (table.scales, table.sources[1], 10)):
            packed = bytearray(len(ids) * width)
            for index, row in enumerate(ids):
                if row < 0:
                    continue
                plane, local = divmod(row, ROWS_PER_PLANE)
                source = sources[plane]
                if source.path not in opened:
                    opened[source.path] = files.enter_context(source.path.open('rb', buffering=0))
                file = opened[source.path]
                file.seek(source.offset + local * width)
                value = file.read(width)
                if len(value) != width:
                    raise RuntimeError(f'short PLE row {row} in {source.path}')
                packed[index * width:(index + 1) * width] = value
            destination[:len(ids)].copy_(torch.frombuffer(packed, dtype=torch.uint8).view(len(ids), width))
    table.ready.record()


def fill_prefill(inputs, requests, state, tokens, generation):
    """Admit the prompt on row 0 and fill the decoder inputs for one prefill chunk."""
    from ignition.recipes.qwen38.sampling import RequestAdmission

    state.reset_for_preparation()
    requests.admit(0, RequestAdmission(generation, 1, max_new_tokens=1, top_k=1), tokens)
    inputs.request_state_rows.zero_()
    inputs.request_state_rows[0, 0] = 1
    inputs.row_slot_ids.fill_(-1)
    inputs.row_slot_ids[0] = 0
    inputs.row_generations[0] = generation
    inputs.live_rows.fill_(1)
    inputs.query_lengths[0] = len(tokens)
    inputs.sample_enabled.zero_()
    inputs.token_ids[:len(tokens)] = torch.tensor(tokens, device='cuda')
    inputs.prefill_positions[:len(tokens)] = torch.arange(len(tokens), device='cuda')[:, None].expand(-1, 3)
    inputs.vision_token_indices.fill_(-1)
    pages = -(-len(tokens) // PAGE_TOKENS)
    inputs.page_tables[0, :pages] = torch.arange(1, pages + 1, device='cuda', dtype=inputs.page_tables.dtype)


@torch.no_grad()
def run(args):
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.ple import PLETable
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.state import HybridStatePool
    from ignition.recipes.qwen38_dflash.bootstrap import PreparationConfig, prepare_model, state_layout
    from ignition.recipes.qwen38_dflash.recipe import RECIPE
    from ignition.recipes.qwen38_dflash.speculative_model import PreparedPrefillContext

    # The report names the target this recipe declares (what `ignition export qwen3.8-flash-dflash2`
    # prepares), not qwen38's nvidia pin.
    target = RECIPE.checkpoints[0]
    device()
    traces = [torch.load(path, map_location='cpu', weights_only=True) for path in args.trace]
    spans = []
    for trace in traces:
        # With a dump folder the whole teacher-forced sequence is prefilled, as the acceptance trace was.
        tokens = list(trace['prompt_tokens']) + (list(trace['continuation_tokens']) if args.dump_features else [])
        if trace['features'].shape[0] < len(tokens):
            raise ValueError('trace features hold fewer rows than the prefilled tokens')
        if args.prefill_chunk_size < len(tokens) or args.context < len(tokens):
            raise ValueError(f'context and prefill-chunk-size must hold the {len(tokens)}-token sequence')
        spans.append(tokens)
    config = PreparationConfig(draft_checkpoint=draft_snapshot(args), max_active=1, context=args.context,
        prefill_chunk_size=args.prefill_chunk_size, pages=args.pages, retain_graphs=True,
        snapshot_budget_bytes=max(args.snapshot_budget_bytes, state_layout(True).row_bytes),
        compile_workers=args.compile_workers, kernel_cache=args.kernel_cache)
    reports, generation = [], 0
    with PackageStorage(world=1) as storage:
        with prepare_model(args.snapshot, config, compile_decoder=not args.no_compile, package_storage=storage,
                           progress=lambda message: print(message, flush=True)) as model:
            def owner(kind):
                return next(value for value in model.resources if isinstance(value, kind))
            state, requests, inputs = owner(HybridStatePool), owner(DeviceRequestState), owner(DecoderInputs)
            for trace, tokens in zip(traces, spans, strict=True):
                name = prefill_graph(config.prefill_capacities, len(tokens))
                # This recipe retains no ``forwards`` map; the captured callables sit in the resources tuple.
                forward = next(value[name] for value in model.resources
                               if isinstance(value, dict) and isinstance(value.get(name), PreparedPrefillContext))
                graph = model.graphs[name]
                # The first replay resolves the PLE rows the prompt needs; the second runs with them staged.
                for _ in range(2):
                    generation += 1
                    fill_prefill(inputs, requests, state, tokens, generation=generation)
                    graph.replay()
                    torch.cuda.synchronize()
                    stage_ple_rows(owner(PLETable))
                auxiliary = forward.decoder.auxiliary[:len(tokens)].cpu().clone()
                slots = forward.context.slots[:len(tokens)].cpu().clone()
                errors = layer_errors(auxiliary, trace['features'][:len(tokens)])
                ring = ring_liveness(slots, len(tokens))
                report = tap_report(trace, errors, ring, prepared_model_id=target.repository,
                                    prepared_revision=target.revision)
                report['prefilled_tokens'] = len(tokens)
                print(json.dumps(plain(report)), flush=True)
                reports.append(report)
                if args.dump_features:
                    args.dump_features.mkdir(parents=True, exist_ok=True)
                    torch.save({**trace, 'features': auxiliary.contiguous(), 'features_source': 'prepared'},
                               args.dump_features / f"{trace['case']}-{trace['digest']}.pt")
    return reports[0] if len(reports) == 1 else {'traces': reports}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True, nargs='+', help='trace .pt files written by dflash2_acceptance trace')
    parser.add_argument('--dump-features', type=Path, default=None,
                        help='write each trace with the prepared tap features over prompt and continuation into this folder')
    parser.add_argument('--output', type=Path, required=True, help='JSON report path')
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--draft', type=Path, default=None,
                        help='DFlash2 draft directory (default: the pinned hub-cache snapshot)')
    parser.add_argument('--hub-cache', type=Path, default=HUB_CACHE)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--prefill-chunk-size', type=int, default=256)
    parser.add_argument('--pages', type=int, default=None)
    parser.add_argument('--snapshot-budget-bytes', type=int, default=0)
    parser.add_argument('--compile-workers', type=int, default=None)
    parser.add_argument('--kernel-cache', type=Path, default=Path('~/.cache/flare/kernels-qwen').expanduser())
    parser.add_argument('--no-compile', action='store_true',
                        help='prepare the decoder without Inductor compilation (eager prepared operators)')
    return parser


def main():
    args = build_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(plain(result), indent=2))


if __name__ == '__main__':
    main()
