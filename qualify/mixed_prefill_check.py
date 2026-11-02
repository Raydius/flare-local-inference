"""Compare a mixed graph's prompt sampling against the prefill graph's on one prompt.

Prepares a small qwen38_dflash catalog with mixed rounds (two requests, one prefill
chunk), then runs the same prompt alone through each prefill graph and each mixed graph
whose prompt capacity holds it, with no decode rows live. For every graph it reports the
token the prompt sampler emitted, the top ids of the prompt's logit row, and that row's
relative error against the plain prefill graph's. A mixed graph whose prompt row differs
from the prefill graph's is the served fault "fixed garbage token then EOS".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qualify.dflash2_tap_check import HUB_CACHE, draft_snapshot, fill_prefill, stage_ple_rows
from qualify.mtp import device
from qualify.provenance import plain, write_json

SNAPSHOT = Path('~/models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD').expanduser()
PROMPT = 'The three primary colors are red, blue and'


def graphs_for(names, prompt_tokens):
    """The prefill and mixed graphs whose prompt capacity holds the prompt, smallest capacity first."""
    chosen = []
    for name in names:
        if name.startswith('prefill'):
            capacity = int(name[len('prefill'):].split('-')[0])
        elif name.startswith('mixed'):
            capacity = int(name.split('p')[1].split('-')[0])
        else:
            continue
        if capacity >= prompt_tokens:
            chosen.append((capacity, name))
    return [name for _, name in sorted(chosen)]


def fill(inputs, requests, state, tokens, generation, *, prompt_row):
    """Admit the prompt on row 0 as the only live request; ``prompt_row`` marks it for a mixed graph."""
    inputs.query_lengths.zero_()
    inputs.page_tables.zero_()
    fill_prefill(inputs, requests, state, tokens, generation, max_new_tokens=4)
    inputs.sample_enabled[0] = 1
    inputs.prefill_row.fill_(prompt_row)


def prompt_row(function):
    """The logit row and sample-hidden row the prompt sampler of this graph reads."""
    from ignition.recipes.qwen38.mixed_model import PreparedMixedRound
    from ignition.recipes.qwen38_dflash.speculative_model import PreparedPrefillContext

    if isinstance(function, PreparedMixedRound):
        scratch = function.decoder.scratch
        return scratch.logits[-1], scratch.sample_hidden[-1]
    decoder = function.decoder if isinstance(function, PreparedPrefillContext) else function
    decoder = getattr(decoder, 'forward', decoder)
    return decoder.scratch.logits[0], decoder.scratch.sample_hidden[0]


def norms(tensor, count):
    return [round(v, 3) for v in tensor[:count].float().norm(dim=-1).tolist()]


def trace(function, pool, count):
    """Where the prompt's hidden row goes missing: packing masks, role metadata, and per-row norms."""
    from ignition.recipes.qwen38.mixed_model import PreparedMixedRound
    from ignition.recipes.qwen38_dflash.speculative_model import PreparedPrefillContext

    out = {'last_hidden_row1_norm': pool.last_hidden[1].float().norm().item()}
    if isinstance(function, PreparedMixedRound):
        m, p, d = function.packing, function.prompt_state, function.decoder
        st, sc = d.state, d.scratch
        out.update(prompt_valid=int(m.prompt_valid.sum()), prompt_indices=m.prompt_indices[:count].tolist(),
                   cu=m.cu.tolist(), is_prompt=int(m.is_prompt.sum()), row_ids=m.row_ids[:count].tolist(),
                   prompt_qsa_request_ids=p.qsa.request_ids[:count].tolist(), prompt_cu=p.gdn.query_start_loc.tolist(),
                   prompt_active=p.token_state.active.tolist(), prompt_live_rows=p.inputs.live_rows.tolist(),
                   prompt_slot=p.inputs.row_slot_ids[:2].tolist(), prompt_query_lengths=p.inputs.query_lengths[:2].tolist(),
                   mixed_qsa_request_ids=st.qsa.request_ids[:count].tolist(), mixed_token_ids=st.token_ids[:count].tolist(),
                   mixed_cu=st.gdn.query_start_loc.tolist(), mixed_num_tokens=st.gdn.num_tokens.tolist(),
                   embedding=norms(sc.embedding, count), residual=norms(sc.residual, count), hidden=norms(sc.hidden, count),
                   prompt_hidden=norms(st.prompt_hidden, count), sample_hidden=norms(sc.sample_hidden, sc.sample_hidden.shape[0]),
                   decode_hidden=norms(st.decode_hidden, st.decode_hidden.shape[0]))
    else:
        decoder = function.decoder if isinstance(function, PreparedPrefillContext) else function
        decoder = getattr(decoder, 'forward', decoder)
        sc, st = decoder.scratch, decoder.state
        out.update(qsa_request_ids=st.qsa.request_ids[:count].tolist(), cu=st.gdn.query_start_loc.tolist(),
                   active=st.token_state.active.tolist(), embedding=norms(sc.embedding, count),
                   residual=norms(sc.residual, count), hidden=norms(sc.hidden, count),
                   sample_hidden=norms(sc.sample_hidden, sc.sample_hidden.shape[0]))
    return out


@torch.no_grad()
def run(args):
    from transformers import AutoTokenizer

    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.ple import PLETable
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.state import HybridStatePool
    from ignition.recipes.qwen38_dflash.bootstrap import PreparationConfig, prepare_model, state_layout

    device()
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot))
    tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    config = PreparationConfig(draft_checkpoint=draft_snapshot(args), max_active=2, mixed_prefill=True,
        context=args.context, prefill_chunk_size=args.prefill_chunk_size, pages=args.pages, retain_graphs=True,
        snapshot_budget_bytes=state_layout(True).row_bytes, compile_workers=args.compile_workers,
        kernel_cache=args.kernel_cache)
    results, generation = {}, 0
    with PackageStorage(world=1) as storage:
        with prepare_model(args.snapshot, config, compile_decoder=not args.no_compile, package_storage=storage,
                           progress=lambda message: print(message, flush=True)) as model:
            def owner(kind):
                return next(value for value in model.resources if isinstance(value, kind))
            state, requests, inputs = owner(HybridStatePool), owner(DeviceRequestState), owner(DecoderInputs)
            functions = next(value for value in model.resources
                             if isinstance(value, dict) and any(isinstance(key, str) and key.startswith('prefill') for key in value))
            names = graphs_for(functions, len(tokens))
            print('graphs:', names, flush=True)
            reference = None
            for name in names:
                mixed = name.startswith('mixed')
                for _ in range(2):
                    generation += 1
                    fill(inputs, requests, state, tokens, generation, prompt_row=0 if mixed else -1)
                    model.graphs[name].replay()
                    torch.cuda.synchronize()
                    stage_ple_rows(owner(PLETable))
                logits, hidden = prompt_row(functions[name])
                logits, hidden = logits.float().cpu().clone(), hidden.float().cpu().clone()
                sampled = int(requests.pending_token[0])
                top = logits.topk(5).indices.tolist()
                entry = {'sampled': sampled, 'sampled_text': tokenizer.decode([sampled]) if sampled >= 0 else None,
                         'top5': top, 'top5_text': [tokenizer.decode([t]) for t in top],
                         'hidden_norm': hidden.norm().item(), 'status': int(requests.status[0]),
                         'error_code': int(requests.error_code[0]), 'consumed': int(requests.consumed_tokens[0])}
                if reference is None:
                    reference = (name, logits, hidden)
                else:
                    _, ref_logits, ref_hidden = reference
                    entry['logits_relative_error'] = ((logits - ref_logits).norm() / ref_logits.norm()).item()
                    entry['hidden_relative_error'] = ((hidden - ref_hidden).norm() / ref_hidden.norm()).item()
                    entry['argmax_matches_reference'] = bool(logits.argmax() == ref_logits.argmax())
                entry['trace'] = trace(functions[name], state, len(tokens))
                results[name] = entry
                print(name, json.dumps(plain(entry)), flush=True)
    return {'prompt': args.prompt, 'prompt_tokens': len(tokens), 'reference': reference[0], 'compiled': not args.no_compile,
            'graphs': results}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='JSON report path')
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--draft', type=Path, default=None)
    parser.add_argument('--hub-cache', type=Path, default=HUB_CACHE)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--prefill-chunk-size', type=int, default=256)
    parser.add_argument('--pages', type=int, default=None)
    parser.add_argument('--compile-workers', type=int, default=None)
    parser.add_argument('--kernel-cache', type=Path, default=Path('~/.cache/flare/kernels-qwen').expanduser())
    parser.add_argument('--no-compile', action='store_true', help='eager prepared operators instead of Inductor forwards')
    return parser


def main():
    args = build_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import math
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return value
    result = finite(plain(result))
    write_json(args.output, result)
    print(json.dumps(plain(result), indent=2))


if __name__ == '__main__':
    main()
