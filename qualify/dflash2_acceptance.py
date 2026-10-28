"""Teacher-forced acceptance of DFlash2 and MTP drafts on the Qwen3.8-Flash-Next target."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import torch

from qualify.mtp import device
from qualify.performance import connection
from qualify.provenance import plain, write_json
from qualify.speculative_benchmark import read_sse
from qualify.speculative_workload import MODEL_ID, build_workload, json_bytes, token_sha

SNAPSHOT = Path('/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD')
TAPS = (5, 15, 25, 35, 45)
RESIDUAL_LAYER = 47
STREAMS, WIDTH = 4, 2560
GREEDY = dict(temperature=0.0, top_k=1, top_p=1.0)


def token_digest(tokens):
    return hashlib.sha256(struct.pack(f'<{len(tokens)}q', *tokens)).hexdigest()[:16]


# generate: greedy continuations from the serving process

def text_cases(tokenizer):
    """Rendered text-only workload cases: id, category, request, prompt_tokens."""
    return [{k: case[k] for k in ('id', 'category', 'request', 'prompt_tokens')}
            for case in build_workload(tokenizer)['cases']
            if case['category'] != 'long_context' and not case.get('images')]


def greedy_request(case, max_tokens, model=None):
    request = deepcopy(case['request'])
    request.update(GREEDY, max_tokens=max_tokens, stream=True,
                   stream_options={'include_usage': True}, return_token_ids=True)
    if model is not None:
        request['model'] = model
    return request


def request_events(server, request, timeout):
    """Send one chat completion and return its SSE events, in order."""
    conn, prefix = connection(server, timeout)
    try:
        conn.request('POST', prefix + '/v1/chat/completions', body=json_bytes(request),
                     headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f'HTTP status {response.status}: {response.read(4096)!r}')
        start = time.perf_counter_ns()
        return list(read_sse(response, io.BytesIO(), elapsed_ns=lambda: time.perf_counter_ns() - start))
    finally:
        conn.close()


def event_tokens(events):
    """Return (prompt_token_ids, continuation token IDs, finish_reason) from one stream."""
    prompt, ids, finish = None, [], None
    for event in events:
        payload = event['data']
        if payload == '[DONE]':
            continue
        if 'error' in payload:
            raise ValueError(f'server error: {payload["error"]}')
        if payload.get('prompt_token_ids') is not None:
            if prompt is not None:
                raise ValueError('prompt token IDs reported twice')
            prompt = list(payload['prompt_token_ids'])
        for choice in payload.get('choices', []):
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
            if choice.get('token_ids'):
                ids.extend(choice['token_ids'])
    if prompt is None or finish is None or not ids:
        raise ValueError('stream lacks prompt token IDs, generated tokens, or a terminal event')
    return prompt, ids, finish


def generate_cases(cases, server, *, max_tokens, timeout, send=request_events, model=None):
    records = []
    for case in cases:
        prompt, ids, finish = event_tokens(send(server, greedy_request(case, max_tokens, model), timeout))
        # The server's own rendering (its chat template may differ) is what the target consumed.
        records.append({'id': case['id'], 'category': case['category'], 'prompt_tokens': prompt,
                        'continuation_tokens': ids, 'finish_reason': finish,
                        'prompt_tokens_sha256': token_sha(prompt),
                        'rendered_prompt_matches': prompt == case['prompt_tokens']})
    return {'model_id': MODEL_ID, 'served_model': model or (cases[0]['request']['model'] if cases else None),
            'server': server, 'max_tokens': max_tokens, 'cases': records}


def generate(args):
    from transformers import AutoTokenizer

    destination = args.output / 'tokens.json'
    if destination.exists():
        raise FileExistsError(f'output already exists: {destination}')
    cases = text_cases(AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True))
    document = generate_cases(cases, args.server, max_tokens=args.max_tokens, timeout=args.timeout,
                              model=args.model)
    write_json(destination, document)
    return {'tokens': str(destination), 'cases': len(document['cases']),
            'continuation_tokens': {c['id']: len(c['continuation_tokens']) for c in document['cases']}}


# trace: target features and residual under teacher forcing

class LayerTap:
    """Keep the stream-averaged tapped layers and the whole final residual of one forward."""

    def __init__(self):
        self.taps, self.residual = {}, None

    def __call__(self, layer, output):
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] != STREAMS * WIDTH:
            raise ValueError(f'layer {layer}: expected a [1, tokens, {STREAMS * WIDTH}] four-stream residual, '
                             f'got {list(output.shape)}')
        if layer in TAPS:
            self.taps[layer] = output[0].float().view(-1, STREAMS, WIDTH).mean(1).to(torch.bfloat16)
        elif layer == RESIDUAL_LAYER:
            self.residual = output[0].to(torch.bfloat16).clone()

    def collect(self):
        if sorted(self.taps) != list(TAPS) or self.residual is None:
            raise RuntimeError('reference forward did not report every tapped layer')
        features = torch.cat([self.taps[layer] for layer in TAPS], dim=1)
        residual, self.taps, self.residual = self.residual, {}, None
        return features, residual


def canonical_reference(args, layer_output):
    from ignition.preparation import inspect_qwen38_checkpoint
    from ignition.reference.qwen38_decoder import CanonicalTextReference

    dev = device()
    manifest = inspect_qwen38_checkpoint(args.snapshot)
    return CanonicalTextReference(manifest, device=str(dev), byte_budget=args.byte_budget,
                                  progress=lambda message: print(message, flush=True),
                                  layer_output=layer_output)


def trace(args, *, reference_factory=None):
    from ignition.preparation.qwen38 import MODEL_ID as CHECKPOINT_ID, REVISION

    document = json.loads(Path(args.tokens).read_text())
    folder = args.output / 'traces'
    folder.mkdir(parents=True, exist_ok=True)
    summary, pending = [], []
    for case in document['cases']:
        tokens = case['prompt_tokens'] + case['continuation_tokens']
        digest = token_digest(tokens)
        path = folder / f"{case['id']}-{digest}.pt"
        summary.append({'id': case['id'], 'cached': path.exists(), 'tokens': len(tokens), 'path': str(path)})
        if not path.exists():
            pending.append((case, tokens, digest, path))
    if pending:
        tap = LayerTap()
        factory = reference_factory or (lambda layer_output: canonical_reference(args, layer_output))
        reference = factory(layer_output=tap)
        for case, tokens, digest, path in pending:
            output = reference.forward(torch.tensor([tokens], dtype=torch.int64), logits_to_keep=1)
            features, residual = tap.collect()
            last_logits = output.logits.reshape(-1, output.logits.shape[-1])[-1].float().cpu().clone()
            torch.save({'schema_version': 1, 'model_id': CHECKPOINT_ID, 'revision': REVISION,
                        'case': case['id'], 'category': case['category'],
                        'prompt_tokens': case['prompt_tokens'], 'continuation_tokens': case['continuation_tokens'],
                        'digest': digest, 'features': features, 'residual': residual,
                        'last_logits': last_logits}, path)
    return {'cases': summary}


# measure: greedy teacher-forced acceptance of each draft arm at each width

def simulate(propose, tokens, prompt_length, k):
    """Teacher-forced rounds over tokens: propose(t) drafts up to k ids after pending tokens[t]."""
    t, rounds, accepted = prompt_length, 0, 0
    while t < len(tokens) - 1:
        proposals, teacher = list(propose(t))[:k], tokens[t + 1:t + 1 + k]
        matched = 0
        while matched < min(len(proposals), len(teacher)) and proposals[matched] == teacher[matched]:
            matched += 1
        rounds, accepted, t = rounds + 1, accepted + matched, t + matched + 1
    return stats(rounds, accepted)


def stats(rounds, accepted):
    return {'acceptance_length': (accepted + rounds) / rounds if rounds else 0.0,
            'rounds': rounds, 'accepted': accepted}


def trace_tokens(trace):
    return trace['prompt_tokens'] + trace['continuation_tokens']


def shuffled(rows, args):
    if not args.shuffle_features:
        return rows
    order = torch.randperm(rows.shape[0], generator=torch.Generator().manual_seed(args.seed))
    return rows[order.to(rows.device)]


def load_traces(folder):
    paths = sorted(Path(folder).glob('*.pt'))
    if not paths:
        raise FileNotFoundError(f'no traces in {folder}')
    return [torch.load(path, map_location='cpu', weights_only=True) for path in paths]


def target_tensors(args, dev):
    """The target's BF16 embedding and head through the exact-checkpoint reader, without the native loader."""
    from ignition.preparation import inspect_qwen38_checkpoint
    from ignition.preparation.qwen38 import TEXT_PREFIX
    from ignition.reference.qwen38_quant import CheckpointTensors

    with CheckpointTensors(inspect_qwen38_checkpoint(args.snapshot), byte_budget=4 << 30) as reader:
        return (reader.read(f'{TEXT_PREFIX}.embed_tokens.weight').to(dev, torch.bfloat16),
                reader.read('lm_head.weight').to(dev, torch.bfloat16))


@contextmanager
def dflash2_arm(args, traces, dev, target):
    from ignition.reference.qwen38_dflash2 import DFlash2Reference

    draft = DFlash2Reference.load(args.draft, dev)
    embedding, head = target

    def for_trace(trace):
        tokens = trace_tokens(trace)
        full = draft.context(shuffled(trace['features'].to(dev), args), torch.arange(len(tokens)))

        def propose(t, k):
            context = SimpleNamespace(layers=[(key[:t], value[:t]) for key, value in full.layers],
                                      positions=full.positions[:t])
            return draft.propose(tokens[t], t, context, embedding, head).tolist()[:k]
        return propose
    yield for_trace


MTP_CHUNK, PAGE = 64, 64


@contextmanager
def mtp_arm(args, traces, dev, target):
    """The checkpoint's MTP head run eagerly: absorb real residuals, then recurse k rows per round."""
    from b12x.preparation import PreparationSession

    from ignition.preparation import inspect_qwen38_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.attention import QSAMetadata, QSAState, RotaryTables
    from ignition.recipes.qwen38.mtp import MTPScratch, PreparedMTP
    from ignition.recipes.qwen38.mtp_weights import MTPWeights

    embedding, head = target
    weights = MTPWeights.load(inspect_qwen38_checkpoint(args.snapshot), dev,
                              embedding_weight=embedding, lm_head_weight=head)
    max_tokens = max(len(trace_tokens(trace)) for trace in traces) + 8
    pages = -(-max_tokens // PAGE) + 1
    cache = QSAState.allocate(pages=pages, slots=1, device=dev, draft_tokens=7)
    rotary = RotaryTables.allocate(weights.module.config, pages * PAGE, dev)
    arena = ScratchArena(dev)
    table = torch.zeros(pages, dtype=torch.int32, device=dev)
    table[:pages - 1] = torch.arange(1, pages, dtype=torch.int32, device=dev)
    with PreparationSession(device=dev, autotune=False, compile_workers=0) as session:
        ops = {}
        for name, capacity, cache_only in (('absorb', MTP_CHUNK, True), ('propose', 1, False)):
            scratch = MTPScratch.allocate(capacity, 1, dev)
            hidden = torch.zeros((capacity, 10240), dtype=torch.bfloat16, device=dev)
            tokens = torch.zeros(capacity, dtype=torch.int64, device=dev)
            selected = torch.zeros(1, dtype=torch.int64, device=dev)
            metadata = QSAMetadata.allocate(tokens=capacity, batch=1, context=pages * PAGE, device=dev)
            op = PreparedMTP(weights, scratch, tokens, hidden, cache, metadata, rotary, arena,
                             output_indices=selected, session=session, name=f'mtp.{name}',
                             max_cache_pages=pages, cache_only=cache_only)
            ops[name] = (op, tokens, hidden, selected, metadata)
        arena.finalize(); arena.freeze()

        def run(name, rows, ids, start):
            op, tokens, hidden, selected, metadata = ops[name]
            count = ids.shape[0]
            positions = torch.arange(start, start + count, device=dev)
            tokens.zero_(); hidden.zero_()
            tokens[:count].copy_(ids); hidden[:count].copy_(rows)
            selected.fill_(count - 1)
            metadata.main_block_table[0].copy_(table)
            metadata.compressed_block_table.copy_(metadata.main_block_table)
            metadata.raw_state_slot_ids.fill_(0)
            metadata.request_ids.fill_(-1); metadata.request_ids[:count].fill_(0)
            metadata.query_positions.fill_(-1); metadata.query_positions[:count].copy_(positions)
            metadata.rope_positions.fill_(-1)
            metadata.rope_positions[:count].copy_(positions[:, None].expand(-1, 3))
            metadata.sequence_lengths.fill_(start + count)
            metadata.query_start_loc[0] = 0; metadata.query_start_loc[-1] = count
            metadata.num_accepted_tokens.fill_(1)
            metadata.is_prefilling.fill_(True)
            metadata.slot_mapping.fill_(-1)
            metadata.slot_mapping[:count].copy_(positions + PAGE)
            return op()

        def for_trace(trace):
            tokens = trace_tokens(trace)
            ids = torch.tensor(tokens, dtype=torch.int64, device=dev)
            residual = shuffled(trace['residual'].to(dev), args)
            for field in cache.__dataclass_fields__:
                value = getattr(cache, field)
                value.fill_(-1 if value.dtype == torch.int64 else 0)
            absorbed = [0]

            def propose(t, k):
                while absorbed[0] < t - 1:
                    a, count = absorbed[0], min(MTP_CHUNK, t - 1 - absorbed[0])
                    run('absorb', residual[a:a + count], ids[a + 1:a + 1 + count], a)
                    absorbed[0] += count
                output = run('propose', residual[t - 1:t], ids[t:t + 1], t - 1)
                absorbed[0] = t
                drafts = [int(output.logits[0].float().argmax())]
                for r in range(1, k):
                    rows = ops['propose'][0].scratch.residual[0].clone()[None]
                    output = run('propose', rows, torch.tensor([drafts[-1]], device=dev), t - 1 + r)
                    drafts.append(int(output.logits[0].float().argmax()))
                return drafts
            return propose
        yield for_trace


ARMS = {'dflash2': dflash2_arm, 'mtp': mtp_arm}


def csv(value, kind=str):
    return [kind(item.strip()) for item in str(value).split(',') if item.strip()]


def measure(args, *, arms=None):
    """arms maps name -> context-manager factory (args, traces) yielding for_trace(trace) -> propose(t, k)."""
    names, widths = csv(args.arms), csv(args.widths, int)
    traces = load_traces(args.traces)
    if arms is None:
        unknown = sorted(set(names) - set(ARMS))
        if unknown or any(width < 2 for width in widths):
            raise ValueError(f'unknown arms {unknown} or width below 2 in {widths}')
        dev = device()
        target = target_tensors(args, dev)
        arms = {name: (lambda a, t, arm=ARMS[name]: arm(a, t, dev, target)) for name in names}
    results = {}
    for name in names:
        results[name] = {str(width): {'cases': {}} for width in widths}
        with arms[name](args, traces) as for_trace:
            for trace in traces:
                propose = for_trace(trace)
                tokens = trace_tokens(trace)
                for width in widths:
                    k = width - 1
                    results[name][str(width)]['cases'][trace['case']] = simulate(
                        lambda t: propose(t, k), tokens, len(trace['prompt_tokens']), k)
        for width in widths:
            report = results[name][str(width)]
            totals = {}
            for trace in traces:
                case = report['cases'][trace['case']]
                for key in ('overall', trace['category']):
                    rounds, accepted = totals.get(key, (0, 0))
                    totals[key] = (rounds + case['rounds'], accepted + case['accepted'])
            report['overall'] = stats(*totals.pop('overall'))
            report['categories'] = {key: stats(*totals[key]) for key in sorted(totals)}
            for key, value in (('all', report['overall']), *report['categories'].items()):
                print(f"{name:8} w={width} {key:24} acceptance {value['acceptance_length']:.3f} "
                      f"rounds {value['rounds']}", flush=True)
    digests = sorted(trace['digest'] for trace in traces)
    return {'arms': names, 'widths': widths, 'shuffle_features': bool(args.shuffle_features), 'seed': args.seed,
            'traces': [{'case': trace['case'], 'category': trace['category'], 'digest': trace['digest'],
                        'tokens': len(trace_tokens(trace)), 'prompt_tokens': len(trace['prompt_tokens'])}
                       for trace in traces],
            'trace_digest': hashlib.sha256(''.join(digests).encode()).hexdigest()[:16],
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'trace', 'measure'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--server', help='generate: base URL of the serving process')
    parser.add_argument('--model', help='generate: served model name (default: the case request model)')
    parser.add_argument('--tokens', type=Path, help='trace: tokens.json written by generate')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--byte-budget', type=int, default=128 << 30)
    parser.add_argument('--traces', type=Path, help='measure: traces folder written by trace')
    parser.add_argument('--draft', type=Path, default=Path('~/models/incoai/Qwen3.8-Flash-Next-DFlash2').expanduser())
    parser.add_argument('--widths', default='4,8', help='measure: comma-separated draft widths')
    parser.add_argument('--arms', default='dflash2,mtp', help='measure: comma-separated arms')
    parser.add_argument('--shuffle-features', action='store_true',
                        help='measure: control that permutes the trace rows fed to the drafts')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'generate' and args.server is None:
        parser.error('generate requires --server')
    if args.command == 'trace' and args.tokens is None:
        parser.error('trace requires --tokens')
    if args.command == 'measure' and args.traces is None:
        parser.error('measure requires --traces')
    if args.max_tokens <= 0 or args.timeout <= 0 or args.byte_budget <= 0:
        parser.error('max-tokens, timeout, and byte-budget must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    result = globals()[args.command](args)
    write_json(args.output / f'{args.command}.json', result)
    print(json.dumps(plain(result), indent=2))


if __name__ == '__main__':
    main()
