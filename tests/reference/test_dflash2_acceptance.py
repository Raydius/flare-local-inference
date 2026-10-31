import contextlib
import io
import json
from types import SimpleNamespace

import pytest
import torch

from qualify import dflash2_acceptance as da
from qualify.speculative_benchmark import read_sse

VOCAB = 8


def layer_tensor(layer, tokens):
    position = torch.arange(tokens, dtype=torch.float32)[:, None, None]
    stream = torch.arange(da.STREAMS, dtype=torch.float32)[None, :, None]
    value = layer * 1000 + stream * 10 + position
    return value.expand(tokens, da.STREAMS, da.WIDTH).reshape(1, tokens, da.STREAMS * da.WIDTH).clone()


class StubReference:
    calls = 0

    def __init__(self, *, layer_output, layers=48):
        self.layer_output, self.layers = layer_output, layers

    def forward(self, input_ids, *, logits_to_keep=0):
        type(self).calls += 1
        tokens = input_ids.shape[1]
        for layer in range(self.layers):
            self.layer_output(layer, layer_tensor(layer, tokens))
        return SimpleNamespace(logits=torch.arange(VOCAB, dtype=torch.float32).reshape(1, 1, VOCAB))


def tokens_file(tmp_path):
    document = {'model_id': da.MODEL_ID, 'server': 'http://stub', 'max_tokens': 4, 'cases': [
        {'id': 'alpha', 'category': 'chat_reasoning', 'prompt_tokens': [1, 2, 3], 'continuation_tokens': [4, 5],
         'finish_reason': 'stop', 'prompt_tokens_sha256': ''},
        {'id': 'beta', 'category': 'code_edit', 'prompt_tokens': [7], 'continuation_tokens': [6, 5, 4],
         'finish_reason': 'length', 'prompt_tokens_sha256': ''}]}
    path = tmp_path / 'tokens.json'
    path.write_text(json.dumps(document))
    return path, document


def test_trace_averages_tapped_streams_and_keeps_final_residual(tmp_path):
    path, document = tokens_file(tmp_path)
    args = SimpleNamespace(tokens=path, output=tmp_path, snapshot=None, byte_budget=1)
    StubReference.calls = 0
    summary = da.trace(args, reference_factory=StubReference)
    assert StubReference.calls == 2
    assert [c['cached'] for c in summary['cases']] == [False, False]
    for record, case in zip(summary['cases'], document['cases'], strict=True):
        tokens = case['prompt_tokens'] + case['continuation_tokens']
        assert record['tokens'] == len(tokens)
        assert record['path'].endswith(f"{case['id']}-{da.token_digest(tokens)}.pt")
        saved = torch.load(record['path'], weights_only=True)
        expected = torch.cat([layer_tensor(l, len(tokens))[0].view(-1, 4, 2560).mean(1) for l in da.TAPS], 1)
        assert saved['features'].dtype == torch.bfloat16 and saved['features'].shape == (len(tokens), 12800)
        assert torch.equal(saved['features'].float(), expected.bfloat16().float())
        assert saved['residual'].dtype == torch.bfloat16 and saved['residual'].shape == (len(tokens), 10240)
        assert torch.equal(saved['residual'].float(), layer_tensor(47, len(tokens))[0].bfloat16().float())
        assert saved['last_logits'].shape == (VOCAB,) and saved['last_logits'].dtype == torch.float32
        assert saved['digest'] == da.token_digest(tokens) and saved['case'] == case['id']
        assert saved['prompt_tokens'] == case['prompt_tokens']


def test_trace_rejects_single_stream_layer_output():
    tap = da.LayerTap()
    with pytest.raises(ValueError, match='10240'):
        tap(5, torch.zeros(1, 3, 2560))
    for layer in range(47):
        tap(layer, layer_tensor(layer, 2))
    with pytest.raises(RuntimeError, match='every tapped layer'):
        tap.collect()


def test_trace_skips_cached_cases(tmp_path):
    path, _ = tokens_file(tmp_path)
    args = SimpleNamespace(tokens=path, output=tmp_path, snapshot=None, byte_budget=1)
    da.trace(args, reference_factory=StubReference)
    StubReference.calls = 0
    summary = da.trace(args, reference_factory=StubReference)
    assert StubReference.calls == 0
    assert all(c['cached'] for c in summary['cases'])


def sse(payloads):
    body = b''.join(b'data: ' + json.dumps(p).encode() + b'\n\n' for p in payloads) + b'data: [DONE]\n\n'
    return list(read_sse(io.BytesIO(body), io.BytesIO(), elapsed_ns=lambda: 0))


def events_for(prompt, ids):
    return sse([{'prompt_token_ids': prompt, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]},
                {'choices': [{'index': 0, 'delta': {'content': 'a'}, 'token_ids': ids[:1]}]},
                {'choices': [{'index': 0, 'delta': {'content': 'b'}, 'token_ids': ids[1:], 'finish_reason': 'stop'}]},
                {'choices': [], 'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(ids)}}])


def cases():
    request = {'model': da.MODEL_ID, 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 1.0,
               'top_k': 20, 'top_p': 0.95, 'seed': 1, 'max_tokens': 512, 'return_token_timestamps': True}
    return [{'id': 'alpha', 'category': 'chat_reasoning', 'request': request, 'prompt_tokens': [1, 2, 3]},
            {'id': 'beta', 'category': 'code_edit', 'request': request, 'prompt_tokens': [7, 8]}]


def test_generate_collects_prompt_and_continuation_tokens():
    sent = []

    def send(server, request, timeout):
        sent.append(request)
        return events_for([1, 2, 3] if len(sent) == 1 else [7, 8], [10, 11, 12])

    document = da.generate_cases(cases(), 'http://stub', max_tokens=4, timeout=1, send=send)
    assert document['model_id'] == da.MODEL_ID and document['server'] == 'http://stub' and document['max_tokens'] == 4
    assert [c['id'] for c in document['cases']] == ['alpha', 'beta']
    alpha = document['cases'][0]
    assert alpha['prompt_tokens'] == [1, 2, 3] and alpha['continuation_tokens'] == [10, 11, 12]
    assert alpha['finish_reason'] == 'stop' and alpha['prompt_tokens_sha256'] == da.token_sha([1, 2, 3])
    assert set(alpha) == {'id', 'category', 'prompt_tokens', 'continuation_tokens', 'finish_reason',
                          'prompt_tokens_sha256', 'rendered_prompt_matches'}
    assert alpha['rendered_prompt_matches'] is True
    request = sent[0]
    assert (request['temperature'], request['top_k'], request['top_p'], request['max_tokens']) == (0.0, 1, 1.0, 4)
    assert request['stream'] and request['stream_options'] == {'include_usage': True} and request['return_token_ids']
    assert request['seed'] == 1 and request['return_token_timestamps'] is True


def test_generate_keeps_server_prompt_and_flags_rendering_mismatch():
    document = da.generate_cases(cases()[:1], 'http://stub', max_tokens=4, timeout=1,
                                 send=lambda *_: events_for([1, 2, 9], [10, 11]))
    record = document['cases'][0]
    assert record['prompt_tokens'] == [1, 2, 9] and record['rendered_prompt_matches'] is False


def test_generate_overrides_served_model_name():
    sent = []
    document = da.generate_cases(cases()[:1], 'http://stub', max_tokens=4, timeout=1, model='Qwen3.6',
                                 send=lambda s, r, t: sent.append(r) or events_for([1, 2, 3], [10, 11]))
    assert sent[0]['model'] == 'Qwen3.6' and document['served_model'] == 'Qwen3.6'
    document = da.generate_cases(cases()[:1], 'http://stub', max_tokens=4, timeout=1,
                                 send=lambda s, r, t: sent.append(r) or events_for([1, 2, 3], [10, 11]))
    assert sent[1]['model'] == da.MODEL_ID and document['served_model'] == da.MODEL_ID


# measure

def test_simulate_counts_rounds_and_accepted_prefix():
    tokens = list(range(100, 112))
    perfect = da.simulate(lambda t: tokens[t + 1:t + 4], tokens, 4, 3)
    assert perfect == {'acceptance_length': 4.0, 'rounds': 2, 'accepted': 6}
    wrong = da.simulate(lambda t: [-1, -1, -1], tokens, 4, 3)
    assert wrong == {'acceptance_length': 1.0, 'rounds': 7, 'accepted': 0}
    partial = da.simulate(lambda t: tokens[t + 1:t + 2] + [-1, -1], tokens, 4, 3)
    assert partial == {'acceptance_length': 2.0, 'rounds': 4, 'accepted': 4}
    capped = da.simulate(lambda t: tokens[t + 1:t + 8], tokens, 4, 3)
    assert capped == {'acceptance_length': 4.0, 'rounds': 2, 'accepted': 6}
    assert da.simulate(lambda t: [], tokens, 11, 3) == {'acceptance_length': 0.0, 'rounds': 0, 'accepted': 0}


def trace_files(tmp_path):
    folder = tmp_path / 'traces'
    folder.mkdir()
    specs = [('alpha', 'chat_reasoning', [1, 2], list(range(10, 20))),
             ('beta', 'code_edit', [3], list(range(30, 36)))]
    for case, category, prompt, continuation in specs:
        tokens = prompt + continuation
        torch.save({'case': case, 'category': category, 'prompt_tokens': prompt, 'continuation_tokens': continuation,
                    'digest': da.token_digest(tokens), 'features': torch.zeros(len(tokens), 4, dtype=torch.bfloat16),
                    'residual': torch.zeros(len(tokens), 4, dtype=torch.bfloat16),
                    'last_logits': torch.zeros(VOCAB)}, folder / f'{case}-{da.token_digest(tokens)}.pt')
    return folder


def stub_arm(correct):
    """An arm whose proposals match the teacher for the first `correct` tokens of every round."""
    @contextlib.contextmanager
    def factory(args, traces):
        def for_trace(trace):
            tokens = trace['prompt_tokens'] + trace['continuation_tokens']
            return lambda t, k: (tokens[t + 1:t + 1 + correct] + [-1] * k)[:k]
        yield for_trace
    return factory


def measure_args(folder, **overrides):
    values = dict(traces=folder, arms='dflash2,mtp', widths='4,8', shuffle_features=False, e4m3_kv=False, seed=0)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_measure_reports_arms_by_width_with_means_over_rounds(tmp_path):
    folder = trace_files(tmp_path)
    arms = {'dflash2': stub_arm(7), 'mtp': stub_arm(1)}
    result = da.measure(measure_args(folder), arms=arms)
    assert result['arms'] == ['dflash2', 'mtp'] and result['widths'] == [4, 8]
    assert result['shuffle_features'] is False and result['seed'] == 0
    assert [t['case'] for t in result['traces']] == ['alpha', 'beta']
    assert result['traces'][0] == {'case': 'alpha', 'category': 'chat_reasoning', 'tokens': 12, 'prompt_tokens': 2,
                                   'digest': da.token_digest([1, 2] + list(range(10, 20)))}
    assert len(result['trace_digest']) == 16
    perfect = result['results']['dflash2']
    # alpha: N=12, prompt 2 -> w=4: t=2,6 accept 3, t=10 accepts the last token; w=8: t=2 accept 7, t=10 accept 1
    assert perfect['4']['cases']['alpha'] == {'acceptance_length': 10 / 3, 'rounds': 3, 'accepted': 7}
    assert perfect['8']['cases']['alpha'] == {'acceptance_length': 5.0, 'rounds': 2, 'accepted': 8}
    # beta: N=7, prompt 1 -> w=4: t=1 accept 3, t=5 accept 1; w=8: t=1 accept 5
    assert perfect['4']['cases']['beta'] == {'acceptance_length': 3.0, 'rounds': 2, 'accepted': 4}
    assert perfect['8']['cases']['beta'] == {'acceptance_length': 6.0, 'rounds': 1, 'accepted': 5}
    assert perfect['4']['overall'] == {'acceptance_length': 16 / 5, 'rounds': 5, 'accepted': 11}
    assert perfect['4']['categories'] == {'chat_reasoning': perfect['4']['cases']['alpha'],
                                          'code_edit': perfect['4']['cases']['beta']}
    one = result['results']['mtp']['4']
    assert one['cases']['alpha'] == {'acceptance_length': 2.0, 'rounds': 5, 'accepted': 5}
    assert one['overall']['rounds'] == 8 and one['overall']['acceptance_length'] == 2.0
    written = json.loads(json.dumps(da.plain(result)))
    assert set(written['results']) == {'dflash2', 'mtp'} and set(written['results']['mtp']) == {'4', '8'}
    assert set(written['results']['mtp']['8']) == {'overall', 'categories', 'cases'}


def test_measure_arm_subset_and_shuffle_flag(tmp_path):
    folder = trace_files(tmp_path)
    result = da.measure(measure_args(folder, arms='dflash2', widths='4', shuffle_features=True, seed=3),
                        arms={'dflash2': stub_arm(0)})
    assert list(result['results']) == ['dflash2'] and list(result['results']['dflash2']) == ['4']
    assert result['shuffle_features'] is True and result['seed'] == 3
    assert result['results']['dflash2']['4']['overall'] == {'acceptance_length': 1.0, 'rounds': 14, 'accepted': 0}


def test_shuffled_permutes_rows_deterministically():
    rows = torch.arange(6)[:, None].expand(6, 2).clone()
    off = da.shuffled(rows, SimpleNamespace(shuffle_features=False, seed=0))
    assert off is rows
    first = da.shuffled(rows, SimpleNamespace(shuffle_features=True, seed=0))
    assert torch.equal(first, da.shuffled(rows, SimpleNamespace(shuffle_features=True, seed=0)))
    assert sorted(first[:, 0].tolist()) == list(range(6)) and not torch.equal(first, rows)
