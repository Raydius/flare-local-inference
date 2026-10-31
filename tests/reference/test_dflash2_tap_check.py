import math

import pytest
import torch

from qualify import dflash2_tap_check as tc

TOKENS = 6


def features(tokens=TOKENS):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(tokens, len(tc.TAPS) * tc.WIDTH, generator=generator).to(torch.bfloat16)


def test_tap_columns_follow_trace_order():
    assert tc.tap_columns(5) == slice(0, 2560)
    assert tc.tap_columns(45) == slice(4 * 2560, 5 * 2560)
    with pytest.raises(ValueError):
        tc.tap_columns(6)


def test_exact_match_reports_zero_errors():
    reference = features()
    errors = tc.layer_errors(reference.clone(), reference)
    assert sorted(errors) == sorted(str(layer) for layer in tc.TAPS)
    for layer in errors.values():
        assert layer == {'max_relative_error': 0.0, 'max_abs_difference': 0.0}


def test_perturbed_layer_reports_its_relative_error_alone():
    reference = features()
    auxiliary = reference.clone()
    columns = tc.tap_columns(25)
    auxiliary[:, columns] = (reference[:, columns].float() * 1.5).to(torch.bfloat16)
    errors = tc.layer_errors(auxiliary, reference)
    block = reference[:, columns].double()
    expected = ((auxiliary[:, columns].double() - block).norm() / block.norm()).item()
    assert math.isclose(errors['25']['max_relative_error'], expected, rel_tol=1e-9)
    assert math.isclose(errors['25']['max_abs_difference'],
                        (auxiliary[:, columns].double() - block).abs().max().item())
    assert errors['25']['max_relative_error'] > 0
    for layer in ('5', '15', '35', '45'):
        assert errors[layer] == {'max_relative_error': 0.0, 'max_abs_difference': 0.0}


def test_layer_errors_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        tc.layer_errors(features(4), features(5))
    with pytest.raises(ValueError):
        tc.layer_errors(torch.zeros(4, 2560), torch.zeros(4, 2560))


def test_ring_liveness_short_prompt_is_fully_live():
    slots = torch.arange(10, dtype=torch.int64)
    slots[7:] = -1  # rows past the prompt do not count
    ring = tc.ring_liveness(slots, 7, window=2048)
    assert ring == {'live': 7, 'expected_live': 7, 'first_live_position': 0,
                    'first_expected_position': 0, 'matches': True}
    slots[3] = -1
    assert tc.ring_liveness(slots, 7, window=2048)['matches'] is False


def test_ring_liveness_long_prompt_keeps_the_last_window():
    prompt = 2048 + 300
    slots = torch.full((prompt,), -1, dtype=torch.int64)
    slots[300:] = torch.arange(300, prompt) % 2048
    ring = tc.ring_liveness(slots, prompt, window=2048)
    assert ring == {'live': 2048, 'expected_live': 2048, 'first_live_position': 300,
                    'first_expected_position': 300, 'matches': True}
    slots[299] = 5  # one position too many is a wrap over a live slot
    ring = tc.ring_liveness(slots, prompt, window=2048)
    assert ring['live'] == 2049 and ring['matches'] is False
    assert tc.ring_liveness(torch.full((4,), -1, dtype=torch.int64), 4)['first_live_position'] is None


def test_prefill_graph_picks_the_smallest_fitting_speculative_capacity():
    assert tc.prefill_graph((128, 256, 512), 198) == 'prefill256'
    assert tc.prefill_graph((128, 256), 128) == 'prefill128'
    with pytest.raises(ValueError):
        tc.prefill_graph((128,), 129)


def test_report_shape():
    trace = {'case': 'chat-capacity', 'digest': 'abc123', 'prompt_tokens': [1, 2, 3],
             'model_id': 'trace/model', 'revision': 'deadbeef', 'features': features(3)}
    errors = tc.layer_errors(trace['features'], trace['features'])
    ring = tc.ring_liveness(torch.tensor([0, 1, 2]), 3)
    report = tc.tap_report(trace, errors, ring, prepared_model_id='prepared/model', prepared_revision='cafe')
    assert report['trace'] == 'chat-capacity-abc123'
    assert report['prompt_tokens'] == 3
    assert set(report['layers']) == {'5', '15', '25', '35', '45'}
    assert set(report['layers']['5']) == {'max_relative_error', 'max_abs_difference'}
    assert set(report['ring_slots']) >= {'live', 'expected_live', 'first_live_position'}
    assert (report['model_id'], report['revision']) == ('prepared/model', 'cafe')
    assert (report['trace_model_id'], report['trace_revision']) == ('trace/model', 'deadbeef')
    from qualify.provenance import plain
    import json
    json.dumps(plain(report), allow_nan=False)


def test_parser_requires_trace_and_output(capsys):
    parser = tc.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--output', 'out.json'])
    with pytest.raises(SystemExit):
        parser.parse_args(['--trace', 'trace.pt'])
    args = parser.parse_args(['--trace', 'trace.pt', '--output', 'out.json'])
    assert args.draft is None and args.context == 256 and args.prefill_chunk_size == 256
