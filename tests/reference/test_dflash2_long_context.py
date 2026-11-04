"""Prompt sizing, retrieval check, draft buckets, and criterion of the long-context probe; no container runs."""

import os
from pathlib import Path
from types import SimpleNamespace

from qualify import dflash2_long_context as lc
from qualify import dflash2_served as ds
from qualify.speculative_workload import _catalog


def test_records_for_sizes_the_catalog_from_the_frozen_long_case():
    assert lc.records_for(51_172) == 768
    assert lc.records_for(320_000) > lc.records_for(200_000) > 768
    assert lc.records_for(320_000) * lc.TOKENS_PER_RECORD > lc.NATIVE_CONTEXT
    assert lc.records_for(0) == 3


def test_prompt_names_three_records_over_one_catalog():
    case = lc.prompt(4800, model="ckpt")
    messages = case["request"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert messages[1]["content"] == _catalog(4800) and messages[1]["content"].count("service-4799:") == 1
    assert "service-0007, service-2400, and service-4799" in messages[2]["content"]
    assert case["request"]["max_tokens"] == 512 and case["request"]["model"] == "ckpt"
    assert case["request"]["return_token_ids"] is False and case["request"]["return_token_timestamps"] is False
    assert case["prompt_tokens"] == [] and case["id"] == "long-services-4800"
    assert case["request_sha256"] == lc.value_sha(case["request"])


def test_expected_values_follow_the_catalog_and_retrieval_checks_substrings():
    records = 4800
    expected = lc.expected_values(records)
    catalog = _catalog(records)
    for record, fields in expected.items():
        line = next(line for line in catalog.splitlines() if line.startswith(record + ":"))
        for field, value in fields.items():
            assert f"{field}={value}" in line
    answer = " ".join(f"{record}: " + ", ".join(f"{k} {v}" for k, v in fields.items()) for record, fields in expected.items())
    assert lc.check_retrieval(answer, expected) == {"passed": True, "missing": {}}
    partial = answer.replace(expected["service-4799"]["owner"], "team-??")
    result = lc.check_retrieval(partial, expected)
    assert result["passed"] is False and result["missing"] == {"service-4799": ["owner"]}


def delta(rounds, offered, accepted, width=4):
    return {f'flare_speculative_rounds_total{{batch="1",width="{width}"}}': rounds,
            f'flare_speculative_offered_drafts_total{{batch="1",width="{width}"}}': offered,
            f'flare_speculative_accepted_drafts_total{{batch="1",width="{width}"}}': accepted,
            'flare_speculative_drafts_total{depth="0",source="mtp",outcome="accepted"}': accepted,
            'flare_speculative_rounds_total{batch="1",width="1"}': 10}


def observation(content, prompt_tokens, generated=100, elapsed_ns=2_000_000_000):
    return {"case": "x", "elapsed_ns": elapsed_ns,
            "analysis": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": generated},
                         "finish_reason": "stop", "content": content,
                         "checks": {"no_stream_error": True, "one_valid_terminal": True}}}


def test_buckets_and_summary_apply_the_criterion():
    expected = lc.expected_values(4800)
    good = " ".join(v for fields in expected.values() for v in fields.values())
    past = lc.bucket(observation(good, 320_500), expected, delta(100, 300, 180))
    sub = lc.bucket(observation(good, 200_100), expected, delta(100, 300, 190))
    assert past["accepted_per_round"] == 180 / 110 and past["acceptance_rate"] == 0.6
    assert past["drafts_by_source"] == {"mtp": {"accepted": 180}} and past["output_tokens_per_second"] == 50.
    assert past["completed"] and past["retrieval"]["passed"]
    result = lc.summary(past, sub)
    assert result["ratio"] == (180 / 110) / (190 / 110) and result["criterion_met"] is True
    assert result["past_native_exceeds_native_context"] is True

    weak = lc.bucket(observation(good, 320_500), expected, delta(100, 300, 150))
    assert lc.summary(weak, sub)["criterion_met"] is False
    short = lc.bucket(observation(good, 250_000), expected, delta(100, 300, 190))
    assert lc.summary(short, sub)["criterion_met"] is False
    blind = lc.bucket(observation("nothing", 320_500), expected, delta(100, 300, 190))
    assert lc.summary(blind, sub)["retrieval_passed"] is False and lc.summary(blind, sub)["criterion_met"] is False


def test_run_serves_once_without_history_drafts_and_sends_past_native_first(tmp_path):
    started, sent = [], []

    class Process:
        def __init__(self, command, *, stdout, stderr):
            started.append(command)
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def measure(base_url, case, folder, *, timeout, model_id):
        sent.append(case["id"])
        records = int(case["id"].rsplit("-", 1)[1])
        expected = lc.expected_values(records)
        content = " ".join(v for fields in expected.values() for v in fields.values())
        return observation(content, round(records * lc.TOKENS_PER_RECORD)), delta(100, 300, 180)

    stopped = []
    args = SimpleNamespace(image="img", package=tmp_path / "pk/models/d/e/model.safetensors",
                           publisher_keys=tmp_path / "keys.json", trust_store=tmp_path / "trust.toml",
                           output=tmp_path / "out", model_id="ckpt", max_context_tokens=1_048_576,
                           past_native_tokens=320_000, sub_native_tokens=200_000, port=8111, timeout=1, ready_timeout=1)
    result = lc.run(args, popen=Process, stop=lambda command, **_: stopped.append(command), ready=lambda url: True, measure=measure)
    assert sent == ["long-services-4803", "long-services-3002"]
    command = started[0]
    assert "--history-drafts" not in command and command[command.index("--max-context-tokens") + 1] == "1048576"
    assert command == ds.container_command("img", args.package, name="dflash2-long-context", uid=os.getuid(), gid=os.getgid(),
                                           home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                           port=8111, history_drafts=False, max_context_tokens=1_048_576)
    assert stopped == [["docker", "stop", "-t", "120", "dflash2-long-context"]]
    assert result["criterion_met"] is True and (tmp_path / "out/summary.json").is_file()
