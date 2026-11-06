"""Needle placement, pass criterion, grid, and serving order of the passkey probe; no container runs."""

import os
from pathlib import Path
from types import SimpleNamespace

from qualify import dflash2_served as ds
from qualify import long_context_passkey as pk
from qualify.speculative_workload import _catalog


def test_prompt_places_the_needle_at_depth_over_the_catalog():
    case = pk.prompt(4800, 0.5, "48213", model="ckpt")
    messages = case["request"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    lines = messages[1]["content"].split("\n")
    assert lines[2400] == pk.needle("48213")
    assert "\n".join(lines[:2400] + lines[2401:]) == _catalog(4800)
    assert messages[2]["content"] == pk.QUESTION
    request = case["request"]
    assert request["top_k"] == 1 and request["max_tokens"] == 32 and request["model"] == "ckpt"
    assert request["return_token_ids"] is False and request["return_token_timestamps"] is False
    assert case["id"] == "passkey-4800-0.5" and case["prompt_tokens"] == []
    assert case["request_sha256"] == pk.value_sha(request)


def test_needle_at_the_edges_stays_inside_the_catalog():
    first = pk.prompt(100, 0.0, "11111", model="ckpt")["request"]["messages"][1]["content"].split("\n")
    last = pk.prompt(100, 1.0, "11111", model="ckpt")["request"]["messages"][1]["content"].split("\n")
    assert first[0] == pk.needle("11111") and last[-1] == pk.needle("11111") and len(first) == len(last) == 101


def test_passkey_is_five_digits_absent_from_the_catalog_and_stable():
    records = pk.records_for(200_000)
    key = pk.passkey_for(48_300, 200_000, 0.1, records)
    assert len(key) == 5 and key.isdigit() and key not in _catalog(records)
    assert key == pk.passkey_for(48_300, 200_000, 0.1, records)
    assert key != pk.passkey_for(48_300, 200_000, 0.5, records)


def test_check_requires_the_exact_passkey():
    assert pk.check("48213.", "48213")
    assert pk.check("The pass key is 48213", "48213")
    assert not pk.check("I do not know", "48213")
    assert not pk.check("48212", "48213")


def observation(content, prompt_tokens, *, generated=4, elapsed_ns=2_000_000_000):
    return {"case": "x", "elapsed_ns": elapsed_ns,
            "analysis": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": generated},
                         "finish_reason": "stop", "content": content,
                         "checks": {"no_stream_error": True, "one_valid_terminal": True}}}


def delta(rounds, offered, accepted, width=4):
    return {f'flare_speculative_rounds_total{{batch="1",width="{width}"}}': rounds,
            f'flare_speculative_offered_drafts_total{{batch="1",width="{width}"}}': offered,
            f'flare_speculative_accepted_drafts_total{{batch="1",width="{width}"}}': accepted,
            'flare_speculative_drafts_total{depth="0",source="mtp",outcome="accepted"}': accepted,
            'flare_speculative_rounds_total{batch="1",width="1"}': 10}


def test_cells_and_summary_build_the_grid():
    good = pk.cell(observation("48213.", 320_500), delta(100, 300, 180), length=320_000, depth=0.5, passkey="48213")
    assert good["passed"] and good["completed"] and good["request_seconds"] == 2.
    assert good["accepted_per_round"] == 180 / 110
    bad = pk.cell(observation("no idea", 200_100), delta(100, 300, 180), length=200_000, depth=0.1, passkey="77777")
    result = pk.summary([good, bad])
    assert result["grid"] == {"320000": {"0.5": True}, "200000": {"0.1": False}}
    assert result["all_passed"] is False
    assert pk.summary([good])["all_passed"] is True and pk.summary([])["all_passed"] is False


class Process:
    def __init__(self, command, *, stdout, stderr):
        self.command = command
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def run_with_fakes(tmp_path, lengths, max_context_tokens, reply):
    started, sent, stopped = [], [], []

    def popen(command, *, stdout, stderr):
        process = Process(command, stdout=stdout, stderr=stderr)
        started.append(command)
        return process

    def measure(base_url, case, folder, *, timeout, model_id):
        sent.append(case["id"])
        records = int(case["id"].split("-")[1])
        passkey = case["request"]["messages"][1]["content"].split("The pass key is ")[1].split(".")[0]
        return observation(reply(passkey), round(records * 66.6)), delta(100, 300, 180)

    args = SimpleNamespace(image="img", package=tmp_path / "pk/models/d/e/model.safetensors",
                           publisher_keys=tmp_path / "keys.json", trust_store=tmp_path / "trust.toml",
                           output=tmp_path / "out", model_id="ckpt", max_context_tokens=max_context_tokens,
                           lengths=lengths, depths=(0.5, 0.1, 0.9), seed=48_300, port=8111, timeout=1, ready_timeout=1)
    result = pk.run(args, popen=popen, stop=lambda command, **_: stopped.append(command), ready=lambda url: True, measure=measure)
    return result, started, sent, stopped, args


def test_run_serves_once_and_sends_the_longest_length_first(tmp_path):
    result, started, sent, stopped, args = run_with_fakes(tmp_path, (200_000, 1_000_000, 320_000, 520_000), 1_048_576,
                                                          lambda passkey: f"{passkey}.")
    r = pk.records_for
    assert sent[0] == f"passkey-{r(1_000_000)}-0.1" and sent[-1] == f"passkey-{r(200_000)}-0.9"
    assert sent == [f"passkey-{r(length)}-{depth}" for length in (1_000_000, 520_000, 320_000, 200_000)
                    for depth in (0.1, 0.5, 0.9)]
    assert len(started) == 1
    command = started[0]
    assert "--history-drafts" not in command and command[command.index("--max-context-tokens") + 1] == "1048576"
    assert command == ds.container_command("img", args.package, name="long-context-passkey", uid=os.getuid(), gid=os.getgid(),
                                           home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                           port=8111, history_drafts=False, max_context_tokens=1_048_576)
    assert stopped == [["docker", "stop", "-t", "120", "long-context-passkey"]]
    assert result["all_passed"] is True and (tmp_path / "out/summary.json").is_file()
    assert sorted(result["grid"]) == sorted(["1000000", "520000", "320000", "200000"])


def test_run_without_a_context_ceiling_omits_the_flag(tmp_path):
    result, started, sent, stopped, args = run_with_fakes(tmp_path, (200_000,), None, lambda passkey: passkey)
    assert "--max-context-tokens" not in started[0] and result["all_passed"] is True


def test_native_baseline_run_takes_one_length_and_fails_on_a_miss(tmp_path):
    result, started, sent, stopped, args = run_with_fakes(tmp_path, (200_000,), 262_144, lambda passkey: "no idea")
    assert len(sent) == 3 and list(result["grid"]) == ["200000"] and result["grid"]["200000"] == {"0.1": False, "0.5": False, "0.9": False}
    assert result["all_passed"] is False and started[0][started[0].index("--max-context-tokens") + 1] == "262144"
