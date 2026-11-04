"""Arm table, trial lifecycle, curve rows, and table of the draft-count sweep; no container runs."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualify import dflash2_draft_sweep as sweep
from qualify import dflash2_served as ds
from qualify.provenance import write_json

from .test_dflash2_served import Process


def test_arms_are_adaptive_then_fixed_counts_with_their_serving_flags():
    policies = sweep.arms("ckpt")
    assert [a["id"] for a in policies] == ["adaptive"] + [f"fixed-{n}" for n in range(8)]
    adaptive, floor, four = policies[0], policies[1], policies[5]
    assert adaptive["speculative_fixed_width"] is None and adaptive["mtp_draft_tokens"] is None and adaptive["adaptive_verification"]
    assert (floor["speculative_fixed_width"], floor["mtp_draft_tokens"], floor["adaptive_verification"]) == (1, 0, False)
    assert (four["id"], four["speculative_fixed_width"], four["mtp_draft_tokens"]) == ("fixed-4", 5, 4)
    assert all(a["model_id"] == "ckpt" and a["drafter"] == "dflash2" and a["history_drafts"] is False for a in policies)
    assert all(a["history_drafts"] for a in sweep.arms("ckpt", history_drafts=True))
    assert [a["id"] for a in sweep.select_arms(policies, ["fixed-4", "adaptive"])] == ["adaptive", "fixed-4"]
    with pytest.raises(ValueError, match="unknown arm"):
        sweep.select_arms(policies, ["fixed-9"])


def measurements(widths, *, offered=100, accepted=60, median=300., transport=True):
    delta = {}
    for width in widths:
        delta[f'flare_speculative_rounds_total{{batch="1",width="{width}"}}'] = 10
        delta[f'flare_speculative_offered_drafts_total{{batch="1",width="{width}"}}'] = offered // len(widths)
        delta[f'flare_speculative_accepted_drafts_total{{batch="1",width="{width}"}}'] = accepted // len(widths)
        delta[f'flare_speculative_emitted_tokens_total{{batch="1",width="{width}"}}'] = 10 + accepted // len(widths)
    delta['flare_speculative_drafts_total{depth="0",source="mtp",outcome="accepted"}'] = accepted
    return {"passed": transport, "transport_passed": transport, "errors": [],
            "groups": [{"metrics_delta": delta}],
            "measurements": {"code_edit": {"observed_group_output_tokens_per_second": {"count": 8, "median": median, "p99": median},
                                           "speculation": {"shapes": []}}}}


def test_trial_rows_carry_the_curve_columns_and_flag_only_overruns():
    policies = {a["id"]: a for a in sweep.arms("ckpt")}
    three = sweep.trial_row(measurements([4, 5]), policies["fixed-3"])
    assert three["fixed_width"] == 4 and three["width_overrun"] is True
    assert three["accepted_per_round"] == 60 / 20 and three["acceptance_rate"] == 0.6
    assert three["drafts_by_source"] == {"mtp": {"accepted": 60}} and three["code_edit_output_tokens_per_second"]["median"] == 300.
    # Per-width rows keep offered and accepted drafts beside rounds and emitted tokens.
    assert [(r["width"], r["rounds"], r["offered_drafts"], r["accepted_drafts"], r["emitted_tokens"]) for r in three["rounds_by_shape"]] == [
        (4, 10, 50, 30, 40), (5, 10, 50, 30, 40)]
    six = sweep.trial_row(measurements([3, 7]), policies["fixed-6"])
    assert six["fixed_width"] == 7 and six["width_overrun"] is False
    adaptive = sweep.trial_row(measurements([1, 8]), policies["adaptive"])
    assert adaptive["fixed_width"] is None and adaptive["width_overrun"] is False
    floor = sweep.trial_row(measurements([], offered=0, accepted=0), policies["fixed-0"])
    assert floor["accepted_per_round"] is None and floor["acceptance_rate"] is None and floor["width_overrun"] is False


def test_summary_and_table_cover_every_arm_and_ceiling(tmp_path):
    policies = sweep.select_arms(sweep.arms("ckpt"), ["adaptive", "fixed-0", "fixed-2"])
    for trial, widths, median in (("adaptive/c1-r0", [2, 3], 250.), ("adaptive/c8-r0", [2, 3], 500.), ("fixed-2/c1-r0", [3], 280.)):
        (tmp_path / trial).mkdir(parents=True)
        write_json(tmp_path / trial / "measurements.json", measurements(widths, median=median))
    (tmp_path / "fixed-0/c1-r0").mkdir(parents=True)
    write_json(tmp_path / "fixed-0/c1-r0/measurements.json", measurements([], offered=0, accepted=0, median=150.))
    summary = sweep.summarize(tmp_path, policies)
    assert summary["trials"] == 4 and set(summary["arms"]) == {"adaptive", "fixed-0", "fixed-2"}
    assert summary["arms"]["fixed-2"]["1"]["0"]["fixed_width"] == 3 and "8" not in summary["arms"]["fixed-2"]
    assert json.loads((tmp_path / "summary.json").read_text())["trials"] == 4
    text = sweep.table(summary)
    assert text.splitlines()[0].startswith("arm") and "c1:" in text and "c8:" in text
    assert any(line.startswith("adaptive") and "250.0" in line and "500.0" in line for line in text.splitlines())
    assert any(line.startswith("fixed-2") and "280.0" in line and "-" in line for line in text.splitlines())
    # The no-speculation floor has no acceptance figures; the table prints dashes, not a crash.
    assert any(line.startswith("fixed-0") and "150.0" in line and line.count("-") >= 2 for line in text.splitlines())


def test_run_serves_one_container_per_trial_in_sweep_order_with_the_arm_flags(tmp_path):
    started, stopped, benchmarked = [], [], []

    def popen(command, **kwargs):
        process = Process(command, **kwargs)
        started.append(process)
        return process

    def stop(command, **kwargs):
        stopped.append(command)
        assert started[-1].returncode is None

    def benchmark(args):
        assert len(started) == len(stopped) + 1
        benchmarked.append(args)
        widths = [args.arm_policy["speculative_fixed_width"] or 3]
        result = measurements(widths)
        args.output.mkdir(parents=True)
        write_json(args.output / "measurements.json", result)
        return result

    args = SimpleNamespace(workload=tmp_path / "workload", image="img", package=tmp_path / "pk/models/d/e/model.safetensors",
                           catalog=tmp_path / "catalog.json", publisher_keys=tmp_path / "ctx/keys.json",
                           trust_store=tmp_path / "ctx/trust.toml", output=tmp_path / "out", model_id="ckpt",
                           arms=None, concurrencies=[1, 8], repetitions=1, history_drafts=False, port=8111, timeout=1, ready_timeout=1)
    results = sweep.run(args, popen=popen, stop=stop, ready=lambda url: True, benchmark=benchmark)
    assert [(r["arm"], r["concurrency"]) for r in results] == [(a, c) for a in ["adaptive"] + [f"fixed-{n}" for n in range(8)] for c in (1, 8)]
    assert len(started) == len(stopped) == 18 and all(r["transport_passed"] and not r["width_overrun"] for r in results)
    assert stopped[0] == ["docker", "stop", "-t", "120", "dflash2-sweep-adaptive-c1-r0"]
    first, floor, four = started[0].command, started[2].command, started[10].command
    assert "--history-drafts" not in first and "--mtp-draft-tokens" not in first and "--no-mtp-adaptive-verification" not in first
    assert floor[-3:] == ["--mtp-draft-tokens", "0", "--no-mtp-adaptive-verification"]
    assert four[-3:] == ["--mtp-draft-tokens", "4", "--no-mtp-adaptive-verification"]
    assert four == ds.container_command("img", args.package, name="dflash2-sweep-fixed-4-c1-r0", uid=os.getuid(), gid=os.getgid(),
                                        home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                        port=8111, history_drafts=False, mtp_draft_tokens=4, adaptive_verification=False)
    assert benchmarked[10].arm == "fixed-4" and benchmarked[10].arm_policy["speculative_fixed_width"] == 5
    assert benchmarked[10].output == tmp_path / "out/fixed-4/c1-r0" and benchmarked[10].model_id == "ckpt"
    assert (tmp_path / "out/summary.json").is_file()

    started.clear(), stopped.clear(), benchmarked.clear()
    narrowed = SimpleNamespace(**{**vars(args), "output": tmp_path / "narrow", "arms": ["fixed-4", "adaptive"], "concurrencies": [8], "history_drafts": True})
    results = sweep.run(narrowed, popen=popen, stop=stop, ready=lambda url: True, benchmark=benchmark)
    assert [(r["arm"], r["concurrency"]) for r in results] == [("adaptive", 8), ("fixed-4", 8)]
    assert "--history-drafts" in started[0].command and benchmarked[0].arm_policy["history_drafts"] is True
