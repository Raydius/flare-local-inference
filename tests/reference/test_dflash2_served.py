"""Trial selection, container command, lifecycle, and summary of the served comparison driver; no container runs."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from qualify import dflash2_served as ds
from qualify.provenance import write_json


ARMS = ["mtp-adaptive", "dflash2-adaptive"]


def plan():
    return {"trials": [{"repetition": repetition, "concurrency_ceiling": concurrency,
                        "arm_order": ARMS if repetition % 2 == 0 else list(reversed(ARMS))}
                       for repetition in range(2) for concurrency in (1, 2, 4, 8)]}


def test_trials_follow_plan_order_and_selection_keeps_plan_order():
    ordered = ds.trials(plan())
    assert len(ordered) == 16
    assert ordered[:4] == [(0, 1, "mtp-adaptive"), (0, 1, "dflash2-adaptive"), (0, 2, "mtp-adaptive"), (0, 2, "dflash2-adaptive")]
    assert ordered[8:10] == [(1, 1, "dflash2-adaptive"), (1, 1, "mtp-adaptive")]
    screen = ds.trials(plan(), ["0:8:dflash2-adaptive", "0:1:mtp-adaptive", "0:1:dflash2-adaptive", "0:8:mtp-adaptive"])
    assert screen == [(0, 1, "mtp-adaptive"), (0, 1, "dflash2-adaptive"), (0, 8, "mtp-adaptive"), (0, 8, "dflash2-adaptive")]
    with pytest.raises(ValueError, match="not in the execution plan"):
        ds.trials(plan(), ["0:3:mtp-adaptive"])
    with pytest.raises(ValueError, match="not in the execution plan"):
        ds.trials(plan(), ["2:1:mtp-adaptive"])


def test_container_command_serves_the_package_with_the_probe_flags():
    package = Path("/home/u/models/.cache/flare/packages/models/org_2Fmodel/gb202-tp1-1/model.safetensors")
    command = ds.container_command("local/flare-ignition:f829c06f", package, name="trial", uid=1000, gid=1000, home="/home/u",
                                   publisher_keys="/home/u/.cache/ctx/development-publisher.json", trust_store="/home/u/.cache/ctx/trust.toml")
    assert command[:5] == ["docker", "run", "--rm", "--name", "trial"]
    assert "--network" in command and command[command.index("--network") + 1] == "host"
    assert command[command.index("--gpus") + 1] == "device=0"
    assert command[command.index("-u") + 1] == "1000:1000"
    mounts = [command[i + 1] for i, item in enumerate(command) if item == "-v"]
    assert "/home/u/models/.cache/flare/packages:/home/u/models/.cache/flare/packages:ro" in mounts
    assert "/home/u/.config/flare:/home/u/.config/flare" in mounts
    assert "/home/u/.cache:/home/u/.cache" in mounts
    assert "/home/u/.cache/ctx:/home/u/.cache/ctx" in mounts
    assert command[command.index("-e") + 1] == "HOME=/home/u"
    entry = command.index("--entrypoint")
    assert command[entry + 1:entry + 5] == ["/usr/local/bin/flare", "local/flare-ignition:f829c06f", "run", str(package)]
    tail = command[entry + 5:]
    for flag, value in [("--gpus", "0"), ("--max-concurrency", "8"), ("--host", "0.0.0.0"), ("--port", "8111"),
                        ("--publisher-keys", "/home/u/.cache/ctx/development-publisher.json"),
                        ("--trust-store", "/home/u/.cache/ctx/trust.toml")]:
        assert tail[tail.index(flag) + 1] == value
    assert "--history-drafts" in tail and "--offline" in tail
    # `--dangerously-accept-publisher` requires `--dangerously-approve-key`; the trust store carries the approval instead.
    assert "--dangerously-accept-publisher" not in tail
    assert "--served-model-name" not in tail and "--mtp-draft-tokens" not in tail


def test_container_command_can_drop_history_drafts_and_raise_the_served_context():
    package = Path("/home/u/models/.cache/flare/packages/models/org_2Fmodel/gb202-tp1-1/model.safetensors")
    options = dict(name="probe", uid=1000, gid=1000, home="/home/u",
                   publisher_keys="/home/u/.cache/ctx/development-publisher.json", trust_store="/home/u/.cache/ctx/trust.toml")
    default = ds.container_command("img", package, **options)
    assert default == ds.container_command("img", package, history_drafts=True, max_context_tokens=None, **options)
    probe = ds.container_command("img", package, history_drafts=False, max_context_tokens=1_048_576, **options)
    tail = probe[probe.index("--entrypoint"):]
    assert "--history-drafts" not in tail and tail[tail.index("--max-context-tokens") + 1] == "1048576"
    assert "--max-context-tokens" not in default
    assert [item for item in default if item != "--history-drafts"] == probe[:-2]


class Process:
    def __init__(self, command, *, stdout, stderr):
        assert stderr is subprocess.STDOUT
        self.command, self.stdout, self.returncode = command, stdout, None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def driver(tmp_path, **overrides):
    write_json(tmp_path / "plan.json", plan())
    args = SimpleNamespace(workload=tmp_path / "workload", plan=tmp_path / "plan.json", image="img",
                           package_mtp=tmp_path / "pk/models/m/e/model.safetensors",
                           package_dflash2=tmp_path / "pk/models/d/e/model.safetensors",
                           catalog_mtp=tmp_path / "mtp.json", catalog_dflash2=tmp_path / "dflash2.json",
                           output=tmp_path / "out", trials=None, port=8111, timeout=12.5, ready_timeout=5,
                           publisher_keys=tmp_path / "ctx/development-publisher.json", trust_store=tmp_path / "ctx/trust.toml",
                           dflash2_model_id=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_run_starts_one_container_per_trial_and_stops_it_after_the_benchmark(tmp_path):
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
        return {"passed": args.arm == "mtp-adaptive"}

    args = driver(tmp_path, trials=["0:1:mtp-adaptive", "0:8:dflash2-adaptive"], dflash2_model_id="ckpt")
    results = ds.run(args, popen=popen, stop=stop, ready=lambda url: True, benchmark=benchmark)
    assert [(r["repetition"], r["concurrency"], r["arm"], r["passed"]) for r in results] == [
        (0, 1, "mtp-adaptive", True), (0, 8, "dflash2-adaptive", False)]
    assert len(started) == len(stopped) == 2
    assert stopped[0] == ["docker", "stop", "-t", "120", "dflash2-served-mtp-adaptive-c1-r0"]
    assert started[1].command[started[1].command.index("--entrypoint") + 4] == str(args.package_dflash2)
    first, second = benchmarked
    assert (first.arm, first.concurrency, first.repetition, first.catalog) == ("mtp-adaptive", 1, 0, args.catalog_mtp)
    assert (second.arm, second.concurrency, second.catalog) == ("dflash2-adaptive", 8, args.catalog_dflash2)
    # Only the DFlash2 arm carries the directory export's model id.
    assert (first.model_id, second.model_id) == (None, "ckpt")
    assert first.output == tmp_path / "out" / "mtp-adaptive" / "c1-r0" and first.base_url == "http://127.0.0.1:8111"
    assert first.workload == args.workload and first.timeout == 12.5
    assert (tmp_path / "out" / "mtp-adaptive" / "c1-r0.server.log").exists()
    assert all(process.returncode == 0 for process in started)


def test_container_is_stopped_when_the_benchmark_fails_or_the_server_never_answers(tmp_path):
    stopped = []

    def stop(command, **kwargs):
        stopped.append(command[-1])

    def failing(args):
        raise RuntimeError("benchmark failed")

    args = driver(tmp_path, trials=["0:1:mtp-adaptive"])
    with pytest.raises(RuntimeError, match="benchmark failed"):
        ds.run(args, popen=Process, stop=stop, ready=lambda url: True, benchmark=failing)
    assert stopped == ["dflash2-served-mtp-adaptive-c1-r0"]

    class Exited(Process):
        def poll(self):
            self.returncode = 3
            return 3

    args = driver(tmp_path, trials=["0:1:dflash2-adaptive"], output=tmp_path / "out2")
    with pytest.raises(RuntimeError, match="exited with 3"):
        ds.run(args, popen=Exited, stop=stop, ready=lambda url: False, benchmark=lambda a: pytest.fail("no benchmark"))
    assert stopped[-1] == "dflash2-served-dflash2-adaptive-c1-r0"


def measurements(arm, passed=True, mixed=0, failed=0):
    deltas = [{'flare_speculative_rounds_total{batch="1",width="8"}': 3,
               'flare_speculative_emitted_tokens_total{batch="1",width="8"}': 15,
               'flare_speculative_drafts_total{depth="1",source="mtp",outcome="offered"}': 4,
               'flare_speculative_drafts_total{depth="1",source="mtp",outcome="accepted"}': 3,
               'flare_speculative_drafts_total{depth="2",source="copy",outcome="offered"}': 2,
               'flare_sampling_graph_launches_total{kind="mixed",mode="general"}': mixed,
               "flare_requests_failed_total": failed},
              {'flare_speculative_rounds_total{batch="8",width="4"}': 2,
               'flare_speculative_emitted_tokens_total{batch="8",width="4"}': 20,
               'flare_speculative_drafts_total{depth="1",source="mtp",outcome="offered"}': 1,
               'flare_sampling_graph_launches_total{kind="mixed",mode="general"}': mixed}]
    return {"arm": {"id": arm}, "passed": passed, "transport_passed": passed, "errors": [],
            "groups": [{"metrics_delta": delta} for delta in deltas],
            "measurements": {"code_edit": {
                "observed_group_output_tokens_per_second": {"count": 1, "median": 250.0, "p99": 250.0, "minimum": 250.0, "maximum": 250.0},
                "speculation": {"shapes": [{"batch": 8, "width": 4, "rounds": 2, "emitted_tokens": 20}]}}}}


def test_summary_is_keyed_by_arm_and_concurrency_and_never_pooled(tmp_path):
    for arm, concurrency, repetition, mixed in [("mtp-adaptive", 1, 0, 0), ("mtp-adaptive", 8, 0, 5),
                                                ("mtp-adaptive", 8, 1, 6), ("dflash2-adaptive", 8, 0, 7)]:
        folder = tmp_path / arm / f"c{concurrency}-r{repetition}"
        folder.mkdir(parents=True)
        write_json(folder / "measurements.json", measurements(arm, passed=arm == "mtp-adaptive", mixed=mixed, failed=repetition))
    summary = ds.summarize(tmp_path)
    assert summary == json.loads((tmp_path / "summary.json").read_text())
    assert summary["trials"] == 4
    assert set(summary["arms"]) == {"mtp-adaptive", "dflash2-adaptive"}
    assert set(summary["arms"]["mtp-adaptive"]) == {"1", "8"}
    assert set(summary["arms"]["mtp-adaptive"]["8"]) == {"0", "1"}
    trial = summary["arms"]["mtp-adaptive"]["8"]["1"]
    assert trial["passed"] and trial["mixed_graph_launches"] == 12 and trial["requests_failed"] == 1
    assert trial["code_edit_output_tokens_per_second"]["median"] == 250.0
    assert trial["code_edit_rounds_by_shape"] == [{"batch": 8, "width": 4, "rounds": 2, "emitted_tokens": 20}]
    assert trial["rounds_by_shape"] == [{"batch": 1, "width": 8, "rounds": 3, "emitted_tokens": 15},
                                        {"batch": 8, "width": 4, "rounds": 2, "emitted_tokens": 20}]
    assert trial["drafts_by_source"] == {"mtp": {"offered": 5, "accepted": 3}, "copy": {"offered": 2}}
    assert not summary["arms"]["dflash2-adaptive"]["8"]["0"]["passed"]
    assert "pooled" not in json.dumps(summary)


def test_help_lists_every_run_flag(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["dflash2_served", "run", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        ds.main()
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--workload", "--plan", "--image", "--package-mtp", "--package-dflash2", "--catalog-mtp",
                 "--catalog-dflash2", "--output", "--trials", "--publisher-keys", "--trust-store"):
        assert flag in text
