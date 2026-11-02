"""Run the served DFlash2 versus MTP comparison on one GPU, one serving container per trial.

Each trial of the paired execution plan starts `flare run PACKAGE` in the serving
container for the arm's package, waits for `/health`, records the frozen speculation
workload through `qualify.speculative_benchmark`, and stops the container before the
next trial starts. The driver is an HTTP client around the container; it never starts
a model from Python. `summarize` reduces the per-trial measurements to one JSON keyed
by arm and concurrency ceiling, never pooled across ceilings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from qualify import speculative_benchmark
from qualify.performance import connection
from qualify.provenance import write_json
from qualify.speculative_workload import arms

PORT = 8111
MIXED_LAUNCH = 'flare_sampling_graph_launches_total{kind="mixed"'
_DRAFT = re.compile(r'^flare_speculative_drafts_total\{depth="(\d+)",source="([^"]+)",outcome="([^"]+)"\}$')


def trials(plan, selection=None):
    """The plan's trials in execution order as (repetition, concurrency, arm); `selection` keeps the named ones."""
    ordered = [(trial["repetition"], trial["concurrency_ceiling"], arm)
               for trial in plan["trials"] for arm in trial["arm_order"]]
    if selection is None:
        return ordered
    wanted = set()
    for item in selection:
        repetition, concurrency, arm = item.split(":")
        key = (int(repetition), int(concurrency), arm)
        if key not in ordered:
            raise ValueError(f"trial {item} is not in the execution plan")
        wanted.add(key)
    return [trial for trial in ordered if trial in wanted]


def container_command(image, package, *, name, uid, gid, home, publisher_keys, trust_store, port=PORT):
    """The 3a probe's serving container: host network, GPU 0, the package cache and trust files at their host paths.

    The trust store already approves the development signer that exported both packages, so no
    first-use acceptance flag is passed.
    """
    package, home = Path(package), Path(home)
    publisher_keys, trust_store = Path(publisher_keys).resolve(), Path(trust_store).resolve()
    cache_root = package.parents[3]
    mounts = {cache_root: "ro", home / ".cache": None, home / ".config" / "flare": None,
              publisher_keys.parent: None, trust_store.parent: None}
    command = ["docker", "run", "--rm", "--name", name, "--gpus", "device=0",
               "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1:-1",
               "--ipc", "host", "--shm-size", "8g", "--network", "host", "-u", f"{uid}:{gid}"]
    for path, mode in mounts.items():
        command += ["-v", f"{path}:{path}" + (f":{mode}" if mode else "")]
    return command + ["-e", f"HOME={home}", "--entrypoint", "/usr/local/bin/flare", image,
                      "run", str(package), "--gpus", "0", "--history-drafts", "--max-concurrency", "8",
                      "--host", "0.0.0.0", "--port", str(port), "--offline",
                      "--publisher-keys", str(publisher_keys), "--trust-store", str(trust_store)]


def healthy(base_url):
    conn, prefix = connection(base_url, 5)
    try:
        conn.request("GET", prefix + "/health")
        return conn.getresponse().status == 200
    except OSError:
        return False
    finally:
        conn.close()


class Server:
    """One serving container, stopped with SIGTERM through `docker stop` however the trial ends."""

    def __init__(self, command, name, base_url, log, *, popen, stop, ready, ready_timeout):
        self.command, self.name, self.base_url, self.log = command, name, base_url, Path(log)
        self.popen, self.stop, self.ready, self.ready_timeout = popen, stop, ready, ready_timeout
        self.process = self.stream = None

    def __enter__(self):
        self.stream = self.log.open("wb")
        self.process = self.popen(self.command, stdout=self.stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.ready_timeout
        try:
            while not self.ready(self.base_url):
                if self.process.poll() is not None:
                    raise RuntimeError(f"serving container exited with {self.process.returncode}; see {self.log}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"serving container was not healthy within {self.ready_timeout}s; see {self.log}")
                time.sleep(1)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        try:
            self.stop(["docker", "stop", "-t", "120", self.name], check=False, capture_output=True)
            self.process.wait(timeout=180)
        finally:
            self.stream.close()


def run(args, *, popen=subprocess.Popen, stop=subprocess.run, ready=healthy, benchmark=speculative_benchmark.run):
    plan = json.loads(args.plan.read_text())
    packages = {"mtp-adaptive": (args.package_mtp, args.catalog_mtp),
                "dflash2-adaptive": (args.package_dflash2, args.catalog_dflash2)}
    base_url = f"http://127.0.0.1:{args.port}"
    results = []
    for repetition, concurrency, arm in trials(plan, args.trials):
        package, catalog = packages[arm]
        folder = args.output / arm / f"c{concurrency}-r{repetition}"
        folder.parent.mkdir(parents=True, exist_ok=True)
        name = f"dflash2-served-{arm}-c{concurrency}-r{repetition}"
        command = container_command(args.image, package, name=name, uid=os.getuid(), gid=os.getgid(),
                                    home=Path.home(), publisher_keys=args.publisher_keys,
                                    trust_store=args.trust_store, port=args.port)
        with Server(command, name, base_url, folder.parent / f"{folder.name}.server.log",
                    popen=popen, stop=stop, ready=ready, ready_timeout=args.ready_timeout):
            measurements = benchmark(SimpleNamespace(
                workload=args.workload, arm=arm, concurrency=concurrency, base_url=base_url,
                catalog=catalog, output=folder, repetition=repetition, timeout=args.timeout))
        results.append({"repetition": repetition, "concurrency": concurrency, "arm": arm,
                        "passed": measurements["passed"], "output": str(folder)})
        print(json.dumps(results[-1]), flush=True)
    return results


def _shapes(rows):
    return [{"batch": row["batch"], "width": row["width"], "rounds": row.get("rounds", 0),
             "emitted_tokens": row.get("emitted_tokens", 0)} for row in rows]


def trial_summary(measurements):
    totals = {}
    for group in measurements["groups"]:
        for key, value in group["metrics_delta"].items():
            totals[key] = totals.get(key, 0) + value
    drafts = {}
    for key, value in totals.items():
        if match := _DRAFT.fullmatch(key):
            _, source, outcome = match.groups()
            outcomes = drafts.setdefault(source, {})
            outcomes[outcome] = outcomes.get(outcome, 0) + value
    code_edit = measurements["measurements"].get("code_edit", {})
    return {"passed": measurements["passed"], "transport_passed": measurements.get("transport_passed", False),
            "errors": measurements.get("errors", []),
            "code_edit_output_tokens_per_second": code_edit.get("observed_group_output_tokens_per_second"),
            "code_edit_rounds_by_shape": _shapes(code_edit.get("speculation", {}).get("shapes", [])),
            "rounds_by_shape": _shapes(speculative_benchmark.round_metrics(totals)["shapes"]),
            "drafts_by_source": drafts,
            "mixed_graph_launches": sum(value for key, value in totals.items() if key.startswith(MIXED_LAUNCH)),
            "requests_failed": totals.get("flare_requests_failed_total", 0)}


def summarize(output):
    output = Path(output)
    summary = {"arms": {}, "trials": 0}
    for arm in (a["id"] for a in arms()):
        for path in sorted((output / arm).glob("c*-r*/measurements.json")):
            concurrency, repetition = path.parent.name[1:].split("-r")
            summary["arms"].setdefault(arm, {}).setdefault(concurrency, {})[repetition] = trial_summary(json.loads(path.read_text()))
            summary["trials"] += 1
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="run the execution plan's trials, one serving container each")
    runner.add_argument("--workload", type=Path, required=True, help="generated workload directory")
    runner.add_argument("--plan", type=Path, required=True, help="execution-plan JSON")
    runner.add_argument("--image", required=True, help="serving image tag, such as local/flare-ignition:f829c06f")
    runner.add_argument("--package-mtp", type=Path, required=True, help="MTP package model.safetensors")
    runner.add_argument("--package-dflash2", type=Path, required=True, help="DFlash2 package model.safetensors")
    runner.add_argument("--catalog-mtp", type=Path, required=True, help="flare inspect --json --graphs of the MTP package")
    runner.add_argument("--catalog-dflash2", type=Path, required=True, help="flare inspect --json --graphs of the DFlash2 package")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--trials", nargs="+", metavar="REPETITION:CONCURRENCY:ARM",
                        help="run only these trials of the plan (default: all, in plan order)")
    runner.add_argument("--publisher-keys", type=Path, required=True, help="publisher profile JSON that signed both packages")
    runner.add_argument("--trust-store", type=Path, required=True, help="trust store approving that signer")
    runner.add_argument("--port", type=int, default=PORT)
    runner.add_argument("--timeout", type=float, default=600, help="per-request deadline in seconds")
    runner.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for /health")
    reducer = sub.add_parser("summarize", help="write summary.json from the trials under --output")
    reducer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        results = run(args)
        raise SystemExit(0 if results and all(r["passed"] for r in results) else 1)
    summary = summarize(args.output)
    print(json.dumps({"trials": summary["trials"], "output": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
