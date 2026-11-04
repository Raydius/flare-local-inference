"""Sweep the served draft count of one DFlash2 package, one serving container per trial.

Arms are the adaptive width controller as shipped and fixed draft counts 0 through 7
(`flare run --mtp-draft-tokens N --no-mtp-adaptive-verification`, every round at width N + 1
where proposals and pending inputs allow; `fixed-0` serves without speculation). Each arm runs
the frozen speculation workload through `qualify.speculative_benchmark` at concurrency ceilings
1 and 8 on a fresh server with an empty prefix cache, history copies off unless asked, and is
reduced with the served comparison's `trial_summary`. `summary.json` and the printed table
give code_edit tokens per second, accepted drafts per round, and acceptance rate per arm and
ceiling; the operator reads the draft count off the curve. The driver is an HTTP client around
the container; it never starts a model from Python, and the frozen workload, its two comparison
arms, and its execution plan are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from qualify import speculative_benchmark
from qualify.dflash2_long_context import draft_counts
from qualify.dflash2_served import PORT, Server, container_command, healthy, metrics_totals, trial_summary
from qualify.provenance import write_json

DRAFT_COUNTS = range(0, 8)
CONCURRENCIES = (1, 8)


def arms(model_id, *, history_drafts=False):
    """The adaptive controller first, then fixed draft counts 0 through 7, as benchmark arm policies."""
    common = {"drafter": "dflash2", "model_id": model_id, "history_drafts": history_drafts}
    adaptive = {"id": "adaptive", **common, "speculative_fixed_width": None,
                "mtp_draft_tokens": None, "adaptive_verification": True}
    fixed = [{"id": f"fixed-{n}", **common, "speculative_fixed_width": n + 1,
              "mtp_draft_tokens": n, "adaptive_verification": False} for n in DRAFT_COUNTS]
    return [adaptive, *fixed]


def select_arms(policies, names):
    """The named arms in the sweep's order; an unknown name is an error."""
    if names is None:
        return list(policies)
    known = {arm["id"] for arm in policies}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"unknown arm(s) {', '.join(unknown)}; expected {', '.join(sorted(known))}")
    return [arm for arm in policies if arm["id"] in set(names)]


def trial_row(measurements, arm):
    """The served summary of one trial plus the draft-count curve's columns."""
    row = trial_summary(measurements)
    counts = draft_counts(metrics_totals(measurements))
    width = arm["speculative_fixed_width"]
    # The per-width rows keep offered and accepted drafts, which the served summary strips.
    row.update(arm=arm["id"], fixed_width=width,
               **{key: counts[key] for key in ("rounds_by_shape", "offered_drafts", "accepted_drafts",
                                                "speculative_rounds", "accepted_per_round", "acceptance_rate")},
               width_overrun=width is not None and any(r["width"] > width for r in counts["rounds_by_shape"]))
    return row


def summarize(output, policies, *, concurrencies=CONCURRENCIES, repetitions=1):
    """`summary.json` keyed arm → ceiling → repetition from the trial measurements under `output`."""
    output = Path(output)
    summary = {"arms": {}, "trials": 0}
    for arm in policies:
        for concurrency in concurrencies:
            for repetition in range(repetitions):
                path = output / arm["id"] / f"c{concurrency}-r{repetition}" / "measurements.json"
                if not path.is_file():
                    continue
                row = trial_row(json.loads(path.read_text()), arm)
                summary["arms"].setdefault(arm["id"], {}).setdefault(str(concurrency), {})[str(repetition)] = row
                summary["trials"] += 1
    write_json(output / "summary.json", summary)
    return summary


def _cell(value, digits):
    return "-" if value is None else f"{value:.{digits}f}"


def table(summary):
    """Arm × ceiling → code_edit median tokens per second, accepted per round, acceptance rate."""
    ceilings = sorted({c for arm in summary["arms"].values() for c in arm}, key=int)
    lines = ["arm        " + "".join(f"  c{c}: tok/s  acc/round   rate  " for c in ceilings)]
    for arm, by_ceiling in summary["arms"].items():
        cells = []
        for c in ceilings:
            rows = list(by_ceiling.get(c, {}).values())
            if not rows:
                cells.append(f"  {'-':>9}  {'-':>9}  {'-':>5}  ")
                continue
            row = rows[0]
            tokens = _cell((row["code_edit_output_tokens_per_second"] or {}).get("median"), 1)
            flag = "!" if row["width_overrun"] or not row["transport_passed"] else " "
            cells.append(f"  {tokens:>9}  {_cell(row['accepted_per_round'], 2):>9}  {_cell(row['acceptance_rate'], 3):>5}{flag} ")
        lines.append(f"{arm:<10}" + "".join(cells))
    return "\n".join(lines)


def run(args, *, popen=subprocess.Popen, stop=subprocess.run, ready=healthy, benchmark=speculative_benchmark.run):
    policies = select_arms(arms(args.model_id, history_drafts=args.history_drafts), args.arms)
    base_url = f"http://127.0.0.1:{args.port}"
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for repetition in range(args.repetitions):
        for arm in policies:
            for concurrency in args.concurrencies:
                folder = args.output / arm["id"] / f"c{concurrency}-r{repetition}"
                folder.parent.mkdir(parents=True, exist_ok=True)
                name = f"dflash2-sweep-{arm['id']}-c{concurrency}-r{repetition}"
                command = container_command(args.image, args.package, name=name, uid=os.getuid(), gid=os.getgid(),
                                            home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                            port=args.port, history_drafts=arm["history_drafts"],
                                            mtp_draft_tokens=arm["mtp_draft_tokens"],
                                            adaptive_verification=arm["adaptive_verification"])
                with Server(command, name, base_url, folder.parent / f"{folder.name}.server.log",
                            popen=popen, stop=stop, ready=ready, ready_timeout=args.ready_timeout):
                    measurements = benchmark(SimpleNamespace(
                        workload=args.workload, arm=arm["id"], arm_policy=arm, concurrency=concurrency,
                        base_url=base_url, catalog=args.catalog, output=folder, repetition=repetition,
                        timeout=args.timeout, model_id=args.model_id))
                row = trial_row(measurements, arm)
                results.append({"repetition": repetition, "concurrency": concurrency, "arm": arm["id"],
                                "transport_passed": row["transport_passed"], "width_overrun": row["width_overrun"],
                                "code_edit_median_tokens_per_second": (row["code_edit_output_tokens_per_second"] or {}).get("median"),
                                "accepted_per_round": row["accepted_per_round"], "output": str(folder)})
                print(json.dumps(results[-1]), flush=True)
    summary = summarize(args.output, policies, concurrencies=args.concurrencies, repetitions=args.repetitions)
    print(table(summary), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True, help="generated frozen workload directory")
    parser.add_argument("--image", required=True, help="serving image tag, such as local/flare-ignition:a945c147")
    parser.add_argument("--package", type=Path, required=True, help="DFlash2 package model.safetensors")
    parser.add_argument("--catalog", type=Path, required=True, help="flare inspect --json --graphs of the package")
    parser.add_argument("--publisher-keys", type=Path, required=True, help="publisher profile JSON that signed the package")
    parser.add_argument("--trust-store", type=Path, required=True, help="trust store approving that signer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="ckpt", help="model id the package's responses carry (a directory export reports its directory name)")
    parser.add_argument("--arms", nargs="+", metavar="ARM", help="run only these arms (adaptive, fixed-0 … fixed-7), in sweep order")
    parser.add_argument("--concurrencies", nargs="+", type=int, choices=(1, 2, 4, 8), default=list(CONCURRENCIES))
    parser.add_argument("--repetitions", type=int, default=1, help="repeat the whole arm order this many times")
    parser.add_argument("--history-drafts", action="store_true", help="serve with history copies on, as the served comparison did")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--timeout", type=float, default=600, help="per-request deadline in seconds")
    parser.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for /health")
    args = parser.parse_args()
    results = run(args)
    raise SystemExit(0 if results and all(r["transport_passed"] and not r["width_overrun"] for r in results) else 1)


if __name__ == "__main__":
    main()
