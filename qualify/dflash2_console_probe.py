"""One trial of the frozen speculation workload against an instance the LocalAI console serves.

Runs `qualify.speculative_benchmark` with the draft sweep's fixed-7 arm policy (seven DFlash2 drafts
every round) at one concurrency ceiling and writes OUTPUT/c<concurrency>-r<repetition>/. It starts and
stops no server. The benchmark refuses a server that has already admitted requests, so each trial
needs a fresh console start. The package check is `validate_package`, which accepts any
prepared concurrency and packages without mixed graphs. The printed row carries the served acceptance counters and the ungated
code_edit rate of `qualify.dflash2_console_rate`.

    python -m qualify.dflash2_console_probe --workload DIR --catalog INSPECT.json --output DIR \\
        --concurrency 8 --repetition 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from qualify import speculative_benchmark
from qualify.dflash2_console_rate import rate
from qualify.dflash2_draft_sweep import arms, select_arms, trial_row


def validate_package(catalog, arm):
    """Check the inspected package against the trial: its model id and a speculative graph for every round.

    `speculative_benchmark.validate_arm` also requires eight request slots and mixed graphs, the shape
    of the packages its paired comparison was frozen on. Packages prepared at another concurrency, or
    by a recipe without mixed prefill, are measured here with the same workload and ceilings.
    """
    if catalog["model"]["model_id"] != arm["model_id"]:
        raise ValueError("package model_id differs from arm")
    graphs = catalog["ranks"][0]["execution_graphs"]
    speculative = {graph["name"] for graph in graphs if graph["kind"] == "speculative"}
    missing = [f"speculative{batch}x{width}" for batch in (1, 2, 4, 8) for width in range(1, 9)
               if f"speculative{batch}x{width}" not in speculative]
    if missing:
        raise ValueError(f"speculative graphs missing: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", type=Path, required=True, help="generated frozen workload directory")
    parser.add_argument("--catalog", type=Path, required=True,
                        help="JSON of `flare inspect PACKAGE --json --graphs --unverified` for the served package")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model-id", default="ckpt", help="model id the served responses carry")
    parser.add_argument("--timeout", type=float, default=600, help="per-request deadline in seconds")
    args = parser.parse_args()
    arm, = select_arms(arms(args.model_id), ["fixed-7"])
    folder = args.output / f"c{args.concurrency}-r{args.repetition}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    measurements = speculative_benchmark.run(SimpleNamespace(
        workload=args.workload, arm=arm["id"], arm_policy=arm, concurrency=args.concurrency,
        base_url=args.base_url, catalog=args.catalog, output=folder, repetition=args.repetition,
        timeout=args.timeout, model_id=args.model_id), validate=validate_package)
    row = trial_row(measurements, arm)
    print(json.dumps({"trial": str(folder), **rate(measurements), "width_overrun": row["width_overrun"],
                      "accepted_per_round": row["accepted_per_round"], "acceptance_rate": row["acceptance_rate"]}))


if __name__ == "__main__":
    main()
