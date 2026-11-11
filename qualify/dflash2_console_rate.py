"""code_edit output tokens per second of speculation-benchmark trials served by the LocalAI console.

`qualify.speculative_benchmark` withholds its rate when `http_integrity` fails, and a console-served
instance always fails it: the console's chat template and generation defaults change the prompt token
ids, so `exact_prompt_ids` cannot hold. `rate` computes the benchmark's figure without that gate:
emitted tokens over the end-to-end span of each code_edit group, reduced to median, minimum and maximum
over the groups. Compare its numbers only with another instance measured the same way.

Given several trial directories, the command also prints the median and the min-max spread of the
trials' medians.

    python -m qualify.dflash2_console_rate TRIAL_DIR [TRIAL_DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def rate(measurements):
    """One trial's ungated code_edit rate, request failures, and DFlash2 acceptance counters."""
    groups = [group for group in measurements["groups"] if group["category"] == "code_edit"]
    if not groups:
        raise ValueError("the trial has no code_edit group")
    rates = [group["actual_output_tokens"] * 1e9 / group["end_to_end_group_span_ns"] for group in groups]
    observations = measurements["observations"]
    failed = sum(o["status"] != 200 or not o["analysis"]["checks"]["no_stream_error"] for o in observations)
    return {"groups": len(groups), "code_edit_median_tok_s": round(statistics.median(rates), 1),
            "min": round(min(rates), 1), "max": round(max(rates), 1),
            "requests": len(observations), "requests_failed": failed,
            "failing_checks": sorted({name for o in observations
                                      for name, passed in o["analysis"]["checks"].items() if not passed})}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", type=Path, help="trial directories holding measurements.json")
    args = parser.parse_args()
    medians = []
    for trial in args.trials:
        row = rate(json.loads((trial / "measurements.json").read_text()))
        medians.append(row["code_edit_median_tok_s"])
        print(json.dumps({"trial": str(trial), **row}))
    if len(medians) > 1:
        print(json.dumps({"trials": len(medians), "median_tok_s": statistics.median(medians),
                          "min": min(medians), "max": max(medians)}))


if __name__ == "__main__":
    main()
