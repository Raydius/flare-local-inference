"""Passkey retrieval at chosen depths of prompts below and beyond the native 262,144-token context.

One serving container per run (`flare run PACKAGE`, history drafts off). Each cell of a
lengths × depths grid sends the frozen workload's service catalog, sized to the target prompt
tokens, with one passkey sentence spliced between catalog lines at the depth fraction, and asks
for the pass key under `top_k=1` (the runtime refuses `temperature=0`). A cell passes when the
reply contains the passkey. Cells run longest length first, depths ascending, so no prompt is a
prefix of an earlier one. `summary.json` holds every cell, a pass/fail grid, and `all_passed`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess

from qualify.dflash2_long_context import draft_counts, measure, records_for
from qualify.dflash2_served import PORT, Server, container_command, healthy
from qualify.provenance import write_json
from qualify.speculative_workload import SAMPLING, _catalog, _message, value_sha

LENGTHS = (200_000, 320_000, 520_000, 1_000_000)
DEPTHS = (0.1, 0.5, 0.9)
QUESTION = "What is the pass key? The pass key is"


def needle(passkey):
    return f"The pass key is {passkey}. Remember it. {passkey} is the pass key."


def passkey_for(seed, length, depth, records):
    """A five-digit passkey for one cell, redrawn while it occurs in that cell's catalog text."""
    generator = random.Random(f"{seed}:{length}:{depth}")
    catalog = _catalog(records)
    while True:
        candidate = f"{generator.randrange(10_000, 100_000)}"
        if candidate not in catalog:
            return candidate


def prompt(records, depth, passkey, *, model, seed=0):
    """The catalog with the needle at depth, then the pass-key question, as a benchmark case."""
    lines = _catalog(records).split("\n")
    at = round(depth * records)
    haystack = "\n".join(lines[:at] + [needle(passkey)] + lines[at:])
    request = dict(model=model, messages=[
        _message("system", "Use only the supplied service records."),
        _message("user", haystack),
        _message("user", QUESTION)],
        tools=[], chat_template_kwargs={"enable_thinking": False, "add_generation_prompt": True},
        **{**SAMPLING, "top_k": 1}, seed=seed, max_tokens=32, stream=True, stream_options={"include_usage": True},
        # No token-id echo: a prompt past the native context would render the first event beyond
        # the benchmark reader's 1 MiB line bound.
        return_token_ids=False, return_token_timestamps=False)
    return dict(id=f"passkey-{records}-{depth}", request=request, request_sha256=value_sha(request),
                prompt_tokens=[], expected_output_semantics={"kind": "text"})


def check(text, passkey):
    return passkey in text


def cell(observation, delta, *, length, depth, passkey):
    analysis = observation["analysis"]
    usage = analysis.get("usage") or {}
    elapsed = observation.get("elapsed_ns")
    content = analysis.get("content") or ""
    return {"case": observation["case"], "length": length, "depth": depth, "passkey": passkey,
            "prompt_tokens": usage.get("prompt_tokens"), "generated_tokens": usage.get("completion_tokens") or 0,
            "reply": content, "passed": check(content, passkey), "finish_reason": analysis.get("finish_reason"),
            "completed": "client_error" not in observation and analysis["checks"]["no_stream_error"]
                         and analysis["checks"]["one_valid_terminal"],
            "client_error": observation.get("client_error"),
            "request_seconds": elapsed / 1e9 if elapsed else None,
            "accepted_per_round": draft_counts(delta)["accepted_per_round"]}


def summary(cells):
    grid = {}
    for row in cells:
        grid.setdefault(str(row["length"]), {})[str(row["depth"])] = row["passed"]
    return {"cells": cells, "grid": grid, "all_passed": bool(cells) and all(row["passed"] for row in cells)}


def run(args, *, popen=subprocess.Popen, stop=subprocess.run, ready=healthy, measure=measure):
    base_url = f"http://127.0.0.1:{args.port}"
    args.output.mkdir(parents=True, exist_ok=True)
    command = container_command(args.image, args.package, name="long-context-passkey", uid=os.getuid(), gid=os.getgid(),
                                home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                port=args.port, history_drafts=False, max_context_tokens=args.max_context_tokens)
    cells = []
    with Server(command, "long-context-passkey", base_url, args.output / "server.log",
                popen=popen, stop=stop, ready=ready, ready_timeout=args.ready_timeout):
        for length in sorted(args.lengths, reverse=True):
            records = records_for(length)
            for depth in sorted(args.depths):
                passkey = passkey_for(args.seed, length, depth, records)
                case = prompt(records, depth, passkey, model=args.model_id, seed=args.seed)
                folder = args.output / f"{length}-{depth}"
                observation, delta = measure(base_url, case, folder, timeout=args.timeout, model_id=args.model_id)
                row = cell(observation, delta, length=length, depth=depth, passkey=passkey)
                cells.append(row)
                print(json.dumps({k: row[k] for k in ("length", "depth", "prompt_tokens", "passed", "completed",
                                                      "request_seconds", "accepted_per_round")}), flush=True)
    result = summary(cells)
    write_json(args.output / "summary.json", result)
    print(json.dumps({"grid": result["grid"], "all_passed": result["all_passed"],
                      "output": str(args.output / "summary.json")}), flush=True)
    return result


def _numbers(kind):
    return lambda text: tuple(kind(item) for item in text.split(",") if item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="serving image tag, such as local/flare-ignition:2ca523b2")
    parser.add_argument("--package", type=Path, required=True, help="package model.safetensors")
    parser.add_argument("--publisher-keys", type=Path, required=True, help="publisher profile JSON that signed the package")
    parser.add_argument("--trust-store", type=Path, required=True, help="trust store approving that signer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="ckpt", help="model id the package's responses carry")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="served context to request from the package (omit to serve the package's prepared ceiling; the flag is the fork's, upstream flare has none)")
    parser.add_argument("--lengths", type=_numbers(int), default=LENGTHS, help="target prompt tokens, comma-separated")
    parser.add_argument("--depths", type=_numbers(float), default=DEPTHS, help="needle depth fractions, comma-separated")
    parser.add_argument("--seed", type=int, default=48_300, help="passkey and sampling seed")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--timeout", type=float, default=900, help="per-request deadline in seconds")
    parser.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for /health")
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
