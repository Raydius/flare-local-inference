"""Probe a rope-scaled DFlash2 package past the native 262,144-token context, one serving container.

Two retrieval prompts built from the frozen workload's service-record catalog are sent to
`flare run PACKAGE` through the native HTTP service: one past the native context (sent first,
so no prefix is reused) and one below it. History copies are off, so the drafter counters
measure the DFlash2 draft alone. Each prompt records whether the three quoted records came
back, the offered and accepted draft counts, accepted drafts per speculative round, and output
tokens per second; `summary.json` puts the two buckets side by side with their ratio against
the 10 % criterion. The driver is an HTTP client around the container; it never starts a model
from Python, and the frozen 28-case workload is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading

from qualify import speculative_benchmark
from qualify.dflash2_served import PORT, Server, container_command, drafts_by_source, healthy
from qualify.provenance import write_json
from qualify.speculative_workload import SAMPLING, _catalog, _message, json_bytes, value_sha

NATIVE_CONTEXT = 262_144
# Measured on the frozen workload: `long-services-768` renders to 51,172 prompt tokens.
TOKENS_PER_RECORD = 51_172 / 768
CRITERION = 0.9


def records_for(tokens, tokens_per_record=TOKENS_PER_RECORD):
    """The catalog size whose prompt renders to about `tokens` prompt tokens at `tokens_per_record`."""
    return max(3, round(tokens / tokens_per_record))


def record_ids(records):
    return (7, records // 2, records - 1)


def prompt(records, *, model):
    """The workload's long-services request over a `records`-record catalog, as a benchmark case."""
    first, middle, last = record_ids(records)
    request = dict(model=model, messages=[
        _message("system", "Use only the supplied service records."),
        _message("user", _catalog(records)),
        _message("user", f"Compare service-{first:04d}, service-{middle:04d}, and service-{last:04d}. Quote each "
                         "timeout, retry limit, and owner, then recommend which deserves the first retry-policy review.")],
        tools=[], chat_template_kwargs={"enable_thinking": False, "add_generation_prompt": True},
        **SAMPLING, seed=48200 + records, max_tokens=512, stream=True, stream_options={"include_usage": True},
        # No token-id echo: the first event would carry the whole prompt's ids, and a prompt past the
        # native context renders that line beyond the benchmark reader's 1 MiB line bound.
        return_token_ids=False, return_token_timestamps=False)
    return dict(id=f"long-services-{records}", request=request, request_sha256=value_sha(request),
                prompt_tokens=[], expected_output_semantics={"kind": "text"})


def expected_values(records):
    """The `timeout_ms`, `retry_limit`, and `owner` values of the three quoted records, as `_catalog` prints them."""
    return {f"service-{i:04d}": {"timeout_ms": str(700 + i % 13 * 50), "retry_limit": str(i % 4), "owner": f"team-{i % 17:02d}"}
            for i in record_ids(records)}


def check_retrieval(text, expected):
    """Every expected value appears in the answer; records whose values are missing are listed."""
    missing = {record: [field for field, value in fields.items() if value not in text]
               for record, fields in expected.items()}
    return {"passed": not any(missing.values()), "missing": {k: v for k, v in missing.items() if v}}


def draft_counts(delta):
    rounds = speculative_benchmark.round_metrics(delta)["shapes"]
    offered = sum(row.get("offered_drafts", 0) for row in rounds)
    accepted = sum(row.get("accepted_drafts", 0) for row in rounds)
    launches = sum(row.get("rounds", 0) for row in rounds)
    return {"offered_drafts": offered, "accepted_drafts": accepted, "speculative_rounds": launches,
            "accepted_per_round": accepted / launches if launches else None,
            "acceptance_rate": accepted / offered if offered else None,
            "rounds_by_shape": rounds, "drafts_by_source": drafts_by_source(delta)}


def bucket(observation, expected, delta):
    analysis = observation["analysis"]
    usage = analysis.get("usage") or {}
    elapsed = observation.get("elapsed_ns")
    generated = usage.get("completion_tokens") or 0
    return {"case": observation["case"], "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": generated, "finish_reason": analysis["finish_reason"],
            "completed": "client_error" not in observation and analysis["checks"]["no_stream_error"]
                         and analysis["checks"]["one_valid_terminal"],
            "client_error": observation.get("client_error"),
            # Over the whole request, prefill included; no per-token timestamps are requested.
            "request_seconds": elapsed / 1e9 if elapsed else None,
            "output_tokens_per_second": generated / (elapsed / 1e9) if elapsed else None,
            "retrieval": check_retrieval(analysis["content"], expected), "expected_values": expected,
            **draft_counts(delta)}


def summary(past_native, sub_native):
    ratio = (past_native["accepted_per_round"] / sub_native["accepted_per_round"]
             if past_native["accepted_per_round"] and sub_native["accepted_per_round"] else None)
    retrieval = past_native["retrieval"]["passed"] and sub_native["retrieval"]["passed"]
    past_native_long_enough = (past_native["prompt_tokens"] or 0) > NATIVE_CONTEXT
    return {"past_native": past_native, "sub_native": sub_native, "ratio": ratio, "criterion": CRITERION,
            "retrieval_passed": retrieval, "past_native_exceeds_native_context": past_native_long_enough,
            "criterion_met": bool(retrieval and past_native_long_enough and ratio is not None and ratio >= CRITERION)}


def measure(base_url, case, folder, *, timeout, model_id):
    folder.mkdir(parents=True, exist_ok=True)
    before = speculative_benchmark.read_metrics(base_url, folder / "metrics-before.txt", timeout)
    observation = speculative_benchmark.stream_request(base_url, case, json_bytes(case["request"]), folder / "request",
                                                       timeout=timeout, barrier=threading.Barrier(1), model_id=model_id)
    after = speculative_benchmark.read_metrics(base_url, folder / "metrics-after.txt", timeout)
    return observation, speculative_benchmark.metric_delta(before, after)


def run(args, *, popen=subprocess.Popen, stop=subprocess.run, ready=healthy, measure=measure):
    base_url = f"http://127.0.0.1:{args.port}"
    args.output.mkdir(parents=True, exist_ok=True)
    command = container_command(args.image, args.package, name="dflash2-long-context", uid=os.getuid(), gid=os.getgid(),
                                home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                port=args.port, history_drafts=False, max_context_tokens=args.max_context_tokens)
    buckets = {}
    with Server(command, "dflash2-long-context", base_url, args.output / "server.log",
                popen=popen, stop=stop, ready=ready, ready_timeout=args.ready_timeout):
        for name, tokens in (("past_native", args.past_native_tokens), ("sub_native", args.sub_native_tokens)):
            records = records_for(tokens)
            case = prompt(records, model=args.model_id)
            observation, delta = measure(base_url, case, args.output / name, timeout=args.timeout, model_id=args.model_id)
            buckets[name] = bucket(observation, expected_values(records), delta)
            print(json.dumps({name: {k: buckets[name][k] for k in ("prompt_tokens", "generated_tokens", "completed",
                                                                    "accepted_per_round", "acceptance_rate")}
                              | {"retrieval_passed": buckets[name]["retrieval"]["passed"]}}), flush=True)
    result = summary(buckets["past_native"], buckets["sub_native"])
    write_json(args.output / "summary.json", result)
    print(json.dumps({"ratio": result["ratio"], "criterion": CRITERION, "criterion_met": result["criterion_met"],
                      "output": str(args.output / "summary.json")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="serving image tag, such as local/flare-ignition:a2971de5")
    parser.add_argument("--package", type=Path, required=True, help="rope-scaled DFlash2 package model.safetensors")
    parser.add_argument("--publisher-keys", type=Path, required=True, help="publisher profile JSON that signed the package")
    parser.add_argument("--trust-store", type=Path, required=True, help="trust store approving that signer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="ckpt", help="model id the package's responses carry (a directory export reports its directory name)")
    parser.add_argument("--max-context-tokens", type=int, default=1_048_576, help="served context to request from the package")
    parser.add_argument("--past-native-tokens", type=int, default=320_000, help="target prompt size past the native context")
    parser.add_argument("--sub-native-tokens", type=int, default=200_000, help="target prompt size below the native context")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--timeout", type=float, default=600, help="per-request deadline in seconds")
    parser.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for /health")
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(0 if result["criterion_met"] else 1)


if __name__ == "__main__":
    main()
