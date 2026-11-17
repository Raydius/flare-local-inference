# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Elephant-1 decision probabilities from a served core with the vendor runtime.

Elephant-1 answers a question by one prefill: the prompt ends at an answer line, and the option
probabilities are the next-token distribution restricted to the option letters. The vendor runtime
(OneJev `qev.engine.DecisionEngine` at the commit and patch the checkpoint's `setup.sh` pins) runs
the eight full-attention layers with the trained mask, bidirectional inside the shared prefix and
inside the question suffix. A prepared `elephant-1-9b` core runs every layer causally and stores the
body projections as NVFP4. Two stages separate those deviations on one fixed probe set:

  stage A  the BF16 checkpoint with causal attention against the vendor runtime as trained: the mask
  stage B  the served core against the BF16 checkpoint with causal attention: storage precision and
           the prepared kernels

The BF16 causal forward is the vendor runtime's own Transformers model with the trained mask
switched off, so both BF16 sides come from one loaded model and differ only in the mask. Every
branch runs alone, without the vendor's batched prefix fork, whose BF16 rounding differs by batch.

`reference` runs in the vendor environment and needs a GPU that holds the 18.8 GB BF16 checkpoint;
`--render-only` writes the same prompts without loading weights. `--masks` scores the trained mask
twice and three ablations of it instead of the two stage A sides, for `masks` to compare. `core`,
`report` and `masks` use only the standard library:

    python -m qualify.elephant1_decision_parity reference --checkpoint DIR --output reference.json
    python -m qualify.elephant1_decision_parity core --probes reference.json \\
        --url http://127.0.0.1:8120 --output core.json
    python -m qualify.elephant1_decision_parity report --reference reference.json --core core.json \\
        --output qualify/results/elephant1-decision-parity
    python -m qualify.elephant1_decision_parity masks --reference masks-reference.json \\
        --output qualify/results/elephant1-mask-feasibility/masks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.error
import urllib.request

TRAINED_CONTEXT = 16384
STAGE_A_ARGMAX, STAGE_A_MEAN_DELTA, STAGE_B_MEAN_DELTA = 0.99, 0.02, 0.01
# Mask runs in scoring order: the side's name, and which blocks keep one block id for all their tokens.
# A block whose tokens each take their own id is causal.
MASK_RUNS = (("trained", ("prefix", "suffix")), ("trained_repeat", ("prefix", "suffix")), ("causal", ()),
             ("suffix", ("suffix",)), ("prefix", ("prefix",)))
MASK_BASELINE, MASK_NOISE_FLOOR = "trained", "trained_repeat"
STATE_END = "</state>\n\n"

_YES_NO = {"type": "noul"}


def _noul(instructions):
    return dict(_YES_NO, instructions=instructions)


def _choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


# Each domain applies one question set to every one of its states.
DOMAINS = {
    "support": {
        "states": [
            "I was charged twice this month, please refund one of them.",
            "The app crashes every time I open the camera tab on my Pixel 9. Reinstalling did not help.",
            "Someone in another country logged into my account last night and changed my email address.",
            "Thanks for the quick fix yesterday, everything works again. You can close the ticket.",
            "Bonjour, je n'arrive pas à télécharger ma facture de mars depuis le portail client.",
            "Our whole warehouse team is locked out since 6am and we cannot ship orders. This is costing us"
            " thousands per hour. Fix it NOW.",
        ],
        "questions": {
            "refund": _noul("Is the customer asking for a refund?"),
            "team": _choice("Which team should handle this?", {
                "billing": "Charges, refunds, invoices", "technical": "App or site faults",
                "fraud": "Suspected unauthorised use", "none": "No further handling needed"}),
            "urgency": _score("How urgent is this?", ["Can wait", "Today", "Blocking the customer now"]),
            "mood": _score("How does the customer feel?",
                           ["Delighted", "Content", "Neutral", "Annoyed", "Furious"]),
            "english": _noul("Is the message written in English?"),
        },
    },
    "moderation": {
        "states": [
            "you are a worthless idiot and everyone in this thread knows it",
            "WIN A FREE IPHONE!!! click bit.ly/xx9 and enter your card number to claim, only 3 left",
            "Does anyone have a good recipe for sourdough starter? Mine keeps going flat after two days.",
            "I don't see the point of anything anymore. I've been thinking about ending it all.",
            "If Dave from accounting parks in my spot again I will find his car and slash every tire.",
            "The referee was blind last night, that penalty call was a robbery and the league should be"
            " ashamed.",
        ],
        "questions": {
            "violates": _noul("Does this post violate a community policy against abuse, threats, spam or"
                              " self-harm content?"),
            "category": _choice("Which category describes the post?", {
                "harassment": "Insults or demeans a person", "self_harm": "Expresses intent to self-harm",
                "spam": "Unsolicited promotion or scams", "violence": "Threatens harm to people or property",
                "sexual": "Sexually explicit content", "benign": "None of the other categories"}),
            "severity": _score("How severe is the content?", ["Harmless", "Mild", "Serious", "Critical"]),
            "review": _noul("Should a human moderator review this post before any action?"),
            "targeted": _noul("Is the post aimed at a specific, identifiable person?"),
        },
    },
    "code": {
        "states": [
            "def mean(xs):\n    return sum(xs) / len(xs)",
            "for (int i = 0; i <= n; i++) {\n    total += values[i];\n}",
            "query = \"SELECT * FROM users WHERE name = '\" + request.args['name'] + \"'\"\ncursor.execute(query)",
            "fn add(a: u8, b: u8) -> u8 {\n    a.checked_add(b).expect(\"overflow\")\n}",
            "const total = items.reduce((sum, item) => sum + item.price * item.qty, 0);",
            "func worker(ch chan int, wg *sync.WaitGroup) {\n    for v := range ch {\n        counter += v\n    }\n"
            "    wg.Done()\n}",
        ],
        "questions": {
            "bug": _noul("Does this code contain a defect that can produce a wrong result, a crash or a"
                         " vulnerability?"),
            "defect": _choice("Which defect class fits best?", {
                "bounds": "Out-of-range index or off-by-one", "injection": "Unsanitised input reaches an interpreter",
                "race": "Unsynchronised shared state", "empty_input": "Fails on empty input",
                "none": "No defect"}),
            "risk": _score("How risky is shipping this code unchanged?", ["Safe", "Low", "High", "Unacceptable"]),
            "language": _choice("Which language is this?", {
                "python": None, "c": None, "rust": None, "javascript": None, "go": None, "java": None}),
            "tests": _noul("Would a unit test with an empty collection as input be a useful addition?"),
        },
    },
    "routing": {
        "states": [
            {"user": "what's the weather in Lisbon tomorrow?"},
            {"user": "delete every file in the build directory and force push to main"},
            {"user": "book it"},
            {"user": "Summarise the attached 40-page contract and list every termination clause."},
            {"user": "what is 17% of 2,340?"},
            {"user": "Find the email from Priya about the Q3 budget and reply that I agree.",
             "attachments": 0, "session": {"mail_connected": True}},
        ],
        "questions": {
            "tool": _choice("Which tool should the assistant call first?", {
                "web_search": "Look up public information", "calculator": "Evaluate arithmetic",
                "shell": "Run commands on the user's machine", "mail": "Search, read or send email",
                "calendar": "Read or create calendar events", "document_reader": "Read an attached document",
                "ask_user": "Ask the user a clarifying question", "none": "Answer directly without a tool"}),
            "clarify": _noul("Is the request too ambiguous to act on without asking the user a question?"),
            "destructive": _noul("Could carrying out the request destroy data or be hard to undo?"),
            "effort": _score("How much work does the request need?",
                             ["Trivial", "Small", "Moderate", "Large", "Very large"]),
            "tier": _choice("Which model tier should handle it?", {
                "small": "Fast, cheap, simple lookups", "medium": "Multi-step tasks",
                "large": "Long documents or careful reasoning"}),
        },
    },
    "retrieval": {
        "states": [
            {"query": "boiling point of water at sea level",
             "passage": "At standard atmospheric pressure water boils at 100 degrees Celsius (212 F)."},
            {"query": "boiling point of water at sea level",
             "passage": "Mount Everest rises 8,849 metres above sea level on the border of Nepal and China."},
            {"query": "how to reset a forgotten router password",
             "passage": "Hold the recessed reset button for ten seconds until the lights blink; the router"
                        " returns to the factory login printed on its label."},
            {"query": "side effects of ibuprofen",
             "passage": "Ibuprofen is a nonsteroidal anti-inflammatory drug sold under many brand names."},
            {"query": "python sort list of dicts by key",
             "passage": "Use sorted(rows, key=lambda r: r['age']) or rows.sort(key=operator.itemgetter('age'))."},
            {"query": "capital of Australia",
             "passage": "Sydney is the largest city in Australia, with over five million residents."},
        ],
        "questions": {
            "relevant": _noul("Is the passage relevant to the query?"),
            "grade": _score("How well does the passage answer the query?",
                            ["Not at all", "Mentions the topic", "Partly answers", "Fully answers"]),
            "complete": _noul("Could a reader answer the query from the passage alone?"),
            "topic": _choice("Which topic is the query about?", {
                "science": None, "geography": None, "technology": None, "medicine": None,
                "programming": None, "history": None, "sport": None}),
            "numeric": _noul("Does the passage contain a number?"),
        },
    },
}

_COLOURS = ("amber", "black", "blue", "bronze", "brown", "crimson", "cyan", "gold", "green", "grey", "indigo",
            "ivory", "lilac", "magenta", "maroon", "navy", "olive", "orange", "pink", "purple", "red", "silver",
            "teal", "violet", "white", "yellow")
MANY_OPTIONS_STATE = "The car in the listing is painted a deep navy, with silver trim on the mirrors."


def _log_state(lines):
    """A deterministic service log; line 7 of every 1000 carries the only ERROR entries."""
    services = ("auth", "billing", "search", "ingest", "render")
    rows = []
    for index in range(lines):
        status = "ERROR" if index % 1000 == 7 else "ok"
        rows.append(f"{index:06d} service={services[index % 5]} status={status}"
                    f" latency_ms={(index * 37) % 211 + 3}")
    return "\n".join(rows)


_LOG_QUESTIONS = {
    "errors": _noul("Does the log contain at least one line with status=ERROR?"),
    "worst": _choice("Which service has an ERROR line?", {
        "auth": None, "billing": None, "search": None, "ingest": None, "render": None}),
}
# Log line counts placing the rendered prompt near 4,000, 12,000 and 16,000 tokens.
LONG_LOGS = {"log4k": 215, "log12k": 645, "log16k": 875}
# One prompt over the trained 16,384 tokens and one over the vendor engine's 32,768-token branch limit.
REFUSAL_LOGS = {"log17k": 920, "log34k": 1800}


def requests():
    """Every probe request as (identifier, state, questions, expects_refusal)."""
    for domain, entry in DOMAINS.items():
        for index, state in enumerate(entry["states"]):
            yield f"{domain}{index}", state, entry["questions"], False
    yield "options20", MANY_OPTIONS_STATE, {"colour": _choice(
        "Which colour is the car's body?", {name: None for name in _COLOURS[:20]})}, False
    yield "options26", MANY_OPTIONS_STATE, {"colour": _choice(
        "Which colour is the car's body?", {name: None for name in _COLOURS})}, False
    for name, lines in LONG_LOGS.items():
        yield name, _log_state(lines), _LOG_QUESTIONS, False
    for name, lines in REFUSAL_LOGS.items():
        yield name, _log_state(lines), {"errors": _LOG_QUESTIONS["errors"]}, True


def _softmax(logits):
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def reference(args):
    """Render every branch by the vendor rules and score it with the vendor mask and the causal mask."""
    from qev.prompt import LETTERS, build_messages, render_question
    from qev.schema import SystemOneRequest

    checkpoint = str(args.checkpoint.resolve())
    engine = None
    if args.render_only:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    else:
        from qev.engine import DecisionEngine
        engine = DecisionEngine(checkpoint, device=args.device)
        tokenizer = engine.tokenizer
        trained = engine.readout.config.attention
        if trained != "noncausal":
            raise ValueError(f"checkpoint declares {trained} attention; stage A needs the trained noncausal mask")
        fork_check = engine.info.fork_mode, engine.info.fork_max_abs_diff
        engine.info.fork_mode = "sequential"
    slot_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in slot_ids):
        raise ValueError("an option letter is not a single token")
    slot_ids = [ids[0] for ids in slot_ids]

    def decide(request, attention, whole_blocks=None):
        engine.readout.config.attention = attention
        if whole_blocks is not None:
            import torch
            from qev.readout import blocks

            def block_ids(prefixes, lengths, width):
                # One branch per forward: `prefixes` holds the one prefix length, and no row is padded.
                position = torch.arange(width, device=engine.device)
                prefix = torch.zeros_like(position) if "prefix" in whole_blocks else position
                suffix = torch.full_like(position, prefixes[0]) if "suffix" in whole_blocks else position
                return blocks(torch.where(position < prefixes[0], prefix, suffix)[None])

            engine._blocks = block_ids
        try:
            _, meta = engine.decide(request, debug=True)
        except ValueError as error:
            if "limit is" not in str(error):
                raise
            return None, str(error)
        finally:
            engine.readout.config.attention = trained
            vars(engine).pop("_blocks", None)
        return {qid: {"probabilities": _softmax(entry["slot_logits"][0]),
                      "letter_mass": entry["allowed_token_mass"]}
                for qid, entry in meta["diagnostics"].items()}, None

    branches = []
    for name, state, questions, expects_refusal in requests():
        request = SystemOneRequest(state=state, questions=questions)
        scored, prefix_tokens = {}, None
        if engine is not None and args.masks:
            scored = {side: decide(request, "noncausal", whole_blocks) for side, whole_blocks in MASK_RUNS}
            prefix_tokens = len(engine._prefix_and_suffix_ids(
                state, [render_question(qid, question) for qid, question in request.questions.items()])[0])
        elif engine is not None:
            scored = {side: decide(request, attention)
                      for side, attention in (("vendor", "noncausal"), ("causal", "causal"))}
        for qid, question in request.questions.items():
            rendered = render_question(qid, question)
            prompt = tokenizer.apply_chat_template(build_messages(state, rendered.suffix), tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            branch = {"id": f"{name}.{qid}", "kind": rendered.kind, "labels": rendered.labels,
                      "slot_ids": slot_ids[:rendered.n_slots], "tokens": tokens, "expects_refusal": expects_refusal}
            if prefix_tokens is not None:
                branch["prefix_tokens"] = prefix_tokens
            for side, (answers, refusal) in scored.items():
                branch[side] = {"refused": refusal} if answers is None else answers[qid]
            branches.append(branch)
        print(f"{name}: {len(request.questions)} questions, longest prompt"
              f" {max(len(b['tokens']) for b in branches[-len(request.questions):])} tokens", flush=True)
    meta = {"checkpoint": checkpoint, "rendered_only": engine is None, "masks": args.masks,
            "state_end_ids": tokenizer.encode(STATE_END, add_special_tokens=False)}
    if engine is not None:
        import torch
        meta.update(device=torch.cuda.get_device_name(engine.device), dtype=engine.info.dtype,
                    head_dtype=engine.head_dtype, trained_attention=trained,
                    transformers=engine.info.transformers_version, torch=engine.info.torch_version,
                    vendor_fork_check={"mode": fork_check[0], "max_probability_difference": fork_check[1]},
                    branch_token_limit=engine.max_branch_tokens)
    args.output.write_text(json.dumps({"meta": meta, "branches": branches}))
    print(f"wrote {len(branches)} branches to {args.output}")


def core(args):
    """Score every rendered branch on a served core through Completions generation logprobs.

    The option probabilities come from `logprob_token_ids`, which returns the letters' full-vocabulary
    logprobs whatever their rank. A second request for the twenty most likely tokens counts how many of
    the letters a caller reading only that list would receive.
    """
    probes = json.loads(args.probes.read_text())["branches"]
    base = args.url.rstrip("/")
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
        model = json.load(response)["data"][0]

    def alternatives(branch, **selection):
        body = json.dumps({"model": model["id"], "prompt": branch["tokens"], "max_tokens": 1, "logprobs": 20,
                           "return_tokens_as_token_ids": True, **selection}).encode()
        request = urllib.request.Request(f"{base}/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)["choices"][0]["logprobs"]["top_logprobs"][0]

    results = []
    for branch in probes:
        try:
            letters = alternatives(branch, logprob_token_ids=branch["slot_ids"])
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace")
            if error.code != 400:
                raise RuntimeError(f"{branch['id']}: HTTP {error.code}: {message}") from error
            results.append({"id": branch["id"], "refused": message})
            print(f"{branch['id']}: refused", flush=True)
            continue
        mass = [math.exp(letters[f"token_id:{slot}"]) for slot in branch["slot_ids"]]
        total = sum(mass)
        top = alternatives(branch)
        results.append({"id": branch["id"], "probabilities": [value / total for value in mass],
                        "letters_in_top20": sum(f"token_id:{slot}" in top for slot in branch["slot_ids"]),
                        "letter_mass": total})
        print(f"{branch['id']}: {len(branch['tokens'])} tokens, letter mass {total:.4f}", flush=True)
    meta = {"url": base, "served_model": model["id"], "max_model_len": model.get("max_model_len"),
            "flare": {key: model.get("flare", {}).get(key) for key in ("max_top_logprobs", "max_concurrency")}}
    args.output.write_text(json.dumps({"meta": meta, "branches": results}))
    print(f"wrote {len(results)} branches to {args.output}")


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: -values[index])
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _spearman(left, right):
    count = len(left)
    distance = sum((a - b) ** 2 for a, b in zip(_ranks(left), _ranks(right)))
    return 1 - 6 * distance / (count * (count * count - 1))


def compare(pairs):
    """Agreement between two sides over [(branch, candidate probabilities, baseline probabilities)]."""
    rows, deltas, argmax, ordered, correlations = [], [], [], [], []
    for branch, candidate, baseline in pairs:
        delta = [abs(a - b) for a, b in zip(candidate, baseline)]
        same = max(range(len(baseline)), key=baseline.__getitem__) == max(range(len(candidate)),
                                                                          key=candidate.__getitem__)
        row = {"id": branch["id"], "kind": branch["kind"], "labels": branch["labels"],
               "tokens": len(branch["tokens"]),
               "token_sha256": hashlib.sha256(json.dumps(branch["tokens"]).encode()).hexdigest(),
               "candidate": candidate, "baseline": baseline, "absolute_delta": delta, "argmax_agrees": same}
        if branch["kind"] == "score":
            row["rank_agrees"] = _ranks(candidate) == _ranks(baseline)
            row["rank_correlation"] = _spearman(candidate, baseline)
            ordered.append(row["rank_agrees"])
            correlations.append(row["rank_correlation"])
        rows.append(row)
        deltas += delta
        argmax.append(same)
    summary = {"questions": len(rows), "options": len(deltas),
               "argmax_agreement": sum(argmax) / len(argmax),
               "mean_absolute_probability_delta": sum(deltas) / len(deltas),
               "maximum_absolute_probability_delta": max(deltas),
               "score_questions": len(ordered),
               "score_rank_agreement": sum(ordered) / len(ordered),
               "score_mean_rank_correlation": sum(correlations) / len(correlations)}
    return summary, rows


def _verdict(passed):
    return "pass" if passed else "failed acceptance"


def report(args):
    """Write the two stage reports and evaluate the acceptance thresholds."""
    source = json.loads(args.reference.read_text())
    served = json.loads(args.core.read_text())
    if source["meta"]["rendered_only"] or source["meta"].get("masks"):
        raise ValueError("the reference file holds no stage A sides; run `reference` with the checkpoint and"
                         " without --masks")
    by_id = {branch["id"]: branch for branch in served["branches"]}
    stage_a, stage_b, refusals, coverage = [], [], [], []
    for branch in source["branches"]:
        packed = by_id[branch["id"]]
        vendor, causal = branch["vendor"], branch["causal"]
        if branch["expects_refusal"] or any("refused" in side for side in (vendor, causal, packed)):
            refusals.append({"id": branch["id"], "tokens": len(branch["tokens"]),
                             "over_trained_context": len(branch["tokens"]) + 1 > TRAINED_CONTEXT,
                             "vendor_refused": "refused" in vendor, "core_refused": "refused" in packed})
            if "refused" not in vendor and "refused" not in causal:
                stage_a.append((branch, causal["probabilities"], vendor["probabilities"]))
            continue
        stage_a.append((branch, causal["probabilities"], vendor["probabilities"]))
        stage_b.append((branch, packed["probabilities"], causal["probabilities"]))
        coverage.append({"id": branch["id"], "options": len(branch["labels"]),
                         "letters_in_top20": packed["letters_in_top20"],
                         "letter_mass": {"core": packed["letter_mass"], "bf16_causal": causal["letter_mass"],
                                         "vendor": vendor["letter_mass"]}})
    args.output.mkdir(parents=True, exist_ok=True)
    summary_a, rows_a = compare(stage_a)
    summary_a["verdicts"] = {
        f"argmax agreement >= {STAGE_A_ARGMAX}": _verdict(summary_a["argmax_agreement"] >= STAGE_A_ARGMAX),
        f"mean absolute probability delta <= {STAGE_A_MEAN_DELTA}":
            _verdict(summary_a["mean_absolute_probability_delta"] <= STAGE_A_MEAN_DELTA)}
    (args.output / "stage-a-mask.json").write_text(json.dumps({
        "candidate": "BF16 checkpoint, causal attention in every layer (vendor Transformers model, one branch per"
                     " forward)",
        "baseline": "vendor DecisionEngine with the trained block-bidirectional mask (one branch per forward)",
        "reference": source["meta"], "summary": summary_a, "questions": rows_a}, indent=1))
    summary_b, rows_b = compare(stage_b)
    summary_b["verdicts"] = {
        f"mean absolute probability delta <= {STAGE_B_MEAN_DELTA}":
            _verdict(summary_b["mean_absolute_probability_delta"] <= STAGE_B_MEAN_DELTA)}
    complete = [entry["letters_in_top20"] == entry["options"] for entry in coverage if entry["options"] <= 20]
    summary_b["top20_letter_coverage"] = {
        "questions_with_at_most_20_options": len(complete), "all_letters_returned": sum(complete)}
    summary_b["mean_letter_mass"] = {
        side: sum(entry["letter_mass"][side] for entry in coverage) / len(coverage)
        for side in ("core", "bf16_causal", "vendor")}
    agree = [entry["vendor_refused"] == entry["core_refused"] for entry in refusals]
    summary_b["length_refusals"] = {"probes": refusals, "agreement": sum(agree) / len(agree) if agree else None}
    (args.output / "stage-b-precision.json").write_text(json.dumps({
        "candidate": "served core, generation logprobs of the option letters at the answer position",
        "baseline": "BF16 checkpoint, causal attention in every layer (vendor Transformers model, one branch per"
                    " forward)",
        "core": served["meta"], "reference": source["meta"], "summary": summary_b, "coverage": coverage,
        "questions": rows_b}, indent=1))
    for stage, summary in (("stage A (mask)", summary_a), ("stage B (precision)", summary_b)):
        print(f"{stage}: {summary['questions']} questions, {summary['options']} options;"
              f" argmax agreement {summary['argmax_agreement']:.4f},"
              f" mean |dp| {summary['mean_absolute_probability_delta']:.4f},"
              f" max |dp| {summary['maximum_absolute_probability_delta']:.4f},"
              f" score rank agreement {summary['score_rank_agreement']:.4f}")
        for threshold, verdict in summary["verdicts"].items():
            print(f"  {threshold}: {verdict}")
    print(f"top-20 letter coverage: {summary_b['top20_letter_coverage']}")
    print(f"mean probability on the option letters: {summary_b['mean_letter_mass']}")
    for entry in refusals:
        print(f"length {entry['id']}: {entry['tokens']} tokens, vendor refused {entry['vendor_refused']},"
              f" core refused {entry['core_refused']}")


def _state_end(tokens, marker):
    """The token index after the last rendered end of the state block, from the prompt's tokens alone."""
    for start in range(len(tokens) - len(marker), -1, -1):
        if tokens[start:start + len(marker)] == marker:
            return start + len(marker)
    return None


def masks(args):
    """Compare each mask run with the first trained run and evaluate the mask acceptance bounds."""
    source = json.loads(args.reference.read_text())
    if not source["meta"].get("masks"):
        raise ValueError("the reference file holds no mask runs; run `reference --masks` with the checkpoint")
    answered = [branch for branch in source["branches"] if "refused" not in branch[MASK_BASELINE]]
    variants = {}
    for side, _ in MASK_RUNS[1:]:
        summary, rows = compare([(branch, branch[side]["probabilities"], branch[MASK_BASELINE]["probabilities"])
                                 for branch in answered])
        if side != MASK_NOISE_FLOOR:
            summary["verdicts"] = {
                f"argmax agreement >= {STAGE_A_ARGMAX}": _verdict(summary["argmax_agreement"] >= STAGE_A_ARGMAX),
                f"mean absolute probability delta <= {STAGE_A_MEAN_DELTA}":
                    _verdict(summary["mean_absolute_probability_delta"] <= STAGE_A_MEAN_DELTA)}
        variants[side] = {"summary": summary, "questions": rows}
    marker = source["meta"]["state_end_ids"]
    boundaries = [(branch["id"], branch["prefix_tokens"], _state_end(branch["tokens"], marker))
                  for branch in source["branches"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "baseline": "vendor DecisionEngine, first run with the trained mask: the prefix one block and the suffix"
                    " one block (one branch per forward)",
        "masks": {"trained_repeat": "the trained mask scored a second time: the noise floor",
                  "causal": "every token its own block",
                  "suffix": "each prefix token its own block, the suffix one block",
                  "prefix": "the prefix one block, each suffix token its own block"},
        "reference": source["meta"],
        "refused": [branch["id"] for branch in source["branches"] if "refused" in branch[MASK_BASELINE]],
        "prefix_boundary": {
            "rule": f"the token after the last {STATE_END!r} in the prompt's tokens equals the vendor engine's"
                    " prefix length",
            "probes": len(boundaries), "matching": sum(vendor == found for _, vendor, found in boundaries),
            "differing": [{"id": name, "vendor": vendor, "from_tokens": found}
                          for name, vendor, found in boundaries if vendor != found]},
        "variants": variants}, indent=1))
    for side, entry in variants.items():
        summary = entry["summary"]
        print(f"{side}: {summary['questions']} questions; argmax agreement {summary['argmax_agreement']:.4f},"
              f" mean |dp| {summary['mean_absolute_probability_delta']:.6f},"
              f" max |dp| {summary['maximum_absolute_probability_delta']:.6f}")
        for threshold, verdict in summary.get("verdicts", {}).items():
            print(f"  {threshold}: {verdict}")
    print(f"prefix boundary from tokens: {sum(vendor == found for _, vendor, found in boundaries)} of"
          f" {len(boundaries)} probes match")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("reference", help="render and score the probe set in the vendor environment")
    command.add_argument("--checkpoint", type=Path, required=True, help="baselabs/elephant-1 snapshot directory")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--render-only", action="store_true", help="write prompts without loading weights")
    command.add_argument("--masks", action="store_true",
                         help="score the trained mask twice and its causal, suffix-only and prefix-only ablations")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=reference)
    command = commands.add_parser("core", help="score the rendered probe set on a served core")
    command.add_argument("--probes", type=Path, required=True, help="file written by `reference`")
    command.add_argument("--url", required=True, help="base URL of `flare run` serving the core")
    command.add_argument("--timeout", type=float, default=300)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=core)
    command = commands.add_parser("report", help="write both stage reports and their verdicts")
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--core", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True, help="report directory")
    command.set_defaults(run=report)
    command = commands.add_parser("masks", help="write the mask ablation report and its verdicts")
    command.add_argument("--reference", type=Path, required=True, help="file written by `reference --masks`")
    command.add_argument("--output", type=Path, required=True, help="report file")
    command.set_defaults(run=masks)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
