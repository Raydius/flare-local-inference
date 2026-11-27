<!-- managed by radflow — rendered into the radflow schema by adopt.sh. Skeleton for openspec/lessons.md,
     the per-repo challenging-scenarios log. The store is created from this skeleton on
     FIRST CAPTURE (by /rad-reframe, or an advisory grind capture) — never at adopt time.
     It stays OUTSIDE radflow-lint's document model: free prose here never trips
     trigger-token checks; only '## <slug>' anchor headings are validated, from the
     change side (design.md's 'Lesson: openspec/lessons.md#<slug>' marker).

     One entry per challenging scenario, appended at the end. Grammar:

## <kebab-slug>

- Scenario: <what was being attempted>
- Trap: <what made it challenging — the wrong assumption or the grind>
- Resolution: <the corrected model, grounded in real code (cite file:line)>
- Accelerator: <what a future agent should do FIRST in a similar situation>
- Source: <change name> (<YYYY-MM-DD>)
-->
# Lessons — challenging scenarios

Entries are appended by /rad-reframe (mandatory on every completed reframe) and by
advisory grind captures (a FAIL/BLOCKED task, a single correction that lands deep,
repeated failed verification attempts). Consult this log at propose/design/reframe
time: grep for paths or problem shapes overlapping your blast radius, and cite the
entries you consulted.

## serve-from-the-operators-running-profile

- Scenario: exporting and serving a new proposal method (DSpark on the dense Qwen recipe) on the operator's GPU 0 through the console, for measurement at the operator's usage.
- Trap: the plan described the serving configuration from notes ("concurrency under 8", a fresh profile name, the checkpoint's own chat template, the branch as it stood) instead of from the instance the operator had been serving. Four corrections followed: 16 sequences, rebase onto deploy before the image build, the Quinn chat template, the served name `Qwen3.6`. Two exports were started and stopped.
- Resolution: the operator's default console profile is the specification of a served package. `GET /api/profiles` on the console (port 8081) returns it; on 2026-11-18 it held `max_num_seqs` 16, `served_model_name` "Qwen3.6", `chat_template_path` `~/models/raydius/quinn-template/chat_template.jinja`, and run fields for generation defaults, prefix-cache memory and scheduler policy (`ignition/cli/serve.py:310` prints the plan a profile produces). A new package changes only what the change is about (model folder, recipe, draft, method, image); every other field is copied. Images are built from a branch that sits on `deploy`.
- Accelerator: before any console export, read the default profile and the running instance's `flare run` arguments (`ps`), diff the planned profile against it, and show the operator the fields that differ; check `git rev-list --count HEAD..deploy` is 0 before `build.sh`.
- Source: add-qwen38-dense-dspark (2026-11-19)

## a-replacement-core-keeps-the-replaced-profiles-capabilities

- Scenario: exporting a core from a different recipe (the dense Qwen recipe) to replace the operator's served profile (the Flash Next recipe), by copying the profile and changing the recipe, model and draft.
- Trap: the profile diff listed fields, not capabilities. The dense recipe prepared text only, so the copied profile's `export:--language-model-only false` had no effect and the served core rejected images. No artifact of the change mentioned image input, and no step compared `/v1/models` of the two cores. The operator found it after approval: "every single other build we've done includes vision".
- Resolution: image input was added to the dense recipe in the same change, before the re-export.
- Accelerator: at propose time, read the target recipe's `max_images`, `language_model_only` and context beside the replaced profile's recipe; after serving, diff the `flare` section of `/v1/models` (`max_images_per_prompt`, `input_modalities`, `max_model_len`, `max_concurrency`) against the replaced core's before asking for approval.
- Source: add-qwen38-dense-dspark (2026-11-19)

## a-declared-cost-may-be-a-reservation-not-an-allocation

- Scenario: a test that ties a recipe's declared `RankCosts.state_row_bytes` to the prepared continuation layout ran for the dense Qwen recipe and found 193,865,984 declared bytes against a 186,001,664-byte prepared row.
- Trap: the mismatch was reported as a recipe defect, and the test was changed to demand equality at the prepared width, without reading the commit or documentation that set the constant. The operator asked "Did he document a reason for expanding it?"; the commit `7303efb1` and `docs/model-preparation.md` both state it.
- Resolution: the declaration is the planning reservation for the widest verification the recipe supports (`max_draft_tokens + 1` positions); narrower preparations allocate a smaller row. The test compares the declaration with the layout at that maximum and requires the prepared row to be smaller.
- Accelerator: before reporting a declared constant as wrong, run `git log -S<constant>` for the commit that set it and grep `docs/` for the number; a recipe comment beside the constant is part of the declaration.
- Source: tie-dense-qwen-row-to-reserved-maximum (2026-11-22)

## check-upstream-for-the-fix-before-writing-it

- Scenario: a Qwen core exported on the `v0.3.2` stack failed every request with device error 1, and the cause was traced to FlashInfer's plan compiler at the fork's pin `d73da5c4`, which builds the prepared grouped RMSNorm kernel without its FP32-gain setting.
- Trap: the plan read the defect only at the pinned revision and at older ones, and scheduled a hand-written kernel-construction fix, a model switch, a pin, an image and an export. FlashInfer's `origin/main` had not been fetched since the pin was cut. Upstream had fixed the same line one day earlier in `c3ab1d04`, with a cache-version bump the planned fix did not include, and upstream's recipes already locked a revision containing it.
- Resolution: the fix commit is a cherry-pick of `c3ab1d04`; once upstream's newer commits were listed, the operator moved both forks to upstream `main`, where the fix arrives with the base.
- Accelerator: when a defect is located in a pinned dependency, run `git fetch` and `git log <pin>..origin/main -- <file>` before designing a fix, and compare the pin with the revision upstream's own lock names.
- Source: debug-v0-3-2-device-error-1 (2026-11-26)

## a-base-move-is-proven-by-a-served-export

- Scenario: moving both forks to a newer upstream base and releasing the result as a fork revision.
- Trap: the 2026-11-24 move took baseline-relative CPU and component gates as proof and put exports out of scope. Those gates compare the fork with unpatched upstream, so a defect upstream's own locked dependency carries cannot fail them: FlashInfer `0aa01681` built the grouped RMSNorm kernel wrongly, and every request of the first export failed after the tag was published. A second, independent replay on the same base and pin failed identically.
- Resolution: the `fork-base-move` capability requires the operator's profile to export and serve before a release; `FORK.md` "Moving to a newer upstream" ends with that step.
- Accelerator: write the operator's export and its served checks into the proposal as tasks that precede any tag; before the first export, run the pinned dependency's own GPU tests for the kernels the served recipe prepares on the spare GPU (`openspec/changes/debug-v0-3-2-device-error-1/handoff.md` holds the recipe).
- Source: rebase-forks-onto-upstream-v0-3-2 (2026-11-26)

## judge-a-patch-against-what-upstream-added-not-what-it-lacks

- Scenario: deciding after a base move whether each fork patch is still needed and still whole.
- Trap: the 2026-11-24 move marked a patch "Still needed: Yes" when upstream lacked its flag or feature. Three patches merged without a textual conflict next to a mechanism upstream added in the range and were bypassed by it: the rebuild record never emitted `--rope-parameters`, a second context-limit site skipped the persisting L2 bound, and the host acceptance population took history-copy depths. In this change's own second move, two fork recipes kept passing settings upstream had removed, again with no conflict and no failing CPU test.
- Resolution: each verdict shows the need on unpatched upstream and searches every upstream enumeration of the thing the patch extends; `tests/cli/test_reproduction_arguments.py` fails when an export flag is neither recorded nor listed as not prepared.
- Accelerator: for each patch run `git log <old base>..<new base>` over its files, grep the range for enumerations (preparation flags, context-limit sites, acceptance estimates, shared configuration fields), and run `~/models/.cache/rebase-v0.3.2/tools/check-kwargs.py` over the fork-changed files before any gate.
- Source: rebase-forks-onto-upstream-v0-3-2 (2026-11-26)

## a-console-script-runs-again-in-spawned-workers

- Scenario: the preparation image installs two packages that both declare the `ignition` script, so the image recipe writes the launcher by hand.
- Trap: the launcher called `main()` at import. FlashInfer starts compiler workers with `multiprocessing` spawn, which imports the parent's main script as `__mp_main__`; every worker re-ran the export command and exited, and every export died in "Prepare and tune kernels". No test starts those workers; only an export does.
- Resolution: the launcher calls `main()` under `if __name__ == "__main__"`, and the image build imports it as `__mp_main__` and fails when the argument parser is reached (`build/flare-ignition/check_ignition_launcher.py` in `vllm-env`).
- Accelerator: when a build step writes an entry-point script, add a build-time import of it under the name `__mp_main__` before trusting the image.
- Source: rebase-forks-onto-upstream-v0-3-2 (2026-11-26)
