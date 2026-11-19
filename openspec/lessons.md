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

