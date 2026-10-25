# Adding a script (call type)

A script is one YAML file in `scripts/`. Code owns the conversation; the script only declares steps, clips and questions.
Jev never writes words or picks clips.

## Structure
```yaml
name: my_call
language: english            # selects turn-silence settings (english / default)
thresholds: {default: 0.7}    # per question key; per-step overrides allowed
max_retries: 1                # re-asks before handing off
steps:
  - id: ask_something
    slot: something           # omit for a terminal step (last step must be terminal)
    clips:
      ask:    [{id: s.ask.1, text: "..."}, ...]      # 3+ variants; no variant plays twice in a row
      re_ask: [{id: s.re_ask.1, text: "..."}, ...]   # rephrased, 3+ variants
    questions:
      slot_value:
        type: choice
        instruction: "What the caller said, in terms of the options."
        options: [a, b, not_stated]   # must include an escape: none / unclear / not_stated
  - id: closing
    clips: {ask: [{id: closing.ask.1, text: "..."}]}
shared:
  bridge:   [...]   # played while Phase B starts (required)
  fallback: [...]   # played when the LLM or speech output fails (required)
  ack:      [...]   # short "Ji" / "Theek hai" at end of turn (optional)
  disclosure: [...] # mandatory wording, see docs/compliance.md
```
Quote `"yes"` / `"no"` options: unquoted, YAML reads them as booleans and the loader rejects the file.

## Checklist
1. Write the script; `load_script` validates it (unique ids, escapes present, last step terminal, thresholds).
2. Clip ids are unique across the whole script, including `shared`.
3. Read `docs/labelling_guide.md`; label 100-150 real or role-played calls at turn level into `data/<script>.jsonl`.
4. `freeze`, then evaluate on validation: `python -m jevagent.eval.cli --data data/<script>.jsonl --script scripts/<script>.yaml`.
   Fix wording by adding versions under `questions/<script>/<key>@vN.yaml`; promote only on a validation pass
   (`eval.versions.promote`). Check rare-class recall (corrections, off-script, handoff), not just accuracy.
5. Tune thresholds on the tuning split (`eval.report.tune_*`) and store them for the pinned model (`save_thresholds`).
6. Render clips: `python scripts/gen_clips.py scripts/<script>.yaml`; warnings list steps with fewer than 3 variants.
7. Run the mock-caller tests pattern (`tests/unit/test_frontend.py`) with your script's transcripts, then a console run.
8. Set required disclosures in `compliance.yaml`; run `compliance.audit`.
9. Held-out evaluation once, at promotion (`get_split(..., allow_heldout=True, reason=...)` is audited).
