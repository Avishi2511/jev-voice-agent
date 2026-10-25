# Labelling guide

Labels are per **turn** (one caller utterance), judged against the agent clip that preceded it.
Label only what the utterance says; do not use later turns. Freeze labels (`jevagent.eval.dataset.freeze`)
before running any evaluation (E-2); relabelling after seeing results invalidates the held-out split.

## Per-turn fields

| Field | Definition |
| --- | --- |
| `step` | The script step the agent was on when the caller spoke. |
| `stored` | Slot values already stored before this turn (`slot: value`). |
| `retries_used` | How many re-asks were already played on this step. |
| `answered_step` | `yes` if the utterance directly answers the question asked in the last clip, `no` if it does not, `unclear` if you cannot tell. A partial or hedged answer ("shayad kal") is `unclear`. |
| `slot_value:<step>` | The option from the step's list that the utterance states. `not_stated` / `unclear` when the options do not apply. Label other steps' slots only if the caller actually said them (early fill). |
| `answers_other_step` | The id of a later step the caller also answered, else `none`. |
| `corrects_stored` | `yes` only if the caller contradicts or changes a value in `stored` ("nahi, kal nahi, parso"). Repeating or confirming a value is `no`. |
| `turn_type` | `on_script`: direct answer or acknowledgement relevant to the last clip. `question`: caller asks something. `objection`: caller disputes or pushes back (e.g. penalty, amount). `complaint`: caller is upset about service or conduct. `wants_human`: caller asks for a person. `wrong_person`: caller is not the customer. `unclear`: unintelligible or silent. |
| `should_handoff` | `true` if a careful human supervisor would say the scripted agent should stop here: any `turn_type` other than `on_script`/`unclear`, an unresolved correction, or a state the script does not cover. A single unclear answer on the first try is **not** a handoff (re-ask first). |

## Rules for ambiguous cases
- Mixed utterance (answer plus question): `turn_type` is the non-script part; label the slot value too.
- Hinglish number or date words count as stated values.
- Background chatter, silence or "hello?": `answered_step=no`, `turn_type=unclear`.
- When two labellers disagree, discuss and add the resolution to this file.

## File format
One JSON object per line (`data/*.jsonl`), schema in `jevagent.eval.dataset.LabelledTurn`. Use call ids that do not
reveal customers; keep raw recordings and transcripts with PII out of git (`data/raw/` is ignored).
