# Question catalog

## Jev wire format (verified against https://api.typesafe.ai/openapi.json)

`POST {JEV_BASE_URL}/v1/systemone`, `Authorization: Bearer <key>`.

Request: `{"model", "state", "questions": {name: {type, instructions, criteria}}}`.
- `choice`: `criteria` is `{option: description}`; `noul`: `instructions` only; `score`: `criteria` is an ordered list.
- Question names containing `:` are sent with `__` instead (`slot_value:pay_when` -> `slot_value__pay_when`) and mapped back.

Response: `{"model", "answers": {name: ...}, "usage"}`.
- choice -> `{choice, confidence, probabilities}`; noul -> `{noul: P(yes)}`; score -> `{score, confidence, probabilities}`.
- Noul is normalised to `yes`/`no` with confidence `max(p, 1-p)`.

Failure policy: timeouts, 408/429/5xx (after retries) and malformed bodies yield low-confidence answers
(`selected=None`, `confidence=0`), so the engine re-asks once then hands off. 4xx other than 408/429 raise `JevConfigError`.

## Standard per-turn questions

| Key | Type | Options | Used for |
| --- | --- | --- | --- |
| `answered_step` | choice | yes, no, unclear | gate for filling the current slot |
| `slot_value:<step>` | choice | from script (must include an escape) | slot value; also asked for later unfilled steps |
| `answers_other_step` | choice | later unfilled step ids, none | early slot fill (A-7) |
| `corrects_stored` | noul | yes/no | handoff trigger 2 (only asked once a value is stored) |
| `turn_type` | choice | on_script, question, objection, complaint, wants_human, wrong_person, unclear | handoff trigger 1 |

Escape options: `none`, `unclear`, `not_stated`. Thresholds per question key live in the script (`thresholds`, per-step override),
and must be re-tuned when the pinned Jev model changes.

## Candidate parsers (A-5)

Dates (numeric, "15th March", "March 15", "the 15th"), amounts (digits, `k`, thousand, lakh, English number words such as
"five thousand"), phones (10-digit Indian mobiles, spoken digit words). Number words are additive ("twenty five" = 25) with
hundred / thousand / lakh / million multipliers; a date needs its day as digits.

## Catalog status (emi_reminder, pinned model `jev-1.14.0`)

| Question | Definition | Threshold | Wording version | Last eval |
| --- | --- | --- | --- | --- |
| `answered_step` | Latest utterance directly answers the question in `last_clip` (`docs/labelling_guide.md`). | 0.70 (script default) | built-in | not evaluated |
| `slot_value:<step>` | The option the caller stated; escape when none applies. | 0.70 | script YAML | not evaluated |
| `answers_other_step` | A later step the caller also answered, else `none` (early fill). | 0.70 | built-in | not evaluated |
| `corrects_stored` | Contradicts a stored value (confirming or repeating is `no`). | 0.70 | built-in | not evaluated |
| `turn_type` | on_script / question / objection / complaint / wants_human / wrong_person / unclear. Fires handoff at confidence 0.50 (`OFF_SCRIPT_THRESHOLD`, early-handoff bias). | 0.50 | built-in | not evaluated |

**No labelled data exists yet**, so every threshold above is the unvalidated default and no question has an evaluation
score. Fill the "Last eval" column from `logs/eval_validation.md` after `python -m jevagent.eval.cli`; store tuned thresholds
with `eval.versions.save_thresholds` (keyed by Jev model version) and keep wording versions in `questions/<script>/`.

## Post-call judge (not on the call path)
`handoff_timing` (choice: too_early / right_time / too_late / not_needed / unclear) and `caller_repeated` (noul), asked over
the whole transcript by `eval.judge.judge_call`. These may use cross-turn context because they run after the call.
