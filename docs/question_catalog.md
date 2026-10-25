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

Dates (numeric, "15th March", "March 15", "pandrah tareekh"), amounts (digits, `k`, lakh/hazaar, Hindi number words),
phones (10-digit Indian mobiles, spoken digit words). Hindi number words cover 0-31, round tens, and sau/hazaar/lakh;
other compound numbers (e.g. 47) are not parsed yet.
