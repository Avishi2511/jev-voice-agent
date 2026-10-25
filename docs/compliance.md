# Compliance and data handling

**Status: wording and retention policy are not set.** Nothing in this repo contains legal text; the checks below
enforce whatever you supply. Until you do, `compliance.yaml` does not exist and no call is checked.

## Mandatory disclosures (B-4)
- Put the required wording in `compliance.yaml` (see `jevagent.compliance.ComplianceConfig`):
  ```yaml
  disclosures:
    - {id: disclosure.1, text: "<exact required wording>", required_on: all}   # or: completed
  ```
- Add a clip with that id under `shared: disclosure:` in the script. The agent can only play disclosures as
  pre-recorded clips (`play_clip`); it is told never to speak them, and `play_clip` rejects ids outside the allowed set.
- `compliance.check_script` fails if the clip text does not contain the wording verbatim.
- `compliance.audit(script, config, summaries)` lists calls that should have played a disclosure but did not
  (`CallSummary.clips_played` records every clip sent).
- This verifies the clip *text* and that the clip was *sent*. It does not transcribe the call audio; if you need proof
  of what the caller heard, add recording plus an STT pass on the outbound leg.

## Data retention and consent
| Data | Where | Default |
| --- | --- | --- |
| Call transcripts (PII) | `logs/transcripts.jsonl` | **off** (`SAVE_TRANSCRIPTS=false`) |
| Decision logs (caller utterances sent to Jev) | `logs/decisions.jsonl` | on, local, gitignored |
| Per-call summaries and latency | `logs/calls.jsonl`, `logs/metrics.jsonl` | on, no caller text |
| Labelled turns | `data/` | gitignored when under `data/raw/` |

Decision logs contain what callers said. Set a retention period and a deletion job before real calls; recording and
consent rules differ by market and use case (PRD open question 7). Utterances are also sent to Jev (TypeSafe) and
Gemini (Google); check both vendors' data terms against your policy.

## Open items for you
1. Required disclosure wording per script and per market.
2. Retention period, consent flow, and whether transcripts may be stored.
3. Whether outbound-audio verification (above) is required.
