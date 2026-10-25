"""Compliance checks for mandatory disclosures (B-4).

The required wording comes from you (compliance.yaml); nothing here invents legal text. Two
checks: the script's disclosure clips must contain the wording verbatim, and every call that
should have played a disclosure must have played that clip (not a spoken paraphrase).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from jevagent.observability.summary import CallSummary
from jevagent.script.schema import Script


class Disclosure(BaseModel):
    id: str  # id of the clip that must carry it
    text: str  # required wording, verbatim
    required_on: str = "all"  # "all" calls, or "completed" calls only


class ComplianceConfig(BaseModel):
    disclosures: list[Disclosure]


def load_config(path: str | Path) -> ComplianceConfig:
    return ComplianceConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _norm(text: str) -> str:
    return " ".join(text.split())


def check_script(script: Script, config: ComplianceConfig) -> list[str]:
    """Problems with the script itself: missing clip, or wording that is not verbatim."""
    clips = {c.id: c.text for c in script.all_clips()}
    problems = []
    for d in config.disclosures:
        if d.id not in clips:
            problems.append(f"{d.id}: no such clip in the script")
        elif _norm(d.text) not in _norm(clips[d.id]):
            problems.append(f"{d.id}: clip text does not contain the required wording verbatim")
    return problems


def check_call(summary: CallSummary, config: ComplianceConfig) -> list[str]:
    """Disclosures this call should have played but did not."""
    missing = []
    for d in config.disclosures:
        applies = d.required_on == "all" or summary.outcome == "completed"
        if applies and d.id not in summary.clips_played:
            missing.append(d.id)
    return missing


def audit(
    script: Script, config: ComplianceConfig, summaries: list[CallSummary]
) -> dict[str, list[str]]:
    """call_id -> missing disclosure ids, for non-compliant calls only; '' holds script problems."""
    out = {s.call_id: m for s in summaries if (m := check_call(s, config))}
    if problems := check_script(script, config):
        out[""] = problems
    return out
