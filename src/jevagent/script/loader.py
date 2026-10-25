"""Load and validate scripts. Fails loudly on any structural problem."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from jevagent.models import ESCAPE_OPTIONS, QuestionType
from jevagent.script.schema import Script


class ScriptError(ValueError):
    pass


def load_script(path: str | Path) -> Script:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        script = Script.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ScriptError(f"cannot load script {path}: {exc}") from exc
    validate_script(script)
    return script


def validate_script(script: Script) -> None:
    if not script.steps:
        raise ScriptError("script has no steps")
    if "default" not in script.thresholds:
        raise ScriptError("script.thresholds must define 'default'")

    ids = [s.id for s in script.steps]
    if len(set(ids)) != len(ids):
        raise ScriptError(f"duplicate step ids: {ids}")
    slots = [s.slot for s in script.steps if s.slot]
    if len(set(slots)) != len(slots):
        raise ScriptError(f"duplicate slot names: {slots}")

    for step in script.steps:
        if not step.clips.ask:
            raise ScriptError(f"step {step.id!r} has no ask clips")
        if step.slot:
            if not step.clips.re_ask:
                raise ScriptError(f"step {step.id!r} has a slot but no re_ask clips")
            if "slot_value" not in step.questions:
                raise ScriptError(f"step {step.id!r} needs a 'slot_value' question")
        if step.on_filled and step.on_filled not in ids:
            raise ScriptError(f"step {step.id!r} on_filled references unknown step")
        for key, q in step.questions.items():
            if q.type is QuestionType.CHOICE and not ESCAPE_OPTIONS & set(q.options):
                raise ScriptError(
                    f"step {step.id!r} question {key!r} has no escape option "
                    f"(one of {sorted(ESCAPE_OPTIONS)})"
                )

    clip_ids = [c.id for c in script.all_clips()]
    if len(set(clip_ids)) != len(clip_ids):
        raise ScriptError("clip ids must be unique across the script")

    if script.steps[-1].slot is not None:
        raise ScriptError("the last step must be a terminal step without a slot")
