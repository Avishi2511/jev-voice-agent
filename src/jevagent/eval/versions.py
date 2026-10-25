"""Versioned question artifacts and model-pinned thresholds (E-5).

Layout: questions/<script>/<question>@vN.yaml, plus thresholds.yaml keyed by Jev model version.
A question version is promoted only after a validation-split report passes the targets.
"""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from jevagent.eval.run_eval import EvalReport
from jevagent.eval.splits import Split
from jevagent.script.schema import Script


class PromotionError(ValueError):
    pass


class QuestionVersion(BaseModel):
    key: str
    version: int
    instruction: str
    status: Literal["draft", "promoted"] = "draft"
    model_version: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)


class PromotionTargets(BaseModel):
    min_accuracy: float = 0.9
    min_rare_recall: float = 0.9
    min_handoff_recall: float = 0.9
    max_false_handoff_rate: float = 0.15


def _safe(key: str) -> str:
    return key.replace(":", "__")


def version_path(root: str | Path, script: str, key: str, version: int) -> Path:
    return Path(root) / script / f"{_safe(key)}@v{version}.yaml"


def save_version(root: str | Path, script: str, qv: QuestionVersion) -> Path:
    path = version_path(root, script, qv.key, qv.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(qv.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def load_version(path: str | Path) -> QuestionVersion:
    return QuestionVersion.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_instructions(root: str | Path, script: str, promoted_only: bool = True) -> dict[str, str]:
    """Latest (promoted) instruction per key, for build_questions(instructions=...)."""
    best: dict[str, QuestionVersion] = {}
    for path in (Path(root) / script).glob("*@v*.yaml"):
        qv = load_version(path)
        if promoted_only and qv.status != "promoted":
            continue
        if qv.key not in best or qv.version > best[qv.key].version:
            best[qv.key] = qv
    return {k: v.instruction for k, v in best.items()}


def promote(
    path: str | Path,
    report: EvalReport,
    split: Split,
    targets: PromotionTargets | None = None,
) -> QuestionVersion:
    t = targets or PromotionTargets()
    if split != "validation":
        raise PromotionError(f"promotion is decided on the validation split, got {split!r}")
    qv = load_version(path)
    group = qv.key.split(":", 1)[0]
    failures: list[str] = []

    g = report.groups.get(group)
    if g is None:
        failures.append(f"no labels for {group!r} in report")
    else:
        if g.accuracy < t.min_accuracy:
            failures.append(f"accuracy {g.accuracy:.2f} < {t.min_accuracy}")
        if not g.beats_baseline:
            failures.append("does not beat majority-class baseline")
    rare_name = {"corrects_stored": "corrects_stored_yes", "turn_type": "off_script"}.get(group)
    if rare_name:
        recall = report.rare[rare_name].recall
        if recall is None or recall < t.min_rare_recall:
            failures.append(f"{rare_name} recall {recall} < {t.min_rare_recall}")
    h = report.handoff
    if h.recall is None or h.recall < t.min_handoff_recall:
        failures.append(f"handoff recall {h.recall} < {t.min_handoff_recall}")
    if h.false_handoff_rate is not None and h.false_handoff_rate > t.max_false_handoff_rate:
        failures.append(
            f"false handoff rate {h.false_handoff_rate:.2f} > {t.max_false_handoff_rate}"
        )
    if failures:
        raise PromotionError(f"{path}: " + "; ".join(failures))

    qv.status = "promoted"
    qv.model_version = report.model_version
    qv.scores = {"accuracy": g.accuracy if g else 0.0}
    save_version(Path(path).parents[1], Path(path).parent.name, qv)
    return qv


# -- thresholds pinned to a Jev model version ---------------------------------------------


def save_thresholds(path: str | Path, model: str, thresholds: dict[str, float]) -> None:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    data = data or {}
    data[model] = thresholds
    p.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def thresholds_for(path: str | Path, model: str) -> dict[str, float]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if model not in data:
        raise PromotionError(
            f"no thresholds tuned for model {model!r} (have {sorted(data)}); "
            "re-tune before upgrading"
        )
    return {str(k): float(v) for k, v in data[model].items()}


def apply_thresholds(script: Script, thresholds: dict[str, float]) -> Script:
    return script.model_copy(update={"thresholds": {**script.thresholds, **thresholds}})


_VERSION_RE = re.compile(r"@v(\d+)\.yaml$")


def next_version(root: str | Path, script: str, key: str) -> int:
    versions = [
        int(m.group(1))
        for p in (Path(root) / script).glob(f"{_safe(key)}@v*.yaml")
        if (m := _VERSION_RE.search(p.name))
    ]
    return max(versions, default=0) + 1
