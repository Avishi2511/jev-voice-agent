from pathlib import Path

import pytest
from pydantic import SecretStr

from jevagent.config import Settings
from jevagent.eval.dataset import (
    DatasetError,
    LabelledTurn,
    freeze,
    load_turns,
    save_turns,
    validate_turn,
    verify_frozen,
)
from jevagent.eval.report import to_markdown, tune_precision_threshold, tune_recall_threshold
from jevagent.eval.run_eval import run_eval
from jevagent.eval.splits import HeldOutAccessError, assign, get_split, split_turns
from jevagent.eval.versions import (
    PromotionError,
    PromotionTargets,
    QuestionVersion,
    apply_thresholds,
    load_instructions,
    next_version,
    promote,
    save_thresholds,
    save_version,
    thresholds_for,
)
from jevagent.models import JevAnswer, JevQuestion, JsonDict
from jevagent.script.loader import load_script

SCRIPT = load_script(Path(__file__).parents[2] / "scripts" / "emi_reminder_en.yaml")
SETTINGS = Settings(_env_file=None, jev_api_key=SecretStr("k"))  # type: ignore[call-arg]


def turn(call: str, n: int, utt: str, step: str = "pay_when", **over: object) -> LabelledTurn:
    labels = {"answered_step": "yes", f"slot_value:{step}": "tomorrow", "turn_type": "on_script"}
    base = dict(
        call_id=call,
        turn=n,
        step=step,
        utterance=utt,
        stored={"identity": "confirmed"},
        labels=labels,
        should_handoff=False,
    )
    base.update(over)
    return LabelledTurn.model_validate(base)


def dataset() -> list[LabelledTurn]:
    turns = [turn(f"c{i}", 1, f"tomorrow {i}") for i in range(30)]
    turns.append(
        turn(
            "obj",
            1,
            "why the penalty",
            labels={
                "answered_step": "no",
                "slot_value:pay_when": "not_stated",
                "turn_type": "objection",
            },
            should_handoff=True,
        )
    )
    turns.append(
        turn(
            "cor",
            1,
            "wrong, the day after",
            labels={
                "answered_step": "yes",
                "slot_value:pay_when": "later",
                "turn_type": "on_script",
                "corrects_stored": "yes",
            },
            should_handoff=True,
        )
    )
    return turns


class OracleJev:
    """Answers from the labels, optionally getting some questions wrong."""

    def __init__(self, turns: list[LabelledTurn], wrong: set[str] | None = None) -> None:
        self.by_utt = {t.utterance: t for t in turns}
        self.wrong = wrong or set()
        self.calls = 0

    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        self.calls += 1
        t = self.by_utt[state["utterance"]]
        out = {}
        for q in questions:
            label = t.labels.get(q.key, "no" if q.type.value == "noul" else "none")
            if q.key.split(":")[0] in self.wrong and t.utterance in self.wrong:
                label = "unclear"
            out[q.key] = JevAnswer(key=q.key, selected=label, confidence=0.95)
        return out


# -- dataset ---------------------------------------------------------------


def test_valid_turn_passes_and_roundtrips(tmp_path: Path) -> None:
    turns = dataset()
    path = tmp_path / "t.jsonl"
    save_turns(path, turns)
    assert load_turns(path, SCRIPT) == turns


@pytest.mark.parametrize(
    "mutation",
    [
        {"step": "nope"},
        {"labels": {"answered_step": "yes", "turn_type": "on_script"}},  # no slot label
        {
            "labels": {
                "answered_step": "maybe",
                "slot_value:pay_when": "today",
                "turn_type": "on_script",
            }
        },
        {"labels": {"answered_step": "no", "turn_type": "on_script", "bogus": "x"}},
        {
            "labels": {
                "answered_step": "no",
                "slot_value:pay_when": "cash",
                "turn_type": "on_script",
            }
        },
        {"labels": {"answered_step": "no"}},  # no turn_type
    ],
)
def test_invalid_labels_rejected(mutation: dict[str, object]) -> None:
    with pytest.raises(DatasetError):
        validate_turn(SCRIPT, turn("c", 1, "x", **mutation))


def test_load_rejects_bad_json_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=":1:"):
        load_turns(path, SCRIPT)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no turns"):
        load_turns(path, SCRIPT)


def test_freeze_detects_changes(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    save_turns(path, dataset())
    with pytest.raises(DatasetError, match="not frozen"):
        verify_frozen(path)
    freeze(path)
    verify_frozen(path)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(DatasetError, match="changed"):
        verify_frozen(path)


# -- splits ----------------------------------------------------------------


def test_split_is_deterministic_and_by_call() -> None:
    turns = [turn(f"call{i % 40}", i, f"u{i}") for i in range(200)]
    a, b = split_turns(turns), split_turns(turns)
    assert {k: [t.utterance for t in v] for k, v in a.items()} == {
        k: [t.utterance for t in v] for k, v in b.items()
    }
    seen: dict[str, str] = {}
    for name, ts in a.items():
        for t in ts:
            assert seen.setdefault(t.call_id, name) == name  # no call leaks across splits
    assert sum(map(len, a.values())) == 200
    assert all(len(v) > 0 for v in a.values())
    assert assign("c1") == assign("c1")
    assert assign("c1", salt="other") in {"tuning", "validation", "heldout"}


def test_heldout_guarded_and_audited(tmp_path: Path) -> None:
    turns = [turn(f"call{i}", 1, f"u{i}") for i in range(50)]
    audit = tmp_path / "audit.log"
    with pytest.raises(HeldOutAccessError):
        get_split(turns, "heldout", audit_log=audit)
    with pytest.raises(HeldOutAccessError):
        get_split(turns, "heldout", allow_heldout=True, audit_log=audit)  # no reason
    assert not audit.exists()
    got = get_split(turns, "heldout", allow_heldout=True, reason="promote v2", audit_log=audit)
    assert got and "promote v2" in audit.read_text(encoding="utf-8")
    assert get_split(turns, "tuning", audit_log=audit)  # other splits need no flag


# -- run_eval --------------------------------------------------------------


async def test_perfect_oracle_scores_perfectly() -> None:
    turns = dataset()
    report = await run_eval(SCRIPT, turns, OracleJev(turns), SETTINGS, shuffle_seeds=[1, 2])
    assert report.groups["answered_step"].accuracy == 1.0
    assert report.groups["slot_value"].accuracy == 1.0
    assert report.rare["corrects_stored_yes"].recall == 1.0
    assert report.rare["off_script"].recall == 1.0
    assert report.handoff.recall == 1.0
    assert report.handoff.false_handoff_rate == 0.0
    assert report.position_stability == 1.0
    assert report.n_turns == len(turns)


async def test_rare_class_misses_hidden_by_overall_accuracy() -> None:
    turns = dataset()
    jev = OracleJev(turns, wrong={"turn_type", "why the penalty"})
    report = await run_eval(SCRIPT, turns, jev, SETTINGS)
    tt = report.groups["turn_type"]
    assert tt.accuracy > 0.9  # looks great overall...
    assert not tt.beats_baseline  # ...but only ties the majority-class baseline
    assert report.rare["off_script"].recall == 0.0  # ...and misses every objection
    assert report.handoff.recall is not None and report.handoff.recall < 1.0
    assert "**NO**" in to_markdown(report)


async def test_position_bias_detected() -> None:
    turns = dataset()

    class FirstOption:
        async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
            return {
                q.key: JevAnswer(
                    key=q.key, selected=q.options[0] if q.options else "no", confidence=0.9
                )
                for q in questions
            }

    report = await run_eval(SCRIPT, turns, FirstOption(), SETTINGS, shuffle_seeds=[1, 2, 3])
    assert report.position_stability is not None and report.position_stability < 0.9


async def test_instruction_override_reaches_jev() -> None:
    turns = dataset()[:2]
    seen: list[str] = []

    class Spy(OracleJev):
        async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
            seen.extend(q.instruction for q in questions if q.key == "turn_type")
            return await super().ask(state, questions)

    await run_eval(SCRIPT, turns, Spy(turns), SETTINGS, instructions={"turn_type": "V2 WORDING"})
    assert seen == ["V2 WORDING"] * 2


# -- tuner -----------------------------------------------------------------


def test_precision_tuner_picks_lowest_passing_threshold() -> None:
    samples = [(0.55, False), (0.6, False), (0.75, True), (0.8, False), (0.9, True), (0.95, True)]
    assert tune_precision_threshold(samples, 1.0) == 0.81
    assert tune_precision_threshold(samples, 0.5) == 0.5
    assert tune_precision_threshold([(0.6, False)], 0.9) is None


def test_recall_tuner_picks_highest_threshold_meeting_recall() -> None:
    samples = [(0.9, True), (0.8, True), (0.7, True), (0.6, True), (0.99, False)]
    assert tune_recall_threshold(samples, 1.0) == 0.6
    assert tune_recall_threshold(samples, 0.5) == 0.8
    assert tune_recall_threshold([(0.9, False)], 0.9) is None


# -- versions / promotion --------------------------------------------------


async def good_report(turns: list[LabelledTurn]):  # type: ignore[no-untyped-def]
    return await run_eval(SCRIPT, turns, OracleJev(turns), SETTINGS)


async def test_promotion_requires_validation_pass(tmp_path: Path) -> None:
    turns = dataset()
    path = save_version(
        tmp_path, "emi", QuestionVersion(key="turn_type", version=1, instruction="v1")
    )
    report = await good_report(turns)
    with pytest.raises(PromotionError, match="validation"):
        promote(path, report, "heldout")
    with pytest.raises(PromotionError, match="accuracy"):
        promote(path, report, "validation", PromotionTargets(min_accuracy=1.01))
    assert load_instructions(tmp_path, "emi") == {}  # still draft
    promoted = promote(path, report, "validation")
    assert promoted.status == "promoted" and promoted.model_version == "jev-1.14.0"
    assert load_instructions(tmp_path, "emi") == {"turn_type": "v1"}


async def test_failed_rare_recall_blocks_promotion(tmp_path: Path) -> None:
    turns = dataset()
    report = await run_eval(
        SCRIPT, turns, OracleJev(turns, wrong={"turn_type", "why the penalty"}), SETTINGS
    )
    path = save_version(
        tmp_path, "emi", QuestionVersion(key="turn_type", version=1, instruction="v1")
    )
    with pytest.raises(PromotionError, match="off_script"):
        promote(path, report, "validation")


def test_latest_promoted_wins_and_keys_are_filename_safe(tmp_path: Path) -> None:
    for v, status in [(1, "promoted"), (2, "draft"), (3, "promoted")]:
        save_version(
            tmp_path,
            "emi",
            QuestionVersion(
                key="slot_value:pay_when", version=v, instruction=f"v{v}", status=status
            ),
        )
    assert load_instructions(tmp_path, "emi") == {"slot_value:pay_when": "v3"}
    assert load_instructions(tmp_path, "emi", promoted_only=False) == {"slot_value:pay_when": "v3"}
    assert next_version(tmp_path, "emi", "slot_value:pay_when") == 4
    assert next_version(tmp_path, "emi", "turn_type") == 1


def test_thresholds_pinned_to_model(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    save_thresholds(path, "jev-1.14.0", {"slot_value": 0.8})
    assert thresholds_for(path, "jev-1.14.0") == {"slot_value": 0.8}
    with pytest.raises(PromotionError, match="re-tune"):
        thresholds_for(path, "jev-1.15.0")
    tuned = apply_thresholds(SCRIPT, {"slot_value": 0.8})
    assert tuned.thresholds["slot_value"] == 0.8 and tuned.thresholds["default"] == 0.7
    assert SCRIPT.thresholds == {"default": 0.7}
