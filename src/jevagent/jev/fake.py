"""Offline Jev fakes for tests and demos."""

from collections.abc import Callable

from jevagent.models import ESCAPE_OPTIONS, JevAnswer, JevQuestion, JsonDict, QuestionType

Handler = Callable[[JsonDict, list[JevQuestion]], dict[str, JevAnswer]]


class ScriptedJev:
    """Returns pre-baked answer sets in order, or delegates to a handler. Records requests."""

    def __init__(
        self, turns: list[dict[str, JevAnswer]] | None = None, handler: Handler | None = None
    ) -> None:
        self._turns = list(turns or [])
        self._handler = handler
        self.requests: list[tuple[JsonDict, list[JevQuestion]]] = []

    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        self.requests.append((state, questions))
        if self._handler is not None:
            return self._handler(state, questions)
        return self._turns.pop(0)


def answer(key: str, selected: str, confidence: float = 0.95) -> JevAnswer:
    return JevAnswer(key=key, selected=selected, confidence=confidence)


class RuleBasedJev:
    """Keyword-driven stand-in: slot options and turn types matched by substring."""

    def __init__(
        self,
        slot_keywords: dict[str, dict[str, list[str]]],
        turn_keywords: dict[str, list[str]] | None = None,
        correction_words: list[str] | None = None,
    ) -> None:
        self._slots = slot_keywords  # {step_id: {option: [words]}}
        self._turns = turn_keywords or {}
        self._corrections = correction_words or []

    async def ask(self, state: JsonDict, questions: list[JevQuestion]) -> dict[str, JevAnswer]:
        text = str(state["utterance"]).lower()
        step = str(state["current_step"])
        return {q.key: self._answer(q, text, step) for q in questions}

    def _answer(self, q: JevQuestion, text: str, step: str) -> JevAnswer:
        if q.key.startswith("slot_value:"):
            hit = self._match(self._slots.get(q.key.split(":", 1)[1], {}), text)
            return answer(q.key, hit or self._escape(q), 0.95 if hit else 0.9)
        if q.key == "turn_type":
            return answer(q.key, self._match(self._turns, text) or "on_script")
        if q.key == "answered_step":
            current = self._slots.get(step, {})
            return answer(q.key, "yes" if self._match(current, text) else "no")
        if q.key == "corrects_stored":
            return answer(q.key, "yes" if any(w in text for w in self._corrections) else "no")
        if q.type is QuestionType.CHOICE:
            return answer(q.key, self._escape(q))
        return answer(q.key, "no")

    @staticmethod
    def _match(table: dict[str, list[str]], text: str) -> str | None:
        for option, words in table.items():
            if any(w in text for w in words):
                return option
        return None

    @staticmethod
    def _escape(q: JevQuestion) -> str:
        return next((o for o in q.options if o in ESCAPE_OPTIONS), "unclear")
