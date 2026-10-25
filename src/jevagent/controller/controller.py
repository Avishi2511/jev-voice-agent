"""Call controller: owns CallState and the A -> B mode switch (CT-1, H-6).

Per turn: build state -> Jev (one request) -> triggers -> engine -> action.
After handoff the controller never consults Jev or the script again.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from jevagent.config import Settings
from jevagent.controller.packet import HandoffPacket, build_packet
from jevagent.controller.triggers import answer_triggers, uncovered_state
from jevagent.jev.client import JevClient, low_confidence_answers
from jevagent.jev.questions import build_questions, build_state
from jevagent.models import (
    Advance,
    CallState,
    Handoff,
    JevAnswer,
    JevQuestion,
    JsonDict,
    Mode,
    ReAsk,
)
from jevagent.observability.decision_log import DecisionLog, DecisionRecord
from jevagent.script.engine import ScriptEngine
from jevagent.script.schema import Script

log = structlog.get_logger(__name__)

MAX_RECENT_TURNS = 6


class PlayClip(BaseModel):
    clip_id: str
    end_call: bool = False


class StartHandoff(BaseModel):
    packet: HandoffPacket


class PassToPhaseB(BaseModel):
    utterance: str


Action = PlayClip | StartHandoff | PassToPhaseB


class Controller:
    def __init__(
        self,
        script: Script,
        jev: JevClient,
        settings: Settings,
        decision_log: DecisionLog,
        call_id: str,
        tools: list[str] | None = None,
        seed: int | None = None,
        shuffle_seed: int | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.script = script
        self.state: CallState
        self._engine = ScriptEngine(script, seed=seed)
        self._jev = jev
        self._settings = settings
        self._log = decision_log
        self._call_id = call_id
        self._tools = tools or []
        self._shuffle_seed = shuffle_seed
        self._clock = clock
        self._turn = 0
        self.state = self._engine.initial_state()

    def start(self) -> PlayClip:
        clip = self._engine.opening_clip(self.state)
        self._remember_agent(clip)
        return PlayClip(clip_id=clip)

    async def on_caller_turn(self, utterance: str) -> Action:
        self._remember(f"caller: {utterance}")
        if self.state.mode is Mode.B:
            return PassToPhaseB(utterance=utterance)

        self._turn += 1
        state_sent = build_state(self.script, self.state, utterance)
        uncovered = uncovered_state(self.script, self.state)
        if uncovered is not None:
            questions: list[JevQuestion] = []
            answers: dict[str, JevAnswer] = {}
            handoff: Handoff | None = uncovered
        else:
            questions = build_questions(self.script, self.state, self._shuffle_seed)
            answers = await self._ask_jev(state_sent, questions)
            handoff = answer_triggers(self.script, self.state, answers, self._settings)

        action: Action
        if handoff is None:
            decision = self._engine.apply_answers(self.state, answers)
            if isinstance(decision, Handoff):
                handoff = decision
            elif isinstance(decision, (Advance, ReAsk)):
                self._remember_agent(decision.clip_id)
                action = PlayClip(
                    clip_id=decision.clip_id,
                    end_call=isinstance(decision, Advance) and decision.terminal,
                )
        if handoff is not None:
            self.state.mode = Mode.B
            action = StartHandoff(packet=build_packet(self.state, handoff, self._tools))
            log.info("handoff", call_id=self._call_id, reason=handoff.detail or handoff.reason)

        self._record(state_sent, questions, answers, action)
        return action

    async def _ask_jev(
        self, state_sent: JsonDict, questions: list[JevQuestion]
    ) -> dict[str, JevAnswer]:
        answers = await self._jev.ask(state_sent, questions)
        # Missing keys are treated as failures, never as agreement.
        fallback = low_confidence_answers(questions)
        return {q.key: answers.get(q.key, fallback[q.key]) for q in questions}

    def _remember_agent(self, clip_id: str) -> None:
        self._remember(f"agent: {self.script.clip_text(clip_id)}")

    def _remember(self, line: str) -> None:
        self.state.recent_turns.append(line)
        del self.state.recent_turns[:-MAX_RECENT_TURNS]

    def _record(
        self,
        state_sent: JsonDict,
        questions: list[JevQuestion],
        answers: dict[str, JevAnswer],
        action: Action,
    ) -> None:
        self._log.write(
            DecisionRecord(
                call_id=self._call_id,
                turn=self._turn,
                timestamp=self._clock(),
                model_version=self._settings.jev_model,
                state_sent=state_sent,
                questions=[q.model_dump(mode="json") for q in questions],
                answers=answers,
                action={"type": type(action).__name__, **action.model_dump(mode="json")},
            )
        )
