"""Paired with/without-skill A/B evaluation.

Static suites often inflate gains when the agent never loads the skill.
Run the *same* cases twice — skill offered vs withheld — and report causal
lift plus trigger rate. Pure functions; no I/O, no framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .cases import Call, Case, ToolSpec
from .score import Scorecard, score

AgentFn = Callable[[Case, "SkillSpec | None"], Call]
TriggerFn = Callable[[Case, Call, "SkillSpec"], bool]


@dataclass(frozen=True)
class SkillSpec:
    """A skill offered to the agent (e.g. contents of a SKILL.md)."""

    name: str
    text: str = ""
    trigger_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmResult:
    """Scorecard for one arm of the A/B pair."""

    label: str
    scorecard: Scorecard
    trigger_rate: float
    n_triggered: int
    n_eligible: int

    def as_dict(self) -> dict[str, float | int | str | dict[str, float]]:
        return {
            "label": self.label,
            "n_eligible": self.n_eligible,
            "n_triggered": self.n_triggered,
            "scorecard": self.scorecard.as_dict(),
            "trigger_rate": self.trigger_rate,
        }


@dataclass(frozen=True)
class ABReport:
    """Causal lift: with_skill minus without_skill, plus trigger rate."""

    skill: str
    with_skill: ArmResult
    without_skill: ArmResult
    lift: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "lift": dict(self.lift),
            "skill": self.skill,
            "with_skill": self.with_skill.as_dict(),
            "without_skill": self.without_skill.as_dict(),
        }


def _ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def _lift(with_card: Scorecard, without_card: Scorecard) -> dict[str, float]:
    return {
        "refusal": round(with_card.refusal - without_card.refusal, 4),
        "schema": round(with_card.schema - without_card.schema, 4),
        "selection": round(with_card.selection - without_card.selection, 4),
    }


def default_trigger(case: Case, call: Call, skill: SkillSpec) -> bool:
    """Heuristic: skill "fired" if a call arm used the expected tool, or hints hit.

    Prefer passing an explicit ``triggered`` sequence from your harness when you
    can observe skill load events on a real CLI.
    """
    if case.expect == "refuse":
        return False
    if call.refused or call.tool is None:
        return False
    if case.tool is not None and call.tool == case.tool:
        return True
    blob = f"{case.prompt}\n{skill.text}".lower()
    return any(h.lower() in blob for h in skill.trigger_hints)


def compare_skill(
    tools: Sequence[ToolSpec],
    cases: Sequence[Case],
    *,
    skill: SkillSpec,
    calls_with: Sequence[Call],
    calls_without: Sequence[Call],
    triggered: Sequence[bool] | None = None,
) -> ABReport:
    """Compare aligned with-skill vs without-skill call lists.

    ``triggered`` (optional) marks, per case on the *with* arm, whether the
    skill was actually loaded/used. When omitted, :func:`default_trigger` is
    used. Trigger rate is computed only over ``expect == "call"`` cases.
    """
    if len(cases) != len(calls_with) or len(cases) != len(calls_without):
        raise ValueError("cases, calls_with, and calls_without must align")
    if triggered is not None and len(triggered) != len(cases):
        raise ValueError("triggered must align with cases")

    with_card = score(tools, cases, calls_with)
    without_card = score(tools, cases, calls_without)

    flags: list[bool]
    if triggered is None:
        flags = [default_trigger(c, w, skill) for c, w in zip(cases, calls_with, strict=True)]
    else:
        flags = list(triggered)

    eligible = 0
    fired = 0
    for case, flag in zip(cases, flags, strict=True):
        if case.expect != "call":
            continue
        eligible += 1
        if flag:
            fired += 1

    rate = _ratio(fired, eligible)
    with_arm = ArmResult(
        label="with_skill",
        scorecard=with_card,
        trigger_rate=rate,
        n_triggered=fired,
        n_eligible=eligible,
    )
    without_arm = ArmResult(
        label="without_skill",
        scorecard=without_card,
        trigger_rate=0.0,
        n_triggered=0,
        n_eligible=eligible,
    )
    return ABReport(
        skill=skill.name,
        with_skill=with_arm,
        without_skill=without_arm,
        lift=_lift(with_card, without_card),
    )


def run_ab(
    tools: Sequence[ToolSpec],
    cases: Sequence[Case],
    skill: SkillSpec,
    agent: AgentFn,
    *,
    detect_trigger: TriggerFn | None = None,
) -> ABReport:
    """Run ``agent(case, skill|None)`` on both arms and build an :class:`ABReport`.

    The agent receives the :class:`SkillSpec` on the with-arm and ``None`` on the
    without-arm. No network I/O is performed here — your callable owns that.
    """
    calls_with = [agent(case, skill) for case in cases]
    calls_without = [agent(case, None) for case in cases]
    detect = detect_trigger or default_trigger
    triggered = [
        detect(case, call, skill) for case, call in zip(cases, calls_with, strict=True)
    ]
    return compare_skill(
        tools,
        cases,
        skill=skill,
        calls_with=calls_with,
        calls_without=calls_without,
        triggered=triggered,
    )
