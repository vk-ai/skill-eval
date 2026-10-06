"""Trigger precision/recall for skill descriptions.

The A/B harness measures trigger *rate* over cases that should use the skill,
which is recall only. A description that fires on everything scores 100%
there. This module adds the other half: **near-miss negatives** (prompts that
look related but should not load the skill), **k repeated runs** per case
(skill loading is stochastic), and optional **competing skills** offered at
the same time.

Pure functions; your ``agent`` callable owns any model/CLI I/O.
"""

from __future__ import annotations

import random
from collections.abc import Collection
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from .ab import SkillSpec

AgentResult = Union[bool, str, None, Collection[str]]
TriggerAgent = Callable[["TriggerCase", Sequence[SkillSpec], int], AgentResult]


@dataclass(frozen=True)
class TriggerCase:
    """A prompt plus whether the skill under test *should* load for it.

    Use ``should_trigger=False`` for near-misses: prompts that share words or
    a domain with the skill but belong to another skill or to no skill.
    """

    tid: str
    prompt: str
    should_trigger: bool
    note: str = ""


@dataclass(frozen=True)
class CaseOutcome:
    """Per-case result over ``runs`` repetitions."""

    case: TriggerCase
    runs: int
    fired: int
    threshold: float

    @property
    def rate(self) -> float:
        return round(self.fired / self.runs, 4) if self.runs else 0.0

    @property
    def triggered(self) -> bool:
        """Counted as triggered when the run rate is at least ``threshold``."""
        return self.runs > 0 and self.fired / self.runs >= self.threshold

    @property
    def correct(self) -> bool:
        return self.triggered == self.case.should_trigger

    @property
    def flaky(self) -> bool:
        """Fired on some runs but not all."""
        return 0 < self.fired < self.runs

    @property
    def kind(self) -> str:
        if self.case.should_trigger:
            return "TP" if self.triggered else "FN"
        return "FP" if self.triggered else "TN"


def _ratio(a: int | float, b: int | float) -> float:
    return round(a / b, 4) if b else 0.0


@dataclass(frozen=True)
class TriggerReport:
    """Precision / recall / false-trigger rate for one skill description."""

    skill: str
    k: int
    threshold: float
    outcomes: tuple[CaseOutcome, ...]

    def _count(self, kind: str) -> int:
        return sum(1 for o in self.outcomes if o.kind == kind)

    @property
    def tp(self) -> int:
        return self._count("TP")

    @property
    def fp(self) -> int:
        return self._count("FP")

    @property
    def fn(self) -> int:
        return self._count("FN")

    @property
    def tn(self) -> int:
        return self._count("TN")

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def false_trigger_rate(self) -> float:
        return _ratio(self.fp, self.fp + self.tn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, len(self.outcomes))

    @property
    def run_level(self) -> dict[str, float]:
        """Raw per-run firing rates, before the per-case threshold."""
        pos = [o for o in self.outcomes if o.case.should_trigger]
        neg = [o for o in self.outcomes if not o.case.should_trigger]
        return {
            "negative_fire_rate": _ratio(sum(o.fired for o in neg), sum(o.runs for o in neg)),
            "positive_fire_rate": _ratio(sum(o.fired for o in pos), sum(o.runs for o in pos)),
        }

    @property
    def flaky(self) -> tuple[CaseOutcome, ...]:
        return tuple(o for o in self.outcomes if o.flaky)

    def failures(self) -> tuple[CaseOutcome, ...]:
        """False negatives and false triggers."""
        return tuple(o for o in self.outcomes if not o.correct)

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "counts": {"fn": self.fn, "fp": self.fp, "tn": self.tn, "tp": self.tp},
            "f1": self.f1,
            "false_trigger_rate": self.false_trigger_rate,
            "flaky": [o.case.tid for o in self.flaky],
            "k": self.k,
            "precision": self.precision,
            "recall": self.recall,
            "run_level": self.run_level,
            "skill": self.skill,
            "threshold": self.threshold,
        }

    def table(self) -> str:
        """Markdown summary plus a per-case table (worst first)."""
        lines = [
            f"# Trigger eval: `{self.skill}` (k={self.k}, threshold={self.threshold:g})",
            "",
            f"precision {self.precision:.2f} · recall {self.recall:.2f} · "
            f"false-trigger rate {self.false_trigger_rate:.2f} · F1 {self.f1:.2f} · "
            f"flaky {len(self.flaky)}",
            "",
            "| case | should | fired | rate | result | note |",
            "|---|---|---|---|---|---|",
        ]
        order = {"FP": 0, "FN": 1, "TP": 2, "TN": 3}
        for o in sorted(self.outcomes, key=lambda o: (order[o.kind], o.case.tid)):
            should = "yes" if o.case.should_trigger else "no"
            lines.append(
                f"| `{o.case.tid}` | {should} | {o.fired}/{o.runs} | {o.rate:.2f} | "
                f"{o.kind}{' (flaky)' if o.flaky else ''} | {o.case.note} |"
            )
        return "\n".join(lines) + "\n"


def _fired(result: AgentResult, skill: SkillSpec) -> bool:
    if isinstance(result, bool):
        return result
    if result is None:
        return False
    if isinstance(result, str):
        return result == skill.name
    if isinstance(result, Collection):
        return skill.name in result
    raise TypeError(f"agent must return bool, skill name, names, or None; got {type(result).__name__}")


def run_triggers(
    cases: Sequence[TriggerCase],
    skill: SkillSpec,
    agent: TriggerAgent,
    *,
    k: int = 3,
    threshold: float = 0.5,
    competing: Sequence[SkillSpec] = (),
) -> TriggerReport:
    """Run each case ``k`` times and score trigger precision/recall.

    ``agent(case, offered, run_index)`` gets the skills offered for this run
    (``skill`` first, then ``competing``) and returns which skill(s) loaded:
    a skill name, a collection of names, ``None``, or a bool meaning
    "``skill`` loaded". A case counts as triggered when it fired on at least
    ``threshold`` of its runs.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not cases:
        raise ValueError("need at least one case")
    seen: set[str] = set()
    for c in cases:
        if c.tid in seen:
            raise ValueError(f"duplicate case id: {c.tid!r}")
        seen.add(c.tid)
    if any(s.name == skill.name for s in competing):
        raise ValueError("competing skills must have different names from the skill under test")
    offered = (skill, *competing)
    outcomes = []
    for case in cases:
        fired = sum(1 for i in range(k) if _fired(agent(case, offered, i), skill))
        outcomes.append(CaseOutcome(case=case, runs=k, fired=fired, threshold=threshold))
    return TriggerReport(skill=skill.name, k=k, threshold=threshold, outcomes=tuple(outcomes))


def from_observations(
    cases: Sequence[TriggerCase],
    skill: str,
    fired: Sequence[Sequence[bool]],
    *,
    threshold: float = 0.5,
) -> TriggerReport:
    """Score pre-recorded runs: ``fired[i]`` is the per-run flags for ``cases[i]``."""
    if len(cases) != len(fired):
        raise ValueError("cases and fired must align")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    outcomes = tuple(
        CaseOutcome(case=c, runs=len(runs), fired=sum(1 for f in runs if f), threshold=threshold)
        for c, runs in zip(cases, fired, strict=True)
    )
    ks = {o.runs for o in outcomes}
    return TriggerReport(skill=skill, k=max(ks) if ks else 0, threshold=threshold, outcomes=outcomes)


def split_cases(
    cases: Sequence[TriggerCase],
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[TriggerCase], list[TriggerCase]]:
    """Deterministic train/validation split, stratified by ``should_trigger``.

    Iterate on the description against ``train``; report on ``val`` so the
    description isn't overfit to the cases you looked at.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = random.Random(seed)
    train: list[TriggerCase] = []
    val: list[TriggerCase] = []
    for label in (True, False):
        group = [c for c in cases if c.should_trigger is label]
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    key = {c.tid: i for i, c in enumerate(cases)}
    return sorted(train, key=lambda c: key[c.tid]), sorted(val, key=lambda c: key[c.tid])
