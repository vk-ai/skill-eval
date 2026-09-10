from __future__ import annotations

import pytest

from skill_eval import (
    Call,
    Case,
    SkillSpec,
    ToolSpec,
    compare_skill,
    run_ab,
    score,
)

TOOLS = [ToolSpec("search", required=("q",), optional=("k",))]


def test_compare_skill_reports_positive_lift_and_trigger_rate() -> None:
    skill = SkillSpec("web-search", text="Use search with q=...", trigger_hints=("search",))
    cases = [
        Case("c1", "find cafes", "call", tool="search", required_args={"q": "cafes"}),
        Case("c2", "write a poem", "refuse"),
    ]
    # Without skill: wrong tool / no refusal
    without = [
        Call("calendar", {"date": "x"}),
        Call("search", {"q": "poem"}),
    ]
    # With skill: correct selection + refusal
    with_calls = [
        Call("search", {"q": "cafes"}),
        Call(None, {}, refused=True),
    ]
    report = compare_skill(
        TOOLS,
        cases,
        skill=skill,
        calls_with=with_calls,
        calls_without=without,
        triggered=[True, False],
    )
    assert report.skill == "web-search"
    assert report.with_skill.scorecard.selection == 1.0
    assert report.without_skill.scorecard.selection == 0.0
    assert report.lift["selection"] == 1.0
    assert report.lift["refusal"] == 1.0
    assert report.with_skill.trigger_rate == 1.0
    assert report.with_skill.n_eligible == 1
    assert report.as_dict()["lift"]["schema"] == report.lift["schema"]


def test_compare_skill_requires_alignment() -> None:
    skill = SkillSpec("x")
    cases = [Case("c1", "q", "call", tool="search")]
    with pytest.raises(ValueError, match="align"):
        compare_skill(
            TOOLS,
            cases,
            skill=skill,
            calls_with=[Call("search", {"q": "a"})],
            calls_without=[],
        )


def test_run_ab_invokes_agent_both_arms() -> None:
    skill = SkillSpec("web-search", text="prefer search tool")
    cases = [
        Case("c1", "find trains", "call", tool="search", required_args={"q": "trains"}),
        Case("c2", "poem please", "refuse"),
    ]
    seen: list[str | None] = []

    def agent(case: Case, offered: SkillSpec | None) -> Call:
        seen.append(None if offered is None else offered.name)
        if offered is None:
            if case.expect == "refuse":
                return Call("search", {"q": "nope"})
            return Call(None, {}, refused=True)
        if case.expect == "refuse":
            return Call(None, {}, refused=True)
        return Call("search", {"q": "trains"})

    report = run_ab(TOOLS, cases, skill, agent, detect_trigger=lambda c, call, s: call.tool == "search")
    assert seen == ["web-search", "web-search", None, None]
    assert report.lift["selection"] == 1.0
    assert report.lift["refusal"] == 1.0
    assert report.with_skill.trigger_rate == 1.0


def test_existing_score_unchanged() -> None:
    cases = [Case("c1", "find cafes", "call", tool="search")]
    calls = [Call("search", {"q": "cafes"})]
    s = score(TOOLS, cases, calls)
    assert s.selection == 1.0
