from __future__ import annotations

import pytest

from skill_eval import (
    SkillSpec,
    TriggerCase,
    from_observations,
    run_triggers,
    split_cases,
)

PDF = SkillSpec("pdf-forms", text="Fill and extract PDF form fields.")
DOCX = SkillSpec("docx", text="Edit Word documents.")

CASES = [
    TriggerCase("p1", "fill the form fields in invoice.pdf", True),
    TriggerCase("p2", "extract the text boxes from this PDF form", True),
    TriggerCase("p3", "what fields does tax-form.pdf have?", True),
    TriggerCase("n1", "convert report.docx to PDF", False, note="near-miss: mentions PDF"),
    TriggerCase("n2", "summarize this PDF article", False, note="near-miss: PDF but no form"),
    TriggerCase("n3", "fill in the blanks in this sentence", False, note="near-miss: 'fill'"),
]


def greedy_agent(case: TriggerCase, offered, run: int):
    """A description that fires on any mention of 'pdf' or 'fill'."""
    text = case.prompt.lower()
    return "pdf-forms" if ("pdf" in text or "fill" in text) else None


def test_recall_only_hides_false_triggers_precision_exposes_them() -> None:
    r = run_triggers(CASES, PDF, greedy_agent, k=3)
    assert r.recall == 1.0  # what the A/B trigger rate would report
    assert r.fp == 3 and r.tn == 0
    assert r.false_trigger_rate == 1.0
    assert r.precision == 0.5
    assert r.f1 == pytest.approx(0.6667, abs=1e-4)
    assert {o.case.tid for o in r.failures()} == {"n1", "n2", "n3"}


def test_k_repeats_threshold_and_flaky_cases() -> None:
    # p1 fires 2/3 runs, n1 fires 1/3 runs, others deterministic.
    pattern = {"p1": [True, True, False], "n1": [False, True, False]}

    def agent(case, offered, run):
        if case.tid in pattern:
            return pattern[case.tid][run]
        return case.should_trigger

    r = run_triggers(CASES, PDF, agent, k=3, threshold=0.5)
    assert r.tp == 3 and r.fp == 0  # 2/3 ≥ 0.5 triggers; 1/3 does not
    assert sorted(o.case.tid for o in r.flaky) == ["n1", "p1"]
    assert r.run_level["positive_fire_rate"] == pytest.approx(8 / 9, abs=1e-4)
    assert r.run_level["negative_fire_rate"] == pytest.approx(1 / 9, abs=1e-4)
    strict = run_triggers(CASES, PDF, agent, k=3, threshold=1.0)
    assert strict.fn == 1  # p1 no longer counts at a strict threshold


def test_competing_skills_are_offered_and_names_are_resolved() -> None:
    seen = []

    def agent(case, offered, run):
        seen.append(tuple(s.name for s in offered))
        if "docx" in case.prompt:
            return "docx"  # picked the competitor — not a trigger for pdf-forms
        if case.tid == "p2":
            return {"pdf-forms", "docx"}  # both loaded counts as fired
        return case.should_trigger

    r = run_triggers(CASES, PDF, agent, k=2, competing=[DOCX])
    assert set(seen) == {("pdf-forms", "docx")}
    assert len(seen) == len(CASES) * 2
    assert r.precision == 1.0 and r.recall == 1.0 and r.false_trigger_rate == 0.0


def test_table_and_dict_list_worst_first() -> None:
    r = run_triggers(CASES, PDF, greedy_agent, k=1)
    md = r.table()
    assert md.index("`n1`") < md.index("`p1`")  # false triggers listed first
    assert "near-miss: mentions PDF" in md
    d = r.as_dict()
    assert d["counts"] == {"fn": 0, "fp": 3, "tn": 0, "tp": 3}
    assert d["skill"] == "pdf-forms" and d["k"] == 1


def test_from_observations_scores_recorded_runs() -> None:
    fired = [[True] * 3, [True, False, True], [False] * 3, [False] * 3, [True] * 3, [False] * 3]
    r = from_observations(CASES, "pdf-forms", fired)
    assert (r.tp, r.fn, r.fp, r.tn) == (2, 1, 1, 2)
    assert r.k == 3
    with pytest.raises(ValueError):
        from_observations(CASES, "pdf-forms", fired[:2])


def test_split_cases_is_stratified_and_deterministic() -> None:
    train, val = split_cases(CASES, val_fraction=0.34, seed=7)
    assert len(train) + len(val) == len(CASES)
    assert {c.tid for c in train}.isdisjoint({c.tid for c in val})
    assert any(c.should_trigger for c in val) and any(not c.should_trigger for c in val)
    assert any(c.should_trigger for c in train) and any(not c.should_trigger for c in train)
    assert split_cases(CASES, val_fraction=0.34, seed=7) == (train, val)


def test_validation() -> None:
    with pytest.raises(ValueError):
        run_triggers(CASES, PDF, greedy_agent, k=0)
    with pytest.raises(ValueError):
        run_triggers(CASES, PDF, greedy_agent, threshold=0)
    with pytest.raises(ValueError):
        run_triggers([CASES[0], CASES[0]], PDF, greedy_agent)
    with pytest.raises(ValueError):
        run_triggers(CASES, PDF, greedy_agent, competing=[PDF])
    with pytest.raises(TypeError):
        run_triggers(CASES, PDF, lambda c, o, i: 3.5)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError):
        split_cases(CASES, val_fraction=1.0)
