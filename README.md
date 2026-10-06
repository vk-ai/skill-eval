# skill-eval

Score **tool selection**, **argument schema**, and **refusal**. No agent framework.

**Live:** [github.com/vk-ai/skill-eval](https://github.com/vk-ai/skill-eval)

![What the scorecard looks like](docs/looks.svg)

## Why this exists

Most evals score the final sentence. Tool-calling fails earlier: wrong tool, extra keys, or answering when it should refuse. skill-eval is that scorecard.

| | skill-eval | Typical stacks |
|---|---|---|
| Unit | one tool call | whole transcript |
| Schema | required / optional keys | JSON-schema engines |
| Refusal | first-class | afterthought |
| Framework | none | LangChain / OpenAI tools |

Design: frozen `ToolSpec` / `Case` / `Call` (value objects), `score()` as a pure function — no I/O, easy to property-test.

## Install — new project

![Install](docs/install.svg)

```bash
git clone https://github.com/vk-ai/skill-eval.git
cd skill-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/quickstart.py
```

## Install — existing project

```bash
pip install skill-eval
```

```python
from skill_eval import Call, Case, ToolSpec, score

print(score(
    [ToolSpec("search", required=("q",))],
    [Case("1", "find trains", "call", tool="search"), Case("2", "write a poem", "refuse")],
    [Call("search", {"q": "trains"}), Call(None, {}, refused=True)],
).as_dict())
```

## With vs without skill (A/B)

Authors ask: *does offering this skill actually help, and did the agent load it?*
Run the same cases twice and measure **lift** + **trigger rate**:

```python
from skill_eval import Call, Case, SkillSpec, ToolSpec, compare_skill, run_ab

tools = [ToolSpec("search", required=("q",))]
cases = [
    Case("1", "find trains", "call", tool="search", required_args={"q": "trains"}),
    Case("2", "write a poem", "refuse"),
]
skill = SkillSpec("web-search", text="For lookup questions, call search with q=...")

def agent(case, offered):
    # Wire to your model/CLI. offered is SkillSpec or None.
    if offered is None:
        return Call(None, {}, refused=case.expect == "refuse")
    if case.expect == "refuse":
        return Call(None, {}, refused=True)
    return Call("search", {"q": "trains"})

report = run_ab(tools, cases, skill, agent)
print(report.lift)                 # selection / schema / refusal deltas
print(report.with_skill.trigger_rate)
```

Or score pre-recorded call lists with `compare_skill(..., calls_with=..., calls_without=..., triggered=...)`.
Still zero dependencies — your agent callable owns any CLI/network I/O.

## Trigger precision / recall (near-miss negatives, k runs)

`trigger_rate` above only counts cases that *should* use the skill, so it
measures recall. A description that fires on everything gets 100% there.
`run_triggers` adds **should-not-trigger near-misses**, **k repeated runs**
per case and optional **competing skills**:

```python
from skill_eval import SkillSpec, TriggerCase, run_triggers, split_cases

skill = SkillSpec("pdf-forms", text="Fill or read form fields in PDF files.")
cases = [
    TriggerCase("fill", "fill the form fields in invoice.pdf", True),
    TriggerCase("convert", "convert report.docx to PDF", False, note="near-miss"),
    TriggerCase("summary", "summarize this PDF article", False, note="near-miss"),
]

def agent(case, offered, run):
    # Call your model/CLI with `offered` skills; return the loaded skill name(s),
    # None, or a bool meaning "this skill loaded".
    ...

report = run_triggers(cases, skill, agent, k=3, competing=[SkillSpec("docx")])
print(report.precision, report.recall, report.false_trigger_rate, report.f1)
print(report.table())             # per case: fired k/k, TP/FP/FN/TN, flaky
train, val = split_cases(cases)   # tune on train, report on val
```

- A case counts as triggered when it fired on at least `threshold` (default 0.5)
  of its `k` runs. `report.run_level` gives the raw per-run fire rates, and
  `report.flaky` lists the cases that fired on only some runs.
- Already have logs? Use `from_observations(cases, "pdf-forms", fired=[[True, False, True], ...])`.
- Undefined ratios (for example precision when nothing fired) are `0.0`.
  The counts are always in `as_dict()["counts"]`.

See [`examples/trigger_eval.py`](examples/trigger_eval.py), which compares a
vague and a specific description.

MIT. Python 3.10+.
