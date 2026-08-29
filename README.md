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

MIT. Python 3.10+.
