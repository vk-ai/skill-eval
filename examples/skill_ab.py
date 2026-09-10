"""Paired with/without-skill eval — measure lift and trigger rate."""

from skill_eval import Call, Case, SkillSpec, ToolSpec, run_ab

tools = [ToolSpec("search", required=("q",))]
cases = [
    Case("1", "find trains to Madrid", "call", tool="search", required_args={"q": "trains Madrid"}),
    Case("2", "write a poem about rain", "refuse"),
]
skill = SkillSpec(
    "web-search",
    text="For factual lookup questions, call search with q set to the query.",
)


def agent(case: Case, offered: SkillSpec | None) -> Call:
    # Stand-in for a real model/CLI. Without the skill, selection/refusal degrade.
    if offered is None:
        if case.expect == "refuse":
            return Call("search", {"q": "poem"})
        return Call(None, {}, refused=True)
    if case.expect == "refuse":
        return Call(None, {}, refused=True)
    return Call("search", {"q": "trains Madrid"})


if __name__ == "__main__":
    report = run_ab(tools, cases, skill, agent)
    print(report.as_dict())
