from skill_eval import Call, Case, ToolSpec, score


TOOLS = [
    ToolSpec("search", required=("q",), optional=("k",)),
    ToolSpec("calendar", required=("date",)),
]


def test_selection_schema_refusal() -> None:
    cases = [
        Case("c1", "find cafes", "call", tool="search", required_args={"q": "cafes"}),
        Case("c2", "what is the meaning of life", "refuse"),
        Case("c3", "book Tuesday", "call", tool="calendar", required_args={"date": "tue"}),
    ]
    calls = [
        Call("search", {"q": "cafes"}),
        Call(None, {}, refused=True),
        Call("calendar", {"date": "tue"}),
    ]
    s = score(TOOLS, cases, calls)
    assert s.selection == 1.0
    assert s.schema == 1.0
    assert s.refusal == 1.0


def test_wrong_tool() -> None:
    cases = [Case("c1", "find cafes", "call", tool="search")]
    calls = [Call("calendar", {"date": "tue"})]
    s = score(TOOLS, cases, calls)
    assert s.selection == 0.0
