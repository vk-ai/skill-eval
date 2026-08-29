from skill_eval import Call, Case, ToolSpec, score

tools = [ToolSpec("search", required=("q",))]
cases = [
    Case("1", "find trains to Madrid", "call", tool="search", required_args={"q": "trains Madrid"}),
    Case("2", "write a poem about rain", "refuse"),
]
# Wire these to your model. Here we fake the policy.
calls = [
    Call("search", {"q": "trains Madrid"}),
    Call(None, {}, refused=True),
]
print(score(tools, cases, calls).as_dict())
