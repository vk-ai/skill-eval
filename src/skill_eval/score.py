from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .cases import Call, Case, ToolSpec


@dataclass
class Scorecard:
    n: int
    selection: float
    schema: float
    refusal: float

    def as_dict(self) -> dict[str, float]:
        return {"n": float(self.n), "selection": self.selection, "schema": self.schema, "refusal": self.refusal}


def score(tools: Sequence[ToolSpec], cases: Sequence[Case], calls: Sequence[Call]) -> Scorecard:
    if len(cases) != len(calls):
        raise ValueError("cases and calls must align")
    by_name = {t.name: t for t in tools}
    sel = sch = ref = 0
    n_sel = n_sch = n_ref = 0
    for case, call in zip(cases, calls, strict=True):
        if case.expect == "refuse":
            n_ref += 1
            if call.refused or call.tool is None:
                ref += 1
            continue
        n_sel += 1
        if call.tool == case.tool and not call.refused:
            sel += 1
        spec = by_name.get(call.tool or "")
        n_sch += 1
        ok_schema = bool(spec) and spec.allows(call.args)
        if case.required_args:
            ok_schema = ok_schema and all(call.args.get(k) == v for k, v in case.required_args.items())
        if ok_schema:
            sch += 1
    def ratio(a: int, b: int) -> float:
        return round(a / b, 4) if b else 1.0
    return Scorecard(n=len(cases), selection=ratio(sel, n_sel), schema=ratio(sch, n_sch), refusal=ratio(ref, n_ref))
