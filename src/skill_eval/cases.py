from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Outcome = Literal["call", "refuse"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    def allows(self, args: dict[str, Any]) -> bool:
        keys = set(args)
        if not set(self.required).issubset(keys):
            return False
        allowed = set(self.required) | set(self.optional)
        return keys.issubset(allowed)


@dataclass(frozen=True)
class Call:
    tool: str | None
    args: dict[str, Any]
    refused: bool = False


@dataclass(frozen=True)
class Case:
    cid: str
    prompt: str
    expect: Outcome
    tool: str | None = None
    required_args: dict[str, Any] | None = None
