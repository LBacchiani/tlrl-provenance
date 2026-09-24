"""Narrow, optional adapters for the v1/v2 trace and formula containers."""

from __future__ import annotations

from .logic import Formula, Op
from .trace import LabeledTrace


_LEGACY_OPS = {
    "atom": Op.ATOM,
    "top": Op.TOP,
    "bottom": Op.BOTTOM,
    "not": Op.NOT,
    "and": Op.AND,
    "or": Op.OR,
    "implies": Op.IMPLIES,
    "next": Op.NEXT,
    "eventually": Op.EVENTUALLY,
    "globally": Op.GLOBALLY,
    "until": Op.UNTIL,
}


def from_legacy_formula(formula: object) -> Formula:
    """Convert the immutable v1 `tlrl.logic.Formula` without importing it."""

    op_name = getattr(formula, "op", None)
    if op_name not in _LEGACY_OPS:
        raise ValueError(f"unsupported legacy formula operator: {op_name!r}")
    args = tuple(from_legacy_formula(item) for item in getattr(formula, "args", ()))
    name = getattr(formula, "name", None)
    return Formula(_LEGACY_OPS[op_name], args, name)


def from_v2_episode(trace: object) -> LabeledTrace:
    """Copy the observable v2 episode fields into the standalone v3 schema."""

    steps = tuple(getattr(trace, "steps"))
    metadata = dict(getattr(trace, "metadata", {}))
    return LabeledTrace(
        labels=tuple(getattr(step, "labels") for step in steps),
        states=tuple(getattr(step, "state", None) for step in steps),
        observations=tuple(getattr(step, "observation", None) for step in steps),
        actions=tuple(getattr(trace, "actions", ())),
        rewards=tuple(getattr(trace, "rewards", ())),
        policy_id=getattr(trace, "policy_id", None),
        environment_id=getattr(trace, "environment_id", None),
        seed=getattr(trace, "seed", None),
        terminated=metadata.get("terminated", not metadata.get("truncated", False)) is True,
        truncated=metadata.get("truncated", False) is True,
        metadata=metadata,
    )

