"""Simple independent recursive LTLf semantics used as a verification oracle."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, AbstractSet

from .logic import Formula, Op


def evaluate_reference_dp(
    formula: Formula,
    labels: Sequence[AbstractSet[str]],
    time: int = 0,
) -> bool:
    """Linear-time independent truth-only evaluator for production cross-checks."""

    return reference_value_table(formula, labels)[formula][time]


def reference_value_table(
    formula: Formula,
    labels: Sequence[AbstractSet[str]],
) -> dict[Formula, tuple[bool, ...]]:
    if not labels:
        raise ValueError("a finite trace requires at least one position")
    horizon = len(labels) - 1
    table: dict[Formula, tuple[bool, ...]] = {}

    def compute(node: Formula) -> tuple[bool, ...]:
        if node in table:
            return table[node]
        children = [compute(child) for child in node.args]
        op = node.op
        if op is Op.ATOM:
            assert node.name is not None
            values = tuple(node.name in step for step in labels)
        elif op is Op.TOP:
            values = (True,) * len(labels)
        elif op is Op.BOTTOM:
            values = (False,) * len(labels)
        elif op is Op.NOT:
            values = tuple(not value for value in children[0])
        elif op in {Op.AND, Op.OR, Op.IMPLIES, Op.IFF}:
            left, right = children
            operation = {
                Op.AND: lambda a, b: a and b,
                Op.OR: lambda a, b: a or b,
                Op.IMPLIES: lambda a, b: (not a) or b,
                Op.IFF: lambda a, b: a == b,
            }[op]
            values = tuple(operation(a, b) for a, b in zip(left, right))
        elif op in {Op.NEXT, Op.WEAK_NEXT}:
            boundary = op is Op.WEAK_NEXT
            values = children[0][1:] + (boundary,)
        elif op in {Op.EVENTUALLY, Op.GLOBALLY}:
            values_list = [False] * len(labels)
            values_list[horizon] = children[0][horizon]
            for t in range(horizon - 1, -1, -1):
                values_list[t] = (
                    children[0][t] or values_list[t + 1]
                    if op is Op.EVENTUALLY
                    else children[0][t] and values_list[t + 1]
                )
            values = tuple(values_list)
        elif op in {Op.UNTIL, Op.WEAK_UNTIL, Op.RELEASE}:
            left, right = children
            values_list = [False] * len(labels)
            values_list[horizon] = (
                right[horizon] or left[horizon]
                if op is Op.WEAK_UNTIL
                else right[horizon]
            )
            for t in range(horizon - 1, -1, -1):
                if op in {Op.UNTIL, Op.WEAK_UNTIL}:
                    values_list[t] = right[t] or (left[t] and values_list[t + 1])
                else:
                    values_list[t] = right[t] and (left[t] or values_list[t + 1])
            values = tuple(values_list)
        else:
            raise AssertionError(op)
        table[node] = values
        return values

    compute(formula)
    return table


def evaluate_reference(
    formula: Formula,
    labels: Sequence[AbstractSet[str]],
    time: int = 0,
) -> bool:
    if not labels:
        raise ValueError("a finite trace requires at least one position")
    if time < 0 or time >= len(labels):
        raise IndexError(time)
    horizon = len(labels) - 1

    @lru_cache(maxsize=None)
    def value(node: Formula, t: int) -> bool:
        op = node.op
        if op is Op.ATOM:
            assert node.name is not None
            return node.name in labels[t]
        if op is Op.TOP:
            return True
        if op is Op.BOTTOM:
            return False
        if op is Op.NOT:
            return not value(node.args[0], t)
        if op is Op.AND:
            return value(node.args[0], t) and value(node.args[1], t)
        if op is Op.OR:
            return value(node.args[0], t) or value(node.args[1], t)
        if op is Op.IMPLIES:
            return (not value(node.args[0], t)) or value(node.args[1], t)
        if op is Op.IFF:
            return value(node.args[0], t) == value(node.args[1], t)
        if op is Op.NEXT:
            return t < horizon and value(node.args[0], t + 1)
        if op is Op.WEAK_NEXT:
            return t == horizon or value(node.args[0], t + 1)
        if op is Op.EVENTUALLY:
            return any(value(node.args[0], j) for j in range(t, horizon + 1))
        if op is Op.GLOBALLY:
            return all(value(node.args[0], j) for j in range(t, horizon + 1))
        if op is Op.UNTIL:
            left, right = node.args
            return any(
                value(right, j) and all(value(left, k) for k in range(t, j))
                for j in range(t, horizon + 1)
            )
        if op is Op.RELEASE:
            left, right = node.args
            return all(
                value(right, j) or any(value(left, k) for k in range(t, j))
                for j in range(t, horizon + 1)
            )
        if op is Op.WEAK_UNTIL:
            left, right = node.args
            return value(Formula(Op.UNTIL, (left, right)), t) or all(
                value(left, j) for j in range(t, horizon + 1)
            )
        raise AssertionError(op)

    return value(formula, time)
