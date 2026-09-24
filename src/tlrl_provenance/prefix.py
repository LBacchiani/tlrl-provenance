"""Bounded-exact three-valued assessment for right-censored traces."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum

from .logic import Formula, Op, Top, Bottom, unary, binary
from .reference import evaluate_reference_dp
from .trace import LabeledTrace

FORMULA_EQUALITY_PAIR_LIMIT = 20_000
DEFAULT_RESIDUAL_NODE_LIMIT = 2_000
DEFAULT_EXACT_STATE_LIMIT = 5_000
DEFAULT_EXACT_ALPHABET_ATOM_LIMIT = 12


class PrefixStatus(str, Enum):
    DEFINITELY_SATISFIED = "definitely_satisfied"
    DEFINITELY_VIOLATED = "definitely_violated"
    OPEN = "open"


class DecisionOrigin(str, Enum):
    """How ``PrefixAssessment.status`` was reached.

    ``SYNTACTIC_COLLAPSE`` and ``EXACT_SEARCH`` are exact: ``status`` is
    provably correct because a complete case analysis was performed --
    including a proven ``OPEN``, when the search actually witnessed both a
    satisfying and a violating continuation. The remaining values mean the
    procedure was abandoned before it could prove anything: two are guard
    fallbacks (the search would have needed more states or a larger
    alphabet than the configured bound), and ``NOT_COMPUTED`` marks a
    caller-requested placeholder where no search was attempted at all. In
    every one of these incomplete cases ``status`` is unconditionally
    ``OPEN`` -- not because openness was established, but because the
    procedure fails closed rather than guessing. Callers that must not
    conflate "provably open" with "unresolved" should branch on
    :attr:`PrefixAssessment.decision_complete`, not on ``status`` alone.
    """

    SYNTACTIC_COLLAPSE = "syntactic_collapse"
    EXACT_SEARCH = "exact_search"
    RESIDUAL_SIZE_GUARD = "residual_size_guard"
    STATE_LIMIT_GUARD = "state_limit_guard"
    ALPHABET_LIMIT_GUARD = "alphabet_limit_guard"
    NOT_COMPUTED = "not_computed"


_COMPLETE_ORIGINS = frozenset(
    {
        DecisionOrigin.SYNTACTIC_COLLAPSE,
        DecisionOrigin.EXACT_SEARCH,
    }
)


@dataclass(frozen=True, slots=True)
class PrefixAssessment:
    status: PrefixStatus
    finite_trace_verdict: bool
    continuation_residual: Formula
    decision_origin: DecisionOrigin
    states_explored: int = 0
    alphabet_size: int = 0

    @property
    def decision_complete(self) -> bool:
        """False iff ``status`` was not proved -- a guard fired, or the
        search was never run at all (:attr:`DecisionOrigin.NOT_COMPUTED`).

        When this is False, ``status`` is always ``OPEN`` by construction,
        but that ``OPEN`` is not a semantic finding -- it means the exact
        procedure was abandoned or skipped, not that every continuation was
        checked. Treat it as unresolved, not as a decided verdict.
        """

        return self.decision_origin in _COMPLETE_ORIGINS


def assess_prefix(
    formula: Formula,
    trace: LabeledTrace,
    *,
    residual_node_limit: int = DEFAULT_RESIDUAL_NODE_LIMIT,
    exact_state_limit: int = DEFAULT_EXACT_STATE_LIMIT,
    exact_alphabet_atom_limit: int = DEFAULT_EXACT_ALPHABET_ATOM_LIMIT,
) -> PrefixAssessment:
    """Classify a prefix without treating truncation as an ordinary endpoint.

    A trace is DEFINITELY_SATISFIED (respectively DEFINITELY_VIOLATED) iff
    the current finite verdict and every nonempty finite continuation agree
    on that value; otherwise it is OPEN. A constant (TOP/BOTTOM) residual is
    a sound but incomplete syntactic certificate for this: it is checked
    first as a fast path, but a non-constant residual does not by itself
    mean a continuation exists that flips the verdict -- a formula such as
    ``G(X p)`` is unsatisfiable on every nonempty finite trace (the strong
    next is false at any trace's own final position) without its residual
    ever syntactically collapsing. When the fast path is inconclusive,
    ``_exact_definiteness`` decides the question exactly by exploring the
    finite set of formulas reachable from the residual by progression.
    """

    if residual_node_limit < 1:
        raise ValueError("residual_node_limit must be positive")
    finite = evaluate_reference_dp(formula, trace.labels)
    residual = formula
    for labels in trace.labels:
        residual = progress_open(residual, frozenset(labels))
        if _exceeds_formula_node_limit(residual, residual_node_limit):
            return PrefixAssessment(
                PrefixStatus.OPEN,
                finite,
                residual,
                decision_origin=DecisionOrigin.RESIDUAL_SIZE_GUARD,
            )
    if finite and residual.op is Op.TOP:
        return PrefixAssessment(
            PrefixStatus.DEFINITELY_SATISFIED,
            finite,
            residual,
            decision_origin=DecisionOrigin.SYNTACTIC_COLLAPSE,
        )
    if not finite and residual.op is Op.BOTTOM:
        return PrefixAssessment(
            PrefixStatus.DEFINITELY_VIOLATED,
            finite,
            residual,
            decision_origin=DecisionOrigin.SYNTACTIC_COLLAPSE,
        )
    status, origin, states_explored, alphabet_size = _exact_definiteness(
        finite,
        residual,
        state_limit=exact_state_limit,
        alphabet_atom_limit=exact_alphabet_atom_limit,
    )
    return PrefixAssessment(
        status,
        finite,
        residual,
        decision_origin=origin,
        states_explored=states_explored,
        alphabet_size=alphabet_size,
    )


def _exact_definiteness(
    finite: bool,
    residual: Formula,
    *,
    state_limit: int,
    alphabet_atom_limit: int,
) -> tuple[PrefixStatus, DecisionOrigin, int, int]:
    """Exact continuation-quantified status when the syntactic fast path in
    :func:`assess_prefix` is inconclusive.

    A trace is DEFINITELY_SATISFIED iff the current verdict and every
    nonempty finite continuation of the residual evaluate to True;
    DEFINITELY_VIOLATED iff all evaluate to False; OPEN otherwise. The
    search space is the set of formulas reachable from ``residual`` by
    :func:`progress_open`, which is finite because progression only
    recombines subformulas of the original formula. At each reachable state
    we ask, for every letter of the residual's own atom alphabet, whether
    stopping the continuation exactly one step further is decided True or
    False by :func:`~tlrl_provenance.reference.evaluate_reference_dp`
    (equivalently, by the progression-correctness property, whether the
    original formula holds on the trace observed so far extended by that
    one more step).

    Returns ``(status, origin, states_explored, alphabet_size)``. ``origin``
    is :attr:`DecisionOrigin.EXACT_SEARCH` whenever the search itself proved
    ``status`` -- including a proven OPEN, when both a true- and a
    false-stopping continuation were actually witnessed. If instead
    exploration is abandoned before proof -- more than ``state_limit``
    distinct reachable states, or a residual referencing more than
    ``alphabet_atom_limit`` atoms (per-state work scales with that
    alphabet's powerset, independent of the state count) -- ``origin`` is
    one of the guard values and ``status`` is unconditionally OPEN, but that
    OPEN is a fail-closed default, not a proof.
    """

    atoms = residual.atoms()
    if len(atoms) > alphabet_atom_limit:
        return PrefixStatus.OPEN, DecisionOrigin.ALPHABET_LIMIT_GUARD, 0, 2 ** len(atoms)
    alphabet = _powerset(atoms)
    alphabet_size = len(alphabet)
    seen_true = finite
    seen_false = not finite
    visited: set[str] = {repr(residual)}
    frontier = [residual]
    explored = 0
    while frontier:
        if seen_true and seen_false:
            return PrefixStatus.OPEN, DecisionOrigin.EXACT_SEARCH, explored, alphabet_size
        psi = frontier.pop()
        explored += 1
        if explored > state_limit:
            return PrefixStatus.OPEN, DecisionOrigin.STATE_LIMIT_GUARD, explored, alphabet_size
        for label in alphabet:
            if evaluate_reference_dp(psi, [label]):
                seen_true = True
            else:
                seen_false = True
            if seen_true and seen_false:
                return PrefixStatus.OPEN, DecisionOrigin.EXACT_SEARCH, explored, alphabet_size
            successor = progress_open(psi, label)
            key = repr(successor)
            if key not in visited:
                visited.add(key)
                frontier.append(successor)
    final_status = PrefixStatus.DEFINITELY_SATISFIED if seen_true else PrefixStatus.DEFINITELY_VIOLATED
    return final_status, DecisionOrigin.EXACT_SEARCH, explored, alphabet_size


def _powerset(atoms: frozenset[str]) -> tuple[frozenset[str], ...]:
    items = sorted(atoms)
    subsets = tuple(
        frozenset(combo)
        for r in range(len(items) + 1)
        for combo in itertools.combinations(items, r)
    )
    return subsets or (frozenset(),)


def progress_open(formula: Formula, labels: frozenset[str]) -> Formula:
    """Progress one position assuming a proper continuation remains possible."""

    op = formula.op
    if op is Op.ATOM:
        assert formula.name is not None
        return Top() if formula.name in labels else Bottom()
    if op in {Op.TOP, Op.BOTTOM}:
        return formula
    if op is Op.NOT:
        return _simplify(unary(Op.NOT, progress_open(formula.args[0], labels)))
    if op in {Op.AND, Op.OR, Op.IMPLIES, Op.IFF}:
        return _simplify(
            binary(
                op,
                progress_open(formula.args[0], labels),
                progress_open(formula.args[1], labels),
            )
        )
    if op in {Op.NEXT, Op.WEAK_NEXT}:
        return formula.args[0]
    if op is Op.EVENTUALLY:
        return _simplify(
            binary(Op.OR, progress_open(formula.args[0], labels), formula)
        )
    if op is Op.GLOBALLY:
        return _simplify(
            binary(Op.AND, progress_open(formula.args[0], labels), formula)
        )
    if op in {Op.UNTIL, Op.WEAK_UNTIL}:
        left, right = formula.args
        return _simplify(
            binary(
                Op.OR,
                progress_open(right, labels),
                binary(Op.AND, progress_open(left, labels), formula),
            )
        )
    if op is Op.RELEASE:
        left, right = formula.args
        return _simplify(
            binary(
                Op.AND,
                progress_open(right, labels),
                binary(Op.OR, progress_open(left, labels), formula),
            )
        )
    raise AssertionError(op)


def _simplify(formula: Formula) -> Formula:
    op = formula.op
    if op is Op.NOT:
        child = _simplify(formula.args[0])
        if child.op is Op.TOP:
            return Bottom()
        if child.op is Op.BOTTOM:
            return Top()
        if child.op is Op.NOT:
            return _simplify(child.args[0])
        return unary(Op.NOT, child)
    if len(formula.args) == 2:
        left, right = (_simplify(item) for item in formula.args)
        if op is Op.AND:
            if Op.BOTTOM in {left.op, right.op}:
                return Bottom()
            if left.op is Op.TOP:
                return right
            if right.op is Op.TOP:
                return left
            if _same_formula_bounded(left, right):
                return left
            if _contains_same_binary_operand(right, op, left):
                return right
            if _contains_same_binary_operand(left, op, right):
                return left
        elif op is Op.OR:
            if Op.TOP in {left.op, right.op}:
                return Top()
            if left.op is Op.BOTTOM:
                return right
            if right.op is Op.BOTTOM:
                return left
            if _same_formula_bounded(left, right):
                return left
            if _contains_same_binary_operand(right, op, left):
                return right
            if _contains_same_binary_operand(left, op, right):
                return left
        elif op is Op.IMPLIES:
            return _simplify(binary(Op.OR, unary(Op.NOT, left), right))
        elif op is Op.IFF:
            if _same_formula_bounded(left, right):
                return Top()
            if left.op is Op.TOP:
                return right
            if right.op is Op.TOP:
                return left
            if left.op is Op.BOTTOM:
                return _simplify(unary(Op.NOT, right))
            if right.op is Op.BOTTOM:
                return _simplify(unary(Op.NOT, left))
        return binary(op, left, right)
    return formula


def _same_formula_bounded(
    left: Formula,
    right: Formula,
    *,
    pair_limit: int = FORMULA_EQUALITY_PAIR_LIMIT,
) -> bool:
    """Return structural equality without risking recursive stack overflow.

    Equality-based simplifications such as ``phi or phi -> phi`` are optional:
    skipping them preserves soundness, but performing them can keep progressed
    residuals small.  Long prefixes can create very deep residual formulas, so
    Python's dataclass-generated recursive equality is unsafe here.  This
    bounded iterative check returns ``False`` if the comparison itself becomes
    too large, meaning the simplifier simply declines the optional rewrite.
    """

    stack = [(left, right)]
    seen: set[tuple[int, int]] = set()
    checked = 0
    while stack:
        first, second = stack.pop()
        if first is second:
            continue
        key = (id(first), id(second))
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        if checked > pair_limit:
            return False
        if first.op is not second.op or first.name != second.name or len(first.args) != len(second.args):
            return False
        stack.extend(zip(first.args, second.args))
    return True


def _contains_same_binary_operand(formula: Formula, op: Op, candidate: Formula) -> bool:
    if formula.op is not op or len(formula.args) != 2:
        return False
    return any(_same_formula_bounded(child, candidate) for child in formula.args)


def _exceeds_formula_node_limit(formula: Formula, limit: int) -> bool:
    stack = [formula]
    seen: set[int] = set()
    count = 0
    while stack:
        node = stack.pop()
        identity = id(node)
        if identity in seen:
            continue
        seen.add(identity)
        count += 1
        if count > limit:
            return True
        stack.extend(node.args)
    return False
