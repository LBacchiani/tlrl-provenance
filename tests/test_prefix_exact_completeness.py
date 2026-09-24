"""Regression tests for the exact continuation-aware status procedure.

These pin down the completeness defect found and fixed in ``prefix.py``:
the syntactic residual-collapse fast path is sound but not complete for
formulas mixing G/R/W with X/Xw, because a finite trace's own strong-next
boundary condition (X is false, Xw is true, at any trace's final position)
is invisible to progression alone. ``_exact_definiteness`` closes that gap
with an exact search, bounded by explicit guards that fail closed -- marked
via ``PrefixAssessment.decision_origin``/``decision_complete`` -- rather
than silently reusing OPEN for both "proven open" and "gave up."
"""

import hashlib
import itertools
import random

import pytest

from tlrl_provenance import DecisionOrigin, LabeledTrace, PrefixStatus, assess_prefix, parse_formula
from tlrl_provenance.prefix import progress_open
from tlrl_provenance.reference import evaluate_reference_dp


def test_strong_next_top_is_open_not_satisfied():
    """X top on a one-state trace: the empty continuation (b_tau=False,
    since X is false at any trace's final position) and every nonempty
    continuation (X top holds one position later) disagree, so the status
    is OPEN -- not DEFINITELY_SATISFIED, even though every *proper*
    continuation alone would satisfy it."""

    result = assess_prefix(parse_formula("X top"), LabeledTrace([set()], terminated=False, truncated=True))
    assert result.status is PrefixStatus.OPEN
    assert result.finite_trace_verdict is False
    assert result.decision_origin is DecisionOrigin.EXACT_SEARCH
    assert result.decision_complete is True


def test_globally_next_violation_is_unsatisfiable_on_every_finite_trace():
    """G(X(!m)) is definitely violated on every nonempty finite trace,
    independent of whether m ever appears: G forces X(!m) to hold at the
    trace's own final position too, where strong X is always false. The
    syntactic fast path cannot see this (the residual never collapses to
    BOTTOM while m stays absent); the exact search must."""

    for labels in ([set()], [set(), {"b"}], [{"m"}], [set(), {"m"}, set()]):
        result = assess_prefix(parse_formula("G(X(!m))"), LabeledTrace(labels, terminated=False, truncated=True))
        assert result.status is PrefixStatus.DEFINITELY_VIOLATED
        assert result.decision_complete is True


def test_globally_weak_next_is_genuinely_open_while_undecided():
    """G(Xw(!m)) has no such forced violation (weak X is true at a trace's
    final position), so while m has not appeared it is genuinely OPEN: a
    continuation where m eventually appears would violate it, and one where
    m never appears would not. Both are real possibilities, not a guard
    fallback."""

    result = assess_prefix(parse_formula("G(Xw(!m))"), LabeledTrace([set(), {"b"}], terminated=False, truncated=True))
    assert result.status is PrefixStatus.OPEN
    assert result.decision_origin is DecisionOrigin.EXACT_SEARCH
    assert result.decision_complete is True

    # Once m actually appears, it becomes decided (a genuine violation).
    violated = assess_prefix(parse_formula("G(Xw(!m))"), LabeledTrace([set(), {"m"}], terminated=False, truncated=True))
    assert violated.status is PrefixStatus.DEFINITELY_VIOLATED
    assert violated.decision_complete is True


def test_alphabet_limit_guard_fails_closed_and_is_marked_incomplete():
    """A residual referencing more atoms than the configured alphabet limit
    is never explored exactly (its powerset would be exponential); status
    is OPEN but decision_complete is False, distinguishing it from a proven
    OPEN. A 13-atom G(X(...)) is semantically DEFINITELY_VIOLATED by the
    same boundary rule as the 1-atom case above -- the guard must not be
    mistaken for that answer."""

    atoms = [f"p{i}" for i in range(13)]
    formula_text = "G (X (" + " & ".join(f"! {a}" for a in atoms) + "))"
    result = assess_prefix(
        parse_formula(formula_text),
        LabeledTrace([frozenset()], terminated=False, truncated=True),
        exact_alphabet_atom_limit=12,
    )
    assert result.status is PrefixStatus.OPEN
    assert result.decision_origin is DecisionOrigin.ALPHABET_LIMIT_GUARD
    assert result.decision_complete is False
    assert result.alphabet_size == 2**13

    # The same formula, with the guard raised past 13 atoms, is decided.
    unblocked = assess_prefix(
        parse_formula(formula_text),
        LabeledTrace([frozenset()], terminated=False, truncated=True),
        exact_alphabet_atom_limit=13,
    )
    assert unblocked.status is PrefixStatus.DEFINITELY_VIOLATED
    assert unblocked.decision_complete is True


def test_state_limit_guard_fails_closed_and_is_marked_incomplete():
    """An artificially tiny state_limit forces the guard to fire even on a
    formula the search would otherwise resolve quickly; the result must be
    marked incomplete, not silently reported as a proven OPEN."""

    result = assess_prefix(
        parse_formula("G(X(!m))"),
        LabeledTrace([set()], terminated=False, truncated=True),
        exact_state_limit=0,
    )
    assert result.status is PrefixStatus.OPEN
    assert result.decision_origin is DecisionOrigin.STATE_LIMIT_GUARD
    assert result.decision_complete is False


def test_residual_size_guard_is_still_distinguished_from_the_exact_fallback():
    """The pre-existing residual-node-size guard (during progression itself,
    not the exact search) must also report itself as incomplete, not as a
    proven OPEN, now that the two are distinguishable."""

    formula = parse_formula("(!magenta U (blue & (!green U yellow)))")
    trace = LabeledTrace([{"blue"}] * 50, terminated=False, truncated=True)
    result = assess_prefix(formula, trace, residual_node_limit=1)
    assert result.status is PrefixStatus.OPEN
    assert result.decision_origin is DecisionOrigin.RESIDUAL_SIZE_GUARD
    assert result.decision_complete is False


def test_fast_path_syntactic_collapse_is_still_marked_complete():
    """Ordinary decided F/U cases still resolve via the cheap syntactic
    fast path, not the exact search, and are marked complete."""

    safety = assess_prefix(parse_formula("G(not a)"), LabeledTrace([set(), {"a"}], terminated=False, truncated=True))
    reach = assess_prefix(parse_formula("F a"), LabeledTrace([set(), {"a"}], terminated=False, truncated=True))
    assert safety.status is PrefixStatus.DEFINITELY_VIOLATED
    assert safety.decision_origin is DecisionOrigin.SYNTACTIC_COLLAPSE
    assert safety.decision_complete is True
    assert reach.status is PrefixStatus.DEFINITELY_SATISFIED
    assert reach.decision_origin is DecisionOrigin.SYNTACTIC_COLLAPSE
    assert reach.decision_complete is True


# ---------------------------------------------------------------------------
# Independent-oracle agreement property test.
#
# This is a compact, CI-fast version of the verification performed once
# during development against all three confirmatory corpora (149 formulas,
# thousands of trials; see scripts/verify_prefix_exact_completeness.py for
# the full, retained sweep). It reimplements the continuation-quantified
# definition from first principles -- via brute-force enumeration of a
# small alphabet's powerset, not by calling anything in prefix.py's own
# search -- so it cannot pass merely because both sides share a bug.
# ---------------------------------------------------------------------------


def _brute_force_status(formula, prefix_labels, atoms, max_ext=3):
    alphabet = [frozenset(c) for r in range(len(atoms) + 1) for c in itertools.combinations(atoms, r)]
    results = set()
    for ext_len in range(0, max_ext + 1):
        for combo in itertools.product(alphabet, repeat=ext_len):
            full = list(prefix_labels) + list(combo)
            results.add(evaluate_reference_dp(formula, full))
            if len(results) == 2:
                return "open"
    return "definitely_satisfied" if results == {True} else "definitely_violated"


_ORACLE_FORMULAS = [
    "X top",
    "Xw top",
    "X bottom",
    "Xw bottom",
    "G (X (! a))",
    "G (Xw (! a))",
    "(! a) R b",
    "(! a) W b",
    "F a",
    "a U b",
    "F (a & F b)",
    "(!a U (b & (!c U a)))",
    "G (a -> X b)",
]
_ORACLE_ATOMS = ["a", "b", "c"]


@pytest.mark.parametrize("formula_text", _ORACLE_FORMULAS)
def test_agrees_with_independent_brute_force_oracle(formula_text):
    stable_seed = int.from_bytes(
        hashlib.sha256(formula_text.encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(stable_seed)
    formula = parse_formula(formula_text)
    alphabet = [frozenset(), *[frozenset({a}) for a in _ORACLE_ATOMS]]
    for trace_len in (1, 2, 3):
        for _ in range(5):
            labels = [rng.choice(alphabet) for _ in range(trace_len)]
            trace = LabeledTrace(labels, terminated=False, truncated=True)
            impl = assess_prefix(formula, trace).status.value
            oracle = _brute_force_status(formula, labels, _ORACLE_ATOMS)
            assert impl == oracle, (formula_text, labels, impl, oracle)
