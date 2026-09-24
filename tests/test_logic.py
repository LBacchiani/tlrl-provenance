import pytest

from tlrl_provenance.logic import IndexedFormula, Op, parse_formula


@pytest.mark.parametrize(
    "source",
    [
        "p",
        "not p",
        "X p",
        "Xw p",
        "F(p or q)",
        "G(p -> F q)",
        "p U q",
        "p R q",
        "p W q",
        "(p <-> q) and (r or not s)",
    ],
)
def test_parser_round_trip(source):
    formula = parse_formula(source)
    assert parse_formula(formula.to_source()) == formula


def test_repeated_subformula_occurrences_remain_distinct():
    indexed = IndexedFormula.build(parse_formula("p or p"))
    atoms = [item for item in indexed.occurrences if item.formula.op is Op.ATOM]
    assert [item.id for item in atoms] == ["o0", "o1"]
    assert atoms[0].formula == atoms[1].formula
    assert atoms[0].path != atoms[1].path


@pytest.mark.parametrize("source", ["", "p ??? q", "(p", "p q"])
def test_parser_rejects_malformed_input(source):
    with pytest.raises(SyntaxError):
        parse_formula(source)


def test_parser_has_an_explicit_complexity_limit():
    with pytest.raises(SyntaxError, match="token limit"):
        parse_formula("not " * 600 + "p")
