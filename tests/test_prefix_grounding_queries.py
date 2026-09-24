import pytest

from tlrl_provenance import (
    GroundingRecord,
    LabeledTrace,
    PrefixStatus,
    assess_prefix,
    evaluate_with_provenance,
    parse_formula,
)
from tlrl_provenance.queries import Polarity, occurrence_mutation_query, occurrence_polarities


def test_prefix_boundary_semantics_do_not_overclaim_next():
    strong = assess_prefix(parse_formula("X top"), LabeledTrace([set()], terminated=False, truncated=True))
    weak = assess_prefix(parse_formula("Xw bottom"), LabeledTrace([set()], terminated=False, truncated=True))
    assert strong.status is PrefixStatus.OPEN
    assert weak.status is PrefixStatus.OPEN


def test_prefix_can_decide_safety_violation_and_eventuality_success():
    safety = assess_prefix(parse_formula("G(not a)"), LabeledTrace([set(), {"a"}], terminated=False, truncated=True))
    reach = assess_prefix(parse_formula("F a"), LabeledTrace([set(), {"a"}], terminated=False, truncated=True))
    assert safety.status is PrefixStatus.DEFINITELY_VIOLATED
    assert reach.status is PrefixStatus.DEFINITELY_SATISFIED


def test_deep_until_prefix_residual_is_stack_safe():
    formula = parse_formula("(!magenta U (blue & (!green U yellow)))")
    trace = LabeledTrace([{"blue"}] * 1_200, terminated=False, truncated=True)
    assessment = assess_prefix(formula, trace)
    assert assessment.status is PrefixStatus.OPEN


def test_strict_grounding_fails_closed_and_validates_values():
    formula = parse_formula("p")
    with pytest.raises(ValueError, match="missing"):
        evaluate_with_provenance(formula, LabeledTrace([{"p"}]), require_grounding=True)
    with pytest.raises(ValueError, match="labels say"):
        LabeledTrace(
            [{"p"}],
            groundings=[GroundingRecord("p", 0, False, "rule", "1")],
        )
    trace = LabeledTrace(
        [{"p"}],
        groundings=[GroundingRecord("p", 0, True, "rule", "1", ("x",))],
    )
    assert evaluate_with_provenance(formula, trace, require_grounding=True).grounding_complete


@pytest.mark.parametrize("labels", [[{1}], [{"bad-name"}]])
def test_invalid_atomic_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="invalid proposition"):
        LabeledTrace(labels)


def test_invalid_trace_numeric_and_transition_fields_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        LabeledTrace([set(), set(), set()], actions=[0, 1], rewards=[1.0])
    with pytest.raises(ValueError, match="finite"):
        LabeledTrace([set(), set()], rewards=[float("nan")])


def test_v1_pass_relevance_is_recovered_as_occurrence_query():
    formula = parse_formula("F(p or q)")
    polarities = occurrence_polarities(formula)
    assert set(polarities.values()) == {Polarity.POSITIVE}
    trace = LabeledTrace([{"q"}])
    # o0 is p, o1 is q under root/eventually/or preorder paths.
    p_query = occurrence_mutation_query(formula, trace, "o0_0")
    q_query = occurrence_mutation_query(formula, trace, "o0_1")
    assert p_query.pass_necessary is False
    assert q_query.pass_necessary is True


def test_mixed_polarity_is_rejected_instead_of_given_a_fake_mutation():
    formula = parse_formula("p <-> q")
    with pytest.raises(ValueError, match="mixed-polarity"):
        occurrence_mutation_query(formula, LabeledTrace([{"p", "q"}]), "o0")
