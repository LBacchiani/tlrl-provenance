import pytest

from tlrl_provenance.circuit import CircuitNode, NodeKind, ProvenanceCircuit


@pytest.mark.parametrize(
    "nodes",
    [
        (CircuitNode(0, NodeKind.EVIDENCE),),
        (CircuitNode(0, NodeKind.TRUE, children=(0,)),),
        (CircuitNode(0, NodeKind.TRUE), CircuitNode(1, NodeKind.AND, children=(0,))),
        (CircuitNode(0, NodeKind.TRUE), CircuitNode(1, NodeKind.OR, children=(1, 0))),
    ],
)
def test_malformed_circuits_fail_closed(nodes):
    with pytest.raises(ValueError):
        ProvenanceCircuit(nodes, len(nodes) - 1)
