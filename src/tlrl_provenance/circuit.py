"""Hash-consed monotone circuits describing all operational derivations."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .trace import EvidenceFact


class NodeKind(str, Enum):
    TRUE = "true"
    EVIDENCE = "evidence"
    AND = "and"
    OR = "or"


class SupportEnumerationMode(str, Enum):
    EXACT = "exact"
    BOUNDED_PARTIAL = "bounded_partial"


@dataclass(frozen=True, slots=True)
class CircuitNode:
    id: int
    kind: NodeKind
    children: tuple[int, ...] = ()
    evidence: EvidenceFact | None = None


@dataclass(frozen=True, slots=True)
class SupportEnumeration:
    supports: tuple[frozenset[EvidenceFact], ...]
    complete: bool
    limit: int
    work_limit: int
    work_done: int
    mode: SupportEnumerationMode
    intermediate_limit: int | None
    truncation_reasons: tuple[str, ...] = ()
    truncated_node_ids: tuple[int, ...] = ()


DEFAULT_SUPPORT_WORK_LIMIT = 100_000
DEFAULT_INTERMEDIATE_SUPPORT_LIMIT = 16
"""AND-node beam cap used only in explicit bounded-partial mode.

A long temporal fold (`G phi` over many steps compiles to a chain of nested
binary AND nodes, one per position) combines each step's local antichain
against the *entire accumulated suffix* antichain. If every node in that
chain were allowed to grow toward a large requested `limit`, the per-level
combination cost (local branching x accumulated suffix size) compounds
across the whole chain, exhausting the work budget before the final target
is ever reached -- confirmed empirically: intermediate antichains grow
combinatorially until some cap engages, after which growth flattens to
roughly linear-in-trace-length. Capping non-target AND nodes to a small
constant makes that flattening happen almost immediately regardless of how
large a `limit` the caller requests for the final result.

Once flattened, remaining cost scales as roughly
`5-6 x trace_length x this_constant` work units (measured directly against
real and synthetic traces, `G(trigger -> F_leq_K(...))` shape). No fixed
cap removes this scaling law -- it is linear in trace length, not
exponential, but still unbounded as length grows. This default (16) keeps
traces up to roughly 1,000 positions within `DEFAULT_SUPPORT_WORK_LIMIT`;
longer traces need an explicitly larger `work_limit` or a smaller
`intermediate_limit`, and will otherwise report `complete=False` with
`truncation_reasons=("work_limit",)` rather than hang or return a wrong
answer. Linear OR folds such as eventuality witness chains are deliberately
not subject to this cap: they do not construct Cartesian products, and
truncating them would discard cheap witness/counterexample alternatives.
Exact mode does not use this cap at all."""


@dataclass(slots=True)
class _WorkBudget:
    limit: int
    used: int = 0

    def consume(self, amount: int = 1) -> bool:
        if amount < 0:
            raise ValueError("work amount cannot be negative")
        if self.used + amount > self.limit:
            return False
        self.used += amount
        return True


class CircuitBuilder:
    def __init__(self, *, max_nodes: int | None = None) -> None:
        if max_nodes is not None and max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self._max_nodes = max_nodes
        self._nodes: list[CircuitNode] = [CircuitNode(0, NodeKind.TRUE)]
        self._intern: dict[tuple[object, ...], int] = {(NodeKind.TRUE,): 0}

    @property
    def true(self) -> int:
        return 0

    def evidence(self, fact: EvidenceFact) -> int:
        key = (NodeKind.EVIDENCE, fact)
        return self._add(key, NodeKind.EVIDENCE, evidence=fact)

    def all_of(self, children: Iterable[int]) -> int:
        normalized = tuple(sorted({child for child in children if child != self.true}))
        if not normalized:
            return self.true
        if len(normalized) == 1:
            return normalized[0]
        key = (NodeKind.AND, normalized)
        return self._add(key, NodeKind.AND, children=normalized)

    def any_of(self, children: Iterable[int]) -> int:
        normalized_set = set(children)
        if not normalized_set:
            raise ValueError("a derivation disjunction requires at least one viable branch")
        if self.true in normalized_set:
            return self.true
        normalized = tuple(sorted(normalized_set))
        if len(normalized) == 1:
            return normalized[0]
        key = (NodeKind.OR, normalized)
        return self._add(key, NodeKind.OR, children=normalized)

    def freeze(self, root: int) -> "ProvenanceCircuit":
        if root < 0 or root >= len(self._nodes):
            raise ValueError("root does not identify a circuit node")
        return ProvenanceCircuit(nodes=tuple(self._nodes), root=root)

    def _add(
        self,
        key: tuple[object, ...],
        kind: NodeKind,
        *,
        children: tuple[int, ...] = (),
        evidence: EvidenceFact | None = None,
    ) -> int:
        existing = self._intern.get(key)
        if existing is not None:
            return existing
        if self._max_nodes is not None and len(self._nodes) >= self._max_nodes:
            raise RuntimeError(f"provenance circuit exceeded node limit {self._max_nodes}")
        node_id = len(self._nodes)
        self._nodes.append(CircuitNode(node_id, kind, children, evidence))
        self._intern[key] = node_id
        return node_id


@dataclass(frozen=True, slots=True)
class ProvenanceCircuit:
    nodes: tuple[CircuitNode, ...]
    root: int

    def __post_init__(self) -> None:
        if not self.nodes or self.root < 0 or self.root >= len(self.nodes):
            raise ValueError("invalid circuit root")
        for expected, node in enumerate(self.nodes):
            if node.id != expected:
                raise ValueError("circuit nodes must have dense stable IDs")
            if any(child >= node.id for child in node.children):
                raise ValueError("circuit edges must point to earlier nodes")
            if node.kind is NodeKind.EVIDENCE and node.evidence is None:
                raise ValueError("evidence node is missing its fact")
            if node.kind is NodeKind.EVIDENCE and node.children:
                raise ValueError("evidence nodes cannot have children")
            if node.kind is NodeKind.TRUE and (node.children or node.evidence is not None):
                raise ValueError("true nodes cannot have children or evidence")
            if node.kind in {NodeKind.AND, NodeKind.OR}:
                if len(node.children) < 2 or node.evidence is not None:
                    raise ValueError("and/or nodes require at least two children and no evidence")

    def possible_evidence(self, node_id: int | None = None) -> frozenset[EvidenceFact]:
        target = self.root if node_id is None else node_id
        self._validate_node_id(target)
        return frozenset(
            self.nodes[node_id].evidence
            for node_id in self._reachable_node_ids(target)
            if self.nodes[node_id].kind is NodeKind.EVIDENCE
            and self.nodes[node_id].evidence is not None
        )

    def necessary_evidence(self, node_id: int | None = None) -> frozenset[EvidenceFact]:
        """Facts present in every represented operational derivation."""

        target = self.root if node_id is None else node_id
        self._validate_node_id(target)
        fact_to_bit: dict[EvidenceFact, int] = {}
        facts: list[EvidenceFact] = []
        masks: list[int] = []
        for node in self.nodes[: target + 1]:
            if node.kind is NodeKind.TRUE:
                mask = 0
            elif node.kind is NodeKind.EVIDENCE:
                assert node.evidence is not None
                bit = fact_to_bit.get(node.evidence)
                if bit is None:
                    bit = len(facts)
                    fact_to_bit[node.evidence] = bit
                    facts.append(node.evidence)
                mask = 1 << bit
            elif node.kind is NodeKind.AND:
                mask = 0
                for child in node.children:
                    mask |= masks[child]
            else:
                mask = masks[node.children[0]]
                for child in node.children[1:]:
                    mask &= masks[child]
            masks.append(mask)
        root_mask = masks[target]
        return frozenset(fact for bit, fact in enumerate(facts) if root_mask & (1 << bit))

    def minimal_supports(
        self,
        limit: int = 10_000,
        node_id: int | None = None,
        *,
        work_limit: int = DEFAULT_SUPPORT_WORK_LIMIT,
        mode: SupportEnumerationMode | str = SupportEnumerationMode.EXACT,
        intermediate_limit: int | None = None,
    ) -> SupportEnumeration:
        """Enumerate derivational supports with output and work hard caps.

        If ``complete`` is true, ``supports`` is the complete inclusion-minimal
        antichain represented by the target circuit. If it is false, every
        returned set is still a valid derivational support, but unexamined
        derivations may contain additional or smaller supports. No global
        minimality claim is made for an incomplete result.

        ``limit`` bounds retained supports at any expanded node.
        ``work_limit`` bounds candidate construction and subset comparisons,
        preventing a Cartesian-product expansion from doing unbounded work
        before the output cap is applied. Exact mode uses no heuristic
        intermediate cap. Bounded-partial mode separately caps non-target AND
        nodes, where Cartesian products arise; linear OR nodes remain governed
        only by ``limit`` and ``work_limit``.
        """

        if limit < 1:
            raise ValueError("support limit must be positive")
        if work_limit < 1:
            raise ValueError("support work limit must be positive")
        try:
            resolved_mode = SupportEnumerationMode(mode)
        except ValueError as exc:
            raise ValueError(f"unsupported support enumeration mode: {mode!r}") from exc
        if intermediate_limit is not None and intermediate_limit < 1:
            raise ValueError("intermediate support limit must be positive")
        if resolved_mode is SupportEnumerationMode.EXACT and intermediate_limit is not None:
            raise ValueError("intermediate_limit is only valid in bounded_partial mode")
        effective_intermediate_limit = None
        if resolved_mode is SupportEnumerationMode.BOUNDED_PARTIAL:
            effective_intermediate_limit = (
                min(limit, DEFAULT_INTERMEDIATE_SUPPORT_LIMIT)
                if intermediate_limit is None
                else intermediate_limit
            )
        target = self.root if node_id is None else node_id
        self._validate_node_id(target)
        reachable = self._reachable_node_ids(target)
        if not any(self.nodes[current].kind is NodeKind.OR for current in reachable):
            return SupportEnumeration(
                supports=(self.possible_evidence(target),),
                complete=True,
                limit=limit,
                work_limit=work_limit,
                work_done=0,
                mode=resolved_mode,
                intermediate_limit=effective_intermediate_limit,
            )

        budget = _WorkBudget(work_limit)
        memo: dict[
            int,
            tuple[
                tuple[frozenset[EvidenceFact], ...],
                bool,
                frozenset[str],
                frozenset[int],
            ],
        ] = {}
        for node_id in sorted(reachable):
            node = self.nodes[node_id]
            node_limit = limit
            cap_reason = "output_limit"
            if node.kind is NodeKind.TRUE:
                result = ((frozenset(),), True, frozenset(), frozenset())
            elif node.kind is NodeKind.EVIDENCE:
                assert node.evidence is not None
                result = (
                    (frozenset((node.evidence,)),),
                    True,
                    frozenset(),
                    frozenset(),
                )
            elif node.kind is NodeKind.OR:
                accumulator = _AntichainAccumulator(node_limit)
                complete = True
                reasons: set[str] = set()
                truncated_nodes: set[int] = set()
                stopped = False
                for child in node.children:
                    supports, child_complete, child_reasons, child_nodes = memo[child]
                    complete &= child_complete
                    reasons.update(child_reasons)
                    truncated_nodes.update(child_nodes)
                    for support in supports:
                        status = accumulator.add(support, budget)
                        if status == "work_limit":
                            reasons.add("work_limit")
                            truncated_nodes.add(node_id)
                            stopped = True
                            break
                        if status == "support_limit":
                            reasons.add(cap_reason)
                            truncated_nodes.add(node_id)
                            stopped = True
                            break
                    if stopped:
                        break
                if stopped:
                    complete = False
                result = (
                    accumulator.supports(),
                    complete,
                    frozenset(reasons),
                    frozenset(truncated_nodes),
                )
            else:
                child_results = [memo[child] for child in node.children]
                branching_children = sum(len(item[0]) > 1 for item in child_results)
                uses_intermediate_cap = (
                    resolved_mode is SupportEnumerationMode.BOUNDED_PARTIAL
                    and node_id != target
                    and branching_children >= 2
                    and effective_intermediate_limit is not None
                    and effective_intermediate_limit < limit
                )
                if uses_intermediate_cap:
                    node_limit = effective_intermediate_limit
                    cap_reason = "intermediate_limit"
                complete = all(item[1] for item in child_results)
                reasons = set().union(*(item[2] for item in child_results))
                truncated_nodes = set().union(*(item[3] for item in child_results))
                accumulator = _AntichainAccumulator(node_limit)
                stopped = False
                sorted_lists = [item[0] for item in child_results]
                combo_state = _CombinationBudgetState()
                for indices in _ascending_size_combinations(sorted_lists, budget, combo_state):
                    candidate = frozenset(
                        fact
                        for choice_index, lst in zip(indices, sorted_lists)
                        for fact in lst[choice_index]
                    )
                    status = accumulator.add(candidate, budget, candidate_already_counted=True)
                    if status == "work_limit":
                        reasons.add("work_limit")
                        truncated_nodes.add(node_id)
                        stopped = True
                        break
                    if status == "support_limit":
                        reasons.add(cap_reason)
                        truncated_nodes.add(node_id)
                        stopped = True
                        break
                if combo_state.exhausted:
                    reasons.add("work_limit")
                    truncated_nodes.add(node_id)
                    stopped = True
                if stopped:
                    complete = False
                result = (
                    accumulator.supports(),
                    complete,
                    frozenset(reasons),
                    frozenset(truncated_nodes),
                )
            memo[node_id] = result

            # Once the global work budget is exhausted, later circuit nodes
            # cannot be expanded soundly within this call. Return an explicit
            # incomplete result instead of falling off an unbounded-work cliff.
            if "work_limit" in result[2] and node_id != target:
                return SupportEnumeration(
                    supports=(),
                    complete=False,
                    limit=limit,
                    work_limit=work_limit,
                    work_done=budget.used,
                    mode=resolved_mode,
                    intermediate_limit=effective_intermediate_limit,
                    truncation_reasons=("work_limit",),
                    truncated_node_ids=tuple(sorted(result[3] | {node_id})),
                )

        supports, complete, reasons, truncated_nodes = memo[target]
        return SupportEnumeration(
            supports=supports,
            complete=complete,
            limit=limit,
            work_limit=work_limit,
            work_done=budget.used,
            mode=resolved_mode,
            intermediate_limit=effective_intermediate_limit,
            truncation_reasons=tuple(sorted(reasons)),
            truncated_node_ids=tuple(sorted(truncated_nodes)),
        )

    def _validate_node_id(self, node_id: int) -> None:
        if node_id < 0 or node_id >= len(self.nodes):
            raise IndexError(node_id)

    def _reachable_node_ids(self, node_id: int) -> frozenset[int]:
        reachable: set[int] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(self.nodes[current].children)
        return frozenset(reachable)


class _CombinationBudgetState:
    """Out-of-band signal that `_ascending_size_combinations` stopped early
    because the shared work budget ran out mid-generation, distinct from the
    accumulator's own per-candidate outcome."""

    __slots__ = ("exhausted",)

    def __init__(self) -> None:
        self.exhausted = False


def _ascending_size_combinations(
    sorted_lists: list[tuple[frozenset[EvidenceFact], ...]],
    budget: "_WorkBudget",
    state: _CombinationBudgetState,
):
    """Yield one-index-per-list tuples choosing an element from each list, in
    non-decreasing order of the summed cardinality of the chosen elements,
    without materializing the full Cartesian product.

    Each input list is assumed already sorted ascending by element size (true
    of every `_AntichainAccumulator.supports()` result). Summed cardinality is
    an *upper bound* on union cardinality because evidence may overlap across
    children. It is therefore only a deterministic exploration heuristic, not
    a guarantee of ascending union size or global minimality for a truncated
    result. Complete runs still examine every combination and recover the
    exact antichain.

    Consumes one work unit per yielded combination. On budget exhaustion,
    stops without raising and sets `state.exhausted = True`; the caller must
    check that flag after iteration ends.
    """

    k = len(sorted_lists)
    if any(len(lst) == 0 for lst in sorted_lists):
        return
    sizes = [[len(item) for item in lst] for lst in sorted_lists]
    start = (0,) * k
    heap: list[tuple[int, tuple[int, ...]]] = [(sum(sizes[i][0] for i in range(k)), start)]
    visited = {start}
    while heap:
        if not budget.consume():
            state.exhausted = True
            return
        _, indices = heapq.heappop(heap)
        yield indices
        for i in range(k):
            if indices[i] + 1 < len(sorted_lists[i]):
                successor = indices[:i] + (indices[i] + 1,) + indices[i + 1 :]
                if successor not in visited:
                    visited.add(successor)
                    total = sum(sizes[j][successor[j]] for j in range(k))
                    heapq.heappush(heap, (total, successor))


class _AntichainAccumulator:
    """Incremental antichain whose expensive operations consume a budget."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._minimal_by_size: dict[int, list[frozenset[EvidenceFact]]] = {}
        self._count = 0
        self._seen: set[frozenset[EvidenceFact]] = set()

    def add(
        self,
        candidate: frozenset[EvidenceFact],
        budget: _WorkBudget,
        *,
        candidate_already_counted: bool = False,
    ) -> str:
        if not candidate_already_counted and not budget.consume():
            return "work_limit"
        if candidate in self._seen:
            return "accepted"

        candidate_size = len(candidate)
        supersets: list[tuple[int, frozenset[EvidenceFact]]] = []
        for existing_size, existing_supports in self._minimal_by_size.items():
            # Distinct equal-size sets cannot include one another. Keeping
            # size buckets avoids the former O(limit^2) scan when thousands
            # of independent obligations produce equal-cardinality supports.
            if existing_size == candidate_size:
                continue
            for existing in existing_supports:
                if not budget.consume():
                    return "work_limit"
                if existing_size < candidate_size:
                    if existing < candidate:
                        self._seen.add(candidate)
                        return "accepted"
                elif candidate < existing:
                    supersets.append((existing_size, existing))

        resulting_size = self._count - len(supersets) + 1
        if resulting_size > self._limit:
            return "support_limit"

        for existing_size, existing in supersets:
            bucket = self._minimal_by_size[existing_size]
            bucket.remove(existing)
            if not bucket:
                del self._minimal_by_size[existing_size]
        self._minimal_by_size.setdefault(candidate_size, []).append(candidate)
        self._count = resulting_size
        self._seen.add(candidate)
        return "accepted"

    def supports(self) -> tuple[frozenset[EvidenceFact], ...]:
        return tuple(
            sorted(
                (
                    support
                    for bucket in self._minimal_by_size.values()
                    for support in bucket
                ),
                key=lambda support: (
                    len(support),
                    tuple(sorted(fact.stable_key() for fact in support)),
                ),
            )
        )
