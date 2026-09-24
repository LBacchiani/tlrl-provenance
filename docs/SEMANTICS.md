# Semantic contract

## 1. Trace and verdict semantics

A trace is a nonempty finite word
`tau = sigma_0 ... sigma_h`, where each `sigma_t` is a set of atomic
propositions. Absence means false. Evaluation is at a formula occurrence and
time pair `(o,t)`.

Boolean operators have their standard interpretation. `X phi` is false at
`h`; `Xw phi` is true at `h`. The temporal recurrences used by the production
evaluator are:

```text
F phi(t)       = phi(t) OR F phi(t+1)
G phi(t)       = phi(t) AND G phi(t+1)
phi U psi(t)   = psi(t) OR (phi(t) AND (phi U psi)(t+1))
phi R psi(t)   = psi(t) AND (phi(t) OR (phi R psi)(t+1))
phi W psi(t)   = psi(t) OR (phi(t) AND (phi W psi)(t+1))
```

At `h`, `F phi = phi`, `G phi = phi`, `phi U psi = psi`,
`phi R psi = psi`, and `phi W psi = psi OR phi`.

Two truth-only reference evaluators are retained. The exhaustive validation
oracle does not reuse the production recurrence for `F`, `G`, `U`, `R`, or
`W`: it evaluates their finite suffix definitions directly. A separately
implemented linear dynamic program supplies scalable per-trace cross-checks;
the tests differentially check both against the direct oracle.

## 2. Occurrences

Formula occurrences are identified by tree paths. Structurally identical
subformulas at different paths are different occurrences. Implication is not
desugared during provenance evaluation, so its antecedent-false and
consequent-true satisfaction mechanisms remain visible.

## 3. Evidence

An atomic evidence fact is `(proposition, time, truth-value)`. Negative facts
are first-class evidence: a passing certificate for `G(not a)` contains
`not a@t` at every observed time. Strong/weak-next decisions at the finite
boundary additionally use `END@h`.

Every proof node certifies the **observed truth value** of its formula cell.
Thus a violated formula receives a failure certificate rather than an absent
success certificate.

## 4. Provenance DAG

The circuit contains evidence leaves, conjunction nodes, disjunction nodes,
and the evidence-free `true` certificate used for logical constants. For an
observed truth table:

- a true conjunction combines all conjunct proofs; a false conjunction offers
  each false conjunct as an alternative counterproof;
- a true disjunction offers each true disjunct; a false disjunction combines
  both counterproofs;
- a true implication offers antecedent-false and consequent-true mechanisms;
  a false implication combines antecedent-true and consequent-false evidence;
- temporal operators apply the recurrences above to both truth values.

Hash-consing shares identical evidence and subderivations. Evidence-free or
duplicate-dominated alternatives are reduced. Operator decision records still
record every observed branch/witness, including branches that are not needed
by an inclusion-minimal evidence proof.

## 5. Support claims

When `minimal_supports()` returns `complete=True`, it returns the complete
antichain of inclusion-minimal supports of the operational provenance DAG.
These supports are:

- sufficient for the reported verdict under the declared LTLf semantics;
- minimal within the represented proof system;
- not causal explanations;
- not claimed to be globally minimum Boolean prime implicants after arbitrary
  theorem-level rewrites.

Enumeration can be exponential. Two modes therefore make different contracts
explicit:

- `exact` is the raw certificate API default. It applies `limit` to retained
  antichains and `work_limit` to candidate construction plus subset
  comparisons. It uses no heuristic intermediate beam. `complete=True` is the
  only signal that the full inclusion-minimal antichain was recovered.
- `bounded_partial` is the verifier/audit default. It applies the same two hard
  limits and additionally uses `intermediate_limit` (default
  `min(limit, 16)`) at genuinely multiplicative, non-target AND nodes: nodes
  for which at least two child antichains contain alternatives. This is an
  explicit availability/performance mode, not an exact algorithm under a
  different name.

The intermediate beam is deliberately *not* applied to OR nodes or to AND
nodes that merely prepend one fixed support to one branching child. Those are
the shapes underlying linear eventuality, global-counterexample, until, and
weak-until alternatives. Consequently, ordinary formulas such as `F p`, a
failed `G(not p)`, `p U q`, and `p W q` retain all cheap temporal alternatives
when the ordinary output/work limits allow them.

The bounded-partial mode exists because a temporal fold (`G phi` over many
trace positions compiles to a chain of nested binary AND nodes, one per
position) may repeatedly combine a branching local antichain with a branching
accumulated suffix. Without a beam, this multiplicative work can exhaust the
budget before the root is reached even on an ordinary controller trace. Once
the branching AND antichains flatten, remaining measured work is approximately
linear in trace length, but no fixed default stays cheap for arbitrarily long
traces.

The heap traversal used for a branching Cartesian product orders candidates
by the sum of child-support cardinalities. Because child supports may overlap,
that sum is an **upper bound** on union size, not a proof that the first unions
visited are globally smallest. Therefore, if any bound is reached,
`complete=False`: returned sets are valid explored operational derivations,
but they need not themselves be globally inclusion-minimal because an
unexplored smaller support may dominate one later. Downstream code must make
neither a complete-alternatives nor a global-minimality claim for an
incomplete result.

Every result records `mode`, the effective `intermediate_limit`, `work_done`,
distinct `truncation_reasons` (`output_limit`, `intermediate_limit`, or
`work_limit`), and the exact `truncated_node_ids`. A result may contain no
expanded root supports when the work bound is exhausted in an internal node.
The compact DAG, `necessary_evidence`, and `possible_evidence` remain
available and unaffected: they use closed-form bitmask/reachability
computations rather than support enumeration. Certificate validity likewise
does not imply that an intentionally capped support list is complete.

## 6. Truncation

Ordinary finite-trace evaluation and right-censored prefix assessment are
separate APIs. Prefix assessment returns `definitely_satisfied` or
`definitely_violated` only when the finite endpoint verdict agrees with a
constant residual for proper continuations. Otherwise it returns `open`.
This prevents strong/weak-next boundary conditions from being silently
classified as decided.

## 7. Grounding and attribution

In strict mode, every formula atom at every observed time requires a grounding
record with a rule ID, version, raw-field list, and truth value agreeing with
the label trace. This establishes reconstruction provenance, not correctness
of the domain requirement itself.

The package makes no causal training-influence claim and cannot distinguish
policy, environment, or shield agency unless those mechanisms are represented
in separately supplied observations or intervention metadata.

Serialized certificate artifacts retain labels, grounding records, formula
semantics, the provenance circuit, and certificate identity. They deliberately
exclude raw states, observations, actions, rewards, and arbitrary metadata.
Artifact-only replay therefore verifies the stored semantic/grounding-record
layer; full raw-state grounding reconstruction requires the separately retained
source trace and registered proposition rules.

## 8. Statistical unit

Trace prevalence estimates describe sampled executions. The policy-cluster
bootstrap first computes rates within independently trained policy IDs and
then weights policies equally. It refuses to run without at least two policy
IDs. Benchmark-specific crossed policy/scenario inference remains an empirical
design responsibility rather than being guessed by this semantic layer.

Default event profiles include only formula cells participating in at least
one operational derivation of the root verdict. Full truth-table events remain
available through the explicit `all_cells` scope; they are not silently mixed
into verdict explanations.
