"""Benchmark-independent statistical aggregation of semantic certificates."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence
import re

from .evaluator import SemanticCertificate, Verdict, verdict_relevant_cells


@dataclass(frozen=True, slots=True, order=True)
class EventKey:
    occurrence_id: str
    operator: str
    kind: str
    alternative: str


@dataclass(frozen=True, slots=True)
class ProportionEstimate:
    successes: int
    trials: int
    value: float
    lower: float
    upper: float
    confidence: float
    method: str = "clopper_pearson"


@dataclass(frozen=True, slots=True)
class SemanticProfile:
    trace_count: int
    policy_count: int
    satisfaction: ProportionEstimate
    event_prevalence: tuple[tuple[EventKey, ProportionEstimate], ...]
    mean_possible_evidence: float
    mean_necessary_evidence: float


@dataclass(frozen=True, slots=True)
class ClusterBootstrapEstimate:
    value: float
    lower: float
    upper: float
    confidence: float
    policy_count: int
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class PermutationTestResult:
    """Null distribution of a group-comparison statistic under label reshuffling.

    ``p_value`` is ``P(null statistic >= observed)`` -- a one-sided test of
    whether the observed group divergence exceeds what group-label-blind
    noise alone would produce. Use this instead of citing a raw range,
    spread, or CI-overlap heuristic as if it were self-evidently significant.
    """

    observed: float
    null_mean: float
    null_p05: float
    null_median: float
    null_p95: float
    p_value: float
    iterations: int


def certificate_events(
    certificate: SemanticCertificate,
    *,
    scope: str = "verdict",
) -> frozenset[EventKey]:
    """Return trace-presence events, deliberately ignoring absolute time.

    Timing remains available in the certificate.  The default statistical
    profile uses time-invariant event identities so traces of different length
    remain comparable without silently choosing time bins.
    """

    if scope == "verdict":
        selected = verdict_relevant_cells(certificate)
    elif scope == "all_cells":
        selected = None
    else:
        raise ValueError("event scope must be 'verdict' or 'all_cells'")
    return frozenset(
        EventKey(item.occurrence_id, item.operator, item.kind, _normalize_alternative(alternative))
        for item in certificate.decisions
        if selected is None or (item.occurrence_id, item.time) in selected
        for alternative in item.alternatives
    )


def _normalize_alternative(alternative: str) -> str:
    return re.sub(r"@\d+$", "@observed_time", alternative)


def aggregate_certificates(
    certificates: Iterable[SemanticCertificate],
    *,
    confidence: float = 0.95,
) -> SemanticProfile:
    certificates = tuple(certificates)
    if not certificates:
        raise ValueError("cannot aggregate an empty certificate collection")
    formula_sources = {item.formula.to_source() for item in certificates}
    if len(formula_sources) != 1:
        raise ValueError("one semantic profile cannot mix different formulas")
    event_counts: Counter[EventKey] = Counter()
    for certificate in certificates:
        event_counts.update(certificate_events(certificate))
    trace_count = len(certificates)
    policies = {item.trace.policy_id for item in certificates if item.trace.policy_id is not None}
    satisfaction_count = sum(item.verdict is Verdict.SATISFIED for item in certificates)
    return SemanticProfile(
        trace_count=trace_count,
        policy_count=len(policies),
        satisfaction=exact_proportion(satisfaction_count, trace_count, confidence=confidence),
        event_prevalence=tuple(
            (key, exact_proportion(count, trace_count, confidence=confidence))
            for key, count in sorted(event_counts.items())
        ),
        mean_possible_evidence=math.fsum(len(item.possible_evidence) for item in certificates) / trace_count,
        mean_necessary_evidence=math.fsum(len(item.necessary_evidence) for item in certificates) / trace_count,
    )


def exact_proportion(successes: int, trials: int, *, confidence: float = 0.95) -> ProportionEstimate:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    from scipy.special import betaincinv

    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(betaincinv(successes, trials - successes + 1, alpha / 2))
    upper = 1.0 if successes == trials else float(betaincinv(successes + 1, trials - successes, 1 - alpha / 2))
    return ProportionEstimate(successes, trials, successes / trials, lower, upper, confidence)


def policy_cluster_bootstrap_satisfaction(
    certificates: Sequence[SemanticCertificate],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> ClusterBootstrapEstimate:
    """Policy-cluster bootstrap of the equal-policy-weighted satisfaction rate."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_policy: dict[str, list[SemanticCertificate]] = defaultdict(list)
    for certificate in certificates:
        if certificate.trace.policy_id is None:
            raise ValueError("cluster bootstrap requires policy_id on every trace")
        by_policy[certificate.trace.policy_id].append(certificate)
    if len(by_policy) < 2:
        raise ValueError("cluster bootstrap requires at least two policies")
    policy_rates = tuple(
        sum(item.verdict is Verdict.SATISFIED for item in group) / len(group)
        for _, group in sorted(by_policy.items())
    )
    value = math.fsum(policy_rates) / len(policy_rates)
    rng = random.Random(seed)
    replicates = sorted(
        math.fsum(rng.choice(policy_rates) for _ in policy_rates) / len(policy_rates)
        for _ in range(bootstrap_samples)
    )
    alpha = 1.0 - confidence
    lower = _percentile(replicates, alpha / 2)
    upper = _percentile(replicates, 1 - alpha / 2)
    return ClusterBootstrapEstimate(
        value=value,
        lower=min(lower, value),
        upper=max(upper, value),
        confidence=confidence,
        policy_count=len(policy_rates),
        bootstrap_samples=bootstrap_samples,
    )


def policy_cluster_bootstrap_event(
    certificates: Sequence[SemanticCertificate],
    event: EventKey,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> ClusterBootstrapEstimate:
    """Policy-cluster bootstrap for the prevalence of one semantic event."""

    return _policy_cluster_bootstrap(
        certificates,
        outcome=lambda certificate: event in certificate_events(certificate),
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _policy_cluster_bootstrap(
    certificates: Sequence[SemanticCertificate],
    *,
    outcome,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> ClusterBootstrapEstimate:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    by_policy: dict[str, list[SemanticCertificate]] = defaultdict(list)
    for certificate in certificates:
        if certificate.trace.policy_id is None:
            raise ValueError("cluster bootstrap requires policy_id on every trace")
        by_policy[certificate.trace.policy_id].append(certificate)
    if len(by_policy) < 2:
        raise ValueError("cluster bootstrap requires at least two policies")
    policy_rates = tuple(
        sum(bool(outcome(item)) for item in group) / len(group)
        for _, group in sorted(by_policy.items())
    )
    value = math.fsum(policy_rates) / len(policy_rates)
    rng = random.Random(seed)
    replicates = sorted(
        math.fsum(rng.choice(policy_rates) for _ in policy_rates) / len(policy_rates)
        for _ in range(bootstrap_samples)
    )
    alpha = 1.0 - confidence
    lower = _percentile(replicates, alpha / 2)
    upper = _percentile(replicates, 1 - alpha / 2)
    return ClusterBootstrapEstimate(
        value=value,
        lower=min(lower, value),
        upper=max(upper, value),
        confidence=confidence,
        policy_count=len(policy_rates),
        bootstrap_samples=bootstrap_samples,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


# --- Significance testing -------------------------------------------------
#
# A raw cross-policy range or spread (e.g. "P_sat varies 0.65 to 0.85 across
# five policies on this formula") is not by itself evidence of a real
# per-requirement policy effect: at n=20 episodes per policy, binomial noise
# alone produces a nonzero expected range. The functions below quantify that
# noise floor directly via permutation, and provide an exact test (no scipy
# dependency -- computed via `math.comb`) for two-group comparisons, so a
# report can state a p-value instead of an unqualified spread.


def fisher_exact_two_sided(successes_a: int, trials_a: int, successes_b: int, trials_b: int) -> float:
    """Two-sided Fisher exact test p-value for a 2x2 contingency table.

    Computed by exact enumeration of the hypergeometric distribution over all
    tables with the observed row/column margins, summing the probability of
    every table at least as extreme as the observed one (the standard
    "sum of probabilities <= observed probability" definition). No scipy
    dependency.
    """

    if trials_a < 0 or trials_b < 0 or not (0 <= successes_a <= trials_a) or not (0 <= successes_b <= trials_b):
        raise ValueError("require 0 <= successes <= trials for both groups")
    successes_total = successes_a + successes_b
    trials_total = trials_a + trials_b
    if trials_total == 0:
        raise ValueError("at least one trial is required")

    def table_probability(a: int) -> float:
        b = successes_total - a
        return (
            math.comb(trials_a, a)
            * math.comb(trials_b, b)
            / math.comb(trials_total, successes_total)
        )

    observed_probability = table_probability(successes_a)
    lower = max(0, successes_total - trials_b)
    upper = min(trials_a, successes_total)
    # A small relative tolerance guards against floating-point noise causing
    # the observed table itself to be excluded from its own tail sum.
    tolerance = 1e-9
    return math.fsum(
        probability
        for a in range(lower, upper + 1)
        for probability in (table_probability(a),)
        if probability <= observed_probability * (1 + tolerance)
    )


def bonferroni_alpha(alpha: float, num_comparisons: int) -> float:
    """Per-comparison significance threshold under Bonferroni correction."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if num_comparisons < 1:
        raise ValueError("num_comparisons must be positive")
    return alpha / num_comparisons


def _group_range(groups: Sequence[Sequence[bool]]) -> float:
    rates = [sum(1 for value in group if value) / len(group) for group in groups]
    return max(rates) - min(rates)


def permutation_test_group_range(
    groups: Sequence[Sequence[bool]],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> PermutationTestResult:
    """Permutation test for one formula: is the cross-group success-rate range
    larger than group-label-blind noise would produce?

    Pools every outcome across ``groups`` (e.g. one boolean per episode, one
    group per policy) and repeatedly reassigns pooled outcomes to
    same-sized groups uniformly at random, recomputing the range each time.
    The null hypothesis is "group identity carries no information about the
    outcome" -- i.e. pure sampling noise at each group's trial count.
    """

    if len(groups) < 2:
        raise ValueError("need at least two groups to compare")
    if any(len(group) == 0 for group in groups):
        raise ValueError("every group must have at least one observation")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    observed = _group_range(groups)
    sizes = [len(group) for group in groups]
    pool = [value for group in groups for value in group]
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(iterations):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        reshuffled_groups: list[list[bool]] = []
        offset = 0
        for size in sizes:
            reshuffled_groups.append(shuffled[offset : offset + size])
            offset += size
        null_values.append(_group_range(reshuffled_groups))
    null_values.sort()
    p_value = _monte_carlo_p_value(null_values, observed, iterations)
    return PermutationTestResult(
        observed=observed,
        null_mean=math.fsum(null_values) / iterations,
        null_p05=_percentile(null_values, 0.05),
        null_median=_percentile(null_values, 0.50),
        null_p95=_percentile(null_values, 0.95),
        p_value=p_value,
        iterations=iterations,
    )


def _monte_carlo_p_value(null_values: Sequence[float], observed: float, iterations: int) -> float:
    """Add-one Monte Carlo correction: ``(#{null >= observed} + 1) / (B + 1)``.

    A raw ``count / B`` ratio can report an exact ``p = 0`` whenever no null
    draw reaches the observed statistic, which overstates certainty a finite
    number of resamples can never actually provide -- the true tail
    probability could be anywhere in ``(0, 1/(B+1))``, not exactly zero. The
    add-one correction (Davison & Hinkley 1997; North, Curtis & Sham 2002) is
    the standard fix and is always slightly conservative (never yields a
    p-value below what the resamples support).
    """

    exceed_or_equal = sum(1 for value in null_values if value >= observed)
    return (exceed_or_equal + 1) / (iterations + 1)


def permutation_test_paired_group_range(
    blocks: Sequence[Sequence[bool]],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> PermutationTestResult:
    """Permutation test for one formula under a matched-scenario (paired) design.

    Use this instead of :func:`permutation_test_group_range` whenever every
    subject was evaluated on the *same* scenario at a given episode index
    (e.g. a shared ``scenario_seed``/generated map across policies). ``blocks``
    is one row per matched scenario and one column per subject, so
    ``blocks[i][j]`` is subject ``j``'s outcome on scenario ``i``. Only the
    assignment of subject labels to columns is permuted, independently within
    each row; an outcome never moves to a different row. This holds
    within-scenario difficulty fixed and asks only whether subject identity
    carries information beyond that -- the null hypothesis this design was
    actually built to test. A fully pooled permutation (mixing outcomes
    across different scenarios) discards exactly the variance reduction the
    paired design paid for and is not the correct null here.
    """

    if not blocks:
        raise ValueError("need at least one matched block")
    width = len(blocks[0])
    if width < 2:
        raise ValueError("need at least two subjects to compare")
    if any(len(row) != width for row in blocks):
        raise ValueError("every block must have the same number of subjects")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    def block_group_range(rows: Sequence[Sequence[bool]]) -> float:
        columns = list(zip(*rows))
        rates = [sum(1 for value in column if value) / len(column) for column in columns]
        return max(rates) - min(rates)

    observed = block_group_range(blocks)
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(iterations):
        permuted_rows = []
        for row in blocks:
            shuffled_row = list(row)
            rng.shuffle(shuffled_row)
            permuted_rows.append(shuffled_row)
        null_values.append(block_group_range(permuted_rows))
    null_values.sort()
    p_value = _monte_carlo_p_value(null_values, observed, iterations)
    return PermutationTestResult(
        observed=observed,
        null_mean=math.fsum(null_values) / iterations,
        null_p05=_percentile(null_values, 0.05),
        null_median=_percentile(null_values, 0.50),
        null_p95=_percentile(null_values, 0.95),
        p_value=p_value,
        iterations=iterations,
    )


def permutation_test_paired_median_range(
    blocks_by_key: Sequence[Sequence[Sequence[bool]]],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> PermutationTestResult:
    """Aggregate paired-design counterpart to :func:`permutation_test_median_range`.

    ``blocks_by_key`` is one ``blocks`` argument (as in
    :func:`permutation_test_paired_group_range`) per key (e.g. per formula).
    Each iteration reshuffles every key's own blocks independently -- a
    block's outcomes never leak into another key's or another block's null
    draw -- then takes the median range across keys as one null sample for
    the aggregate statistic, mirroring the observed statistic exactly.
    """

    if not blocks_by_key:
        raise ValueError("need at least one key")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    for blocks in blocks_by_key:
        if not blocks:
            raise ValueError("every key must contain at least one matched block")
        width = len(blocks[0])
        if width < 2:
            raise ValueError("need at least two subjects to compare")
        if any(len(row) != width for row in blocks):
            raise ValueError("every block must have the same number of subjects")

    def block_group_range(rows: Sequence[Sequence[bool]]) -> float:
        columns = list(zip(*rows))
        rates = [sum(1 for value in column if value) / len(column) for column in columns]
        return max(rates) - min(rates)

    observed = _percentile(sorted(block_group_range(blocks) for blocks in blocks_by_key), 0.50)
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(iterations):
        per_key_ranges = []
        for blocks in blocks_by_key:
            permuted_rows = []
            for row in blocks:
                shuffled_row = list(row)
                rng.shuffle(shuffled_row)
                permuted_rows.append(shuffled_row)
            per_key_ranges.append(block_group_range(permuted_rows))
        null_values.append(_percentile(sorted(per_key_ranges), 0.50))
    null_values.sort()
    p_value = _monte_carlo_p_value(null_values, observed, iterations)
    return PermutationTestResult(
        observed=observed,
        null_mean=math.fsum(null_values) / iterations,
        null_p05=_percentile(null_values, 0.05),
        null_median=_percentile(null_values, 0.50),
        null_p95=_percentile(null_values, 0.95),
        p_value=p_value,
        iterations=iterations,
    )


def permutation_test_median_range(
    groups_by_key: Sequence[Sequence[Sequence[bool]]],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> PermutationTestResult:
    """Permutation test for an aggregate claim spanning many independent keys
    (e.g. "the median cross-policy range across 50 formulas is X").

    ``groups_by_key`` is one ``groups`` argument (as in
    ``permutation_test_group_range``) per key (e.g. per formula). Each
    iteration reshuffles every key's own pool independently -- a key's
    outcomes never leak into another key's null draw -- then takes the
    median range across keys as one null sample for the aggregate
    statistic. This mirrors the observed statistic exactly (also the median
    of per-key ranges), so the p-value answers "would this many independent
    n=20-scale binomial comparisons produce a median range this large by
    chance alone?" rather than conflating per-key and aggregate noise.
    """

    if len(groups_by_key) < 1:
        raise ValueError("need at least one key")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    observed = _percentile(sorted(_group_range(groups) for groups in groups_by_key), 0.50)
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(iterations):
        per_key_ranges = []
        for groups in groups_by_key:
            sizes = [len(group) for group in groups]
            pool = [value for group in groups for value in group]
            shuffled = pool[:]
            rng.shuffle(shuffled)
            reshuffled_groups: list[list[bool]] = []
            offset = 0
            for size in sizes:
                reshuffled_groups.append(shuffled[offset : offset + size])
                offset += size
            per_key_ranges.append(_group_range(reshuffled_groups))
        null_values.append(_percentile(sorted(per_key_ranges), 0.50))
    null_values.sort()
    p_value = _monte_carlo_p_value(null_values, observed, iterations)
    return PermutationTestResult(
        observed=observed,
        null_mean=math.fsum(null_values) / iterations,
        null_p05=_percentile(null_values, 0.05),
        null_median=_percentile(null_values, 0.50),
        null_p95=_percentile(null_values, 0.95),
        p_value=p_value,
        iterations=iterations,
    )
