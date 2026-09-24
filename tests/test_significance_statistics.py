"""Regression coverage for the significance-testing additions to
``tlrl_provenance.statistics``: an exact two-group test with no scipy
dependency, and paired and unpaired permutation tests for per-key and
aggregate group-range claims.
"""

from __future__ import annotations

import math

import pytest

from tlrl_provenance import (
    bonferroni_alpha,
    fisher_exact_two_sided,
    permutation_test_group_range,
    permutation_test_median_range,
    permutation_test_paired_group_range,
    permutation_test_paired_median_range,
)


def test_fisher_exact_matches_known_textbook_value():
    # Fisher's "lady tasting tea", 8 cups (4 milk-first / 4 tea-first): she
    # correctly identifies 3 of the 4 milk-first cups. The standard
    # "sum of table probabilities no greater than the observed table's"
    # two-sided p-value for this classic example is 34/70 (probability
    # tables a=0,1,3,4 out of C(8,4)=70 total arrangements).
    p = fisher_exact_two_sided(3, 4, 1, 4)
    assert p == pytest.approx(34 / 70, abs=1e-9)


def test_fisher_exact_identical_groups_gives_p_one():
    p = fisher_exact_two_sided(5, 10, 5, 10)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_fisher_exact_perfect_separation_is_small():
    p = fisher_exact_two_sided(0, 10, 10, 10)
    assert p < 1e-4


def test_fisher_exact_rejects_invalid_counts():
    with pytest.raises(ValueError):
        fisher_exact_two_sided(-1, 10, 5, 10)
    with pytest.raises(ValueError):
        fisher_exact_two_sided(11, 10, 5, 10)


def test_bonferroni_alpha_divides_by_comparison_count():
    assert bonferroni_alpha(0.05, 50) == pytest.approx(0.001)
    with pytest.raises(ValueError):
        bonferroni_alpha(0.05, 0)
    with pytest.raises(ValueError):
        bonferroni_alpha(1.5, 10)


def test_permutation_test_group_range_identical_groups_large_p_value():
    groups = [[True] * 10 + [False] * 10 for _ in range(5)]
    result = permutation_test_group_range(groups, iterations=500, seed=0)
    assert result.observed == pytest.approx(0.0)
    # No divergence at all: every reshuffle is at least as extreme.
    assert result.p_value == pytest.approx(1.0)


def test_permutation_test_group_range_perfect_separation_significant():
    groups = [[True] * 20, [False] * 20, [True] * 20, [False] * 20, [True] * 20]
    result = permutation_test_group_range(groups, iterations=2000, seed=0)
    assert result.observed == pytest.approx(1.0)
    assert result.p_value < 0.01
    assert result.p_value >= 1 / 2001


def test_permutation_test_group_range_is_deterministic_for_fixed_seed():
    groups = [[True, True, False], [True, False, False], [True, True, True]]
    a = permutation_test_group_range(groups, iterations=200, seed=42)
    b = permutation_test_group_range(groups, iterations=200, seed=42)
    assert a == b


def test_permutation_test_group_range_rejects_degenerate_input():
    with pytest.raises(ValueError):
        permutation_test_group_range([[True, False]], iterations=10, seed=0)
    with pytest.raises(ValueError):
        permutation_test_group_range([[True], []], iterations=10, seed=0)
    with pytest.raises(ValueError):
        permutation_test_group_range([[True], [False]], iterations=0, seed=0)


def test_permutation_test_median_range_matches_observed_median_of_ranges():
    key_a = [[True] * 20, [False] * 20]  # range 1.0
    key_b = [[True] * 10 + [False] * 10, [True] * 10 + [False] * 10]  # range 0.0
    result = permutation_test_median_range([key_a, key_b], iterations=300, seed=0)
    assert result.observed == pytest.approx(0.5)


def test_permutation_test_median_range_noise_floor_is_positive_at_small_n():
    # At n=20 per group with 5 groups, pure noise alone produces a nonzero
    # expected median range -- this is exactly the phenomenon the paired
    # PointWorld report's statistics section must be honest about.
    rng_groups = [[[True] * 17 + [False] * 3 for _ in range(5)] for _ in range(50)]
    result = permutation_test_median_range(rng_groups, iterations=500, seed=1)
    assert result.null_mean > 0.0


def test_permutation_test_median_range_is_deterministic_for_fixed_seed():
    keys = [[[True, False, True], [False, False, True]] for _ in range(5)]
    a = permutation_test_median_range(keys, iterations=100, seed=7)
    b = permutation_test_median_range(keys, iterations=100, seed=7)
    assert a == b


def test_permutation_test_median_range_rejects_empty_input():
    with pytest.raises(ValueError):
        permutation_test_median_range([], iterations=10, seed=0)


def test_paired_group_range_detects_consistent_within_scenario_separation():
    blocks = [[True, False] for _ in range(20)]
    result = permutation_test_paired_group_range(blocks, iterations=2000, seed=0)
    assert result.observed == pytest.approx(1.0)
    assert 1 / 2001 <= result.p_value < 0.01


def test_paired_group_range_is_deterministic_for_fixed_seed():
    blocks = [[True, False, True], [False, False, True], [True, True, False]]
    a = permutation_test_paired_group_range(blocks, iterations=200, seed=42)
    b = permutation_test_paired_group_range(blocks, iterations=200, seed=42)
    assert a == b


@pytest.mark.parametrize(
    "blocks,iterations",
    [([], 10), ([[True]], 10), ([[True, False], [True]], 10), ([[True, False]], 0)],
)
def test_paired_group_range_rejects_invalid_designs(blocks, iterations):
    with pytest.raises(ValueError):
        permutation_test_paired_group_range(blocks, iterations=iterations, seed=0)


def test_paired_median_range_matches_observed_median_of_ranges():
    key_a = [[True, False] for _ in range(20)]  # range 1.0
    key_b = [[True, True], [False, False]]  # both subject rates are 0.5
    result = permutation_test_paired_median_range([key_a, key_b], iterations=300, seed=0)
    assert result.observed == pytest.approx(0.5)


@pytest.mark.parametrize(
    "keys",
    [[], [[]], [[[True]]], [[[True, False], [True]]]],
)
def test_paired_median_range_rejects_invalid_designs(keys):
    with pytest.raises(ValueError):
        permutation_test_paired_median_range(keys, iterations=10, seed=0)
