import numpy as np
import pytest

from tools.ranking import check_ranks_are_normalised
from tools.ranking import ranks2prob, normalise_rank


def test_check_ranks_are_normalised_raises_on_unnormalised_input():
    bad_input = np.array([1.2, 1, 3])
    with pytest.raises(AssertionError) as e:
        check_ranks_are_normalised(bad_input)
    assert 'must sum to 6' in str(e)


def test_check_ranks_are_normalised_raises_on_out_of_bound_input():
    bad_input = np.array([0.9, 2.1, 3])
    with pytest.raises(AssertionError) as e:
        check_ranks_are_normalised(bad_input)
    assert 'must be 1 < ranks < 3 for all ranks' in str(e)


def test_ranks2prob_outputs_doubly_stochastic_matrix():
    ranks = np.array([2., 1., 3.])
    p = ranks2prob(ranks)
    assert np.allclose(np.sum(p, axis=0), 1.)
    assert np.allclose(np.sum(p, axis=1), 1.)


def test_ranks2prob_outputs_sums_to_expected_ranks():
    expected_rank = np.array([2., 1., 3.])
    p = ranks2prob(expected_rank)
    ranks = np.vstack([[1, 2, 3]] * 3)
    assert np.allclose(np.sum(p * ranks, axis=1), expected_rank)


def test_normalise_rank():
    unnormalised_ranks = np.array([2., 1.7, 1.1])
    out = normalise_rank(unnormalised_ranks)
    check_ranks_are_normalised(out)
    assert np.allclose(out.sum(), 6)
    assert min(out) >= 1 and max(out) <= 3
