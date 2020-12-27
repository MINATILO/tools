import numpy as np
import pytest

from tools.pos2prob import check_positions_are_normalised
from tools.pos2prob import pos2prob, normalise_finishing_positions


def test_check_positions_are_normalised_raises_on_unnormalised_input():
    bad_input = np.array([1.2, 1, 3])
    with pytest.raises(AssertionError) as e:
        check_positions_are_normalised(bad_input)
    assert 'must sum to 6' in str(e)


def test_check_positions_are_normalised_raises_on_out_of_bound_input():
    bad_input = np.array([0.9, 2.1, 3])
    with pytest.raises(AssertionError) as e:
        check_positions_are_normalised(bad_input)
    assert 'must be 1 < positions < 3 for all positions' in str(e)


def test_pos2prob_outputs_doubly_stochastic_matrix():
    positions = np.array([2., 1., 3.])
    p = pos2prob(positions)
    assert np.allclose(np.sum(p, axis=0), 1.)
    assert np.allclose(np.sum(p, axis=1), 1.)


def test_pos2prob_outputs_sums_to_expected_positions():
    expected_positions = np.array([2., 1., 3.])
    p = pos2prob(expected_positions)
    positions = np.vstack([[1, 2, 3]] * 3)
    assert np.allclose(np.sum(p * positions, axis=1), expected_positions)


def test_normalise_finishing_positions():
    unnormalised_positions = np.array([2., 1.7, 1.1])
    out = normalise_finishing_positions(unnormalised_positions)
    check_positions_are_normalised(out)
    assert np.allclose(out.sum(), 6)
    assert min(out) >= 1 and max(out) <= 3
