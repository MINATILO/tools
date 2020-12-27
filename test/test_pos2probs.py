import numpy as np
import pytest

from tools.pos2prob import check_positions_are_normalised


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

