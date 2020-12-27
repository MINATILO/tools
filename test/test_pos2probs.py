import numpy as np
import pytest

from tools.pos2prob import check_positions_are_normalised


def test_check_positions_are_normalised():
    bad_input = np.array([1.2, 1, 3])
    with pytest.raises(AssertionError) as e:
        check_positions_are_normalised(bad_input)
    assert 'must sum to' in str(e)
