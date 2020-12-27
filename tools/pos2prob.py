import numpy as np
import cvxpy as cp


def pos2prob(positions):
    check_positions_are_normalised(positions)
    n = len(positions)

    a = np.arange(n) + 1
    b = np.ones(n)

    A = np.kron(np.eye(n), a)
    B = np.kron(np.eye(n), b)
    C = np.block([np.eye(n)]*n)

    X = np.concatenate([A, B, C])
    m = np.concatenate([positions, np.ones(2*n)])
    p = _cvx_solve(X, m, n)
    return p


def check_positions_are_normalised(positions):
    n = len(positions)
    expected_sum = 0.5 * n * (n+1)
    msg1 = f'positions must sum to {expected_sum} for {n} runners'
    msg2 = f'must be 1 < positions < {n} for all positions'
    assert np.allclose(positions.sum(), expected_sum), msg1
    assert min(positions) >= 1 and max(positions) <= n, msg2


def _cvx_solve(X, m, n):
    p = cp.Variable(n**2)
    objective = cp.Minimize(cp.sum_squares(X@p - m))
    constraints = [0 <= p, p <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    p = p.value.reshape((n, n))
    return p


def normalise_finishing_positions(positions):
    m = positions - np.min(positions) + 1
    n = len(m)
    R = 0.5 * (n**2 + n) / np.sum(m)
    return m * R
