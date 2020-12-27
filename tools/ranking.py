import numpy as np
import cvxpy as cp


def ranks2prob(ranks):
    check_ranks_are_normalised(ranks)
    n = len(ranks)

    M1 = np.kron(np.eye(n), np.arange(n) + 1)
    M2 = np.kron(np.eye(n), np.ones(n))
    M3 = np.block([np.eye(n)]*n)
    M = np.concatenate([M1, M2, M3])

    target = np.concatenate([ranks, np.ones(2*n)])
    p = _cvx_solve(M, target, n)
    return p


def check_ranks_are_normalised(ranks):
    n = len(ranks)
    expected_sum = 0.5 * n * (n+1)
    msg1 = f'ranks must sum to {expected_sum} for {n} runners'
    msg2 = f'must be 1 < ranks < {n} for all ranks'
    assert np.allclose(ranks.sum(), expected_sum), msg1
    assert min(ranks) >= 1 and max(ranks) <= n, msg2


def _cvx_solve(M, target, n):
    p = cp.Variable(n**2)
    objective = cp.Minimize(cp.sum_squares(M@p - target))
    constraints = [0 <= p, p <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    p = p.value.reshape((n, n))
    return p


def normalise_rank(ranks):
    m = ranks - np.min(ranks) + 1
    n = len(m)
    R = 0.5 * (n**2 + n) / np.sum(m)
    return m * R
