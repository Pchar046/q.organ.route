"""Build the QUBO for organ transport mode selection and convert it to an Ising Hamiltonian."""
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from scenario import ORGANS, MODES, LEGS, JET_CAPACITY, variables

W_ISCHEMIA = 10.0   # weight on fraction of ischemia window used
W_COST = 0.1        # weight per $k
LATE_PENALTY = 50.0 # expected arrival past the organ's limit
P_ONEHOT = 40.0     # each organ gets exactly one mode
P_CAPACITY = 40.0   # jet carries at most one organ


def leg_score(organ, mode, legs=LEGS):
    base, p, delay, cost = legs[(organ, mode)]
    expected = base + p * delay
    limit = ORGANS[organ][1]
    score = W_ISCHEMIA * expected / limit + W_COST * cost
    if expected > limit:
        score += LATE_PENALTY
    return score


def build_qubo(legs=LEGS):
    """Return (Q, offset) with energy = x^T Q x + offset, Q upper-triangular."""
    var = variables()
    idx = {v: i for i, v in enumerate(var)}
    n = len(var)
    Q = np.zeros((n, n))
    offset = 0.0

    # objective (linear terms on the diagonal)
    for (o, m), i in idx.items():
        Q[i, i] += leg_score(o, m, legs)

    # one-hot per organ: P * (sum x - 1)^2 = P * (-sum x + 2 sum_{i<j} x_i x_j + 1)
    for o in ORGANS:
        ids = [idx[(o, m)] for m in MODES]
        for i in ids:
            Q[i, i] -= P_ONEHOT
        for i, j in itertools.combinations(ids, 2):
            Q[i, j] += 2 * P_ONEHOT
        offset += P_ONEHOT

    # jet capacity (<= 1): penalize any pair of organs on the jet
    assert JET_CAPACITY == 1
    jets = [idx[(o, "jet")] for o in ORGANS]
    for i, j in itertools.combinations(jets, 2):
        Q[i, j] += P_CAPACITY

    return Q, offset


def energy(x, Q, offset):
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + offset)


def qubo_to_ising(Q, offset):
    """x_i = (1 - z_i)/2. Returns SparsePauliOp H and constant so <H> + const = QUBO energy."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = {}
    const = offset
    for i in range(n):
        q = Q[i, i]
        const += q / 2
        h[i] -= q / 2
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j]
            if q == 0:
                continue
            const += q / 4
            h[i] -= q / 4
            h[j] -= q / 4
            J[(i, j)] = J.get((i, j), 0) + q / 4

    terms = []
    for i in range(n):
        if h[i]:
            terms.append(("Z", [i], h[i]))
    for (i, j), v in J.items():
        terms.append(("ZZ", [i, j], v))
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    return H, const


def decode(bits):
    """bits[i] for qubit i -> {organ: mode} (None if not one-hot)."""
    var = variables()
    plan = {}
    for o in ORGANS:
        chosen = [m for (oo, m), b in zip(var, bits) if oo == o and b]
        plan[o] = chosen[0] if len(chosen) == 1 else None
    return plan


def is_feasible(bits):
    plan = decode(bits)
    if any(v is None for v in plan.values()):
        return False
    return sum(1 for v in plan.values() if v == "jet") <= JET_CAPACITY
