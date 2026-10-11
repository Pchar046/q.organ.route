"""QAOA solver (Qiskit) for the organ transport QUBO.

Supports a plain expectation objective or CVaR (average of the best alpha
fraction of outcomes), which tends to help QAOA on constrained problems.
"""
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz, StatePreparation
from qiskit.quantum_info import Statevector, SparsePauliOp
try:
    from qiskit_aer import AerSimulator
except ImportError:  # e.g. browser runtime without Aer
    AerSimulator = None

from qubo import build_qubo, qubo_to_ising, energy
from scenario import LEGS


def _find_platform_backend():
    """Return the `backend` Qollab pre-creates, if any (checked in main and builtins)."""
    import builtins
    import __main__
    for place in (__main__, builtins):
        b = getattr(place, "backend", None)
        if b is not None and hasattr(b, "run"):
            return b
    return None


def _bits_from_key(key):
    # Qiskit keys are little-endian: qubit 0 is the rightmost character
    return [int(c) for c in reversed(key)]


def _cvar(probs, energies, alpha):
    order = np.argsort(energies)
    total, acc = 0.0, 0.0
    for k in order:
        take = min(probs[k], alpha - acc)
        if take <= 0:
            break
        total += take * energies[k]
        acc += take
    return total / alpha


def _w_state(k):
    """Equal superposition of the k one-hot states (one organ, one transport mode)."""
    vec = np.zeros(2 ** k)
    for i in range(k):
        vec[1 << i] = 1 / np.sqrt(k)
    qc = QuantumCircuit(k)
    qc.append(StatePreparation(vec), range(k))
    return qc


def xy_parts(n, group_size=3):
    """Initial state and ring XY mixer that keep exactly one mode per organ.

    The XY mixer swaps a 1 between qubits of the same organ, so it never
    creates a plan with zero or two modes for one organ.
    """
    init = QuantumCircuit(n)
    terms = []
    for g in range(0, n, group_size):
        qs = list(range(g, g + group_size))
        init.compose(_w_state(group_size), qubits=qs, inplace=True)
        for a in range(group_size):
            i, j = qs[a], qs[(a + 1) % group_size]
            terms.append(("XX", [i, j], 0.5))
            terms.append(("YY", [i, j], 0.5))
    return init, SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def solve(legs=LEGS, reps=2, alpha=1.0, restarts=5, shots=4096, seed=7, noise_model=None,
          sample=True, mixer="x"):
    """mixer='x': standard QAOA (constraints only as penalties).
    mixer='xy': XY mixer + one-hot start state, so every sample has one mode per organ."""
    Q, off = build_qubo(legs)
    H, _ = qubo_to_ising(Q, off)
    n = Q.shape[0]
    scale = np.max(np.abs(H.coeffs))
    if mixer == "xy":
        init, mix = xy_parts(n)
        raw = QAOAAnsatz(H / scale, reps=reps, initial_state=init, mixer_operator=mix)
    else:
        raw = QAOAAnsatz(H / scale, reps=reps)
    # decompose once into native gates so each optimizer step is cheap
    ansatz = transpile(raw, basis_gates=["h", "rz", "rx", "ry", "rzz", "cx"], optimization_level=1)

    # exact energy for every basis state, indexed by integer key
    all_E = np.array([energy(_bits_from_key(format(k, f"0{n}b")), Q, off) for k in range(2 ** n)])

    def objective(theta):
        probs = Statevector(ansatz.assign_parameters(theta)).probabilities()
        if alpha >= 1.0:
            return float(probs @ all_E)
        return _cvar(probs, all_E, alpha)

    rng = np.random.default_rng(seed)
    best = None
    t = time.perf_counter()
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi, ansatz.num_parameters)
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    opt_time = time.perf_counter() - t

    qc = ansatz.assign_parameters(best.x)
    result = {
        "circuit": qc,
        "Q": Q,
        "offset": off,
        "params": best.x,
        "objective": best.fun,
        "opt_time": opt_time,
        "num_qubits": n,
        "depth": qc.depth(),
        "all_energies": all_E,
        "mixer": mixer,
    }
    if sample:
        counts, where = sample_circuits([qc], shots=shots, seed=seed, noise_model=noise_model)
        finish(result, counts[0], where)
    return result


def sample_circuits(circuits, shots=4096, seed=7, noise_model=None):
    """Sample several circuits in ONE job (Qollab allows one job per run).

    Uses the Qollab-provided backend if present, else Aer, else the statevector.
    Returns (list of counts dicts, name of where it ran).
    """
    platform_backend = _find_platform_backend()
    measured = []
    for qc in circuits:
        m = qc.copy()
        m.measure_all()
        measured.append(m)

    if platform_backend is not None:
        # IonQ via Qollab takes ONE circuit per job and ONE job per run, so we
        # place the circuits side by side on separate qubits and split the counts.
        sizes = [qc.num_qubits for qc in circuits]
        wide = QuantumCircuit(sum(sizes))
        start = 0
        for qc, size in zip(circuits, sizes):
            wide.compose(qc, qubits=range(start, start + size), inplace=True)
            start += size
        wide.measure_all()
        tqc = transpile(wide, platform_backend, optimization_level=1)
        counts = platform_backend.run(tqc, shots=shots).result().get_counts()
        return _split_counts(counts, sizes), getattr(platform_backend, "name", "platform backend")
    if AerSimulator is not None:
        sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        res = sim.run(transpile(measured, sim, optimization_level=1), shots=shots).result()
        return [res.get_counts(i) for i in range(len(measured))], "aer_simulator"
    out = []
    for qc in circuits:
        sv = Statevector(qc)
        sv.seed(seed)
        out.append(sv.sample_counts(shots))
    return out, "statevector"


def _split_counts(counts, sizes):
    """Split counts of a side-by-side circuit into one counts dict per sub-circuit."""
    parts = [dict() for _ in sizes]
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]  # qubit 0 first
        start = 0
        for i, size in enumerate(sizes):
            sub = bits[start:start + size][::-1]  # back to Qiskit's little-endian key
            parts[i][sub] = parts[i].get(sub, 0) + c
            start += size
    return parts


def finish(result, counts, where):
    """Attach sampled plans (sorted best first) to a solve() result."""
    Q, off = result["Q"], result["offset"]
    total = sum(counts.values())
    samples = []
    for key, c in counts.items():
        bits = _bits_from_key(key)
        samples.append((energy(bits, Q, off), c / total, bits))
    samples.sort(key=lambda s: s[0])
    result.update(samples=samples, best_bits=samples[0][2],
                  best_energy=samples[0][0], sampled_on=where)
    return result
