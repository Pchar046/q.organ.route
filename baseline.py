"""Classical baselines: exhaustive search over the QUBO, and OR-Tools CP-SAT on the original model."""
import itertools
import time
try:
    from ortools.sat.python import cp_model
except ImportError:  # e.g. browser runtime without OR-Tools
    cp_model = None

from scenario import ORGANS, MODES, LEGS, JET_CAPACITY
from qubo import build_qubo, energy, leg_score, variables


def brute_force(legs=LEGS):
    Q, off = build_qubo(legs)
    n = Q.shape[0]
    t = time.perf_counter()
    best = min(itertools.product([0, 1], repeat=n), key=lambda x: energy(x, Q, off))
    return list(best), energy(best, Q, off), time.perf_counter() - t


def cp_sat(legs=LEGS, scale=1000):
    if cp_model is None:
        return None
    m = cp_model.CpModel()
    x = {(o, md): m.NewBoolVar(f"{o}_{md}") for o in ORGANS for md in MODES}
    for o in ORGANS:
        m.AddExactlyOne(x[(o, md)] for md in MODES)
    m.Add(sum(x[(o, "jet")] for o in ORGANS) <= JET_CAPACITY)
    m.Minimize(sum(int(round(scale * leg_score(o, md, legs))) * x[(o, md)]
                   for o in ORGANS for md in MODES))
    solver = cp_model.CpSolver()
    t = time.perf_counter()
    solver.Solve(m)
    elapsed = time.perf_counter() - t
    plan = {o: md for (o, md), v in x.items() if solver.Value(v)}
    bits = [1 if plan[o] == md else 0 for (o, md) in variables()]
    return bits, solver.ObjectiveValue() / scale, elapsed
