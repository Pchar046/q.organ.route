"""Entry point for Qollab: quantum + classical organ transport planning.

Both scenarios are optimized first, then sampled together in ONE job,
because Qollab allows one circuit job per run.

This effort is supported by Qollab & IonQ.
"""
import warnings
warnings.filterwarnings("ignore")

from scenario import LEGS, apply_disruption, RISK_SOURCE
from qubo import decode, is_feasible
from baseline import brute_force, cp_sat
from qaoa_solve import solve, sample_circuits, finish

REPS, ALPHA = 2, 0.2
MIXER = "xy"  # "xy" = XY mixer (one mode per organ guaranteed), "x" = standard QAOA

jet_delay = LEGS
for organ in ("heart", "liver", "kidney"):
    jet_delay = apply_disruption(jet_delay, organ, "jet", 2.5)

print(f"Commercial flight delay risk from: {RISK_SOURCE}")

scenarios = [("Nominal plan", LEGS), ("Disruption: jet delayed 2.5 h", jet_delay)]

# 1) classical answers + QAOA training (no quantum job yet)
runs = []
for title, legs in scenarios:
    bf_bits, bf_E, bf_t = brute_force(legs)
    r = solve(legs, reps=REPS, alpha=ALPHA, sample=False, mixer=MIXER)
    runs.append((title, legs, bf_bits, bf_E, bf_t, r))

# 2) one job for all circuits
counts_list, where = sample_circuits([r["circuit"] for *_, r in runs])

# 3) report
for (title, legs, bf_bits, bf_E, bf_t, r), counts in zip(runs, counts_list):
    finish(r, counts, where)
    print(f"\n=== {title} ===")
    print(f"Exact (brute force): {decode(bf_bits)}  E={bf_E:.3f}  ({bf_t*1000:.1f} ms)")
    cp = cp_sat(legs)
    if cp:
        print(f"OR-Tools CP-SAT:     {decode(cp[0])}  obj={cp[1]:.3f}  ({cp[2]*1000:.1f} ms)")
    else:
        print("OR-Tools CP-SAT:     skipped (not installed in this runtime)")
    p_opt = sum(p for E, p, b in r["samples"] if abs(E - bf_E) < 1e-6)
    p_feas = sum(p for E, p, b in r["samples"] if is_feasible(b))
    print(f"QAOA p={REPS}, mixer={MIXER}, CVaR alpha={ALPHA}, {r['num_qubits']} qubits, depth {r['depth']}")
    print(f"  sampled on: {where}")
    print(f"  best sampled plan: {decode(r['best_bits'])}  E={r['best_energy']:.3f}")
    print(f"  P(optimal)={p_opt:.3f}  P(feasible)={p_feas:.3f}  "
          f"(random guess P(optimal)={1/2**r['num_qubits']:.4f})")
    print(f"  found optimum: {abs(r['best_energy'] - bf_E) < 1e-6}")

# 4) transport failures: how often an organ misses its time limit (classical simulation, no quantum job)
from failure_sim import report as failure_report
print("\n=== Transport failure simulation ===")
failure_report()
