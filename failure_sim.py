"""Measure transport failures: how often does an organ miss its time limit?

Simulates many transports with random delays and compares three ways of planning:
  A. Basic dispatch: cheapest option that fits the limit on the timetable,
     ignoring delay risk, never re-plans.
  B. Q-Organ Route plan: plans with the AI delay risk (same scoring the QAOA
     optimizes), planned once.
  C. Q-Organ Route + re-planning: like B, but if the jet reports a delay before
     departure, the plan is solved again.

Random events per transport (assumptions marked):
  - Jet mechanical/crew delay before departure: 10% chance, 1-3 h  (assumption)
  - Each leg can run late with its delay probability (commercial: AI model on
    BTS data; jet/ground: planning assumptions). Late legs draw an exponential
    delay with the leg's mean delay-if-late.
A failure is an organ arriving after its cold-ischemia limit.
Plans are found with the exact solver; QAOA returned the same optimal plans
in every recorded run (see results/).
"""
import itertools
import numpy as np

from scenario import ORGANS, MODES, LEGS, JET_CAPACITY
from qubo import leg_score

JET_DISRUPT_P = 0.10          # assumption
JET_DISRUPT_RANGE = (1.0, 3.0)  # hours, assumption


def plans():
    for combo in itertools.product(MODES, repeat=len(ORGANS)):
        if combo.count("jet") <= JET_CAPACITY:
            yield dict(zip(ORGANS, combo))


def with_jet_delay(legs, hours):
    new = dict(legs)
    for o in ORGANS:
        b, p, d, c = new[(o, "jet")]
        new[(o, "jet")] = (b + hours, p, d, c)
    return new


def best_plan(legs):
    """What the optimizer (QAOA / exact solver) picks: lowest total score."""
    return min(plans(), key=lambda pl: sum(leg_score(o, m, legs) for o, m in pl.items()))


def basic_plan(legs):
    """Timetable-only dispatcher: cheapest plan whose scheduled times fit, no risk."""
    def fits(pl):
        return all(legs[(o, m)][0] <= ORGANS[o][1] for o, m in pl.items())
    ok = [pl for pl in plans() if fits(pl)]
    return min(ok, key=lambda pl: sum(legs[(o, m)][3] for o, m in pl.items()))


def simulate(n=10000, seed=42, legs=LEGS):
    rng = np.random.default_rng(seed)
    policies = ["basic", "planner", "planner_replan"]
    fails = {k: {o: 0 for o in ORGANS} for k in policies}
    any_fail = {k: 0 for k in policies}
    spare = {k: [] for k in policies}
    replans = 0
    plan_basic, plan_ai = basic_plan(legs), best_plan(legs)

    for _ in range(n):
        jet_delay = rng.uniform(*JET_DISRUPT_RANGE) if rng.random() < JET_DISRUPT_P else 0.0
        # random lateness per leg, shared by all policies (fair comparison)
        late = {}
        for (o, m), (b, p, d, c) in legs.items():
            late[(o, m)] = rng.exponential(d) if rng.random() < p else 0.0

        replan = best_plan(with_jet_delay(legs, jet_delay)) if jet_delay > 0 else plan_ai
        if replan != plan_ai:
            replans += 1
        for k, pl in (("basic", plan_basic), ("planner", plan_ai), ("planner_replan", replan)):
            failed = False
            for o, m in pl.items():
                t = legs[(o, m)][0] + late[(o, m)] + (jet_delay if m == "jet" else 0.0)
                margin = ORGANS[o][1] - t
                spare[k].append(margin)
                if margin < 0:
                    fails[k][o] += 1
                    failed = True
            any_fail[k] += failed

    return {
        "n": n, "plans": {"basic": plan_basic, "planner": plan_ai},
        "replans": replans,
        "any_fail_rate": {k: any_fail[k] / n for k in policies},
        "organ_fail_rate": {k: {o: fails[k][o] / n for o in ORGANS} for k in policies},
    }


def report(n=10000, seed=42):
    from scenario import RISK_SOURCE
    r = simulate(n=n, seed=seed)
    print(f"Delay risk source: {RISK_SOURCE}")
    print(f"Transports simulated: {r['n']:,}  (jet delayed before departure in ~{JET_DISRUPT_P:.0%})")
    print(f"Basic dispatch plan:      {r['plans']['basic']}")
    print(f"Q-Organ Route plan:       {r['plans']['planner']}")
    print(f"Re-planned in {r['replans']:,} transports\n")
    names = {"basic": "Basic dispatch", "planner": "Q-Organ Route (plan once)",
             "planner_replan": "Q-Organ Route + re-planning"}
    print(f"{'Policy':30s} {'any organ late':>15s} {'heart':>8s} {'liver':>8s} {'kidney':>8s}")
    for k, name in names.items():
        o = r["organ_fail_rate"][k]
        print(f"{name:30s} {r['any_fail_rate'][k]:>14.2%} {o['heart']:>8.2%} {o['liver']:>8.2%} {o['kidney']:>8.2%}")


if __name__ == "__main__":
    report()
