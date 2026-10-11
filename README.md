# Q-Organ Route

**Quantum + AI planning for donor organ transport.**

This effort is supported by Qollab & IonQ.

## The problem

Donor organs are sometimes lost because transport fails, not for medical reasons. A flight is delayed, the plan breaks, and the organ runs past its safe time outside the body (its cold-ischemia limit). A heart has about 4 hours. A liver has about 12. A kidney has about 24.

This is a recognized problem. HRSA asked MITRE to study organ transportation logistics and recommend standards to reduce organs going unused for logistics reasons, and HRSA says it is building the data foundation to better understand transport failures.

Planning organ transport is hard because each organ has its own clock, vehicles are limited, and delays happen after the plan is made. Classical models for multi-organ routing exist. Quantum optimization has been applied to organ *matching* (kidney exchange). We did not find prior work applying it to organ *transport*. That is the gap this project explores.

## What it does

**AI predicts, quantum decides, humans approve.**

1. **AI predicts delay risk.** A gradient-boosted model trained on 582,304 real U.S. flights (BTS, December 2025) predicts how likely each commercial flight is to be late, and by how much.
2. **Quantum decides the plan.** For each organ, the planner chooses charter jet, commercial flight, or ground transport, so every organ arrives inside its limit. The choice is written as a QUBO and solved with QAOA in Qiskit, run on IonQ's Aria 1 noise-model simulator through Qollab.
3. **Re-planning.** When the jet reports a delay, the plan is solved again.
4. **Failure tracking.** A simulation of 10,000 transports measures how often an organ misses its limit, and the simulator app logs every delay, re-plan, arrival and missed limit.
5. **A human approves.** The system suggests a plan. A transport coordinator makes the final call.

## Results

All numbers come from saved runs in the `results/` folder.

### Fewer organs miss their time limit

10,000 simulated transports, same random delays for every method:

| How the plan is made | Any organ late |
|---|---|
| Basic dispatch (cheapest option on the timetable, ignores delay risk) | 12.3% |
| Q-Organ Route (AI risk, plan once) | 5.3% |
| **Q-Organ Route + re-planning** | **1.0%** |

Across 10 runs of 10,000 transports, the averages were 12.5%, 5.1% and 1.0%. Almost all failures are hearts, which have the shortest limit. Proof run on Qollab: `results/qollab_full_run_ionq_aria1.txt`.

### Quantum: how often QAOA returns the best plan

Share of 4,096 samples that were the exact best plan (normal day / jet delayed 2.5 h). Random guessing gives 0.2%.

| Method | Simulator | Normal day | Jet delayed |
|---|---|---|---|
| Standard QAOA | Ideal (Qollab built-in) | 4.0% | 4.8% |
| Standard QAOA | IonQ Aria 1 noise model | 3.1% | 3.4% |
| XY-mixer QAOA | Ideal (Qiskit Aer) | 25.5% | 21.8% |
| XY-mixer QAOA | IonQ Aria 1 noise model | 9.4% | 8.5% |
| **XY-mixer QAOA + AI risk** | **IonQ Aria 1 noise model** | **10.6%** | **9.9%** |

QAOA found the best plan in every run. The XY mixer is about 4 to 6 times better than standard QAOA on an ideal simulator, and about 2.5 to 3 times better under IonQ noise.

### AI: delay prediction

Trained on Dec 1–21, tested on 186,734 flights from Dec 22–31 that the model never saw.

| | AI model | Route average |
|---|---|---|
| Ranking risky flights (AUC, higher is better) | 0.645 | 0.587 |
| Probability error (Brier, lower is better) | 0.1994 | 0.2075 |
| Delay length error (minutes, lower is better) | 57.5 | 60.8 |

The gain is modest. The model has no weather data yet.

## How it works

```
 BTS flight data ──► AI delay model ──► risk_table.py
                                            │
 scenario (organs, limits, vehicles) ◄──────┘
        │
        ▼
     QUBO  (cost = time used vs. limit + cost + penalties)
        │
        ├──► Classical check: brute force, OR-Tools CP-SAT
        │
        └──► QAOA (Qiskit): XY mixer + CVaR
                 │  trained on a statevector simulator
                 ▼
             IonQ Aria 1 noise model (one 18-qubit job: both scenarios side by side)
                 │
                 ▼
             Best plan ──► failure simulation ──► simulator app + transport log
```

**The quantum part, in plain words:**

- **9 qubits.** One qubit per organ-and-mode choice: 3 organs × 3 modes.
- **QUBO.** The plan's score rewards keeping each organ well inside its limit and penalizes cost, late arrivals, giving an organ more or fewer than one mode, and putting two organs on the one jet.
- **XY mixer.** Standard QAOA can wander into impossible plans. The XY mixer only swaps an organ between its own modes, so in an ideal run every sample gives each organ exactly one mode.
- **CVaR.** Training focuses on the best 20% of outcomes instead of the average, which helps on constrained problems.
- **One job per run.** Qollab allows one circuit job per run, so both scenarios are placed side by side in one 18-qubit circuit and the results are split afterward.

## How to run

### On Qollab

1. Open the project and click **Run Project**.
2. Pick a backend: **Built-in simulator** (ideal) or **IonQ Aria 1** (noise model).
3. Click **Run**. It takes a few minutes. The output shows the AI risk source, both scenarios, and the failure simulation.

`requirements.txt` lists only `scipy`. Qiskit and NumPy are already included on Qollab. OR-Tools does not install there, so that check is skipped automatically.

### On your own computer

```
pip install -r requirements-full.txt scikit-learn pandas
python main.py              # full pipeline (ideal Aer simulator, OR-Tools check included)
python failure_sim.py       # failure simulation only
python ai_delay_model.py "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_12.csv"
```

The flight data is free from the [BTS On-Time download page](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) (choose December 2025, Prezipped File).

### Simulator app

`app/index.html` is a self-contained page. Open it in a browser. Move the delay sliders to watch the plan re-solve, press **Play transport** to animate the trip, and press **Simulate 10,000 transports** to measure failures. The quantum numbers in the app are from the recorded IonQ runs. A web page can't call IonQ live, so the sliders check each new plan with an exact solver, which confirms the same answers QAOA found.

Accessibility: text meets color-contrast guidelines in light and dark mode, every control works by keyboard with a visible focus outline, the map has alt text, and status is shown in words as well as color ("Within limit", "Tight", "Over limit"). Animation is skipped for people who turn on reduced motion.

## Files

| File | What it does |
|---|---|
| `main.py` | Runs everything (entry point on Qollab) |
| `scenario.py` | Organs, limits, transport options, AI risk hookup |
| `qubo.py` | Builds the QUBO and converts it to a quantum Hamiltonian |
| `qaoa_solve.py` | QAOA with XY mixer and CVaR; one-job sampling on IonQ |
| `baseline.py` | Classical checks: brute force and OR-Tools |
| `ai_delay_model.py` | Trains the delay model on BTS data |
| `risk_table.py` | The model's predictions, used by the planner |
| `failure_sim.py` | Simulates 10,000 transports and counts missed limits |
| `app/index.html` | Interactive simulator with transport log |
| `results/` | Saved output from every run |

## Honest limits

- **Small problem.** 3 organs and 9 qubits. Classical solvers answer this in milliseconds. The point is to show the quantum method works on this problem and is ready to scale as hardware grows, not to claim a speed advantage today.
- **Illustrative numbers.** Travel times, costs, the jet's delay odds, and the ground options are planning assumptions, not clinical data. Only the commercial-flight risk comes from real data.
- **Simulators, not a real QPU.** Runs used IonQ's Aria 1 noise model.
- **Noise breaks some guarantees.** Under noise, some samples give an organ zero or two modes. QAOA still returned the best plan in every run.
- **Assumed jet disruptions.** In the failure simulation, 10% of transports have a 1–3 hour jet delay. That rate is an assumption.
- **Not medical advice.** This is a research prototype.

## Future work

- Larger problems: several donors, more organs and vehicles, timed handoffs.
- Run on IonQ hardware, and sweep circuit depth under noise.
- Add weather and real-time flight status to the AI model.
- Live re-planning from coordinator messages.
- Tamper-proof, blockchain-style chain-of-custody records for organ transport.
- The same engine for other urgent medical shipments: vaccines, cell therapies, scarce medications.

## Sources

- HRSA, [Organ transplant transportation logistics: vendor summary (MITRE)](https://www.hrsa.gov/sites/default/files/hrsa/optn/vendor-summary-organ-transplant-transportation-logistics.pdf)
- HRSA, [OPTN modernization update, August 2026](https://www.hrsa.gov/optn-modernization/updates/august-2026)
- U.S. DOT Bureau of Transportation Statistics, [Reporting Carrier On-Time Performance (1987–present)](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD)
- Hadfield et al., "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz," *Algorithms*, 2019 (XY mixers)
- Barkoutsos et al., "Improving Variational Quantum Optimization using CVaR," *Quantum*, 2020
- Prior classical multi-organ transport work reviewed: [transplant organ transportation with a bear hibernation algorithm (2020)](https://www.academia.edu/102078671/An_optimization_approach_for_transportation_problem_of_transplant_organ_using_self_adaptive_bear_hibernation_algorithm); [organ transplant logistics metaheuristic (Springer, 2025)](https://link.springer.com/article/10.1007/s10696-025-09625-6)
- [IonQ Qiskit provider](https://docs.ionq.com/sdks/qiskit) and [IonQ noise models](https://docs.ionq.com/guides/simulation-with-noise-models)

## Disclosure

**Libraries:** Qiskit, qiskit-ionq, SciPy, NumPy, scikit-learn, pandas, OR-Tools (local check only), Qiskit Aer (local only). Fonts in the simulator app: Barlow Condensed, Source Sans 3 and JetBrains Mono from Google Fonts.

**Templates:** none. The project was built from scratch during the hackathon window.

**AI assistance:** This project was built with help from Claude (Anthropic). At my direction, Claude researched the literature, wrote the code,  debugged most errors, suggested scenario values. I chose the problem, decided what to research, directed the project, made the design decisions, and ran and verified all results on Qollab and IonQ's simulators.

## License

MIT License. See `LICENSE`.

Author: Patrick Charles (Qollab: Charpat160)
