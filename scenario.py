"""Scenario: one donor in Miami, three organs, three transport modes.

All numbers are illustrative planning values, not clinical data, except the
commercial-flight delay risk, which comes from the AI model (risk_table.py).
"""

ORGANS = {
    # name: (destination, max cold-ischemia hours)
    "heart":  ("Atlanta",   4.0),
    "liver":  ("Charlotte", 12.0),
    "kidney": ("Orlando",   24.0),
}

MODES = ["jet", "commercial", "ground"]

# (organ, mode) -> (base hours, delay probability, delay hours if late, cost $k)
LEGS = {
    ("heart", "jet"):         (2.0, 0.05, 0.5, 18.0),
    ("heart", "commercial"):  (3.5, 0.25, 1.5, 3.0),
    ("heart", "ground"):      (10.0, 0.10, 1.0, 1.5),
    ("liver", "jet"):         (2.3, 0.05, 0.5, 20.0),
    ("liver", "commercial"):  (4.0, 0.25, 1.5, 3.0),
    ("liver", "ground"):      (11.0, 0.10, 1.0, 1.8),
    ("kidney", "jet"):        (1.5, 0.05, 0.5, 12.0),
    ("kidney", "commercial"): (2.8, 0.20, 1.5, 2.5),
    ("kidney", "ground"):     (3.8, 0.10, 0.5, 0.8),
}

JET_CAPACITY = 1  # only one charter jet available

# AI layer: replace the commercial-flight delay risk with predictions from the
# delay model trained on real BTS data (see ai_delay_model.py -> risk_table.py).
# Jet and ground legs keep planning assumptions (no public data for them).
try:
    from risk_table import COMMERCIAL_RISK
    for _organ, _risk in COMMERCIAL_RISK.items():
        _base, _p, _d, _cost = LEGS[(_organ, "commercial")]
        LEGS[(_organ, "commercial")] = (_base, _risk["p_delay"], _risk["delay_hours_if_late"], _cost)
    RISK_SOURCE = "AI model on BTS Dec 2025 data"
except ImportError:
    RISK_SOURCE = "planning assumptions"


def variables():
    """Ordered list of (organ, mode) decision variables -> qubit index."""
    return [(o, m) for o in ORGANS for m in MODES]


def apply_disruption(legs, organ, mode, extra_hours):
    """Return a copy of legs with extra delay on one leg (live re-planning)."""
    new = dict(legs)
    base, p, d, c = new[(organ, mode)]
    new[(organ, mode)] = (base + extra_hours, p, d, c)
    return new
