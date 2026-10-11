"""AI layer: predict flight delay risk from real BTS on-time data.

Trains two gradient-boosted models on U.S. DOT/BTS Reporting Carrier On-Time
Performance data (December 2025, ~582k flights):
  1. classifier: P(flight arrives 15+ min late, or is cancelled/diverted)
  2. regressor:  expected delay in minutes, given that it is late

Honest test: train on Dec 1-21, test on Dec 22-31 (no future leakage).
The predictions for the scenario's commercial legs are written to
risk_table.py, which the quantum planner reads. Training runs offline;
Qollab only needs risk_table.py.

Usage: python ai_delay_model.py path/to/On_Time_..._2025_12.csv
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error

CANCEL_DELAY_MIN = 180  # assumption: a cancelled/diverted flight costs ~3 h to rebook
FEATURES = ["Origin", "Dest", "Reporting_Airline", "dep_hour", "DayOfWeek", "Distance"]
CATEGORICAL = ["Origin", "Dest", "Reporting_Airline"]

# scenario routes (commercial legs) and the departure we plan for
ROUTES = {"heart": "ATL", "liver": "CLT", "kidney": "MCO"}
ORIGIN = "MIA"
PLAN_HOUR = 14        # 2 PM departure
PLAN_DOW = 5          # Friday (BTS: 1=Mon ... 7=Sun)


def load(path):
    cols = ["FlightDate", "DayOfWeek", "Reporting_Airline", "Origin", "Dest",
            "CRSDepTime", "ArrDelay", "ArrDel15", "Cancelled", "Diverted", "Distance"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["dep_hour"] = (df["CRSDepTime"] // 100).clip(0, 23)
    df["disrupted"] = ((df["ArrDel15"] == 1) | (df["Cancelled"] == 1) | (df["Diverted"] == 1)).astype(int)
    df["delay_min"] = np.where(df["Cancelled"].eq(1) | df["Diverted"].eq(1),
                               CANCEL_DELAY_MIN, df["ArrDelay"].clip(lower=0))
    for c in CATEGORICAL:
        # the model handles up to 255 categories: keep the 250 busiest, group the rest
        top = df[c].value_counts().index[:250]
        df[c] = df[c].where(df[c].isin(top), "OTHER").astype("category")
    return df


def main(path):
    df = load(path)
    day = pd.to_datetime(df["FlightDate"]).dt.day
    train, test = df[day <= 21], df[day > 21]

    clf = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=300,
                                         learning_rate=0.1, random_state=0)
    clf.fit(train[FEATURES], train["disrupted"])
    p = clf.predict_proba(test[FEATURES])[:, 1]

    # baseline: historical average for each route (what a simple lookup table gives)
    route_rate = train.groupby(["Origin", "Dest"], observed=True)["disrupted"].mean()
    base = test.set_index(["Origin", "Dest"]).index.map(route_rate).to_numpy(dtype=float)
    base = np.where(np.isnan(base), train["disrupted"].mean(), base)

    late_tr, late_te = train[train.disrupted == 1], test[test.disrupted == 1]
    reg = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=300,
                                        learning_rate=0.1, random_state=0)
    reg.fit(late_tr[FEATURES], late_tr["delay_min"])
    d_pred = reg.predict(late_te[FEATURES])

    metrics = {
        "train_flights": int(len(train)), "test_flights": int(len(test)),
        "test_disruption_rate": round(float(test["disrupted"].mean()), 3),
        "classifier_auc": round(roc_auc_score(test["disrupted"], p), 3),
        "route_average_auc": round(roc_auc_score(test["disrupted"], base), 3),
        "classifier_brier": round(brier_score_loss(test["disrupted"], p), 4),
        "route_average_brier": round(brier_score_loss(test["disrupted"], base), 4),
        "delay_mae_min": round(mean_absolute_error(late_te["delay_min"], d_pred), 1),
        "delay_mae_route_avg_min": round(mean_absolute_error(
            late_te["delay_min"], np.full(len(late_te), late_tr["delay_min"].mean())), 1),
    }

    # predict the scenario's commercial legs: best (lowest-risk) carrier on each route
    table = {}
    for organ, dest in ROUTES.items():
        hist = df[(df.Origin == ORIGIN) & (df.Dest == dest)]
        rows = []
        for airline in hist["Reporting_Airline"].dropna().unique():
            x = pd.DataFrame([{"Origin": ORIGIN, "Dest": dest, "Reporting_Airline": airline,
                               "dep_hour": PLAN_HOUR, "DayOfWeek": PLAN_DOW,
                               "Distance": float(hist["Distance"].iloc[0])}])
            for c in CATEGORICAL:
                x[c] = pd.Categorical(x[c], categories=df[c].cat.categories)
            rows.append((float(clf.predict_proba(x[FEATURES])[0, 1]),
                         float(reg.predict(x[FEATURES])[0]), str(airline),
                         int((hist["Reporting_Airline"] == airline).sum())))
        rows = [r for r in rows if r[3] >= 30]  # carriers with enough flights on the route
        p_late, delay_min, airline, n = min(rows)
        table[organ] = {"route": f"{ORIGIN}-{dest}", "airline": airline, "flights_in_data": n,
                        "p_delay": round(p_late, 3), "delay_hours_if_late": round(delay_min / 60, 2)}

    with open("risk_table.py", "w") as f:
        f.write('"""Generated by ai_delay_model.py from BTS December 2025 data. Do not edit by hand."""\n')
        f.write(f"PLAN = {json.dumps({'origin': ORIGIN, 'dep_hour': PLAN_HOUR, 'day_of_week': PLAN_DOW})}\n")
        f.write(f"COMMERCIAL_RISK = {json.dumps(table, indent=4)}\n")
        f.write(f"MODEL_METRICS = {json.dumps(metrics, indent=4)}\n")

    print(json.dumps(metrics, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
