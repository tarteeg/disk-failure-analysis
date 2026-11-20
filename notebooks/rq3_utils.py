"""Helpers for RQ3: probability calibration, cost model and decision thresholds.

Only numpy, pandas and matplotlib are used (no scikit-learn), so nothing new
is needed in requirements.txt.

Expected input: the daily risk scores of the classification model (GBDT, H=14)
exported by part 9.1 of the notebook, one Parquet file per split
(`gbdt_h14_daily_scores_fold2_val.parquet` for validation,
`gbdt_h14_daily_scores_fold3_test_panel.parquet` for test), with the columns of
`PANEL_COLUMNS`. Healthy disks are a random drive sample, so every statistic is
weighted by `pop_weight`.

Protocol (no leakage):
    1. fit the calibrator and pick thresholds on the validation split only;
    2. report Brier score, reliability and costs on the test split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["serial_number", "date", "score", "y_14", "days_to_failure", "ep_outcome", "pop_weight"]

# Cost ratios c_FN / c_FP: an unplanned failure costs `ratio` times an
# unnecessary replacement (c_FP is the unit cost).
COST_RATIOS = [1, 2, 5, 10, 20, 50]


# ==========================================================================
# Input
# ==========================================================================
def load_panel(path_or_df) -> pd.DataFrame:
    """Load and validate a score panel (Parquet path read with DuckDB, or DataFrame).

    The CSV exports are not accepted: they lack `pop_weight`, and the test CSV
    drops healthy disks with zero critical counters, which would bias every
    weighted statistic.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        if not str(path_or_df).endswith(".parquet"):
            raise ValueError("Use the .parquet export, the .csv files lack pop_weight")
        import duckdb

        df = duckdb.connect().execute(f"SELECT * FROM read_parquet('{path_or_df}')").df()
    missing = set(PANEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df["score"].between(0, 1).all():
        raise ValueError("score must be a probability in [0, 1]")
    if not set(df["y_14"].unique()) <= {0, 1}:
        raise ValueError("y_14 must be 0 or 1")
    if not (df["pop_weight"] > 0).all():
        raise ValueError("pop_weight must be positive")
    return df


# ==========================================================================
# Calibration
# ==========================================================================
# The model was trained with negatives downsampled (50 negatives per positive),
# so its raw scores are inflated: on the real panels the weighted mean score is
# about 1.2% for a weighted prevalence of 0.16% to 0.19%. No analytic prior
# shift correction is used. It would need the exact negative keep rate of the
# training set, which is not exported with the scores. Instead the isotonic
# calibrator is fitted on the validation panel with `pop_weight`, which maps the
# scores to the true population frequency and absorbs the downsampling bias.

class IsotonicCalibrator:
    """Isotonic regression (pool adjacent violators), monotone and non-parametric.

    Preferred over Platt scaling here because the validation set is large
    (millions of drive-days) and the score distribution is very skewed.
    """

    def fit(self, scores, y, sample_weight=None):
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        order = np.argsort(scores, kind="stable")
        s, yy, ww = scores[order], y[order], w[order]

        # Merge ties first so that equal scores get one value.
        uniq, start = np.unique(s, return_index=True)
        wsum = np.add.reduceat(ww, start)
        ysum = np.add.reduceat(yy * ww, start)

        # Pool adjacent violators on blocks (mean, weight, last unique index).
        means: list[float] = []
        weights: list[float] = []
        counts: list[int] = []
        for m, wt in zip(ysum / wsum, wsum):
            means.append(m)
            weights.append(wt)
            counts.append(1)
            while len(means) > 1 and means[-2] > means[-1]:
                wt_new = weights[-2] + weights[-1]
                m_new = (means[-2] * weights[-2] + means[-1] * weights[-1]) / wt_new
                c_new = counts[-2] + counts[-1]
                means[-2:], weights[-2:], counts[-2:] = [m_new], [wt_new], [c_new]

        fitted = np.repeat(means, counts)
        self.x_ = uniq
        self.y_ = fitted
        return self

    def predict(self, scores):
        return np.interp(np.asarray(scores, dtype=float), self.x_, self.y_)


# ==========================================================================
# Calibration metrics
# ==========================================================================
def brier_score(p, y, sample_weight=None) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.average((p - y) ** 2, weights=sample_weight))


def reliability_curve(p, y, n_bins: int = 15, sample_weight=None) -> pd.DataFrame:
    """Reliability table with quantile bins (equal counts), robust to skew.

    Columns: mean_pred, frac_pos, weight. ECE is the weighted mean of
    |mean_pred - frac_pos|.
    """
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:  # constant probabilities: a single point, no curve
        return pd.DataFrame(columns=["mean_pred", "frac_pos", "weight"], dtype=float)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "mean_pred": np.average(p[m], weights=w[m]),
                    "frac_pos": np.average(y[m], weights=w[m]),
                    "weight": w[m].sum(),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    gap = (curve["mean_pred"] - curve["frac_pos"]).abs()
    return float(np.average(gap, weights=curve["weight"]))


# ==========================================================================
# Cost model and thresholds
# ==========================================================================
def elkan_threshold(c_fp: float, c_fn: float) -> float:
    """Theoretical threshold for calibrated probabilities (Elkan 2001)."""
    return c_fp / (c_fp + c_fn)


def cost_at_threshold(p, y, threshold: float, c_fp: float, c_fn: float, sample_weight=None) -> dict:
    """Total cost of alerting on every drive-day with p >= threshold."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    alert = p >= threshold
    fp = w[alert & (y == 0)].sum()
    fn = w[~alert & (y == 1)].sum()
    return {"fp": fp, "fn": fn, "cost": c_fp * fp + c_fn * fn}


def empirical_optimal_threshold(p, y, c_fp: float, c_fn: float, sample_weight=None) -> dict:
    """Threshold that minimizes the total cost, found by one sweep over the scores."""
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    order = np.argsort(-p, kind="stable")
    ps, ys, ws = p[order], y[order], w[order]

    # Alerting on the k highest scores: FP = weight of negatives among them,
    # FN = weight of positives left out. Only cut between distinct scores.
    fp_cum = np.cumsum(ws * (ys == 0))
    pos_total = (ws * (ys == 1)).sum()
    fn_after = pos_total - np.cumsum(ws * (ys == 1))
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]

    cost = c_fp * fp_cum + c_fn * fn_after
    cost = np.where(last_of_tie, cost, np.inf)
    k = int(np.argmin(cost))
    cost_none = c_fn * pos_total  # alert nobody
    if cost_none <= cost[k]:
        return {"threshold": np.inf, "cost": float(cost_none), "n_alerts_weight": 0.0}
    return {"threshold": float(ps[k]), "cost": float(cost[k]), "n_alerts_weight": float(ws[: k + 1].sum())}


def threshold_table(p_cal, y, ratios=COST_RATIOS, sample_weight=None, p_test=None, y_test=None, w_test=None):
    """One row per cost ratio: theoretical vs empirical threshold and costs.

    Thresholds are chosen on (p_cal, y) = validation. If (p_test, y_test) is
    given, the cost of each threshold is also measured on it.
    """
    rows = []
    for r in ratios:
        c_fp, c_fn = 1.0, float(r)
        t_th = elkan_threshold(c_fp, c_fn)
        emp = empirical_optimal_threshold(p_cal, y, c_fp, c_fn, sample_weight)
        row = {
            "ratio_fn_over_fp": r,
            "threshold_elkan": t_th,
            "threshold_empirical": emp["threshold"],
            "val_cost_elkan": cost_at_threshold(p_cal, y, t_th, c_fp, c_fn, sample_weight)["cost"],
            "val_cost_empirical": emp["cost"],
        }
        if p_test is not None:
            for name, t in (("elkan", t_th), ("empirical", emp["threshold"])):
                c = cost_at_threshold(p_test, y_test, t, c_fp, c_fn, w_test)
                row[f"test_cost_{name}"] = c["cost"]
                row[f"test_fp_{name}"] = c["fp"]
                row[f"test_fn_{name}"] = c["fn"]
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Disk-level decisions (one alert per disk)
# ==========================================================================
# The cost functions above count every drive-day. Operationally a disk is
# replaced once, so here the unit is the disk:
#   * a healthy disk alerted at least once is ONE false positive (c_FP);
#   * a failed disk is "planned" if its first alert comes at least L days before
#     the failure; otherwise (late alert, or no alert) it is an unplanned
#     failure (c_FN). A late alert still ends in a failure, not in an
#     unnecessary replacement, so it is not a false positive.
# Failed disks have weight 1; healthy disks carry `pop_weight` (inverse of the
# sampling rate of healthy disks).
#
# Equivalent view used for the threshold sweep: a disk is alerted in time at
# threshold t if and only if the maximum score over its rows at least L days
# before the failure is >= t (for a healthy disk: the maximum over all rows).
# This reduces the problem to one score per disk (`disk_table`), so the
# drive-day sweep functions above can be reused.

DISK_INPUT_COLUMNS = ["serial_number", "date", "ep_outcome", "days_to_failure"]


def _prepare_rows(df: pd.DataFrame, window_end, p_col: str, weight_col: str) -> pd.DataFrame:
    """Normalise columns and drop disks that fail after `window_end`.

    Disks failing after the observation window are removed: their rows stop at
    `window_end`, so a missing alert would be an artefact of the window and not a
    real miss. This keeps the failure count equal to the one of the RQ2 replay
    (785 failures on the test window).
    """
    missing = set(DISK_INPUT_COLUMNS + [p_col, weight_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    d = df[DISK_INPUT_COLUMNS + [p_col, weight_col]].copy()
    d.columns = ["serial_number", "date", "ep_outcome", "dtf", "p", "w"]
    d["date"] = pd.to_datetime(d["date"])
    d["failed"] = d["ep_outcome"] == "failed"
    d["fail_date"] = d["date"] + pd.to_timedelta(d["dtf"], unit="D")
    late = d.loc[d["failed"] & (d["fail_date"] > pd.Timestamp(window_end)), "serial_number"].unique()
    return d[~d["serial_number"].isin(late)]


def _disk_meta(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("serial_number").agg(
        failed=("failed", "max"), w=("w", "first"), fail_date=("fail_date", "max")
    )


def disk_table(df, window_end, L: int = 7, p_col: str = "p", weight_col: str = "pop_weight") -> pd.DataFrame:
    """One row per disk: `score` (maximum score usable at least L days ahead).

    Columns: failed (bool), weight, score, plannable. A failed disk whose rows
    all lie within L days of the failure cannot be alerted in time
    (`plannable` False, score NaN): it is an unplanned failure at any threshold.
    """
    d = _prepare_rows(df, window_end, p_col, weight_col)
    usable = d[~d["failed"] | (d["dtf"] >= L)]
    out = _disk_meta(d).join(usable.groupby("serial_number")["p"].max().rename("score"))
    out["plannable"] = out["score"].notna()
    return out.rename(columns={"w": "weight"})[["failed", "weight", "score", "plannable"]]


def disk_outcomes(
    df, threshold: float, window_end, L: int = 7, c_fp: float = 1.0, c_fn: float = 1.0,
    p_col: str = "p", weight_col: str = "pop_weight",
) -> dict:
    """Disk-level outcome at one threshold, with weighted counts.

    planned: failed, first alert at least L days ahead; late: failed, first alert
    less than L days ahead; missed: failed, never alerted; fp: healthy disks
    alerted at least once. cost = c_fp * fp + c_fn * (late + missed).
    """
    d = _prepare_rows(df, window_end, p_col, weight_col)
    meta = _disk_meta(d)
    first = d[d["p"] >= threshold].groupby("serial_number")["date"].min().rename("first_alert")
    meta = meta.join(first)
    f = meta[meta["failed"]]
    lead = (f["fail_date"] - f["first_alert"]).dt.days
    alerted = f["first_alert"].notna()
    planned = float(f["w"][alerted & (lead >= L)].sum())
    late = float(f["w"][alerted & (lead < L)].sum())
    missed = float(f["w"][~alerted].sum())
    h = meta[~meta["failed"]]
    fp = float(h["w"][h["first_alert"].notna()].sum())
    return {
        "planned": planned,
        "late": late,
        "missed": missed,
        "fp": fp,
        "fn": late + missed,
        "cost": c_fp * fp + c_fn * (late + missed),
        "median_lead_days": float(lead[alerted].median()) if alerted.any() else float("nan"),
        "n_failed": int(len(f)),
    }


def disk_threshold_table(
    val_df, test_df, val_end, test_end, L: int = 7, ratios=COST_RATIOS,
    p_col: str = "p", weight_col: str = "pop_weight",
) -> pd.DataFrame:
    """Elkan vs empirical threshold at disk level, one row per cost ratio.

    Thresholds are chosen on validation only; costs are measured on test with
    `disk_outcomes`. Disks that cannot be alerted in time are a constant number
    of unplanned failures, so they are left out of the sweep and added back.
    """
    val = disk_table(val_df, val_end, L, p_col, weight_col)
    sweep = val[val["failed"].eq(False) | val["plannable"]]
    rows = []
    for r in ratios:
        c_fp, c_fn = 1.0, float(r)
        t_th = elkan_threshold(c_fp, c_fn)
        emp = empirical_optimal_threshold(
            sweep["score"].to_numpy(), sweep["failed"].to_numpy().astype(int), c_fp, c_fn, sweep["weight"].to_numpy()
        )
        row = {"ratio_fn_over_fp": r, "threshold_elkan": t_th, "threshold_empirical": emp["threshold"]}
        for name, t in (("elkan", t_th), ("empirical", emp["threshold"])):
            for split, frame, end in (("val", val_df, val_end), ("test", test_df, test_end)):
                o = disk_outcomes(frame, t, end, L, c_fp, c_fn, p_col, weight_col)
                row[f"{split}_cost_{name}"] = o["cost"]
                row[f"{split}_fp_{name}"] = o["fp"]
                row[f"{split}_fn_{name}"] = o["fn"]
                row[f"{split}_planned_{name}"] = o["planned"]
                row[f"{split}_median_lead_{name}"] = o["median_lead_days"]
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Population weights and bootstrap
# ==========================================================================
def healthy_weight_factor(panel: pd.DataFrame, fleet_counts: pd.DataFrame, weight_col: str = "pop_weight") -> float:
    """Factor that makes the weighted healthy disks of a panel match the real fleet.

    In the exported panels `pop_weight` is N_healthy / len(sampled healthy disks),
    but validation and test only use 30% of the sampled disks (the other 70% are
    the training pool). The weighted healthy population is therefore about 30% of
    the fleet while failed disks are all kept with weight 1. The factor is the mean
    over the days of (active disks of the RQ2 replay) / (weighted healthy disks of
    the panel), `fleet_counts` being `real_test_daily_fleet_counts.csv`
    (column `active_test_drives`). It is applied to healthy disks only.
    """
    day = pd.to_datetime(panel["date"])
    healthy = panel["ep_outcome"] != "failed"
    per_day = panel.loc[healthy, weight_col].astype(float).groupby(day[healthy]).sum()
    fleet = fleet_counts.assign(date=pd.to_datetime(fleet_counts["date"])).set_index("date")["active_test_drives"]
    common = per_day.index.intersection(fleet.index)
    return float((fleet.loc[common] / per_day.loc[common]).mean())


def rescale_healthy_weights(panel: pd.DataFrame, factor: float, weight_col: str = "pop_weight") -> pd.DataFrame:
    """Copy of the panel with the weight of healthy disks multiplied by `factor`."""
    out = panel.copy()
    out[weight_col] = out[weight_col].astype(float)
    healthy = out["ep_outcome"] != "failed"
    out.loc[healthy, weight_col] = out.loc[healthy, weight_col] * factor
    return out


def disk_outcome_frame(
    df, threshold: float, window_end, L: int = 7, p_col: str = "p", weight_col: str = "pop_weight"
) -> pd.DataFrame:
    """One row per disk with its outcome at `threshold`.

    Columns: failed, weight, category in {planned, late, missed, fp, tn}, lead_days
    (failure date minus first alert, NaN without alert). Same rules as `disk_outcomes`.
    """
    d = _prepare_rows(df, window_end, p_col, weight_col)
    meta = _disk_meta(d)
    first = d[d["p"] >= threshold].groupby("serial_number")["date"].min().rename("first_alert")
    meta = meta.join(first)
    lead = (meta["fail_date"] - meta["first_alert"]).dt.days.where(meta["failed"])
    alerted = meta["first_alert"].notna()
    cat = np.where(
        meta["failed"],
        np.where(~alerted, "missed", np.where(lead >= L, "planned", "late")),
        np.where(alerted, "fp", "tn"),
    )
    return pd.DataFrame(
        {"failed": meta["failed"], "weight": meta["w"], "category": cat, "lead_days": lead}, index=meta.index
    )


def bootstrap_disk_cost(frame: pd.DataFrame, c_fp: float, c_fn: float, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Disk bootstrap of the cost, from a `disk_outcome_frame` (thresholds stay fixed).

    One row per resample: cost, fp, unplanned, planned_recall (share of failed disks
    alerted at least L days ahead).
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    w = frame["weight"].to_numpy(dtype=float)
    cat = frame["category"].to_numpy()
    is_fp, is_planned = cat == "fp", cat == "planned"
    is_unplanned = (cat == "late") | (cat == "missed")
    is_failed = frame["failed"].to_numpy(dtype=bool)
    rows = []
    for _ in range(n_boot):
        k = rng.multinomial(n, np.full(n, 1.0 / n))
        fp = float((k * w)[is_fp].sum())
        unplanned = float((k * w)[is_unplanned].sum())
        rows.append(
            {
                "cost": c_fp * fp + c_fn * unplanned,
                "fp": fp,
                "unplanned": unplanned,
                "planned_recall": float((k * w)[is_planned].sum() / (k * w)[is_failed].sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_calibration(
    panel: pd.DataFrame, probs: dict, y_col: str = "y_14", weight_col: str = "pop_weight", n_boot: int = 200, seed: int = 42
) -> pd.DataFrame:
    """Disk bootstrap of the weighted Brier score and ECE, one row per resample and name.

    `probs` maps a name ("raw", "isotonic") to the probability vector of the rows
    of `panel`. Disks (not rows) are resampled, as in the RQ2 replay.
    """
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(panel["serial_number"])
    n = len(uniques)
    y = panel[y_col].to_numpy(dtype=float)
    base_w = panel[weight_col].to_numpy(dtype=float)
    rows = []
    for b in range(n_boot):
        w = base_w * rng.multinomial(n, np.full(n, 1.0 / n))[codes]
        keep = w > 0
        for name, p in probs.items():
            p = np.asarray(p, dtype=float)
            curve = reliability_curve(p[keep], y[keep], sample_weight=w[keep])
            rows.append(
                {
                    "boot": b,
                    "name": name,
                    "brier": brier_score(p[keep], y[keep], w[keep]),
                    "ece": expected_calibration_error(curve),
                }
            )
    return pd.DataFrame(rows)


def group_disk_costs(
    df, group_col: str, ratios, window_end, L: int = 7, min_failed: int = 30, n_boot: int = 200, seed: int = 42,
    p_col: str = "p", weight_col: str = "pop_weight",
) -> pd.DataFrame:
    """Disk-level cost at the Elkan threshold of each cost ratio, split by a disk attribute.

    No refit: the thresholds are those of the whole fleet (1 / (1 + ratio)) and the scores are
    already calibrated, so a difference between groups is a difference of transfer, not of tuning.
    One row per (group, ratio): failed disks, planned recall, false alarms, unplanned failures,
    `relative_cost` = cost / cost of never alerting (below 1 means alerting pays) with a disk
    bootstrap interval. Groups with fewer than `min_failed` failed disks are left out.
    """
    rows = []
    for name, g in df.groupby(group_col):
        for r in ratios:
            frame = disk_outcome_frame(g, elkan_threshold(1.0, float(r)), window_end, L, p_col, weight_col)
            n_failed = float(frame.loc[frame["failed"], "weight"].sum())
            if len(frame[frame["failed"]]) < min_failed:
                break
            boot = bootstrap_disk_cost(frame, 1.0, float(r), n_boot=n_boot, seed=seed)
            fp = float(frame.loc[frame["category"] == "fp", "weight"].sum())
            unplanned = float(frame.loc[frame["category"].isin(["late", "missed"]), "weight"].sum())
            never = r * n_failed
            rows.append(
                {
                    group_col: name,
                    "ratio_fn_over_fp": r,
                    "n_failed": int(len(frame[frame["failed"]])),
                    "planned_recall": float(frame.loc[frame["category"] == "planned", "weight"].sum() / n_failed),
                    "false_alarms": fp,
                    "unplanned": unplanned,
                    "relative_cost": (fp + r * unplanned) / never,
                    "relative_cost_ci_low": float(boot["cost"].quantile(0.025) / never),
                    "relative_cost_ci_high": float(boot["cost"].quantile(0.975) / never),
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# Leave-one-model-out: ranking quality at a fixed false alarm rate
# ==========================================================================
def recall_at_false_alarm_rate(table: pd.DataFrame, rates, counts=None) -> dict:
    """Planned recall when a given share of the healthy disks is alerted, from a `disk_table`.

    Threshold free, so it compares two scorers without calibration. For each rate, the
    threshold is the lowest disk score whose weighted share of healthy disks at or above it
    stays within `rate`; ties are kept together. Failed disks that cannot be alerted L days
    ahead (`plannable` False) count as missed. `counts` multiplies the weights (bootstrap).
    """
    w = table["weight"].to_numpy(dtype=float)
    if counts is not None:
        w = w * np.asarray(counts, dtype=float)
    failed = table["failed"].to_numpy(dtype=bool)
    score = table["score"].to_numpy(dtype=float)
    h_score, h_w = score[~failed], w[~failed]
    order = np.argsort(-h_score, kind="stable")
    h_score, h_w = h_score[order], h_w[order]
    total_h, total_f = h_w.sum(), w[failed].sum()
    # share of healthy disks alerted at each distinct score value (ties together)
    last_of_tie = np.r_[h_score[1:] != h_score[:-1], True]
    share = np.cumsum(h_w)[last_of_tie] / total_h
    values = h_score[last_of_tie]
    f_score = score[failed]
    f_ok = np.isfinite(f_score)
    out = {}
    for rate in rates:
        ok = values[share <= rate]
        if len(ok) == 0 or total_f == 0:
            out[rate] = 0.0
            continue
        t = ok.min()
        out[rate] = float(w[failed][f_ok & (np.where(f_ok, f_score, -np.inf) >= t)].sum() / total_f)
    return out


def bootstrap_recall_gap(table_ref: pd.DataFrame, table_alt: pd.DataFrame, rates, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Disk bootstrap of recall at fixed false alarm rate for two scorers on the same disks.

    Both tables must cover the same disks. The same resample is applied to both, so the gap
    `alt - ref` is paired. One row per (resample, rate): ref, alt, gap.
    """
    alt = table_alt.reindex(table_ref.index)
    if alt["failed"].isna().any():
        raise ValueError("the two tables must cover the same disks")
    rng = np.random.default_rng(seed)
    n = len(table_ref)
    rows = []
    for _ in range(n_boot):
        k = rng.multinomial(n, np.full(n, 1.0 / n))
        a = recall_at_false_alarm_rate(table_ref, rates, counts=k)
        b = recall_at_false_alarm_rate(alt, rates, counts=k)
        rows.extend({"rate": r, "ref": a[r], "alt": b[r], "gap": b[r] - a[r]} for r in rates)
    return pd.DataFrame(rows)


# ==========================================================================
# Plots
# ==========================================================================
def plot_reliability(curves: dict, ax=None):
    """Reliability curves (log-log, because probabilities are tiny)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(5, 5))[1]
    lo = min(c["mean_pred"].min() for c in curves.values())
    hi = max(c["mean_pred"].max() for c in curves.values())
    ax.plot([lo, hi], [lo, hi], color="k", linestyle="dashed", lw=1, label="perfect")
    for name, c in curves.items():
        ax.plot(c["mean_pred"], c["frac_pos"], marker="o", ms=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    return ax


def plot_threshold_shift(table: pd.DataFrame, ax=None):
    """Theoretical and empirical thresholds as the cost ratio grows."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.plot(table["ratio_fn_over_fp"], table["threshold_elkan"], marker="o", label="Elkan c_FP/(c_FP+c_FN)")
    emp = table["threshold_empirical"].replace(np.inf, np.nan)
    ax.plot(table["ratio_fn_over_fp"], emp, marker="s", label="Empirical optimum (validation)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cost ratio c_FN / c_FP")
    ax.set_ylabel("Probability threshold")
    ax.legend()
    return ax
