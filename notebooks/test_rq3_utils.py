"""Smoke tests on synthetic data. Run: python notebooks/test_rq3_utils.py

These checks only validate the code. Their numbers must never appear in the
notebook results or in the report.
"""

import numpy as np
import pandas as pd

from rq3_utils import (
    COST_RATIOS,
    IsotonicCalibrator,
    bootstrap_calibration,
    bootstrap_disk_cost,
    brier_score,
    cost_at_threshold,
    disk_outcome_frame,
    disk_outcomes,
    disk_table,
    disk_threshold_table,
    elkan_threshold,
    empirical_optimal_threshold,
    expected_calibration_error,
    healthy_weight_factor,
    load_panel,
    reliability_curve,
    rescale_healthy_weights,
    threshold_table,
)

rng = np.random.default_rng(0)


def make_data(n, keep_rate=1 / 50):
    """True risk p_true, labels from it, and a score inflated like a 1:50 downsampled model."""
    p_true = np.clip(rng.beta(0.3, 60, n), 1e-6, 0.9)
    y = rng.binomial(1, p_true)
    odds = p_true / (1 - p_true) / keep_rate
    return p_true, y, odds / (1 + odds)


def test_isotonic_is_monotone_and_matches_pav():
    c = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert np.all(np.diff(c.y_) >= 0)
    # 0.2 and 0.3 violate monotonicity and are pooled to 0.5
    assert np.allclose(c.predict([0.1, 0.2, 0.3, 0.4]), [0.0, 0.5, 0.5, 1.0])


def test_isotonic_handles_ties_and_weights():
    c = IsotonicCalibrator().fit([0.5, 0.5, 0.9], [0, 1, 1], sample_weight=[3, 1, 1])
    assert np.allclose(c.predict([0.5, 0.9]), [0.25, 1.0])


def sampled_panel(n, keep_neg=0.05):
    """Population with inflated scores, then like the real panels: all positives
    kept, healthy rows kept at `keep_neg` and weighted by 1/keep_neg."""
    p_true, y, s = make_data(n)
    keep = (y == 1) | (rng.random(n) < keep_neg)
    w = np.where(y[keep] == 1, 1.0, 1.0 / keep_neg)
    return y.mean(), s[keep], y[keep], w


def test_weighted_isotonic_absorbs_sampling_and_downsampling_bias():
    _, s_val, y_val, w_val = sampled_panel(600_000)
    prevalence, s_te, y_te, w_te = sampled_panel(600_000)
    weighted = IsotonicCalibrator().fit(s_val, y_val, sample_weight=w_val)
    unweighted = IsotonicCalibrator().fit(s_val, y_val)
    p_w, p_u = weighted.predict(s_te), unweighted.predict(s_te)
    # the weighted calibrated mean matches the population prevalence, the raw score and
    # the unweighted fit are far above it
    assert abs(np.average(p_w, weights=w_te) - prevalence) < 0.1 * prevalence
    assert np.average(s_te, weights=w_te) > 3 * prevalence
    assert np.average(p_u, weights=w_te) > 3 * prevalence
    assert brier_score(p_w, y_te, w_te) < brier_score(s_te, y_te, w_te)
    ece_raw = expected_calibration_error(reliability_curve(s_te, y_te, sample_weight=w_te))
    ece_cal = expected_calibration_error(reliability_curve(p_w, y_te, sample_weight=w_te))
    assert ece_cal < ece_raw / 10


def test_calibration_lowers_brier_and_ece():
    _, y_val, s_val = make_data(400_000)
    _, y_te, s_te = make_data(400_000)
    cal = IsotonicCalibrator().fit(s_val, y_val)
    p_te = cal.predict(s_te)
    assert brier_score(p_te, y_te) < brier_score(s_te, y_te)
    ece_raw = expected_calibration_error(reliability_curve(s_te, y_te))
    ece_cal = expected_calibration_error(reliability_curve(p_te, y_te))
    assert ece_cal < ece_raw / 10


def test_elkan_threshold():
    assert np.isclose(elkan_threshold(1, 5), 1 / 6)
    assert np.isclose(elkan_threshold(1, 1), 0.5)


def test_empirical_threshold_close_to_elkan_when_calibrated():
    p_true, y, _ = make_data(600_000)
    for ratio in (5, 20):
        emp = empirical_optimal_threshold(p_true, y, 1.0, float(ratio))
        # Brute force check of the sweep.
        grid = np.quantile(p_true, np.linspace(0.9, 0.99999, 400))
        best = min(cost_at_threshold(p_true, y, t, 1.0, ratio)["cost"] for t in grid)
        assert emp["cost"] <= best + 1e-9
        t_th = elkan_threshold(1.0, ratio)
        assert emp["cost"] <= cost_at_threshold(p_true, y, t_th, 1.0, ratio)["cost"] + 1e-9


def test_threshold_table_columns():
    p, y, _ = make_data(50_000)
    t = threshold_table(p, y, p_test=p, y_test=y)
    assert len(t) == len(COST_RATIOS)
    assert {"threshold_elkan", "threshold_empirical", "test_cost_elkan"} <= set(t.columns)


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid input was accepted")


def test_load_panel_validation():
    ok = pd.DataFrame(
        {
            "serial_number": ["a"], "date": ["2025-10-01"], "score": [0.1], "y_14": [0],
            "days_to_failure": [np.nan], "ep_outcome": ["active"], "pop_weight": [19.8],
        }
    )
    load_panel(ok)
    _expect_value_error(lambda: load_panel(ok.assign(score=1.5)))
    _expect_value_error(lambda: load_panel(ok.assign(pop_weight=0.0)))
    _expect_value_error(lambda: load_panel(ok.drop(columns=["pop_weight"])))
    _expect_value_error(lambda: load_panel("gbdt_h14_daily_scores_test.csv"))


def _rows(serial, outcome, p_by_dtf, fail_date=None, start="2025-11-01", weight=1.0):
    """Rows of one disk. For a failed disk `p_by_dtf` maps days-to-failure to score."""
    if outcome == "failed":
        end = pd.Timestamp(fail_date)
        recs = [(serial, end - pd.Timedelta(days=k), outcome, k, p, weight) for k, p in p_by_dtf.items()]
    else:
        days = pd.date_range(start, periods=len(p_by_dtf))
        recs = [(serial, d, outcome, np.nan, p, weight) for d, p in zip(days, p_by_dtf)]
    return pd.DataFrame(recs, columns=["serial_number", "date", "ep_outcome", "days_to_failure", "p", "pop_weight"])


def make_panel():
    parts = [
        _rows("A", "failed", {10: 0.1, 9: 0.6, 5: 0.9, 0: 0.9}, "2025-11-20"),  # alert 9 days ahead: planned
        _rows("B", "failed", {10: 0.1, 5: 0.2, 3: 0.7, 0: 0.9}, "2025-11-20"),  # alert 3 days ahead: late
        _rows("C", "failed", {10: 0.1, 5: 0.2, 0: 0.3}, "2025-11-20"),  # never alerted: missed
        _rows("D", "failed", {10: 0.9, 0: 0.9}, "2025-12-20"),  # fails after the window: excluded
        _rows("E", "active", [0.1, 0.8, 0.9, 0.2], weight=20.0),  # alerted on two days: one false positive
        _rows("F", "active", [0.1, 0.2, 0.3, 0.2], weight=20.0),  # never alerted
        _rows("G", "failed", {5: 0.99, 0: 0.99}, "2025-11-20"),  # only rows within 7 days: cannot be planned
    ]
    return pd.concat(parts, ignore_index=True)


def test_disk_outcomes_hand_checked():
    o = disk_outcomes(make_panel(), 0.5, window_end="2025-11-30", L=7, c_fp=1.0, c_fn=10.0)
    assert o["n_failed"] == 4  # A, B, C, G (D is excluded)
    assert o["planned"] == 1  # A
    assert o["late"] == 2  # B and G
    assert o["missed"] == 1  # C
    assert o["fp"] == 20.0  # E counted once, with its weight
    assert o["cost"] == 1.0 * 20.0 + 10.0 * 3
    assert o["median_lead_days"] == 5.0  # leads of A, B, G are 9, 3, 5 days


def test_disk_table_scores():
    t = disk_table(make_panel(), window_end="2025-11-30", L=7)
    assert set(t.index) == {"A", "B", "C", "E", "F", "G"}
    assert np.isclose(t.loc["A", "score"], 0.6)  # rows at 10 and 9 days ahead
    assert np.isclose(t.loc["B", "score"], 0.1)
    assert np.isclose(t.loc["E", "score"], 0.9)  # healthy disk: maximum over all rows
    assert not t.loc["G", "plannable"]
    assert t.loc["E", "weight"] == 20.0


def test_disk_table_matches_disk_outcomes_on_random_panel():
    rng_l = np.random.default_rng(3)
    parts = []
    for i in range(300):
        n = 40
        if i < 60:
            dtf = np.arange(n)[::-1]
            fail = pd.Timestamp("2025-11-25") - pd.Timedelta(days=int(rng_l.integers(0, 20)))
            p = np.clip(rng_l.beta(1, 8, n) + (n - dtf) / n * rng_l.random() * 0.5, 0, 1)
            parts.append(_rows(f"f{i}", "failed", dict(zip(dtf, p)), fail))
        else:
            parts.append(_rows(f"h{i}", "active", rng_l.beta(1, 20, n), weight=15.0))
    df = pd.concat(parts, ignore_index=True)
    table = disk_table(df, "2025-11-30", L=7)
    sweep = table[~table["failed"] | table["plannable"]]
    for t in (0.05, 0.2, 0.4, 0.7):
        o = disk_outcomes(df, t, "2025-11-30", L=7)
        ref = cost_at_threshold(
            sweep["score"].to_numpy(), sweep["failed"].to_numpy().astype(int), t, 1.0, 1.0, sweep["weight"].to_numpy()
        )
        n_unplannable = int((~table["plannable"]).sum())
        assert np.isclose(o["fp"], ref["fp"])
        assert np.isclose(o["fn"], ref["fn"] + n_unplannable)


def test_disk_threshold_table_runs():
    t = disk_threshold_table(make_panel(), make_panel(), "2025-11-30", "2025-11-30", L=7)
    assert len(t) == len(COST_RATIOS)
    assert {"threshold_elkan", "threshold_empirical", "test_cost_elkan", "val_cost_empirical"} <= set(t.columns)


def test_healthy_weight_factor_restores_fleet_size():
    days = pd.date_range("2025-10-01", periods=3)
    panel = pd.DataFrame(
        {
            "date": np.repeat(days, 3),
            "ep_outcome": ["active", "active", "failed"] * 3,
            "pop_weight": [20.0, 20.0, 1.0] * 3,
        }
    )
    fleet = pd.DataFrame({"date": days, "active_test_drives": [130, 130, 130]})
    factor = healthy_weight_factor(panel, fleet)
    assert np.isclose(factor, 130 / 40)
    out = rescale_healthy_weights(panel, factor)
    assert np.allclose(out.loc[out["ep_outcome"] == "failed", "pop_weight"], 1.0)  # failed disks untouched
    assert np.isclose(out.loc[out["ep_outcome"] != "failed"].groupby("date")["pop_weight"].sum().iloc[0], 130)


def test_disk_outcome_frame_matches_disk_outcomes():
    df = make_panel()
    for t in (0.15, 0.5, 0.95):
        o = disk_outcomes(df, t, "2025-11-30", L=7)
        f = disk_outcome_frame(df, t, "2025-11-30", L=7)
        for cat in ("planned", "late", "missed", "fp"):
            assert np.isclose(f.loc[f["category"] == cat, "weight"].sum(), o[cat])


def test_bootstrap_disk_cost_brackets_point_estimate():
    df = make_panel()
    f = disk_outcome_frame(df, 0.5, "2025-11-30", L=7)
    point = disk_outcomes(df, 0.5, "2025-11-30", L=7, c_fp=1.0, c_fn=10.0)["cost"]
    boot = bootstrap_disk_cost(f, 1.0, 10.0, n_boot=300, seed=1)
    assert len(boot) == 300
    assert boot["cost"].quantile(0.025) <= point <= boot["cost"].quantile(0.975)
    assert boot["planned_recall"].dropna().between(0, 1).all()  # NaN only when a tiny resample has no failed disk


def test_bootstrap_calibration_shape():
    _, s, y, w = sampled_panel(80_000)
    panel = pd.DataFrame({"serial_number": np.arange(len(s)) // 5, "y_14": y, "pop_weight": w})
    out = bootstrap_calibration(panel, {"raw": s, "same": s}, n_boot=5, seed=0)
    assert len(out) == 10 and set(out["name"]) == {"raw", "same"}
    assert (out["brier"] > 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
