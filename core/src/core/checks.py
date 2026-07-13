import polars as pl
from core.features import detect_events, auto_detected_events


def detection_agreement(readouts, events):
    """KPI #1 (validation only): fraction of expert windows overlapped by >=1
    auto-detected event. Runs the existing `detect_events` on the readouts and
    checks interval overlap against each expert window. Returns a summary dict."""
    auto = detect_events(readouts)
    aw = auto.select("start", "end").to_numpy()
    win = events.select("group_start", "group_end").to_numpy()
    recovered = sum(any((a_s <= e) and (a_e >= s) for a_s, a_e in aw) for s, e in win)
    return {
        "n_expert_windows": len(win),
        "n_auto_events": auto.height,
        "n_recovered": recovered,
        "recall": recovered / len(win) if len(win) else float("nan"),
    }


def granularity_reconciliation(readouts, events_covered, moments, peak_min=0.1):
    """Reconcile the THREE event granularities on the covered window:
    expert umbrellas  <  auto-detected events  <  expert moments.

    The expert labels at two levels (coarse multi-day *umbrellas* in `events_covered`
    and fine individual *moments* in `moments`); our `detect_events` sits between them
    (it splits on DO returning to baseline, so it merges sub-peaks the expert resolves).
    Returns `(summary, fanout, n_orphan)`:
      summary  one row per level (n_events, median duration, ratio per umbrella);
      fanout   per expert umbrella, how many auto events overlap it and how many expert
               moments it contains (the nesting that explains the count gap);
      n_orphan auto events (>= peak_min) overlapping no expert window (unlabelled)."""
    auto = auto_detected_events(readouts, peak_min)
    cov_ids = events_covered["event_id"]
    mom_cov = moments.filter(pl.col("event_id").is_in(cov_ids))

    def _median_min(df, a, b):
        return df.select((pl.col(b) - pl.col(a)).dt.total_minutes().median()).item()

    n_umb, n_auto, n_mom = events_covered.height, auto.height, mom_cov.height
    summary = pl.DataFrame(
        {
            "level": [
                "expert umbrellas",
                f"auto events (peak>={peak_min})",
                "expert moments",
            ],
            "n_events": [n_umb, n_auto, n_mom],
            "median_dur_min": [
                _median_min(events_covered, "group_start", "group_end"),
                _median_min(auto, "start", "end"),
                None,
            ],
            "per_umbrella": [1.0, round(n_auto / n_umb, 2), round(n_mom / n_umb, 2)],
        }
    )

    pairs = auto.join_where(
        events_covered.select("event_id", "group_start", "group_end"),
        pl.col("start") <= pl.col("group_end"),
        pl.col("end") >= pl.col("group_start"),
    )
    auto_per = pairs.group_by("event_id").agg(
        pl.col("eid").n_unique().alias("n_auto_events")
    )
    mom_per = mom_cov.group_by("event_id").len(name="n_expert_moments")
    fanout = (
        events_covered.select("event_id", "expert_label")
        .join(auto_per, on="event_id", how="left")
        .join(mom_per, on="event_id", how="left")
        .with_columns(
            pl.col("n_auto_events").fill_null(0),
            pl.col("n_expert_moments").fill_null(0),
        )
        .sort("n_expert_moments", descending=True)
    )
    n_orphan = auto.height - pairs["eid"].n_unique()
    return summary, fanout, n_orphan
