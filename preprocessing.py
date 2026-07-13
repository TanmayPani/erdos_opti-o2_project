from pathlib import Path
import polars as pl

from core.io import read_one_year, read_expert_event_list, reformat_expert_workbook
from core.features import (
    sync_expert_features,
    enrich_expert_events,
    build_units,
    catalogue_orphans,
    finalize_units,
    finalize_moments,
    explode_moments,
    build_curves,
)


def preprocess(readouts, expert_labels, output_dir):
    _r0, _r1 = readouts["Datetime"].min(), readouts["Datetime"].max()
    # raw expert list (all non-ignore events, incl. those outside the readout span) —
    # saved verbatim so eda.py can read it instead of re-parsing the workbook.
    expert_event_list = expert_labels.filter(pl.col("expert_label") != "ignore").sort(
        "group_start"
    )

    _in_range = (pl.col("group_end") >= _r0) & (pl.col("group_start") <= _r1)
    events_covered = expert_event_list.filter(_in_range)

    expert_events, expert_samples = sync_expert_features(readouts, events_covered)
    orphan_events, orphan_samples = catalogue_orphans(
        readouts, events_covered, expert_events["event_id"].to_list()
    )

    expert_events_full = enrich_expert_events(expert_events, events_covered)
    _n_expert_features = expert_events_full.width - expert_events.width

    # `2024-08-19` and `2021-12-08` are now recovered as real expert umbrellas (2024-2o3 /
    # 2021-10o1) by the window-completeness parser, so adopting them here is redundant (verified
    # byte-identical proc either way). `2024-08-19` was also mis-adopted as `hot`; the expert
    # workbook labels event #2's 08-19 an oxic pulse (o3), and it now correctly follows that.
    ADOPT_ORPHANS = {"2023-07-04": "hot"}
    _units, _, _ = build_units(readouts, events_covered, adopt_orphans=ADOPT_ORPHANS)
    _unit_feat, _ = sync_expert_features(
        readouts, _units.rename({"unit_id": "event_id"})
    )

    proc = finalize_units(_unit_feat, _units, expert_events)

    curves = build_curves(readouts, proc).join(
        proc.select("unit_id", "is_public_augmented"), on="unit_id", how="left"
    )

    # MOMENT ROUTE — mirror of the unit route at moment granularity: classify each expert
    # moment (its own expert window + direct hot/oxic label) instead of an auto-detected unit.
    # No orphans, no weak supervision, no `mixed` holdout; hysteresis dropped (MOMENT_FEATURE_COLS).
    mom_frame = explode_moments(events_covered)
    _mom_feat, _ = sync_expert_features(readouts, mom_frame)
    moment_proc = finalize_moments(_mom_feat, mom_frame)
    moment_curves = build_curves(readouts, moment_proc).join(
        moment_proc.select("unit_id", "is_public_augmented"), on="unit_id", how="left"
    )

    output_dir.mkdir(exist_ok=True)

    proc.write_parquet(output_dir / "proc_features.parquet")
    curves.write_parquet(output_dir / "proc_curves.parquet")
    moment_proc.write_parquet(output_dir / "moment_features.parquet")
    moment_curves.write_parquet(output_dir / "moment_curves.parquet")
    readouts.write_parquet(output_dir / "readouts.parquet")
    expert_event_list.write_parquet(output_dir / "expert_event_list.parquet")
    expert_events_full.write_parquet(output_dir / "expert_events.parquet")
    expert_samples.write_parquet(output_dir / "expert_event_samples.parquet")
    orphan_events.write_parquet(output_dir / "orphan_events.parquet")
    orphan_samples.write_parquet(output_dir / "orphan_event_samples.parquet")


def load_public_dataset(path):
    _PUBLIC_WEATHER_RENAME = {
        "SlrFD_kW_Avg": "SlrFD_kW_Avg | kW/m^2 | Avg",
        "AirT_C_Avg": "AirT_C_Avg | Deg C | Avg",
        "BP_hPa": "BP_hPa | hPa | Smp",
        "SlrTF_MJ_Tot": "SlrTF_MJ_Tot | MJ/m^2 | Tot",
        "WS_ms_S_WVT": "WS_ms_S_WVT | meters/second | WVc",
        "VP_hPa_Avg": "VP_hPa_Avg | hPa | Avg",
        "RH": "RH | % | Smp",
    }

    df = pl.read_csv(
        path,
        skip_rows_after_header=2,
        infer_schema_length=10000,
        null_values=["#REF!", ""],
        try_parse_dates=False,
    )
    # the trailing column is an empty spacer left over from the CSV export
    df = df.select(pl.all().exclude(df.columns[-1]))
    return (
        df.with_columns(
            pl.col("Datetime").str.to_datetime("%-m/%-d/%Y %-H:%M").dt.round("5m")
        )
        .drop("WEATHER TIMESTAMP")
        .rename(_PUBLIC_WEATHER_RENAME)
        .drop_nulls("Datetime")
        .unique("Datetime", keep="first")
        .sort("Datetime")
    )


def load_full_dataset(data_dir, fname_glob, public_data=None, canonical_only=False):
    """All water-year workbooks under `data_dir` matching `fname_glob`
    -> one deduped, sorted 5-min series with every column (mirrors the exploratory.py
        load: drop null datetimes, keep first per stamp). `canonical_only=True` restricts to
        the six feature channels."""
    paths = sorted(Path(data_dir).glob(fname_glob))
    if not paths:
        raise FileNotFoundError(
            f"no readout workbooks matching {fname_glob} under {data_dir}"
        )
    df = (
        pl.concat([read_one_year(p, canonical_only=canonical_only) for p in paths])
        .drop_nulls("Datetime")
        .unique("Datetime", keep="first")
        .sort("Datetime")
    )
    if public_data is not None:
        df_pd = load_public_dataset(public_data)
        df_pd_aug = df_pd.filter(pl.col("Datetime") < df["Datetime"].min())
        df_pd_aug = df_pd_aug.select([c for c in df_pd_aug.columns if c in df.columns])
        df = pl.concat(
            [
                df_pd_aug.with_columns(pl.lit(True).alias("is_public_augmented")),
                df.with_columns(pl.lit(False).alias("is_public_augmented")),
            ],
            how="diagonal_relaxed",
        ).sort("Datetime")

    return df


def main():
    DATA_DIR = Path("datasets")
    OUTPUT_DIR = Path("derived")
    DATA_GLOB = "BeaverCreek_DO_Events_Labeled/20*Beaver Creek DO*QCd*.xlsx"
    EVENTS_XLSX = (
        Path("datasets")
        / "BeaverCreek_DO_Events_Labeled"
        / "2019-2026 list of hot & oxic moments rev. 04-29-26.xlsx"
    )
    PUBLIC_DATA = (
        DATA_DIR
        / "BeaverCreekWA_EssDive_26Jun2019-30Sep2024"
        / "data"
        / "2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv"
    )
    dataset = load_full_dataset(DATA_DIR, DATA_GLOB, PUBLIC_DATA)
    # decouple bad-formatting cleanup (-> tidy CSV) from parsing; original xlsx untouched
    reformatted = EVENTS_XLSX.with_name("expert_annotations_reformatted.csv")
    reformat_expert_workbook(EVENTS_XLSX, reformatted)
    expert_labels = read_expert_event_list(reformatted)

    preprocess(dataset, expert_labels, OUTPUT_DIR)


if __name__ == "__main__":
    main()
