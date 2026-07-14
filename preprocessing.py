from pathlib import Path
import polars as pl

from core.io import read_one_year, read_expert_event_list, reformat_expert_workbook
from core.features import splice_readouts, event_samples


def preprocess(readouts, expert_labels, expert_events, output_dir, prefix="auto_"):
    _r0, _r1 = readouts["Datetime"].min(), readouts["Datetime"].max()

    expert_slices_covered = (
        expert_labels.sort("start_time").filter(
            (pl.col("start_time") >= _r0) & (pl.col("start_time") <= _r1)
        )
        if expert_labels is not None
        else None
    )

    expert_events_covered = (
        expert_events.sort("start_time").filter(
            (pl.col("start_time") >= _r0) & (pl.col("start_time") <= _r1)
        )
        if expert_events is not None
        else None
    )

    proc, curves = splice_readouts(
        readouts, events=expert_events_covered, umbrella=expert_slices_covered
    )

    output_dir.mkdir(exist_ok=True)

    print(
        f"Writing {len(proc)} events, {len(curves)} timesteps to {output_dir} / processed_{prefix}*.parquet"
    )
    proc.write_parquet(output_dir / f"processed_{prefix}features.parquet")
    curves.write_parquet(output_dir / f"processed_{prefix}curves.parquet")


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
    dataset.write_parquet(OUTPUT_DIR / "readouts.parquet")
    # decouple bad-formatting cleanup (-> tidy CSV) from parsing; original xlsx untouched
    reformatted = EVENTS_XLSX.with_name("expert_annotations_reformatted.csv")
    reformat_expert_workbook(EVENTS_XLSX, reformatted)
    expert_event_labels, expert_moment_labels = read_expert_event_list(reformatted)
    expert_event_labels.write_parquet(OUTPUT_DIR / "expert_event_list.parquet")
    expert_moment_labels.write_parquet(OUTPUT_DIR / "expert_moment_list.parquet")

    _r0, _r1 = dataset["Datetime"].min(), dataset["Datetime"].max()
    expert_samples = event_samples(
        dataset,
        expert_event_labels.with_columns(pl.col("event_id").alias("unit_id"))
        .sort("start_time")
        .filter((pl.col("start_time") >= _r0) & (pl.col("start_time") <= _r1)),
    )
    expert_samples.write_parquet(OUTPUT_DIR / "expert_samples.parquet")
    print(
        f"Writing {len(expert_samples)} sliced events to {OUTPUT_DIR} / expert_samples.parquet"
    )

    preprocess(
        dataset,
        None,
        expert_event_labels.with_columns(
            pl.col("event_id").alias("unit_id"), pl.lit("expert").alias("source")
        ),
        OUTPUT_DIR,
        prefix="expert_",
    )
    preprocess(dataset, expert_event_labels, None, OUTPUT_DIR, prefix="auto_")
    preprocess(
        dataset,
        expert_event_labels,
        expert_moment_labels,
        OUTPUT_DIR,
        prefix="moments_",
    )


if __name__ == "__main__":
    main()
