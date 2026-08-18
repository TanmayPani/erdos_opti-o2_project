import csv
import re
import zipfile
from datetime import timedelta
from pathlib import Path

import fastexcel
import polars as pl
import polars.selectors as cs

from core.features import DO_COL, WL_COL, SAL_COL, TEMP_COL, PRECIP_COL

_LABEL_RE = re.compile(r"^\d{4}-\d+([a-zA-Z]+\d*)$")
_DT_FMT = "%Y-%m-%d %H:%M:%S%.f"


def _readout_col_names(df, data_start):
    """Clean, unique column name per readout column, built from the banner header
    rows above `data_start` (quantity | units | agg). The copyright glyph the sheet
    uses for '(C)' is normalised; blank-header columns (the weather channels' unlabelled
    Hour-Average companions) fall back to `unnamed_<i>` so every column is addressable."""
    names = []
    for i in range(df.width):
        parts = [df[r, i] for r in range(data_start)]
        nm = " | ".join(p for p in parts if p not in (None, "")).replace("©", "(C)")
        names.append(nm if nm else f"unnamed_{i}")
    return names


def _pick(headers, must_have, must_not=()):
    """Index of the first column whose composite header contains every token in
    `must_have` and none in `must_not`. Raises if nothing matches (fail loud)."""
    for i, h in enumerate(headers):
        hl = h.lower()
        if all(m.lower() in hl for m in must_have) and not any(
            x.lower() in hl for x in must_not
        ):
            return i
    raise KeyError(f"no readout column matched {must_have} (excluding {must_not})")


def read_one_year(source, canonical_only=False):
    """One raw water-year workbook (`source` = the workbook path)
    -> tidy 5-min series with **all** columns.
    Read all-string (mixed dtypes break calamine's inference), find the real data
    start (first col-0 cell that parses as a datetime, below the banner header),
    then keep every column with a clean unique name. The six channels the feature
    code depends on are renamed to their exact canonical names (`DO_COL` etc., note
    the trailing space on `TEMP_COL`); all other channels — O2 %, the Hour-Average
    and Raw variants, the full weather block, submergence flag, water height — ride
    along under their sheet names. `Datetime` (and the parallel `WEATHER TIMESTAMP`)
    parse to datetime, `Datetime` snapped to the 5-min grid (stored values drift by
    ~20 ms); `Datehour`/`Notes` stay strings; everything else casts to Float64. Pass
    `canonical_only=True` to get just the six feature channels + `Datetime`."""
    df = (
        fastexcel.read_excel(source)
        .load_sheet("floodplain well data", header_row=None, dtypes="string")
        .to_polars()
    )
    start = next(
        r
        for r in range(df.height)
        if df[r, 0] and re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:", str(df[r, 0]))
    )
    names = _readout_col_names(df, start)
    # override the feature channels to their exact canonical names so downstream
    # code finds them by name; every other column keeps its cleaned sheet name.
    canonical = {
        0: "Datetime",
        _pick(names, ["Dissolved Oxygen (mg/L)", "Corrected"], ["Raw"]): DO_COL,
        _pick(names, ["Well Salinity (PPT)"], ["Hour Average"]): SAL_COL,
        _pick(
            names, ["Flood plain water level in BGS (cm)"], ["Hour Average", "raw"]
        ): WL_COL,
        _pick(names, ["DO Sensor Temperature"], ["Hour Average"]): TEMP_COL,
        _pick(names, ["Precip (mm)"]): PRECIP_COL,
    }
    final = [canonical.get(i, names[i]) for i in range(df.width)]
    # Four blank-header columns are the Hour-Average companions of the preceding
    # weather channel (verified against Datehour: constant within each hour, equal to
    # the base channel's hourly mean). Name them `<base> | Hour Average`. Indices are
    # stable — all six workbooks share one 34-column layout. Indices 19 and 26 are
    # always-empty spacer columns (100% null in every file) and are dropped; dropping
    # by nullness instead would vary the schema per file (e.g. Precip is all-null in
    # 2024-25 but present elsewhere).
    # WY2020-WY2024 label these four headers themselves (`<channel> | <unit> | Hour Average`);
    # only WY2025 leaves them blank. Rebuild the SAME name by swapping the base channel's agg
    # segment for "Hour Average" — build it from the RAW header (`names`), never from `final`,
    # whose canonical rename has already dropped the unit segment. Appending to `final` instead
    # yields `Precip (mm) over 5 minutes | Hour Average` where the other five workbooks say
    # `… | mm | Hour Average`, and the concat across years then fails on mismatched names.
    for _i in (15, 17, 21, 25):
        if final[_i] == f"unnamed_{_i}" and not names[_i - 1].startswith("unnamed_"):
            _base = names[_i - 1].rsplit(" | ", 1)[0]
            final[_i] = f"{_base} | Hour Average"
    _drop_idx = {19, 26}
    keep = set(canonical.values()) if canonical_only else set(final)

    _dt_idx = {0} | {i for i, n in enumerate(final) if "WEATHER TIMESTAMP" in n}
    # _str_names = {"Datehour", "Notes"}
    _str_names = {
        "Datehour",
    }
    cols = df.columns
    exprs = []
    for i in range(df.width):
        if i in _drop_idx or final[i] not in keep:
            continue
        src = pl.col(cols[i])
        if i == 0:
            e = src.str.to_datetime(_DT_FMT, strict=False).dt.round("5m")
        elif i in _dt_idx:
            e = src.str.to_datetime(_DT_FMT, strict=False)
        elif final[i] in _str_names:
            e = src
        else:
            e = src.cast(pl.Float64, strict=False)
        exprs.append(e.alias(final[i]))
    return df.slice(start).select(exprs)


def _class3(suffix):
    """Fine expert suffix -> 3-class target, per the workbook LEGEND.

    rev 07-31-26 (current):
      `m`  = 'mixed hot moment'              -> mixed  (the deck's Mixed DO Event)
      `h`                                    -> hot
      `o1..o5` = oxic pulse (various)        -> pulse
      `b`, `x`                               -> ignore (dropped)

    rev 04-29-26 (retired, still mapped so older workbooks parse):
      `hx` = 'unknown hot moment'            -> mixed
      `e`  = 'oxygen event during flood'     -> pulse
    The 07-31 revision merged `hx` and `e` into the single explicit `m`.

    Order matters: match `hx` before `h`. The LEGEND glosses `b`='hot moment', but every
    `b`-tagged event (2022-2b, 2022-9b) is BLACKED OUT in the workbook (the expert's ignore
    marker, fill theme=1) and shows no DO excursion, so `b` joins `x` in the ignore pile
    (user decision, 2026-07-11)."""
    s = suffix.lower()
    if s.startswith(("m", "hx")):
        return "mixed"
    if s.startswith(("e", "o")):
        return "pulse"
    if s.startswith("h"):
        return "hot"

    return "ignore"


def _events_datastart(df):
    for r in range(df.height):
        v = df[r, 0]
        if v and _LABEL_RE.match(str(v).strip()):
            return r
    return 0


_BLACK_MIN = (
    5  # a row is "blacked out" (expert ignore) if this many cells carry the dark fill
)


def _blacked_rows(src):
    """Rows the expert BLACKED OUT (solid dark fill = their 'ignore' marker) in the raw workbook —
    `fastexcel`'s value-only read can't see cell fills, so we read them straight from the xlsx zip
    (`styles.xml` + `sheetN.xml`). A fill counts as dark if its solid `fgColor` is `theme=1` (dark1);
    a row is blacked if >= `_BLACK_MIN` of its cells use such a style. Returns
    `{sheet_name: {0-indexed row, ...}}` (0-indexed to match `fastexcel`'s `header_row=None` frame).
    These are e.g. `2023-1o1`'s sub-threshold (<0.1 mg/L) sub-pulses and the whole `2019-8x`/`b`
    ignore events."""
    zf = zipfile.ZipFile(src)
    styles = zf.read("xl/styles.xml").decode("utf8", "replace")
    fills = re.findall(r"<fill>.*?</fill>", styles, re.S)
    dark = {
        i
        for i, f in enumerate(fills)
        if 'patternType="solid"' in f and re.search(r'<fgColor theme="1"\s*/?>', f)
    }
    xfs = re.findall(
        r"<xf\b[^>]*?/?>", re.search(r"<cellXfs.*?</cellXfs>", styles, re.S).group(0)
    )
    black_s = {
        i
        for i, xf in enumerate(xfs)
        if (m := re.search(r'fillId="(\d+)"', xf)) and int(m.group(1)) in dark
    }
    wb = zf.read("xl/workbook.xml").decode("utf8", "replace")
    rels = zf.read("xl/_rels/workbook.xml.rels").decode("utf8", "replace")
    relmap = dict(re.findall(r'Id="([^"]+)"[^>]*Target="(worksheets/[^"]+)"', rels))
    out = {}
    for name, rid in re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
        if rid not in relmap:
            continue
        xml = zf.read("xl/" + relmap[rid]).decode("utf8", "replace")
        cnt = {}
        for row, s in re.findall(r'<c r="[A-Z]+(\d+)"(?:[^>]*?\bs="(\d+)")?', xml):
            if s and int(s) in black_s:
                cnt[int(row)] = cnt.get(int(row), 0) + 1
        out[name] = {
            row - 1 for row, n in cnt.items() if n >= _BLACK_MIN
        }  # xlsx row is 1-indexed
    return out


def _recombine_expr(raw_col, date_col):
    """Heal a per-moment timestamp *list* column, grafting Excel time-only serials onto
    each moment's own start date.

    `raw_col` is a `list[str]` of raw cell strings, `date_col` the aligned
    `list[datetime]` of moment starts. Some peak/inflection/end cells store only a
    time-of-day (rendered `1899-12-31 …` / `1900-01-.. …`); when the parsed year is
    pre-2000 we treat it as time-only and graft that clock time onto the moment's own
    date, rolling to the next day if the result would precede the start. Full datetimes
    pass through. Polars can't zip two aligned list columns in a pure expression, so the
    elementwise pairing runs through a small `map_elements` UDF (n is tiny — one row per
    umbrella)."""

    def _heal(row):
        out = []
        for dt, ref in zip(row["dt"], row["ref_date"]):
            if dt is None:
                out.append(None)
            elif dt.year < 2000 and ref is not None:
                comb = ref.replace(
                    hour=dt.hour,
                    minute=dt.minute,
                    second=dt.second,
                    microsecond=dt.microsecond,
                )
                if comb < ref:
                    comb += timedelta(days=1)
                out.append(comb)
            else:
                out.append(dt)
        return out

    dt_expr = pl.col(raw_col).list.eval(
        pl.element().str.to_datetime(_DT_FMT, strict=False)
    )
    return pl.struct(dt=dt_expr, ref_date=pl.col(date_col)).map_elements(
        _heal, return_dtype=pl.List(pl.Datetime("us"))
    )


def _minutes_between(start_col, end_col):
    """Elementwise whole-minute `end - start` across two aligned `list[datetime]` columns
    (same list-zip limitation as `_recombine_expr`, so it also runs through a UDF)."""

    def _diff(row):
        return [
            None if (a is None or b is None) else int((b - a).total_seconds()) // 60
            for a, b in zip(row["start"], row["end"])
        ]

    return pl.struct(start=pl.col(start_col), end=pl.col(end_col)).map_elements(
        _diff, return_dtype=pl.List(pl.Int64)
    )


# ---------------------------------------------------------------------------
# Raw-workbook column indices (positional; the sheet has no reliable header row).
# Schema of `…rev. 07-31-26.xlsx` (27 cols). That revision renamed the hierarchy —
# `Event -> Individual Hot/Oxic Moments` became `Cluster -> Individual Hot/Oxic Events` —
# but did NOT change it: a cluster row is our umbrella/event, its sub-rows are our moments.
# It also inserted a third `MIXED EVENTS` block and gave the hot block its own
# `Time of Beginning DO` (cols 16-18 and 6; everything from 6 on shifted vs rev 04-29).
# ---------------------------------------------------------------------------
_C_START, _C_END = 2, 3  # Cluster Start / Cluster End
_C_NHOT, _C_NOXIC, _C_NMIXED = 10, 13, 18
_C_PEAK_TS, _C_INFL_TS, _C_SAL, _C_RISE_PCT = 4, 5, 8, 9  # hot-block measurements
_C_PRECIP, _C_FLOOD = 14, 15
_C_PEAK_DO, _C_CONS_RATE = 20, 21
_C_NOTES, _C_CHECKED = 23, 24

# THE moment-typing rule: a moment's class is the column block its DO window sits in.
# The header of each block *is* the class ("HOT MOMENT EVENTS" / "OXIC EVENTS" /
# "MIXED EVENTS"), and no row in the workbook fills more than one block, so this is an
# exact partition — no label-suffix or which-columns-are-non-null heuristics needed.
# A row without a complete window pair in any block is not a moment (blank separators,
# the foot-of-sheet WY summary, the legend glossary all fall out for free).
_MOMENT_BLOCKS = (
    ("hot", 6, 7),  # Time of Beginning DO / Time of End DO
    ("oxic", 11, 12),
    ("mixed", 16, 17),
)
# the tidy CSV schema written by reformat_expert_workbook and read by read_expert_event_list
# `type` categorises every row: `umbrella` for the event row, `hot`/`oxic` for a moment row
# (so an umbrella is just "a moment of type umbrella"). `start_time`/`end_time` is the single
# window column pair shared by both: the reliable umbrella window on an umbrella row, and the
# unified moment window on a moment row (HOT: DO-rise start col 2 / DO-end col 6; OXIC: oxic
# begin/end cols 10-11 — the workbook staggers these hot/oxic blocks, merged here).
_UMB_FIELDS = (
    "year",
    "event_id",
    "expert_subtype",
    "expert_label",
    "type",
    "start_time",
    "end_time",
    "n_hot_moments",
    "n_oxic_pulses",
    "n_mixed_moments",
    "concurrent_precip",
    "flooding",
    "notes",
    "xf_qc_checked",
)
_MOM_FIELDS = (
    # moment-only extras; `type` and `start_time`/`end_time` are the shared columns above.
    # `m_peak_do_ts`/`m_do_inflection_ts` are the hot-only intermediate timings (blank for oxic).
    # `m_subtype` is the suffix on the moment row's OWN label where the expert wrote one
    # (`2019-15m` decomposes into rows labelled `2019-15h` / `2019-15o4`). It does NOT set the
    # moment's `type` — block placement does — but it records the expert's read of that
    # moment's character inside a mixed cluster, which is otherwise lost. Usually null.
    "m_subtype",
    "moment_idx",
    "m_peak_do_ts",
    "m_do_inflection_ts",
    "m_sal_at_peak",
    "m_do_rise_pct",
    "m_peak_do",
    "m_consumption_rate",
)
# Yes/(No|blank) umbrella flag columns -> boolean (normalised in the reformatter, cast in the parser)
_BOOL_COLS = ("concurrent_precip", "flooding", "xf_qc_checked")


def reformat_expert_workbook(src, out_csv):
    """Read the raw, inconsistently-formatted Opti-O2 event workbook and write a TIDY
    one-row-per-moment CSV (`expert_annotations_reformatted.csv`). The original xlsx is never
    modified. ALL of the team's formatting quirks are resolved HERE, so the downstream parser
    (`read_expert_event_list`) stays quirk-free:
      A. a single oxic pulse written on the event row itself (cols 10-11, no sub-row);
      B. sub-rows with a BLANK "Individual Moment #" (col 1) — else silently dropped;
      C. a `# of Hot/Oxic` count that slipped onto a sub-row instead of the event row;
      E. sub-rows that repeat a `<year>-N<type>` label with the index set — folded into the
         current umbrella (NOT new umbrellas); the suffix gives that moment's hot/oxic type.
    The reliable umbrella window (cols 2-3) is never touched. Each moment gets an explicit
    hot/oxic `type` from its label suffix, else the measurement block it fills. Every event
    emits one `type=umbrella` row (umbrella scalars, the umbrella window in `start_time`/
    `end_time`, moment-only fields blank) followed by one `type=hot|oxic` row per moment (its
    own window in `start_time`/`end_time`); a moment-less umbrella is just its umbrella row.
    The umbrella row's `n_hot_moments`/`n_oxic_pulses` are the ACTUAL tally of its typed moment
    rows (not the workbook's declared counts, which disagree with their own moment typing for a
    few events). Every umbrella with ANY data-quality flaw — count mismatch, or a reversed /
    overlapping / out-of-window / untyped moment (via `_audit_flaws` on the parsed output) — has
    its event_id written to a sibling `inconsistent-ids.txt` for reporting back to Opti-O2.
    Returns the output path."""

    def _isnum(v):
        try:
            float(str(v).strip())
            return True
        except (TypeError, ValueError):
            return False

    xl = fastexcel.read_excel(src)
    blacked = _blacked_rows(
        src
    )  # expert 'ignore' rows (dark fill), invisible to fastexcel
    umbrellas = []  # each: umbrella scalars + `_urow_ox` + `moments` (list of dicts)
    for yr in [s for s in xl.sheet_names if re.fullmatch(r"\d{4}", s)]:
        df = xl.load_sheet(yr, header_row=None, dtypes="string").to_polars()
        ds = _events_datastart(df)
        d = df.slice(ds)
        black = blacked.get(yr, set())
        cur = None
        for r in range(d.height):
            if (r + ds) in black:
                continue  # expert blacked this row out (e.g. <0.1 mg/L sub-pulse) -> ignore
            label = d[r, 0]
            mom_idx = d[r, 1]
            label_str = str(label).strip() if label is not None else None
            # the trailing LEGEND block (a=/b=/f=/x= glossary) spills text into cols 5 & 7, which
            # live in _HOT_COLS — without this it is mis-read as a phantom hot moment on the sheet's
            # last umbrella. It always sits below the final event, so stop the sheet here.
            if label_str is not None and label_str.upper() == "LEGEND":
                break
            label_mat = _LABEL_RE.match(label_str) if label_str is not None else None

            # umbrella row: a label with BOTH a start (col 2) AND an end (col 3). A labeled row
            # with a start but a BLANK end is a Pattern-E moment sub-row (a single instant inside
            # the umbrella), not a new umbrella. Window-completeness is the reliable signal: the
            # moment-index cell is unreliable (some umbrella rows carry a stray index — 2024-2o3,
            # 2021-10o1 — while some continuation rows leave it blank), and merging by event number
            # would wrongly fuse distinct events that reuse a number months apart (2019-8h/8x).
            _lab_start, _lab_end = d[r, _C_START], d[r, _C_END]

            # this row's moment window, if any — the block that holds it gives its class
            _typ, _mstart, _mend = None, None, None
            for _t, _a, _b in _MOMENT_BLOCKS:
                if d[r, _a] is not None and d[r, _b] is not None:
                    _typ, _mstart, _mend = _t, d[r, _a], d[r, _b]
                    break

            def _moment(idx, own_label=None):
                return {
                    "moment_idx": idx,
                    "moment_type": _typ,
                    "m_subtype": own_label,
                    "start_time": _mstart,
                    "end_time": _mend,
                    "m_peak_do_ts": d[r, _C_PEAK_TS],
                    "m_do_inflection_ts": d[r, _C_INFL_TS],
                    "m_sal_at_peak": d[r, _C_SAL],
                    "m_do_rise_pct": d[r, _C_RISE_PCT],
                    "m_peak_do": d[r, _C_PEAK_DO],
                    "m_consumption_rate": d[r, _C_CONS_RATE],
                }

            if (
                label_mat is not None
                and _lab_start is not None
                and _lab_end is not None
            ):
                cur = {
                    "year": int(yr),
                    "event_id": label_str,
                    "expert_subtype": label_mat.group(1),
                    "expert_label": _class3(label_mat.group(1)),
                    "start_time": _lab_start,
                    "end_time": _lab_end,
                    "n_hot_moments": d[r, _C_NHOT],
                    "n_oxic_pulses": d[r, _C_NOXIC],
                    "n_mixed_moments": d[r, _C_NMIXED],
                    "concurrent_precip": d[r, _C_PRECIP],
                    "flooding": d[r, _C_FLOOD],
                    "notes": d[r, _C_NOTES],
                    "xf_qc_checked": d[r, _C_CHECKED],
                    "moments": [],
                }
                umbrellas.append(cur)
                # A cluster row often carries its OWN moment window as well as the cluster
                # window — a single-moment event (`2020-1o1`, `2022-4m`) writes its only moment
                # there, and `2020-13m` writes moment 1 on the cluster row with moment 2 below.
                # Same rule as any other row: a complete block window IS a moment.
                if _typ is not None:
                    cur["moments"].append(_moment(mom_idx))
                continue

            if cur is None:
                continue  # stray rows before the first umbrella

            if _typ is None:
                continue  # blank separator / foot-of-sheet WY summary / legend glossary

            # Pattern C: adopt a count that slipped onto this sub-row when the cluster row's
            # is null (guarded by `_typ` above, so the WY-summary block can't donate one).
            if cur["n_hot_moments"] is None and _isnum(d[r, _C_NHOT]):
                cur["n_hot_moments"] = d[r, _C_NHOT]
            if cur["n_oxic_pulses"] is None and _isnum(d[r, _C_NOXIC]):
                cur["n_oxic_pulses"] = d[r, _C_NOXIC]
            if cur["n_mixed_moments"] is None and _isnum(d[r, _C_NMIXED]):
                cur["n_mixed_moments"] = d[r, _C_NMIXED]

            cur["moments"].append(
                _moment(mom_idx, label_mat.group(1) if label_mat else None)
            )

    # NOTE: rev 04-29's two post-passes are gone. "Pattern A" (a single oxic pulse written on
    # the event row itself) is now just the cluster-row moment handled inline above, and the
    # "coerce a single-moment pulse umbrella's moment to oxic" hack is obsolete: it existed
    # only because the old which-columns-are-non-null heuristic mis-typed pulses whose hot
    # measurement block was filled (2024-1o5 / 2025-3o1). Block placement types them directly.

    out_csv = Path(out_csv)
    count_mismatch = []  # event_ids whose workbook-declared #hot/#oxic disagree with the moments
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_UMB_FIELDS) + list(_MOM_FIELDS))
        w.writeheader()
        for u in umbrellas:
            base = {k: u[k] for k in _UMB_FIELDS if k != "type"}
            # umbrella hot/oxic counts: write the ACTUAL tally of typed moment rows, not the
            # workbook's declared `# of Hot/Oxic` (which disagrees with its own per-moment typing
            # for a handful of events — e.g. `hx` umbrellas declare every moment hot). Where the
            # declared count (blank -> 0) differs from the tally, log the event_id for reporting.
            _decl_hot = (
                int(float(base["n_hot_moments"]))
                if _isnum(base["n_hot_moments"])
                else 0
            )
            _decl_oxic = (
                int(float(base["n_oxic_pulses"]))
                if _isnum(base["n_oxic_pulses"])
                else 0
            )
            _decl_mixed = (
                int(float(base["n_mixed_moments"]))
                if _isnum(base["n_mixed_moments"])
                else 0
            )
            _act_hot = sum(m["moment_type"] == "hot" for m in u["moments"])
            _act_oxic = sum(m["moment_type"] == "oxic" for m in u["moments"])
            _act_mixed = sum(m["moment_type"] == "mixed" for m in u["moments"])
            if (_decl_hot, _decl_oxic, _decl_mixed) != (
                _act_hot,
                _act_oxic,
                _act_mixed,
            ):
                count_mismatch.append(u["event_id"])
            base["n_hot_moments"] = _act_hot
            base["n_oxic_pulses"] = _act_oxic
            base["n_mixed_moments"] = _act_mixed
            # Yes/No flag columns -> numeric 1/0 (case-insensitive), cast to bool on read;
            # a blank stays blank (unknown != no), reading back as null.
            for _c in _BOOL_COLS:
                _s = "" if base[_c] is None else str(base[_c]).strip().lower()
                base[_c] = "" if _s == "" else int(_s == "yes")
            # umbrella row: `type`="umbrella", `moment_idx`=0, window = the reliable umbrella
            # window (in `base`)
            w.writerow({**base, "type": "umbrella", "moment_idx": 0})
            for i, m in enumerate(u["moments"], start=1):
                # moment row: `type` = this moment's hot/oxic; `moment_idx` = a clean 1-indexed
                # position within the umbrella (the workbook's own index is gappy — blank for
                # single oxic pulses, stray on some rows). The window + counts + label describe
                # THIS moment (its own start/end overriding the umbrella's), while the umbrella
                # row keeps its own class (e.g. `mixed` for an hx umbrella).
                _typ = m["moment_type"]
                _lab = {"hot": "hot", "oxic": "pulse", "mixed": "mixed"}.get(
                    _typ, base["expert_label"]
                )
                w.writerow(
                    {
                        **base,
                        "type": _typ,
                        "start_time": m["start_time"],
                        "end_time": m["end_time"],
                        **{k: m[k] for k in _MOM_FIELDS},
                        "moment_idx": i,
                        "expert_label": _lab,
                        "n_hot_moments": int(_typ == "hot"),
                        "n_oxic_pulses": int(_typ == "oxic"),
                        "n_mixed_moments": int(_typ == "mixed"),
                    }
                )
    # side-car for reporting back to Opti-O2: EVERY umbrella with any data-quality flaw — the
    # declared-vs-actual count mismatches collected above, plus the window/type integrity flaws
    # found by re-auditing the healed parser output (reversed/overlapping/out-of-window/untyped
    # moments). One event_id per line, sorted; overwritten each run; empty file if all clean.
    # flaws = _audit_flaws(read_expert_event_list(out_csv))
    # for _eid in count_mismatch:
    #    flaws.setdefault(_eid, set()).add("count_mismatch")
    # out_csv.with_name("inconsistent-ids.txt").write_text(
    #    "".join(f"{eid}\n" for eid in sorted(flaws))
    # )
    return out_csv


def read_expert_event_list(path):
    """Parse the TIDY reformatted event CSV (`reformat_expert_workbook`'s output — one row per
    moment, umbrella scalars repeated, a `type` of `umbrella`/`hot`/`oxic`) into one row per TOP-LEVEL event
    (umbrella) with nested `moments.*` list columns. This is the *real* parser: it assumes a
    clean, well-formed input and contains **no** handling of the raw workbook's formatting
    quirks — that all lives in `reformat_expert_workbook`.

    Umbrella scalars: `group_start`/`group_end`, `n_hot_moments`/`n_oxic_pulses`
    (⚠ label-leaking — only hot events fill the former, only oxic the latter),
    `concurrent_precip`, `flooding`, `notes`, `xf_qc_checked`. Per-moment `moments.*`
    lists: `idx`, `type`, the healed timings `start_time`/`do_peak_time`/`do_inflection_time`/
    `end_time` (`start_time`→`end_time` is the unified moment window for BOTH hot and oxic
    moments; `do_peak_time`/`do_inflection_time` are hot-only), the measured
    `sal_at_peak`/`do_rise_pct`/`peak_do`/`consumption_rate`, and derived
    `duration_min`/`min_to_rise`/`min_to_infl`/`min_to_fall`. Aggregate up via
    `core.features.agg_expert_moment` /
    `core.features.enrich_expert_events`."""
    raw = pl.read_csv(path, infer_schema_length=0, null_values=[""]).with_columns(
        pl.col("year").cast(pl.Int64, strict=False),
        pl.col("moment_idx").cast(pl.Int64, strict=False),
        pl.col("start_time").str.to_datetime(_DT_FMT, strict=False),
        pl.col("end_time").str.to_datetime(_DT_FMT, strict=False),
        pl.col("n_hot_moments").cast(pl.Int64, strict=False),
        pl.col("n_oxic_pulses").cast(pl.Int64, strict=False),
        pl.col("n_mixed_moments").cast(pl.Int64, strict=False),
        pl.col("concurrent_precip").cast(pl.Int8, strict=False).cast(pl.Boolean),
        pl.col("flooding").cast(pl.Int8, strict=False).cast(pl.Boolean),
        pl.col("xf_qc_checked").cast(pl.Int8, strict=False).cast(pl.Boolean),
        pl.col("m_peak_do_ts").str.to_datetime(_DT_FMT, strict=False),
        pl.col("m_do_inflection_ts").str.to_datetime(_DT_FMT, strict=False),
        pl.col("m_sal_at_peak").cast(pl.Float64, strict=False),
        pl.col("m_do_rise_pct").cast(pl.Float64, strict=False),
        pl.col("m_peak_do").cast(pl.Float64, strict=False),
        pl.col("m_consumption_rate").cast(pl.Float64, strict=False),
        pl.lit("expert").alias("source"),
    )

    print(f"{len(raw)} read from {path}...")

    # The workbook's hot-block timing columns ("Time of Peak DO" / "Time of DO Inflection
    # Point") sometimes hold something that is not a timestamp of this moment at all. The bulk
    # of it is one mistake: a `=end-start` DURATION formula left in the peak column, carrying
    # the `[h]:mm` elapsed-time number format. Every xlsx reader classifies that format as a
    # time and converts the bare day-fraction against the 1900 date system, so 2h20m comes back
    # as `1899-12-31 02:20:00` (13 moments, all oxic/mixed — events whose hot block should be
    # empty). The remainder are transcription slips: a wrong month or year (3 moments).
    # Neither is repairable from the sheet, and a stray value is worse than a missing one —
    # eda's event viewer layers the peak rule on a SHARED temporal x scale, so a single 1899
    # datum stretches the domain across ~120 years and squeezes the real event into a sub-pixel
    # sliver (the chart looks blank). Anything outside its own moment window becomes null.
    _has_win = pl.col("start_time").is_not_null() & pl.col("end_time").is_not_null()
    for _c in ("m_peak_do_ts", "m_do_inflection_ts"):
        _stray = (
            _has_win
            & pl.col(_c).is_not_null()
            & (
                (pl.col(_c) < pl.col("start_time"))
                | (pl.col(_c) > pl.col("end_time"))
            )
        )
        _ids = raw.filter(_stray)["event_id"].unique().sort().to_list()
        if _ids:
            print(
                f"  out-of-window `{_c}` nulled on {len(_ids)} event(s) "
                f"(report to Opti-O2): {', '.join(_ids)}"
            )
        raw = raw.with_columns(
            pl.when(_stray)
            .then(pl.lit(None, dtype=pl.Datetime("us")))
            .otherwise(pl.col(_c))
            .alias(_c)
        )

    # a moment row is any row whose `type` isn't "umbrella" (hot/oxic, or blank if untyped);
    # umbrella scalars come from each event's first row (the umbrella row, written first). The
    # shared `start_time`/`end_time` are aliased back to the stable `group_start`/`group_end`
    # output names downstream expects.
    is_umb = pl.col("type").fill_null("_moment_") == "umbrella"
    events = raw.filter(is_umb).drop(cs.starts_with("m_"))
    moments = raw.filter(~is_umb).with_columns(
        unit_id=pl.col("event_id") + "#" + pl.col("moment_idx").cast(pl.String)
    )

    print(f"{len(events['moment_idx'])} events with {len(moments)} moments detected")
    return events, moments


def read_workbook_summary(src):
    """The expert's own per-water-year tally from the workbook's `WY…summary` sheet, as a tidy
    frame `[water_year, class, n_expert]` plus a `total` row. `class` uses the MOMENT `type`
    vocabulary (hot / oxic / mixed) so it joins straight onto the parsed moments. Added in
    rev 07-31-26; returns an EMPTY frame for older workbooks that have no such sheet."""
    schema = {"water_year": pl.Int64, "class": pl.String, "n_expert": pl.Int64}
    xl = fastexcel.read_excel(src)
    sheet = next((s for s in xl.sheet_names if "summary" in s.lower()), None)
    if sheet is None:
        return pl.DataFrame(schema=schema)
    d = xl.load_sheet(sheet, header_row=None, dtypes="string").to_polars()
    # header row names the columns; the four data columns are
    # Water Year | Total # DO Events | # Hot Moments | # Oxic Events | # Mixed Events
    cols = ("total", "hot", "oxic", "mixed")
    rows = []
    for r in d.rows():
        if not (r and r[0] and str(r[0]).strip().isdigit()):
            continue  # header / blank
        wy = int(str(r[0]).strip())
        for i, cls in enumerate(cols, start=1):
            v = r[i] if i < len(r) else None
            if v is not None and str(v).strip():
                rows.append(
                    {"water_year": wy, "class": cls, "n_expert": int(float(v))}
                )
    return pl.DataFrame(rows, schema=schema)


def check_moment_counts(events, moments, src):
    """Reconcile the PARSED moments against the expert's own summary sheet.

    Returns a tidy frame `[water_year, class, n_parsed, n_expert, delta]`. Moments are
    attributed to a water year by their **cluster's** start (hence the `events` join, not the
    moment's own start): a cluster straddling 30 Sep — `2019-11h`, `2020-13m` — is counted
    whole by the expert in the water year it begins, and matching that convention is what
    makes the tallies line up. `class` uses our labels (hot / pulse / mixed) plus a `total` row.

    Known residual on rev 07-31-26: WY2021 mixed parses as 3 vs the sheet's 2 — the sheet's own
    WY2021 row reads total 20 while its classes sum to 19, so the 2 is a typo and 3 is what
    makes their total reconcile. Every other water year matches exactly."""
    _cstart = events.select("event_id", pl.col("start_time").alias("_cluster_start"))
    _wy = (
        pl.when(pl.col("_cluster_start").dt.month() >= 10)
        .then(pl.col("_cluster_start").dt.year() + 1)
        .otherwise(pl.col("_cluster_start").dt.year())
        .alias("water_year")
    )
    m = moments.join(_cstart, on="event_id", how="left").with_columns(
        _wy, pl.col("type").alias("class")
    )
    parsed = pl.concat(
        [
            m.group_by("water_year", "class").agg(pl.len().alias("n_parsed")),
            m.group_by("water_year")
            .agg(pl.len().alias("n_parsed"))
            .with_columns(pl.lit("total").alias("class")),
        ],
        how="diagonal",
    )
    return (
        read_workbook_summary(src)
        .join(parsed, on=["water_year", "class"], how="left")
        .with_columns(pl.col("n_parsed").fill_null(0))
        .with_columns((pl.col("n_parsed") - pl.col("n_expert")).alias("delta"))
        .sort("water_year", "class")
    )
