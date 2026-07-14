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
    for _i in (15, 17, 21, 25):
        if final[_i] == f"unnamed_{_i}" and not final[_i - 1].startswith("unnamed_"):
            final[_i] = f"{final[_i - 1]} | Hour Average"
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
    """Fine expert suffix -> 3-class target, per the workbook LEGEND + Charles (2026-07-09):
      `hx` = 'unknown hot moment'            -> mixed  (ambiguous hot; the deck's Mixed Event)
      `e`  = 'oxygen event during flood'     -> pulse  (generally an oxic pulse, NOT a hot moment)
      `h`                                    -> hot
      `o1..o5` = oxic pulse (various)        -> pulse
      `b`, `x`                               -> ignore (dropped)
    Order matters: match `hx` before `h`. The LEGEND glosses `b`='hot moment', but every
    `b`-tagged event (2022-2b, 2022-9b) is BLACKED OUT in the workbook (the expert's ignore
    marker, fill theme=1) and shows no DO excursion, so `b` joins `x` in the ignore pile
    (user decision, 2026-07-11)."""
    s = suffix.lower()
    if s.startswith("hx"):
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


# raw-workbook column indices (positional; the sheet has no reliable header row)
_C_START, _C_END = 2, 3
_C_NHOT, _C_NOXIC = 9, 12
_HOT_COLS = (
    4,
    5,
    6,
    7,
    16,
    17,
)  # peak-DO time / inflection / end-DO time / salinity / peak DO / consumption
_OXIC_COLS = (10, 11)  # oxic begin / end
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
    "concurrent_precip",
    "flooding",
    "notes",
    "xf_qc_checked",
)
_MOM_FIELDS = (
    # moment-only extras; `type` and `start_time`/`end_time` are the shared columns above.
    # `m_peak_do_ts`/`m_do_inflection_ts` are the hot-only intermediate timings (blank for oxic).
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
                    "concurrent_precip": d[r, 13],
                    "flooding": d[r, 14],
                    "notes": d[r, 19],
                    "xf_qc_checked": d[r, 20],
                    "_urow_ox": (d[r, 10], d[r, 11]),
                    "moments": [],
                }
                umbrellas.append(cur)
                continue

            if cur is None:
                continue  # stray rows before the first umbrella

            has_data = any(d[r, c] is not None for c in _HOT_COLS + _OXIC_COLS)
            if mom_idx is None and not has_data and label_mat is None:
                continue  # blank separator / foot-of-sheet WY summary / legend row

            # Pattern C: adopt a count that slipped onto this sub-row when the event row's is null
            if cur["n_hot_moments"] is None and _isnum(d[r, _C_NHOT]):
                cur["n_hot_moments"] = d[r, _C_NHOT]
            if cur["n_oxic_pulses"] is None and _isnum(d[r, _C_NOXIC]):
                cur["n_oxic_pulses"] = d[r, _C_NOXIC]

            if label_mat is not None:  # Pattern E — suffix is this moment's type
                _suf = label_mat.group(1).lower()
                _typ = (
                    "hot"
                    if _suf[0] in ("h", "b")
                    else "oxic"
                    if _suf[0] in ("o", "e")
                    else None
                )
            elif any(d[r, c] is not None for c in _HOT_COLS):
                _typ = "hot"
            elif any(d[r, c] is not None for c in _OXIC_COLS):
                _typ = "oxic"
            else:
                _typ = None
            cur["moments"].append(
                {
                    "moment_idx": d[r, 1],
                    "moment_type": _typ,
                    # unified window: hot fills col 2/6, oxic fills col 10/11 (never both)
                    "start_time": d[r, 2] if d[r, 2] is not None else d[r, 10],
                    "end_time": d[r, 6] if d[r, 6] is not None else d[r, 11],
                    "m_peak_do_ts": d[r, 4],
                    "m_do_inflection_ts": d[r, 5],
                    "m_sal_at_peak": d[r, 7],
                    "m_do_rise_pct": d[r, 8],
                    "m_peak_do": d[r, 16],
                    "m_consumption_rate": d[r, 17],
                }
            )

    # Pattern A: single oxic pulse written on the event row itself (no sub-rows) -> one moment
    for u in umbrellas:
        _ob, _oe = u.pop("_urow_ox")
        if not u["moments"] and (_ob is not None or _oe is not None):
            u["moments"].append(
                {
                    "moment_idx": None,
                    "moment_type": "oxic",
                    "start_time": _ob,
                    "end_time": _oe,
                    "m_peak_do_ts": None,
                    "m_do_inflection_ts": None,
                    "m_sal_at_peak": None,
                    "m_do_rise_pct": None,
                    "m_peak_do": None,
                    "m_consumption_rate": None,
                }
            )

    # A single-moment oxic-pulse umbrella's lone moment IS the pulse -> oxic. The workbook
    # sometimes fills the hot measurement block (salinity-at-peak etc.) for it, which the
    # column heuristic would otherwise mis-type hot (2024-1o5 / 2025-3o1 — readout shapes confirm
    # oxic: rain-driven flood / flat-salinity flood). Multi-moment pulse umbrellas can genuinely
    # contain hot moments (2020-4o4, 2023-3o3), so only single-moment ones are coerced.
    for u in umbrellas:
        if u["expert_label"] == "pulse" and len(u["moments"]) == 1:
            u["moments"][0]["moment_type"] = "oxic"

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
            _act_hot = sum(m["moment_type"] == "hot" for m in u["moments"])
            _act_oxic = sum(m["moment_type"] == "oxic" for m in u["moments"])
            if (_decl_hot, _decl_oxic) != (_act_hot, _act_oxic):
                count_mismatch.append(u["event_id"])
            base["n_hot_moments"] = _act_hot
            base["n_oxic_pulses"] = _act_oxic
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
                _lab = {"hot": "hot", "oxic": "pulse"}.get(_typ, base["expert_label"])
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
