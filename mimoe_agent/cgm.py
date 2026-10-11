"""CGM data loading and deterministic glucose metrics (all values in mmol/L).

Targets follow the International Consensus on Time in Range (Battelino et al., 2019):
in range 3.9-10.0, low <3.9, very low <3.0, high >10.0, very high >13.9 mmol/L.
The LLM never computes these numbers; it only calls the tools that wrap this module.
"""

import csv
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

MGDL_PER_MMOLL = 18.0182
READING_INTERVAL_MIN = 5  # Dexcom / Libre sample every 5 (or 15) minutes

VERY_LOW, LOW, HIGH, VERY_HIGH = 3.0, 3.9, 10.0, 13.9
MIN_EVENT_MINUTES = 15

TimeOfDay = Literal["overnight", "morning", "afternoon", "evening"]


class GlucoseReading(BaseModel):
    timestamp: datetime
    mmol_l: float = Field(ge=1.1, le=33.3)


class CGMData(BaseModel):
    readings: list[GlucoseReading]
    skipped_rows: int = 0
    source_format: Literal["simple", "dexcom"] = "simple"


def time_of_day(ts: datetime) -> TimeOfDay:
    if ts.hour < 6:
        return "overnight"
    if ts.hour < 12:
        return "morning"
    if ts.hour < 18:
        return "afternoon"
    return "evening"


# --------------------------------------------------------------------------- loading


def _parse_dexcom_value(raw: str, unit: Literal["mmol", "mgdl"]) -> float:
    """Dexcom writes 'Low'/'High' when outside the sensor's measurable range."""
    raw = raw.strip()
    if raw.lower() == "low":
        return 2.2
    if raw.lower() == "high":
        return 22.2
    value = float(raw)
    return value / MGDL_PER_MMOLL if unit == "mgdl" else value


def load_cgm(path: Path | str) -> CGMData:
    """Load a simple `timestamp,glucose_mmol_l` CSV or a Dexcom Clarity export.

    Rows that fail to parse or validate are skipped and counted rather than aborting the load.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in CGM file {path}")

    columns = rows[0].keys()
    readings: list[GlucoseReading] = []
    skipped = 0

    if "glucose_mmol_l" in columns:
        fmt: Literal["simple", "dexcom"] = "simple"
        for row in rows:
            try:
                readings.append(GlucoseReading(timestamp=row["timestamp"], mmol_l=row["glucose_mmol_l"]))
            except (ValidationError, KeyError):
                skipped += 1
    elif "Timestamp (YYYY-MM-DDThh:mm:ss)" in columns:
        fmt = "dexcom"
        unit: Literal["mmol", "mgdl"] = "mmol" if "Glucose Value (mmol/L)" in columns else "mgdl"
        value_col = "Glucose Value (mmol/L)" if unit == "mmol" else "Glucose Value (mg/dL)"
        for row in rows:
            if row.get("Event Type") != "EGV":  # header/metadata/calibration rows
                continue
            try:
                readings.append(
                    GlucoseReading(
                        timestamp=row["Timestamp (YYYY-MM-DDThh:mm:ss)"],
                        mmol_l=_parse_dexcom_value(row[value_col], unit),
                    )
                )
            except (ValidationError, ValueError, KeyError):
                skipped += 1
    else:
        raise ValueError(f"Unrecognised CGM CSV format, columns: {list(columns)}")

    if not readings:
        raise ValueError(f"No valid glucose readings in {path}")
    readings.sort(key=lambda r: r.timestamp)
    return CGMData(readings=readings, skipped_rows=skipped, source_format=fmt)


def window(readings: list[GlucoseReading], days: int) -> list[GlucoseReading]:
    """Readings from the last `days` days, anchored to the most recent reading (not wall-clock now)."""
    if not readings:
        return []
    cutoff = readings[-1].timestamp - timedelta(days=days)
    return [r for r in readings if r.timestamp > cutoff]


# --------------------------------------------------------------------------- metrics


class Period(BaseModel):
    days: int
    start: datetime
    end: datetime
    n_readings: int
    coverage_pct: float = Field(description="Share of expected 5-min readings actually present.")


class ReadingPoint(BaseModel):
    mmol_l: float
    timestamp: datetime


class AverageStats(BaseModel):
    period: Period
    mean_mmol_l: float
    sd_mmol_l: float
    cv_pct: float = Field(description="Coefficient of variation; <=36% is considered stable.")
    gmi_pct: float = Field(description="Glucose Management Indicator (estimated HbA1c), in %.")
    gmi_mmol_mol: float = Field(description="Same GMI in IFCC units (mmol/mol).")
    min: ReadingPoint
    max: ReadingPoint


class RangeBreakdown(BaseModel):
    very_low_pct: float
    low_pct: float
    in_range_pct: float
    high_pct: float
    very_high_pct: float


class DailyRange(BaseModel):
    date: str
    in_range_pct: float
    mean_mmol_l: float


class RangeStats(BaseModel):
    period: Period
    target_mmol_l: str = f"{LOW}-{HIGH}"
    overall: RangeBreakdown
    hours_in_range: float
    in_range_pct_by_time_of_day: dict[TimeOfDay, float]
    best_time_of_day: TimeOfDay
    worst_time_of_day: TimeOfDay
    daily: list[DailyRange]


class GlucoseEvent(BaseModel):
    start: datetime
    end: datetime
    duration_min: int
    extreme_mmol_l: float = Field(description="Nadir for lows, peak for highs.")
    time_of_day: TimeOfDay


class EventReport(BaseModel):
    period: Period
    kind: Literal["low", "high"]
    threshold_mmol_l: float
    count: int
    total_minutes: int
    most_common_time_of_day: TimeOfDay | None
    severe_count: int = Field(description="Events crossing the very-low (<3.0) or very-high (>13.9) level.")
    events: list[GlucoseEvent]


def _r(x: float) -> float:
    return round(x, 1)


def _period(readings: list[GlucoseReading], days: int) -> Period:
    expected = days * 24 * 60 / READING_INTERVAL_MIN
    return Period(
        days=days,
        start=readings[0].timestamp,
        end=readings[-1].timestamp,
        n_readings=len(readings),
        coverage_pct=_r(min(100.0, 100 * len(readings) / expected)),
    )


def _pct(n: int, total: int) -> float:
    return _r(100 * n / total) if total else 0.0


def _require(readings: list[GlucoseReading]) -> None:
    if not readings:
        raise ValueError("No glucose readings in the requested period.")


def average(readings: list[GlucoseReading], days: int) -> AverageStats:
    data = window(readings, days)
    _require(data)
    values = [r.mmol_l for r in data]
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    lo = min(data, key=lambda r: r.mmol_l)
    hi = max(data, key=lambda r: r.mmol_l)
    return AverageStats(
        period=_period(data, days),
        mean_mmol_l=_r(mean),
        sd_mmol_l=_r(sd),
        cv_pct=_r(100 * sd / mean),
        # Bergenstal et al. 2018: GMI(%) = 3.38 + 0.02345 * mean glucose (mg/dL).
        gmi_pct=_r(3.38 + 0.02345 * mean * MGDL_PER_MMOLL),
        gmi_mmol_mol=_r(12.71 + 4.70587 * mean),
        min=ReadingPoint(mmol_l=_r(lo.mmol_l), timestamp=lo.timestamp),
        max=ReadingPoint(mmol_l=_r(hi.mmol_l), timestamp=hi.timestamp),
    )


def _in_range(v: float) -> bool:
    return LOW <= v <= HIGH


def time_in_range(readings: list[GlucoseReading], days: int) -> RangeStats:
    data = window(readings, days)
    _require(data)
    values = [r.mmol_l for r in data]
    n = len(values)

    by_tod: dict[TimeOfDay, list[float]] = {"overnight": [], "morning": [], "afternoon": [], "evening": []}
    by_day: dict[str, list[float]] = {}
    for r in data:
        by_tod[time_of_day(r.timestamp)].append(r.mmol_l)
        by_day.setdefault(r.timestamp.date().isoformat(), []).append(r.mmol_l)

    tod_tir = {tod: _pct(sum(map(_in_range, vs)), len(vs)) for tod, vs in by_tod.items() if vs}
    in_range = sum(map(_in_range, values))
    return RangeStats(
        period=_period(data, days),
        overall=RangeBreakdown(
            very_low_pct=_pct(sum(v < VERY_LOW for v in values), n),
            low_pct=_pct(sum(VERY_LOW <= v < LOW for v in values), n),
            in_range_pct=_pct(in_range, n),
            high_pct=_pct(sum(HIGH < v <= VERY_HIGH for v in values), n),
            very_high_pct=_pct(sum(v > VERY_HIGH for v in values), n),
        ),
        hours_in_range=_r(in_range * READING_INTERVAL_MIN / 60),
        in_range_pct_by_time_of_day=tod_tir,
        best_time_of_day=max(tod_tir, key=tod_tir.__getitem__),
        worst_time_of_day=min(tod_tir, key=tod_tir.__getitem__),
        daily=[
            DailyRange(date=d, in_range_pct=_pct(sum(map(_in_range, vs)), len(vs)), mean_mmol_l=_r(statistics.fmean(vs)))
            for d, vs in sorted(by_day.items())
        ],
    )


def find_events(readings: list[GlucoseReading], days: int, kind: Literal["low", "high"]) -> EventReport:
    """Contiguous runs beyond the threshold lasting >= 15 minutes.

    A gap longer than two sample intervals (e.g. sensor dropout) ends an episode.
    """
    data = window(readings, days)
    _require(data)
    threshold = LOW if kind == "low" else HIGH
    severe = VERY_LOW if kind == "low" else VERY_HIGH
    beyond = (lambda v: v < threshold) if kind == "low" else (lambda v: v > threshold)
    max_gap = timedelta(minutes=2 * READING_INTERVAL_MIN)

    runs: list[list[GlucoseReading]] = []
    current: list[GlucoseReading] = []
    for r in data:
        if beyond(r.mmol_l) and (not current or r.timestamp - current[-1].timestamp <= max_gap):
            current.append(r)
            continue
        if current:
            runs.append(current)
        current = [r] if beyond(r.mmol_l) else []
    if current:
        runs.append(current)

    events: list[GlucoseEvent] = []
    for run in runs:
        # A run of k samples spans k intervals of sensor time (each sample represents 5 min).
        duration = len(run) * READING_INTERVAL_MIN
        if duration < MIN_EVENT_MINUTES:
            continue
        pick = min if kind == "low" else max
        extreme = pick(r.mmol_l for r in run)
        events.append(
            GlucoseEvent(
                start=run[0].timestamp,
                end=run[-1].timestamp,
                duration_min=duration,
                extreme_mmol_l=_r(extreme),
                time_of_day=time_of_day(run[0].timestamp),
            )
        )

    tods = Counter(e.time_of_day for e in events)
    is_severe = (lambda e: e.extreme_mmol_l < severe) if kind == "low" else (lambda e: e.extreme_mmol_l > severe)
    return EventReport(
        period=_period(data, days),
        kind=kind,
        threshold_mmol_l=threshold,
        count=len(events),
        total_minutes=sum(e.duration_min for e in events),
        most_common_time_of_day=tods.most_common(1)[0][0] if tods else None,
        severe_count=sum(map(is_severe, events)),
        events=events,
    )
