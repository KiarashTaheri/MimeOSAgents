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

MGDL_PER_MMOLL = 18.0182  # glucose molar mass (180.16 g/mol) / 10
READING_INTERVAL_MIN = 5  # Dexcom / Libre sample every 5 (or 15) minutes
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR

# Consensus thresholds (mmol/L): 54 / 70 / 180 / 250 mg/dL.
VERY_LOW, LOW, HIGH, VERY_HIGH = 3.0, 3.9, 10.0, 13.9
MIN_EVENT_MINUTES = 15  # consensus minimum duration of a low/high episode
STABLE_CV_PCT = 36  # consensus: CV at or below this is stable glucose
# A gap of more than this many sample intervals (sensor dropout) ends an episode.
MAX_EVENT_GAP_INTERVALS = 2

# Plausible reading range (20-600 mg/dL); anything outside is treated as a bad row.
MIN_VALID_MMOL_L, MAX_VALID_MMOL_L = 1.1, 33.3
# Dexcom writes 'Low'/'High' outside its measurable range (40 / 400 mg/dL).
SENSOR_LOW_MMOL_L, SENSOR_HIGH_MMOL_L = 2.2, 22.2

# Glucose Management Indicator (Bergenstal et al. 2018).
GMI_PCT_INTERCEPT, GMI_PCT_PER_MGDL = 3.38, 0.02345  # GMI(%) from mean glucose in mg/dL
GMI_MMOL_MOL_INTERCEPT, GMI_MMOL_MOL_PER_MMOLL = 12.71, 4.70587  # GMI(mmol/mol) from mean in mmol/L

# Time-of-day buckets: overnight 00-06, morning 06-12, afternoon 12-18, evening 18-24.
MORNING_START_HOUR, AFTERNOON_START_HOUR, EVENING_START_HOUR = 6, 12, 18

OUTPUT_DECIMALS = 1  # values the model quotes, and the evals match, are rounded to this

TimeOfDay = Literal["overnight", "morning", "afternoon", "evening"]


class GlucoseReading(BaseModel):
    timestamp: datetime
    mmol_l: float = Field(ge=MIN_VALID_MMOL_L, le=MAX_VALID_MMOL_L)


class CGMData(BaseModel):
    readings: list[GlucoseReading]
    skipped_rows: int = 0
    source_format: Literal["simple", "dexcom"] = "simple"


def time_of_day(timestamp: datetime) -> TimeOfDay:
    if timestamp.hour < MORNING_START_HOUR:
        return "overnight"
    if timestamp.hour < AFTERNOON_START_HOUR:
        return "morning"
    if timestamp.hour < EVENING_START_HOUR:
        return "afternoon"
    return "evening"


# --------------------------------------------------------------------------- loading


def _parse_dexcom_value(raw_value: str, unit: Literal["mmol", "mgdl"]) -> float:
    """Dexcom writes 'Low'/'High' when outside the sensor's measurable range."""
    raw_value = raw_value.strip().lower()
    if raw_value == "low":
        return SENSOR_LOW_MMOL_L
    if raw_value == "high":
        return SENSOR_HIGH_MMOL_L
    value = float(raw_value)
    return value / MGDL_PER_MMOLL if unit == "mgdl" else value


def load_cgm(path: Path | str) -> CGMData:
    """Load a simple `timestamp,glucose_mmol_l` CSV or a Dexcom Clarity export.

    Rows that fail to parse or validate are skipped and counted rather than aborting the load.
    """
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No rows found in CGM file {path}")

    columns = rows[0].keys()
    readings: list[GlucoseReading] = []
    skipped_rows = 0

    if "glucose_mmol_l" in columns:
        source_format: Literal["simple", "dexcom"] = "simple"
        for row in rows:
            try:
                readings.append(GlucoseReading(timestamp=row["timestamp"], mmol_l=row["glucose_mmol_l"]))
            except (ValidationError, KeyError):
                skipped_rows += 1
    elif "Timestamp (YYYY-MM-DDThh:mm:ss)" in columns:
        source_format = "dexcom"
        unit: Literal["mmol", "mgdl"] = "mmol" if "Glucose Value (mmol/L)" in columns else "mgdl"
        value_column = "Glucose Value (mmol/L)" if unit == "mmol" else "Glucose Value (mg/dL)"
        for row in rows:
            if row.get("Event Type") != "EGV":  # header/metadata/calibration rows
                continue
            try:
                readings.append(
                    GlucoseReading(
                        timestamp=row["Timestamp (YYYY-MM-DDThh:mm:ss)"],
                        mmol_l=_parse_dexcom_value(row[value_column], unit),
                    )
                )
            except (ValidationError, ValueError, KeyError):
                skipped_rows += 1
    else:
        raise ValueError(f"Unrecognised CGM CSV format, columns: {list(columns)}")

    if not readings:
        raise ValueError(f"No valid glucose readings in {path}")
    readings.sort(key=lambda reading: reading.timestamp)
    return CGMData(readings=readings, skipped_rows=skipped_rows, source_format=source_format)


def window(readings: list[GlucoseReading], days: int) -> list[GlucoseReading]:
    """Readings from the last `days` days, anchored to the most recent reading (not wall-clock now)."""
    if not readings:
        return []
    cutoff = readings[-1].timestamp - timedelta(days=days)
    return [reading for reading in readings if reading.timestamp > cutoff]


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
    cv_pct: float = Field(description=f"Coefficient of variation; <={STABLE_CV_PCT}% is considered stable.")
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
    severe_count: int = Field(
        description=f"Events crossing the very-low (<{VERY_LOW}) or very-high (>{VERY_HIGH}) level."
    )
    events: list[GlucoseEvent]


def _summarize_period(readings: list[GlucoseReading], days: int) -> Period:
    expected_readings = days * MINUTES_PER_DAY / READING_INTERVAL_MIN
    return Period(
        days=days,
        start=readings[0].timestamp,
        end=readings[-1].timestamp,
        n_readings=len(readings),
        coverage_pct=round(min(100.0, 100 * len(readings) / expected_readings), OUTPUT_DECIMALS),
    )


def _percent(count: int, total: int) -> float:
    return round(100 * count / total, OUTPUT_DECIMALS) if total else 0.0


def _require_readings(readings: list[GlucoseReading]) -> None:
    if not readings:
        raise ValueError("No glucose readings in the requested period.")


def average(readings: list[GlucoseReading], days: int) -> AverageStats:
    recent = window(readings, days)
    _require_readings(recent)
    glucose_values = [reading.mmol_l for reading in recent]
    mean = statistics.fmean(glucose_values)
    std_dev = statistics.pstdev(glucose_values)
    lowest = min(recent, key=lambda reading: reading.mmol_l)
    highest = max(recent, key=lambda reading: reading.mmol_l)
    return AverageStats(
        period=_summarize_period(recent, days),
        mean_mmol_l=round(mean, OUTPUT_DECIMALS),
        sd_mmol_l=round(std_dev, OUTPUT_DECIMALS),
        cv_pct=round(100 * std_dev / mean, OUTPUT_DECIMALS),
        gmi_pct=round(GMI_PCT_INTERCEPT + GMI_PCT_PER_MGDL * mean * MGDL_PER_MMOLL, OUTPUT_DECIMALS),
        gmi_mmol_mol=round(GMI_MMOL_MOL_INTERCEPT + GMI_MMOL_MOL_PER_MMOLL * mean, OUTPUT_DECIMALS),
        min=ReadingPoint(mmol_l=round(lowest.mmol_l, OUTPUT_DECIMALS), timestamp=lowest.timestamp),
        max=ReadingPoint(mmol_l=round(highest.mmol_l, OUTPUT_DECIMALS), timestamp=highest.timestamp),
    )


def _is_in_range(mmol_l: float) -> bool:
    return LOW <= mmol_l <= HIGH


def _in_range_percent(glucose_values: list[float]) -> float:
    return _percent(sum(map(_is_in_range, glucose_values)), len(glucose_values))


def time_in_range(readings: list[GlucoseReading], days: int) -> RangeStats:
    recent = window(readings, days)
    _require_readings(recent)
    glucose_values = [reading.mmol_l for reading in recent]
    n_readings = len(glucose_values)

    values_by_time_of_day: dict[TimeOfDay, list[float]] = {
        "overnight": [], "morning": [], "afternoon": [], "evening": []
    }
    values_by_date: dict[str, list[float]] = {}
    for reading in recent:
        values_by_time_of_day[time_of_day(reading.timestamp)].append(reading.mmol_l)
        values_by_date.setdefault(reading.timestamp.date().isoformat(), []).append(reading.mmol_l)

    in_range_pct_by_time_of_day = {
        part_of_day: _in_range_percent(values) for part_of_day, values in values_by_time_of_day.items() if values
    }
    in_range_count = sum(map(_is_in_range, glucose_values))
    return RangeStats(
        period=_summarize_period(recent, days),
        overall=RangeBreakdown(
            very_low_pct=_percent(sum(v < VERY_LOW for v in glucose_values), n_readings),
            low_pct=_percent(sum(VERY_LOW <= v < LOW for v in glucose_values), n_readings),
            in_range_pct=_percent(in_range_count, n_readings),
            high_pct=_percent(sum(HIGH < v <= VERY_HIGH for v in glucose_values), n_readings),
            very_high_pct=_percent(sum(v > VERY_HIGH for v in glucose_values), n_readings),
        ),
        hours_in_range=round(in_range_count * READING_INTERVAL_MIN / MINUTES_PER_HOUR, OUTPUT_DECIMALS),
        in_range_pct_by_time_of_day=in_range_pct_by_time_of_day,
        best_time_of_day=max(in_range_pct_by_time_of_day, key=in_range_pct_by_time_of_day.__getitem__),
        worst_time_of_day=min(in_range_pct_by_time_of_day, key=in_range_pct_by_time_of_day.__getitem__),
        daily=[
            DailyRange(
                date=date,
                in_range_pct=_in_range_percent(day_values),
                mean_mmol_l=round(statistics.fmean(day_values), OUTPUT_DECIMALS),
            )
            for date, day_values in sorted(values_by_date.items())
        ],
    )


def find_events(readings: list[GlucoseReading], days: int, kind: Literal["low", "high"]) -> EventReport:
    """Contiguous runs beyond the threshold lasting >= MIN_EVENT_MINUTES.

    A gap longer than MAX_EVENT_GAP_INTERVALS sample intervals (e.g. sensor dropout) ends an episode.
    """
    recent = window(readings, days)
    _require_readings(recent)
    is_low = kind == "low"
    threshold = LOW if is_low else HIGH
    severe_threshold = VERY_LOW if is_low else VERY_HIGH

    def is_beyond_threshold(mmol_l: float) -> bool:
        return mmol_l < threshold if is_low else mmol_l > threshold

    def is_severe(event: GlucoseEvent) -> bool:
        return event.extreme_mmol_l < severe_threshold if is_low else event.extreme_mmol_l > severe_threshold

    max_gap = timedelta(minutes=MAX_EVENT_GAP_INTERVALS * READING_INTERVAL_MIN)

    runs: list[list[GlucoseReading]] = []
    current_run: list[GlucoseReading] = []
    for reading in recent:
        continues_run = not current_run or reading.timestamp - current_run[-1].timestamp <= max_gap
        if is_beyond_threshold(reading.mmol_l) and continues_run:
            current_run.append(reading)
            continue
        if current_run:
            runs.append(current_run)
        current_run = [reading] if is_beyond_threshold(reading.mmol_l) else []
    if current_run:
        runs.append(current_run)

    events: list[GlucoseEvent] = []
    for run in runs:
        # A run of k samples spans k intervals of sensor time (each sample represents 5 min).
        duration_min = len(run) * READING_INTERVAL_MIN
        if duration_min < MIN_EVENT_MINUTES:
            continue
        run_values = [reading.mmol_l for reading in run]
        extreme = min(run_values) if is_low else max(run_values)
        events.append(
            GlucoseEvent(
                start=run[0].timestamp,
                end=run[-1].timestamp,
                duration_min=duration_min,
                extreme_mmol_l=round(extreme, OUTPUT_DECIMALS),
                time_of_day=time_of_day(run[0].timestamp),
            )
        )

    events_by_time_of_day = Counter(event.time_of_day for event in events)
    return EventReport(
        period=_summarize_period(recent, days),
        kind=kind,
        threshold_mmol_l=threshold,
        count=len(events),
        total_minutes=sum(event.duration_min for event in events),
        most_common_time_of_day=events_by_time_of_day.most_common(1)[0][0] if events_by_time_of_day else None,
        severe_count=sum(map(is_severe, events)),
        events=events,
    )
