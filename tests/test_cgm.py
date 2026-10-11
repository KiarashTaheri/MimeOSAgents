from datetime import datetime, timedelta

import pytest

from mimoe_agent import cgm

T0 = datetime(2026, 10, 1, 0, 0)


def series(values: list[float], start: datetime = T0) -> list[cgm.GlucoseReading]:
    return [cgm.GlucoseReading(timestamp=start + timedelta(minutes=5 * i), mmol_l=v) for i, v in enumerate(values)]


def test_time_in_range_buckets():
    # 10 readings: 1 very low, 1 low, 6 in range, 1 high, 1 very high
    readings = series([2.8, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0])
    stats = cgm.time_in_range(readings, days=1)
    assert stats.overall.very_low_pct == 10.0
    assert stats.overall.low_pct == 10.0
    assert stats.overall.in_range_pct == 60.0  # 3.9 and 10.0 are inclusive bounds
    assert stats.overall.high_pct == 10.0
    assert stats.overall.very_high_pct == 10.0
    assert stats.hours_in_range == 0.5  # 6 readings x 5 min


def test_average_and_gmi():
    stats = cgm.average(series([7.0] * 12), days=1)
    assert stats.mean_mmol_l == 7.0
    assert stats.cv_pct == 0.0
    # GMI(%) = 3.38 + 0.02345 * 126.1 mg/dL = 6.3 ; IFCC = 12.71 + 4.70587 * 7 = 45.7
    assert stats.gmi_pct == 6.3
    assert stats.gmi_mmol_mol == 45.7


def test_window_is_anchored_to_latest_reading():
    readings = series([5.0] * (3 * 288))  # 3 days of 5-min readings
    assert len(cgm.window(readings, 1)) == 288
    assert cgm.window(readings, 1)[-1] == readings[-1]


def test_low_events_require_15_minutes():
    # 2 readings low (10 min, ignored), back in range, then 4 readings low (20 min, counted)
    readings = series([5.0, 3.5, 3.4, 5.0, 5.0, 3.6, 3.2, 2.9, 3.5, 5.0])
    report = cgm.find_events(readings, days=1, kind="low")
    assert report.count == 1
    event = report.events[0]
    assert event.duration_min == 20
    assert event.extreme_mmol_l == 2.9
    assert report.severe_count == 1  # dropped below 3.0
    assert event.time_of_day == "overnight"


def test_sensor_gap_splits_events():
    before = series([11.0] * 4)
    after = series([11.0] * 4, start=before[-1].timestamp + timedelta(minutes=60))
    report = cgm.find_events(before + after, days=1, kind="high")
    assert report.count == 2


def test_no_readings_in_window_raises():
    with pytest.raises(ValueError):
        cgm.average([], days=7)


def test_load_simple_csv_skips_bad_rows(tmp_path):
    path = tmp_path / "cgm.csv"
    path.write_text("timestamp,glucose_mmol_l\n2026-10-01T00:00:00,5.5\n2026-10-01T00:05:00,not-a-number\n2026-10-01T00:10:00,99\n")
    data = cgm.load_cgm(path)
    assert [r.mmol_l for r in data.readings] == [5.5]
    assert data.skipped_rows == 2  # unparseable + out of physiological range


def test_load_dexcom_mgdl_export_converts_to_mmol(tmp_path):
    path = tmp_path / "clarity.csv"
    path.write_text(
        "Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL)\n"
        "1,,FirstName,\n"
        "2,2026-10-01T00:00:00,EGV,90\n"
        "3,2026-10-01T00:05:00,EGV,Low\n"
        "4,2026-10-01T00:10:00,Calibration,100\n"
    )
    data = cgm.load_cgm(path)
    assert data.source_format == "dexcom"
    assert [round(r.mmol_l, 1) for r in data.readings] == [5.0, 2.2]


def test_unknown_csv_format(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unrecognised"):
        cgm.load_cgm(path)
