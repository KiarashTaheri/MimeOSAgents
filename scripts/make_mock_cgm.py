"""Generate a reproducible 14-day mock CGM trace (5-min readings, mmol/L) at data/cgm_mock.csv.

Patterns baked in so the agent has something real to find:
- baseline ~5.8 mmol/L with smooth sensor noise and a dawn rise (04:00-08:00)
- three meals a day; some dinners are late and carb-heavy (peaks > 11, occasionally > 13.9)
- a few overnight lows (< 3.9), one of them < 3.0, plus one post-exercise afternoon dip
- a ~2 hour sensor gap on day 9
"""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 7
DAYS = 14
STEP = timedelta(minutes=5)
END = datetime(2026, 10, 9, 23, 55)
START = END - timedelta(days=DAYS) + STEP
OUT = Path(__file__).resolve().parent.parent / "data" / "cgm_mock.csv"


def bump(minutes_since: float, amplitude: float, peak_min: float) -> float:
    """Gamma-like glucose excursion: rises to `amplitude` at `peak_min`, then decays over a few hours."""
    if minutes_since < 0 or minutes_since > 6 * peak_min:
        return 0.0
    x = minutes_since / peak_min
    return amplitude * x * math.exp(1 - x)


def build_events(rng: random.Random) -> list[tuple[datetime, float, float]]:
    """(start time, amplitude mmol/L, peak minutes). Negative amplitude = a dip."""
    events = []
    late_dinner_days = {2, 5, 6, 9, 12}
    for d in range(DAYS):
        day = START.replace(hour=0, minute=0) + timedelta(days=d)
        breakfast = day + timedelta(hours=7, minutes=rng.randint(0, 60))
        lunch = day + timedelta(hours=12, minutes=rng.randint(0, 60))
        if d in late_dinner_days:
            dinner, dinner_amp = day + timedelta(hours=21, minutes=rng.randint(0, 40)), rng.uniform(6.0, 9.0)
        else:
            dinner, dinner_amp = day + timedelta(hours=18, minutes=rng.randint(30, 90)), rng.uniform(2.5, 4.0)
        events += [
            (breakfast, rng.uniform(2.5, 4.5), 50),
            (lunch, rng.uniform(2.0, 3.5), 55),
            (dinner, dinner_amp, 60),
        ]
    # Overnight lows (the late-dinner insulin/rebound story) and one exercise dip.
    for d, depth in [(3, -2.4), (7, -3.2), (10, -2.3), (13, -2.5)]:
        night = START.replace(hour=2, minute=0) + timedelta(days=d, minutes=rng.randint(0, 60))
        events.append((night, depth, 45))
    events.append((START.replace(hour=16, minute=0) + timedelta(days=11), -2.4, 35))
    return events


def main() -> None:
    rng = random.Random(SEED)
    events = build_events(rng)
    gap_start = START.replace(hour=10, minute=0) + timedelta(days=9)
    gap_end = gap_start + timedelta(hours=2)

    noise = 0.0
    rows = []
    t = START
    while t <= END:
        noise = 0.85 * noise + rng.gauss(0, 0.08)  # AR(1): smooth, sensor-like jitter
        hour = t.hour + t.minute / 60
        dawn = 0.7 * math.exp(-((hour - 6.5) ** 2) / 2)
        glucose = 5.8 + dawn + noise
        glucose += sum(bump((t - start).total_seconds() / 60, amp, peak) for start, amp, peak in events)
        glucose = min(max(glucose, 2.2), 22.2)
        if not gap_start <= t < gap_end:
            rows.append((t.isoformat(timespec="seconds"), f"{glucose:.1f}"))
        t += STEP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "glucose_mmol_l"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} readings ({START:%Y-%m-%d} to {END:%Y-%m-%d}) to {OUT}")


if __name__ == "__main__":
    main()
