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
from typing import NamedTuple

SEED = 7
DAYS = 14
STEP = timedelta(minutes=5)
END = datetime(2026, 10, 9, 23, 55)
START = END - timedelta(days=DAYS) + STEP
OUT = Path(__file__).resolve().parent.parent / "data" / "cgm_mock.csv"

BASELINE_MMOL_L = 5.8  # normal fasting glucose
NOISE_PERSISTENCE = 0.85  # share of the previous step's noise kept (AR(1)): smooth, sensor-like jitter
NOISE_SD_MMOL_L = 0.08  # new random noise added each 5-min step
DAWN_RISE_MMOL_L = 0.7  # dawn phenomenon: early-morning rise ...
DAWN_PEAK_HOUR = 6.5  # ... centred at 06:30
DAWN_WIDTH_HOURS = 1.0  # ... spread (standard deviation) of the bell curve
BUMP_CUTOFF_PEAKS = 6  # a meal/dip effect is ~0 after 6x its time-to-peak, so stop computing it
SENSOR_MIN_MMOL_L, SENSOR_MAX_MMOL_L = 2.2, 22.2  # real sensors cannot read outside 40-400 mg/dL
GAP_DAY, GAP_START_HOUR, GAP_HOURS = 9, 10, 2  # simulated sensor dropout



class Meal(NamedTuple):
    start_hour: int
    delay_range_min: tuple[int, int]  # random delay after start_hour
    rise_range_mmol_l: tuple[float, float]  # random size of the glucose rise
    minutes_to_peak: int


BREAKFAST = Meal(start_hour=7, delay_range_min=(0, 60), rise_range_mmol_l=(2.5, 4.5), minutes_to_peak=50)
LUNCH = Meal(start_hour=12, delay_range_min=(0, 60), rise_range_mmol_l=(2.0, 3.5), minutes_to_peak=55)
NORMAL_DINNER = Meal(start_hour=18, delay_range_min=(30, 90), rise_range_mmol_l=(2.5, 4.0), minutes_to_peak=60)
# Late and carb-heavy: drives the evening highs.
LATE_DINNER = Meal(start_hour=21, delay_range_min=(0, 40), rise_range_mmol_l=(6.0, 9.0), minutes_to_peak=60)
LATE_DINNER_DAYS = {2, 5, 6, 9, 12}

# Overnight lows as (day index, depth mmol/L). The -3.2 one goes below 3.0 so there is a severe low to find.
OVERNIGHT_LOWS = [(3, -2.4), (7, -3.2), (10, -2.3), (13, -2.5)]
OVERNIGHT_LOW_START_HOUR, OVERNIGHT_LOW_DELAY_MAX_MIN, OVERNIGHT_LOW_MINUTES_TO_NADIR = 2, 60, 45
EXERCISE_DIP_DAY, EXERCISE_DIP_HOUR, EXERCISE_DIP_MMOL_L, EXERCISE_DIP_MINUTES_TO_NADIR = 11, 16, -2.4, 35


def glucose_excursion(minutes_since: float, amplitude: float, peak_min: float) -> float:
    """Gamma-like glucose excursion: rises to `amplitude` at `peak_min`, then decays over a few hours."""
    if minutes_since < 0 or minutes_since > BUMP_CUTOFF_PEAKS * peak_min:
        return 0.0
    time_in_peaks = minutes_since / peak_min
    return amplitude * time_in_peaks * math.exp(1 - time_in_peaks)


def build_events(rng: random.Random) -> list[tuple[datetime, float, float]]:
    """(start time, amplitude mmol/L, peak minutes). Negative amplitude = a dip."""
    events = []
    for day_index in range(DAYS):
        midnight = START.replace(hour=0, minute=0) + timedelta(days=day_index)
        dinner = LATE_DINNER if day_index in LATE_DINNER_DAYS else NORMAL_DINNER

        def meal_start(meal: Meal) -> datetime:
            return midnight + timedelta(hours=meal.start_hour, minutes=rng.randint(*meal.delay_range_min))

        # The order of rng calls is fixed so the seed keeps reproducing the committed CSV exactly.
        breakfast_start, lunch_start, dinner_start = meal_start(BREAKFAST), meal_start(LUNCH), meal_start(dinner)
        dinner_rise = rng.uniform(*dinner.rise_range_mmol_l)
        events += [
            (breakfast_start, rng.uniform(*BREAKFAST.rise_range_mmol_l), BREAKFAST.minutes_to_peak),
            (lunch_start, rng.uniform(*LUNCH.rise_range_mmol_l), LUNCH.minutes_to_peak),
            (dinner_start, dinner_rise, dinner.minutes_to_peak),
        ]
    # Overnight lows (the late-dinner insulin/rebound story) and one exercise dip.
    for day_index, depth in OVERNIGHT_LOWS:
        delay_min = rng.randint(0, OVERNIGHT_LOW_DELAY_MAX_MIN)
        first_night = START.replace(hour=OVERNIGHT_LOW_START_HOUR, minute=0)
        low_start = first_night + timedelta(days=day_index, minutes=delay_min)
        events.append((low_start, depth, OVERNIGHT_LOW_MINUTES_TO_NADIR))
    exercise_start = START.replace(hour=EXERCISE_DIP_HOUR, minute=0) + timedelta(days=EXERCISE_DIP_DAY)
    events.append((exercise_start, EXERCISE_DIP_MMOL_L, EXERCISE_DIP_MINUTES_TO_NADIR))
    return events


def main() -> None:
    rng = random.Random(SEED)
    events = build_events(rng)
    gap_start = START.replace(hour=GAP_START_HOUR, minute=0) + timedelta(days=GAP_DAY)
    gap_end = gap_start + timedelta(hours=GAP_HOURS)

    noise = 0.0
    rows = []
    timestamp = START
    while timestamp <= END:
        noise = NOISE_PERSISTENCE * noise + rng.gauss(0, NOISE_SD_MMOL_L)
        hour = timestamp.hour + timestamp.minute / 60
        dawn = DAWN_RISE_MMOL_L * math.exp(-((hour - DAWN_PEAK_HOUR) ** 2) / (2 * DAWN_WIDTH_HOURS**2))
        glucose = BASELINE_MMOL_L + dawn + noise
        glucose += sum(
            glucose_excursion((timestamp - start).total_seconds() / 60, amplitude, peak)
            for start, amplitude, peak in events
        )
        glucose = min(max(glucose, SENSOR_MIN_MMOL_L), SENSOR_MAX_MMOL_L)
        if not gap_start <= timestamp < gap_end:
            rows.append((timestamp.isoformat(timespec="seconds"), f"{glucose:.1f}"))
        timestamp += STEP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "glucose_mmol_l"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} readings ({START:%Y-%m-%d} to {END:%Y-%m-%d}) to {OUT}")


if __name__ == "__main__":
    main()
