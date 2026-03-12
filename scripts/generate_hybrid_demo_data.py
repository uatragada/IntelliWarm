import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# 2-story house layout with 4 zones.
ROOMS = [
    ("LivingRoom", 1),
    ("Kitchen", 1),
    ("DiningRoom", 1),
    ("Office", 2),
    ("Laundry", 2),
    ("GarageEntry", 2),
    ("MasterBedroom", 3),
    ("KidsBedroom", 3),
    ("GuestBedroom", 3),
    ("UpstairsHall", 4),
    ("Bathroom", 4),
]


def is_weekend(ts: datetime) -> bool:
    return ts.weekday() >= 5


def target_prob(room: str, zone: int, ts: datetime) -> float:
    """Rule-based occupancy profile to demonstrate hybrid zoning savings."""
    h = ts.hour
    weekend = is_weekend(ts)

    # Base low occupancy to keep idle zones mostly off.
    p = 0.04

    # Zone 3: upstairs bedrooms (night-heavy).
    if zone == 3:
        if 22 <= h or h < 6:
            p = 0.92
        elif 6 <= h < 8:
            p = 0.58
        elif 8 <= h < 17:
            p = 0.08 if not weekend else 0.22
        elif 17 <= h < 22:
            p = 0.38

    # Zone 1: downstairs shared spaces (morning/evening peaks).
    elif zone == 1:
        if 6 <= h < 8:
            p = 0.65
        elif 8 <= h < 17:
            p = 0.18 if not weekend else 0.52
        elif 17 <= h < 22:
            p = 0.82
        else:
            p = 0.06

    # Zone 2: office/utility (weekday office heavy, utility sporadic).
    elif zone == 2:
        if room == "Office":
            if 8 <= h < 17:
                p = 0.78 if not weekend else 0.16
            elif 19 <= h < 22:
                p = 0.30
            else:
                p = 0.05
        elif room == "Laundry":
            if (7 <= h < 9) or (18 <= h < 21):
                p = 0.22
            elif weekend and 10 <= h < 16:
                p = 0.34
            else:
                p = 0.07
        else:  # GarageEntry
            if (7 <= h < 9) or (17 <= h < 19):
                p = 0.26
            else:
                p = 0.05

    # Zone 4: upstairs transition areas (short intermittent use).
    elif zone == 4:
        if (6 <= h < 9) or (21 <= h < 23):
            p = 0.30
        elif 12 <= h < 14 and weekend:
            p = 0.22
        else:
            p = 0.09

    # Room-specific nudges.
    if room == "GuestBedroom":
        p *= 0.55
    if room == "DiningRoom" and not weekend and (12 <= h < 13):
        p += 0.18
    if room == "Bathroom":
        p += 0.04

    return max(0.01, min(0.98, p))


def next_state(prev: int, p_target: float) -> int:
    """Markov-like persistence so occupancy does not flicker unrealistically."""
    stickiness = 0.93
    if prev == 1:
        p = stickiness * 1.0 + (1.0 - stickiness) * p_target
    else:
        p = stickiness * 0.0 + (1.0 - stickiness) * p_target
    # Add tiny jitter
    p += random.uniform(-0.015, 0.015)
    p = max(0.0, min(1.0, p))
    return 1 if random.random() < p else 0


def generate_rows(start: datetime, end: datetime):
    states = {room: 0 for room, _ in ROOMS}

    ts = start
    while ts <= end:
        for room, zone in ROOMS:
            p = target_prob(room, zone, ts)
            states[room] = next_state(states[room], p)
            yield {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "room": room,
                "zone": zone,
                "occupied": states[room],
            }
        ts += timedelta(minutes=1)


def main():
    start = datetime(2025, 1, 1, 0, 0)
    end = datetime(2025, 1, 4, 23, 59)  # 4 days for richer timeline behavior

    output_path = "data-analytics/roommate_data/roommates_occupancy.csv"

    rows = list(generate_rows(start, end))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "room", "zone", "occupied"])
        writer.writeheader()
        writer.writerows(rows)

    unique_ts = len({r["timestamp"] for r in rows})
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Rooms: {len(ROOMS)} | Zones: 4 | Timeline points: {unique_ts}")


if __name__ == "__main__":
    main()
