"""Reproducible, schema-disjoint text-to-SQL SFT dataset.

Only synthetic training domains are used. None of the five held-out
benchmark tables/questions/results are loaded by this generator.

Runs on Python 3.10+ with standard library + pandas (already in MVP).
Optional: DuckDB/SQLGlot for full application-level reference validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3

# Imports deliberately reuse the production prompt and schema formatter.
from dataset_utils import load_file
from llm_client import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "data"

# Domain names and schemas are purposely disjoint from the five benchmark
# schemas: retail_transactions, saas_invoices, staff_shifts,
# support_requests, warehouse_inventory.
# Each domain: (subject, item values, locations, channels, statuses, role synonyms)
DOMAINS = {
    "cinema_bookings": (
        "cinema booking", ["Matinee", "Premiere", "LateShow", "Festival"],
        ["Uptown", "Riverside", "Midtown", "OldTown"], ["App", "Kiosk", "Web"],
        ["Confirmed", "Pending", "Cancelled"],
        ["screening", "show_title", "feature", "film_name", "session_name", "program_name"],
    ),
    "cycling_rentals": (
        "bike rental", ["RoadBike", "CityBike", "CargoBike", "EBike"],
        ["DepotA", "DepotB", "DepotC", "DepotD"], ["App", "Counter", "Partner"],
        ["Completed", "Reserved", "Cancelled"],
        ["bike_type", "cycle_model", "ride_class", "rental_item", "cycle_type", "bike_model"],
    ),
    "hotel_stays": (
        "hotel stay", ["Standard", "Deluxe", "Suite", "Family"],
        ["Harbor", "Park", "Central", "Summit"], ["Direct", "Agency", "Web"],
        ["CheckedOut", "Booked", "Cancelled"],
        ["room_class", "suite_type", "lodging_type", "accommodation", "room_category", "stay_type"],
    ),
    "cafe_catering": (
        "catering order", ["Sandwich", "Salad", "Wrap", "Pastry"],
        ["Airport", "Campus", "Downtown", "Marina"], ["Phone", "Online", "WalkIn"],
        ["Delivered", "Preparing", "Cancelled"],
        ["menu_item", "dish_name", "food_item", "meal", "catering_item", "menu_choice"],
    ),
    "pet_grooming": (
        "grooming visit", ["Trim", "Bath", "Nails", "Spa"],
        ["NorthBranch", "SouthBranch", "HillBranch", "LakeBranch"], ["Phone", "App", "Web"],
        ["Finished", "Booked", "NoShow"],
        ["service_name", "treatment", "grooming_type", "package_name", "service_package", "visit_type"],
    ),
    "furniture_orders": (
        "furniture reservation", ["Desk", "Chair", "Sofa", "Lamp"],
        ["ShowroomA", "ShowroomB", "ShowroomC", "ShowroomD"], ["Web", "Store", "Phone"],
        ["Fulfilled", "Open", "Returned"],
        ["furniture_type", "model_name", "piece", "item_type", "furnishing", "catalog_item"],
    ),
    "museum_visits": (
        "museum ticket", ["Gallery", "Exhibition", "Workshop", "Tour"],
        ["EastWing", "WestWing", "Atrium", "Annex"], ["Desk", "Online", "Group"],
        ["Used", "Reserved", "Refunded"],
        ["experience", "attraction", "ticket_type", "exhibit", "program", "museum_activity"],
    ),
    "fitness_classes": (
        "fitness class registration", ["Pilates", "Yoga", "Spin", "HIIT"],
        ["StudioOne", "StudioTwo", "StudioThree", "StudioFour"], ["App", "FrontDesk", "Web"],
        ["Attended", "Booked", "Missed"],
        ["class_title", "workout", "session_type", "class_name", "course", "fitness_session"],
    ),
    "music_lessons": (
        "music lesson", ["Piano", "Violin", "Guitar", "Drums"],
        ["HallA", "HallB", "HallC", "HallD"], ["Phone", "Online", "Desk"],
        ["Complete", "Scheduled", "Skipped"],
        ["instrument", "lesson_type", "music_course", "course_title", "subject_name", "tuition_type"],
    ),
    # Validation domains are completely absent from training.
    "beach_equipment": (
        "equipment rental", ["Umbrella", "Kayak", "Board", "Tent"],
        ["PierA", "PierB", "PierC", "PierD"], ["App", "Desk", "Hotel"],
        ["Returned", "Rented", "Lost"],
        ["gear_type", "equipment", "rental_product", "gear", "asset_type", "equipment_name"],
    ),
    "language_courses": (
        "language enrollment", ["Spanish", "German", "French", "Italian"],
        ["CampusA", "CampusB", "CampusC", "CampusD"], ["Web", "Advisor", "Phone"],
        ["Finished", "Enrolled", "Dropped"],
        ["language", "course_name", "subject", "class_type", "language_course", "curriculum"],
    ),
    "garden_supplies": (
        "garden supply order", ["Compost", "Seeds", "Mulch", "Planter"],
        ["NurseryA", "NurseryB", "NurseryC", "NurseryD"], ["Web", "Shop", "Wholesale"],
        ["Shipped", "Pending", "Refunded"],
        ["supply_type", "product_name", "garden_item", "catalog_name", "planting_item", "supply_name"],
    ),
    # Completely unseen final-test domains: do not use for model selection.
    "art_exhibitions": (
        "art exhibition entry", ["Portraits", "Sculpture", "Photography", "Modern"],
        ["GalleryA", "GalleryB", "GalleryC", "GalleryD"], ["Web", "Desk", "Member"],
        ["Admitted", "Reserved", "Refunded"],
        ["art_program", "art_show", "exhibit_title", "collection", "exhibition", "gallery_exhibit"],
    ),
    "pool_reservations": (
        "pool reservation", ["LapSwim", "FamilySwim", "Diving", "Lessons"],
        ["PoolOne", "PoolTwo", "PoolThree", "PoolFour"], ["App", "Web", "Desk"],
        ["Used", "Reserved", "Cancelled"],
        ["pool_service", "session_name", "aquatics_event", "slot_type", "pool_activity", "program_type"],
    ),
    "vehicle_cleaning": (
        "vehicle wash", ["Basic", "Deluxe", "Wax", "Detail"],
        ["WashA", "WashB", "WashC", "WashD"], ["Phone", "Kiosk", "Web"],
        ["Done", "Scheduled", "Skipped"],
        ["wash_program", "cleaning_package", "wash_type", "service_tier", "wash_option", "detailing_type"],
    ),
}
TRAIN_DOMAINS = tuple(list(DOMAINS)[:9])
VAL_DOMAINS = tuple(list(DOMAINS)[9:12])
TEST_DOMAINS = tuple(list(DOMAINS)[12:])
ROLE_NAMES = ["item", "place", "channel", "units", "charge", "cost", "status", "date"]
ROLE_ALIASES = {
    "place": ["location", "site", "facility", "branch", "venue", "center"],
    "channel": ["booking_source", "order_channel", "source", "request_method", "access_channel", "intake_channel"],
    "units": ["unit_count", "units", "quantity", "seats", "number_of_units", "total_units"],
    "charge": ["charge_amount", "total_amount", "fee", "gross_amount", "amount", "price_total"],
    "cost": ["unit_cost", "purchase_price", "cost_per_unit", "base_cost", "acquisition_cost", "unit_expense"],
    "status": ["record_status", "state", "booking_status", "fulfillment_state", "entry_status", "completion_status"],
    "date": ["recorded_on", "event_date", "created_date", "entry_date", "transaction_date", "date_placed"],
}


def _render_schema_and_csv(domain: str, variant: int, seed: int) -> tuple[str, bytes, dict, dict]:
    subject, items, places, channels, statuses, item_names = DOMAINS[domain]
    roles = {"item": item_names[variant]}
    # Vary role labels independently by domain and variant; do not alias to
    # benchmark table names. All SQL identifier names are simple ASCII.
    for role, names in ROLE_ALIASES.items():
        roles[role] = names[(variant + sum(map(ord, domain))) % len(names)]
    if len(set(roles.values())) != 8:
        raise AssertionError(f"duplicate column for {domain} v{variant}")
    rng = random.Random(int(hashlib.sha256(f"{seed}:{domain}:{variant}".encode()).hexdigest()[:16], 16))
    rows = []
    for i in range(76):
        d = date(2025, 1, 1) + timedelta(days=(i * 9 + variant * 7) % 360)
        row = {
            "item": items[(i + rng.randint(0, 6)) % 4],
            "place": places[(i * 3 + rng.randint(0, 6)) % 4],
            "channel": channels[rng.randrange(3)],
            "units": rng.randint(1, 8),
            "charge": rng.randint(15, 180) * 10,
            "cost": rng.randint(5, 55),
            "status": statuses[rng.randrange(3)],
            "date": d.isoformat(),
        }
        rows.append({roles[role]: row[role] for role in ROLE_NAMES})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=[roles[role] for role in ROLE_NAMES])
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    loaded = load_file(domain + ".csv", data)
    if list(loaded.frame.columns) != list(w.fieldnames):
        raise AssertionError("Training columns did not survive actual CSV loader")
    values = {"subject": subject, "items": items, "places": places,
              "channels": channels, "statuses": statuses}
    return loaded.schema, data, roles, values


def _examples(roles: dict, values: dict) -> list[tuple[str, str, str]]:
    """(skill, natural-language question, gold SQLite/DuckDB SELECT) examples.

    Each question is explicit about requested metrics; no arbitrary assumptions
    about statuses are baked into the SQL. HAVING, top-N, arithmetic and
    percentages are deliberately overrepresented.
    """
    i, p, ch, u, a, c, st, dt = (roles[x] for x in ROLE_NAMES)
    subject = values["subject"]
    item = values["items"][0]
    place = values["places"][1]
    channel = values["channels"][0]
    state = values["statuses"][0]
    # SQLite / DuckDB common subset; output aliases support readable results.
    return [
        ("count", f"How many {subject} records are in the dataset?", "SELECT COUNT(*) AS records FROM dataset"),
        ("count", f"Count {subject} rows, not the number of units.", "SELECT COUNT(*) AS records FROM dataset"),
        ("sum", f"How many total units are recorded for {subject}s?", f"SELECT SUM({u}) AS total_units FROM dataset"),
        ("sum", f"What is the sum of {a} across all records?", f"SELECT SUM({a}) AS total_amount FROM dataset"),
        ("avg", f"What is the average {a} per record?", f"SELECT AVG({a}) AS avg_amount FROM dataset"),
        ("distinct", f"How many distinct {i} values are there?", f"SELECT COUNT(DISTINCT {i}) AS unique_items FROM dataset"),
        ("filter", f"How many {subject} records have {st} equal to {state}?", f"SELECT COUNT(*) FROM dataset WHERE {st} = '{state}'"),
        ("filter", f"Count records booked through {channel}.", f"SELECT COUNT(*) FROM dataset WHERE {ch} = '{channel}'"),
        ("filter_sum", f"Sum {a} for {i} {item}.", f"SELECT SUM({a}) FROM dataset WHERE {i} = '{item}'"),
        ("filter_units", f"How many units are associated with {i} {item}?", f"SELECT SUM({u}) FROM dataset WHERE {i} = '{item}'"),
        ("multi_filter", f"How many {state} records came through {channel}?", f"SELECT COUNT(*) FROM dataset WHERE {st} = '{state}' AND {ch} = '{channel}'"),
        ("multi_filter", f"Total {a} for {place} records with status {state}.", f"SELECT SUM({a}) FROM dataset WHERE {p} = '{place}' AND {st} = '{state}'"),
        ("date", f"How many entries have {dt} from April through June 2025?", f"SELECT COUNT(*) FROM dataset WHERE {dt} >= '2025-04-01' AND {dt} < '2025-07-01'"),
        ("date", f"Total {a} for entries recorded before July 2025.", f"SELECT SUM({a}) FROM dataset WHERE {dt} < '2025-07-01'"),
        ("group", f"Show total {a} for every {p}, ordered alphabetically by {p}.", f"SELECT {p}, SUM({a}) AS total FROM dataset GROUP BY {p} ORDER BY {p}"),
        ("group", f"Count records per {ch} and sort by {ch}.", f"SELECT {ch}, COUNT(*) AS n FROM dataset GROUP BY {ch} ORDER BY {ch}"),
        ("group", f"List all {i} values with their total unit counts, sorted by {i}.", f"SELECT {i}, SUM({u}) AS total_units FROM dataset GROUP BY {i} ORDER BY {i}"),
        ("group", f"What is the average charge per {p}? Sort by {p}.", f"SELECT {p}, AVG({a}) AS avg_charge FROM dataset GROUP BY {p} ORDER BY {p}"),
        ("group_top", f"Which {i} generated the greatest sum of {a}? Include the total.", f"SELECT {i}, SUM({a}) AS total FROM dataset GROUP BY {i} ORDER BY total DESC, {i} LIMIT 1"),
        ("group_top", f"What is the top {p} by number of rows, including the count?", f"SELECT {p}, COUNT(*) AS n FROM dataset GROUP BY {p} ORDER BY n DESC, {p} LIMIT 1"),
        ("group_top", f"Show the top three {i} by total {u} in descending order.", f"SELECT {i}, SUM({u}) AS total FROM dataset GROUP BY {i} ORDER BY total DESC, {i} LIMIT 3"),
        ("group_top", f"Which two {p} values have the largest sum of {a}?", f"SELECT {p}, SUM({a}) AS total FROM dataset GROUP BY {p} ORDER BY total DESC, {p} LIMIT 2"),
        ("group_top", f"Rank {ch} values by number of records, highest first.", f"SELECT {ch}, COUNT(*) AS n FROM dataset GROUP BY {ch} ORDER BY n DESC, {ch}"),
        ("group_top", f"Which {i} has the highest average {a}, and what is that average?", f"SELECT {i}, AVG({a}) AS avg_amount FROM dataset GROUP BY {i} ORDER BY avg_amount DESC, {i} LIMIT 1"),
        ("having", f"Which {i} values have total {a} above 10000? Return the totals.", f"SELECT {i}, SUM({a}) AS total FROM dataset GROUP BY {i} HAVING SUM({a}) > 10000 ORDER BY {i}"),
        ("having", f"Show {p} values with at least 15 records. Include each count.", f"SELECT {p}, COUNT(*) AS n FROM dataset GROUP BY {p} HAVING COUNT(*) >= 15 ORDER BY {p}"),
        ("having", f"List {ch} sources whose total {u} is greater than 80.", f"SELECT {ch}, SUM({u}) AS total FROM dataset GROUP BY {ch} HAVING SUM({u}) > 80 ORDER BY {ch}"),
        ("having", f"Find {i} items with average {a} over 750, sorted by name.", f"SELECT {i}, AVG({a}) AS avg_amount FROM dataset GROUP BY {i} HAVING AVG({a}) > 750 ORDER BY {i}"),
        ("having", f"Which {st} states appear more than 22 times? Show their counts.", f"SELECT {st}, COUNT(*) AS n FROM dataset GROUP BY {st} HAVING COUNT(*) > 22 ORDER BY {st}"),
        ("having", f"Which {i} values have more than 65 units combined?", f"SELECT {i}, SUM({u}) AS total FROM dataset GROUP BY {i} HAVING SUM({u}) > 65 ORDER BY {i}"),
        ("arithmetic", f"Calculate total acquisition cost using each row's {u} multiplied by {c}.", f"SELECT SUM({u} * {c}) AS total_cost FROM dataset"),
        ("arithmetic", f"Calculate the total charge less total unit cost, multiplying {u} by {c} on every row.", f"SELECT SUM({a} - {u} * {c}) AS margin FROM dataset"),
        ("arithmetic", f"What is the average cost per unit, weighting {c} by {u}?", f"SELECT SUM({u} * {c}) * 1.0 / NULLIF(SUM({u}), 0) AS weighted_cost FROM dataset"),
        ("arithmetic", f"Show estimated total unit costs by {p}, as sum of units times cost per unit.", f"SELECT {p}, SUM({u} * {c}) AS total_cost FROM dataset GROUP BY {p} ORDER BY {p}"),
        ("conditional", f"What percent of records have {st} equal to {state}?", f"SELECT ROUND(100.0 * SUM(CASE WHEN {st} = '{state}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS pct FROM dataset"),
        ("conditional", f"What percent of {subject} records used {channel}? Round to two decimals.", f"SELECT ROUND(100.0 * SUM(CASE WHEN {ch} = '{channel}' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS pct FROM dataset"),
        ("conditional", f"How many rows have {a} at least 1000?", f"SELECT SUM(CASE WHEN {a} >= 1000 THEN 1 ELSE 0 END) AS n FROM dataset"),
        ("conditional", f"For each {p}, count records where {st} is {state}.", f"SELECT {p}, SUM(CASE WHEN {st} = '{state}' THEN 1 ELSE 0 END) AS n FROM dataset GROUP BY {p} ORDER BY {p}"),
        ("filter_top", f"Which {i} has the highest total {a} among {state} records?", f"SELECT {i}, SUM({a}) AS total FROM dataset WHERE {st} = '{state}' GROUP BY {i} ORDER BY total DESC, {i} LIMIT 1"),
        ("filter_top", f"Among records from {channel}, show the top two {p} values by number of rows.", f"SELECT {p}, COUNT(*) AS n FROM dataset WHERE {ch} = '{channel}' GROUP BY {p} ORDER BY n DESC, {p} LIMIT 2"),
        ("filter_having", f"Which {p} have more than 10 {state} records? Include counts.", f"SELECT {p}, COUNT(*) AS n FROM dataset WHERE {st} = '{state}' GROUP BY {p} HAVING COUNT(*) > 10 ORDER BY {p}"),
        ("filter_having", f"Which {i} values have a total charge above 2000 from {channel} entries?", f"SELECT {i}, SUM({a}) AS total FROM dataset WHERE {ch} = '{channel}' GROUP BY {i} HAVING SUM({a}) > 2000 ORDER BY {i}"),
        ("no_implicit_filter", f"Sum {a} over ALL records, including cancelled or incomplete ones.", f"SELECT SUM({a}) AS total FROM dataset"),
        ("no_implicit_filter", f"Which {p} has the largest total {u} across ALL statuses?", f"SELECT {p}, SUM({u}) AS total FROM dataset GROUP BY {p} ORDER BY total DESC, {p} LIMIT 1"),
        ("no_implicit_filter", f"How many records exist across ALL {st} values?", "SELECT COUNT(*) AS records FROM dataset"),
        ("no_implicit_filter", f"Rank {i} values by total {a}, regardless of {st}.", f"SELECT {i}, SUM({a}) AS total FROM dataset GROUP BY {i} ORDER BY total DESC, {i}"),
    ]


def _sqlite_validate(raw_csv: bytes, sqls: list[str], numeric_cols: tuple[str, ...]) -> None:
    rdr = csv.DictReader(io.StringIO(raw_csv.decode("utf-8")))
    rows = list(rdr)
    quoted = [f'"{col}" {"REAL" if col in numeric_cols else "TEXT"}' for col in rdr.fieldnames]
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE dataset (" + ", ".join(quoted) + ")")
        cols = ", ".join('"' + col + '"' for col in rdr.fieldnames)
        args = ",".join("?" for _ in rdr.fieldnames)
        conn.executemany(f"INSERT INTO dataset ({cols}) VALUES ({args})", [tuple(row.values()) for row in rows])
        for sql in sqls:
            try:
                conn.execute(sql).fetchall()
            except Exception as exc:
                raise ValueError(f"Bad reference SQL: {sql}\n{exc}") from exc


def _jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build(output: Path = OUT, *, seed: int = 20261015) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    fixtures = output / "fixtures"
    fixtures.mkdir(exist_ok=True)
    splits = {"train": [], "validation": [], "test": []}
    metadata = {"seed": seed, "source": "synthetic, deterministic templates; no benchmark cases used",
                "domain_split": {"train": list(TRAIN_DOMAINS), "validation": list(VAL_DOMAINS),
                                 "test": list(TEST_DOMAINS)},
                "num_schemas": {}, "examples": {}, "skills": {},
                "sha256": {}}
    for split, domains in (("train", TRAIN_DOMAINS), ("validation", VAL_DOMAINS),
                           ("test", TEST_DOMAINS)):
        for domain in domains:
            for variant in range(6):
                schema_id = f"{domain}_v{variant:02d}"
                schema, raw_csv, names, vals = _render_schema_and_csv(domain, variant, seed)
                (fixtures / f"{schema_id}.csv").write_bytes(raw_csv)
                samples = _examples(names, vals)
                _sqlite_validate(raw_csv, [s[2] for s in samples],
                                 (names["units"], names["charge"], names["cost"]))
                for n, (skill, question, sql) in enumerate(samples):
                    # The training format mirrors production's messages and trains
                    # on SQL only, not on the system/user prompt tokens.
                    rec = {
                        "id": f"{schema_id}_{n:02d}",
                        "schema_id": schema_id,
                        "skill": skill,
                        "question": question,
                        "sql": sql,
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"{schema}\n\nUser question: {question}\n\nSQL:"},
                        ],
                        "completion": [{"role": "assistant", "content": sql}],
                    }
                    splits[split].append(rec)
        random.Random(seed).shuffle(splits[split])
        _jsonl(output / f"{split}.jsonl", splits[split])
        metadata["num_schemas"][split] = len(domains) * 6
        metadata["examples"][split] = len(splits[split])
        metadata["skills"][split] = dict(sorted(Counter(r["skill"] for r in splits[split]).items()))
        metadata["sha256"][split] = hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
    domain_ids = [{r["schema_id"] for r in splits[name]} for name in ("train", "validation", "test")]
    assert not (domain_ids[0] & domain_ids[1])
    assert not (domain_ids[0] & domain_ids[2])
    assert not (domain_ids[1] & domain_ids[2])
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    report = build(args.output, seed=args.seed)
    print("Fine-tuning data generated and reference SQL checked with SQLite")
    print(f"Training:   {report['examples']['train']} examples from {report['num_schemas']['train']} schemas")
    print(f"Validation: {report['examples']['validation']} examples from {report['num_schemas']['validation']} unseen schemas")
    print(f"Final test: {report['examples']['test']} examples from {report['num_schemas']['test']} unseen schemas")
    print(f"Manifest:   {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
