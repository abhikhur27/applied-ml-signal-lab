from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


SOURCE_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
SOURCE_PAGE = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"
REUSE_POLICY = "https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html"
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"
ROOT = Path(__file__).resolve().parents[1]
SERIES = {
    "usd": {
        "pair": "EUR/USD",
        "source_column": "USD",
        "unit": "USD per EUR",
    },
    "gbp": {
        "pair": "EUR/GBP",
        "source_column": "GBP",
        "unit": "GBP per EUR",
    },
    "jpy": {
        "pair": "EUR/JPY",
        "source_column": "JPY",
        "unit": "JPY per EUR",
    },
}


def download_source() -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "applied-ml-signal-lab fixture updater"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def extract_source_csv(archive: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        return bundle.read("eurofxref-hist.csv").decode("utf-8-sig")


def extract_rows(source: str, source_column: str) -> list[dict[str, str]]:
    rows = []
    for row in csv.DictReader(io.StringIO(source)):
        date = row["Date"]
        rate = row[source_column]
        if START_DATE <= date <= END_DATE and rate not in {"", "N/A"}:
            rows.append({"date": date, "close": rate})
    rows.sort(key=lambda row: row["date"])
    return rows


def render_fixture(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["date", "close"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_fixture(source: str, quote_currency: str, config: dict[str, str]) -> dict[str, object]:
    fixture_path = ROOT / "data" / f"ecb_eur_{quote_currency}_2012_2024.csv"
    metadata_path = ROOT / "data" / f"ecb_eur_{quote_currency}_2012_2024.metadata.json"
    fixture = render_fixture(extract_rows(source, config["source_column"]))
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(fixture, encoding="utf-8", newline="")

    metadata = {
        "name": f"ECB {config['pair']} reference-rate benchmark fixture",
        "pair": config["pair"],
        "source": "European Central Bank",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "reuse_policy": REUSE_POLICY,
        "date_range": {"start": START_DATE, "end": END_DATE},
        "rows": fixture.count("\n") - 1,
        "columns": {
            "date": "ECB observation date",
            "close": (
                f"{config['unit']} reference rate (ECB {config['source_column']} column; "
                "renamed only for pipeline compatibility)"
            ),
        },
        "transformations": [
            f"selected the Date and {config['source_column']} columns",
            f"removed rows where the {config['source_column']} value was unavailable",
            "limited observations to the fixed inclusive date range",
            "sorted observations chronologically",
            f"renamed Date to date and {config['source_column']} to close",
        ],
        "fixture_sha256": hashlib.sha256(fixture.encode("utf-8")).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata['rows']} {config['pair']} rows to {fixture_path}")
    print(f"SHA-256: {metadata['fixture_sha256']}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the frozen ECB FX benchmark fixtures.")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use a previously downloaded eurofxref-hist.zip instead of downloading it.",
    )
    args = parser.parse_args()

    archive = args.archive.read_bytes() if args.archive else download_source()
    source = extract_source_csv(archive)
    for quote_currency, config in SERIES.items():
        write_fixture(source, quote_currency, config)


if __name__ == "__main__":
    main()
