from __future__ import annotations

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
FIXTURE_PATH = ROOT / "data" / "ecb_eur_usd_2012_2024.csv"
METADATA_PATH = ROOT / "data" / "ecb_eur_usd_2012_2024.metadata.json"


def download_source() -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "applied-ml-signal-lab fixture updater"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def extract_rows(archive: bytes) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        source = bundle.read("eurofxref-hist.csv").decode("utf-8-sig")

    rows = []
    for row in csv.DictReader(io.StringIO(source)):
        date = row["Date"]
        usd_rate = row["USD"]
        if START_DATE <= date <= END_DATE and usd_rate not in {"", "N/A"}:
            rows.append({"date": date, "close": usd_rate})
    rows.sort(key=lambda row: row["date"])
    return rows


def render_fixture(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["date", "close"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    fixture = render_fixture(extract_rows(download_source()))
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(fixture, encoding="utf-8", newline="")

    metadata = {
        "name": "ECB EUR/USD reference-rate benchmark fixture",
        "source": "European Central Bank",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "reuse_policy": REUSE_POLICY,
        "date_range": {"start": START_DATE, "end": END_DATE},
        "rows": fixture.count("\n") - 1,
        "columns": {
            "date": "ECB observation date",
            "close": "USD per EUR reference rate (ECB USD column; renamed only for pipeline compatibility)",
        },
        "transformations": [
            "selected the Date and USD columns",
            "removed rows where the USD value was unavailable",
            "limited observations to the fixed inclusive date range",
            "sorted observations chronologically",
            "renamed Date to date and USD to close",
        ],
        "fixture_sha256": hashlib.sha256(fixture.encode("utf-8")).hexdigest(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata['rows']} rows to {FIXTURE_PATH}")
    print(f"SHA-256: {metadata['fixture_sha256']}")


if __name__ == "__main__":
    main()
