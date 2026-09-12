from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request


SOURCE_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "filetype=csv&from=&label=include&lastobs=&layout=seriescolumn&"
    "rel=H15&series=bf17364827e38702b42a58cf8eaa3f78&to=&type=package"
)
SOURCE_PAGE = "https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15"
REUSE_POLICY = "https://www.federalreserve.gov/disclaimer.htm"
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"
SERIES_ID = "RIFLGFCY10_N.B"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "data" / "frb_us_treasury_10y_2012_2024.csv"
METADATA_PATH = ROOT / "data" / "frb_us_treasury_10y_2012_2024.metadata.json"


def download_source() -> bytes:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "applied-ml-signal-lab fixture updater"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def extract_rows(source: str) -> list[dict[str, str]]:
    parsed = list(csv.reader(io.StringIO(source)))
    header_index = next(
        (index for index, row in enumerate(parsed) if row and row[0] == "Time Period"),
        None,
    )
    if header_index is None:
        raise ValueError("Federal Reserve H.15 source is missing the Time Period header")

    header = parsed[header_index]
    if SERIES_ID not in header:
        raise ValueError(f"Federal Reserve H.15 source is missing {SERIES_ID}")
    date_index = header.index("Time Period")
    value_index = header.index(SERIES_ID)

    rows = []
    for source_row in parsed[header_index + 1 :]:
        if max(date_index, value_index) >= len(source_row):
            continue
        date = source_row[date_index].strip()
        value = source_row[value_index].strip()
        if START_DATE <= date <= END_DATE and value not in {"", "ND", "N/A"}:
            rows.append({"date": date, "close": value})
    rows.sort(key=lambda row: row["date"])
    return rows


def render_fixture(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["date", "close"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_fixture(source: str) -> dict[str, object]:
    fixture = render_fixture(extract_rows(source))
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(fixture, encoding="utf-8", newline="")

    metadata = {
        "name": "Federal Reserve Board 10-year Treasury yield benchmark fixture",
        "pair": "US 10Y Treasury yield",
        "asset_family": "interest_rates",
        "source": "Board of Governors of the Federal Reserve System, H.15 release",
        "source_data_origin": "U.S. Treasury",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "reuse_policy": REUSE_POLICY,
        "series_id": SERIES_ID,
        "date_range": {"start": START_DATE, "end": END_DATE},
        "rows": fixture.count("\n") - 1,
        "columns": {
            "date": "Federal Reserve H.15 observation date",
            "close": (
                "Market yield on U.S. Treasury securities at 10-year constant maturity, "
                "percent per year; renamed only for pipeline compatibility"
            ),
        },
        "transformations": [
            f"selected the Time Period and {SERIES_ID} columns",
            "removed observations reported as unavailable",
            "limited observations to the fixed inclusive date range",
            "sorted observations chronologically",
            f"renamed Time Period to date and {SERIES_ID} to close",
        ],
        "fixture_sha256": hashlib.sha256(fixture.encode("utf-8")).hexdigest(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata['rows']} US 10Y Treasury yield rows to {FIXTURE_PATH}")
    print(f"SHA-256: {metadata['fixture_sha256']}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the frozen Federal Reserve H.15 fixture.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Use a previously downloaded H.15 CSV instead of downloading it.",
    )
    args = parser.parse_args()

    source = (
        args.source.read_text(encoding="utf-8-sig")
        if args.source
        else download_source().decode("utf-8-sig")
    )
    write_fixture(source)


if __name__ == "__main__":
    main()
