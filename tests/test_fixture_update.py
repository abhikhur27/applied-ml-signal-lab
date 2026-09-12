from scripts.update_ecb_fixture import extract_rows, render_fixture
from scripts.update_federal_reserve_fixture import (
    extract_rows as extract_h15_rows,
    render_fixture as render_h15_fixture,
)


def test_fixture_extraction_selects_currency_and_sorts_dates() -> None:
    source = "Date,USD,GBP,JPY\n2024-01-03,1.09,0.86,157.2\n2024-01-02,1.08,N/A,156.4\n"

    rows = extract_rows(source, "GBP")

    assert rows == [{"date": "2024-01-03", "close": "0.86"}]
    assert render_fixture(rows) == "date,close\n2024-01-03,0.86\n"


def test_h15_fixture_extraction_skips_metadata_and_unavailable_rows() -> None:
    source = """\
"Series Description","one year","ten year"
"Unit:","Percent:_Per_Year","Percent:_Per_Year"
"Unique Identifier:","H15/H15/RIFLGFCY01_N.B","H15/H15/RIFLGFCY10_N.B"
"Time Period","RIFLGFCY01_N.B","RIFLGFCY10_N.B"
2024-01-03,4.80,4.05
2024-01-02,4.75,ND
"""

    rows = extract_h15_rows(source)

    assert rows == [{"date": "2024-01-03", "close": "4.05"}]
    assert render_h15_fixture(rows) == "date,close\n2024-01-03,4.05\n"
