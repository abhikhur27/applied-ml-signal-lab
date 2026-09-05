from scripts.update_ecb_fixture import extract_rows, render_fixture


def test_fixture_extraction_selects_currency_and_sorts_dates() -> None:
    source = "Date,USD,GBP,JPY\n2024-01-03,1.09,0.86,157.2\n2024-01-02,1.08,N/A,156.4\n"

    rows = extract_rows(source, "GBP")

    assert rows == [{"date": "2024-01-03", "close": "0.86"}]
    assert render_fixture(rows) == "date,close\n2024-01-03,0.86\n"
