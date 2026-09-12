# Benchmark data

All checked-in series cover the fixed inclusive range from 2012-01-01 through 2024-12-31. They are scalar market observations adapted to the training pipeline's `date,close` input contract. For the Treasury fixture, `bull` and `bear` labels mean rising and falling yield levels; they do not mean rising and falling Treasury bond prices.

## ECB foreign-exchange reference rates

The checked-in `ecb_eur_{usd,gbp,jpy}_2012_2024.csv` files are frozen slices of the European Central Bank's daily euro foreign-exchange reference rates. Their `close` fields are the published quote-currency-per-euro reference rates, not exchange-venue closing trades. They are used only as realistic, non-synthetic financial time series.

Source: [European Central Bank euro reference exchange rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)

The ECB permits free reuse of publicly available ESCB statistics when the source is quoted and the statistics are not modified. The fixtures select, filter, order, and rename columns without changing the published values. See the [ESCB statistics reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html).

To reproduce the CSV and its provenance metadata from the official historical archive:

```bash
python scripts/update_ecb_fixture.py
```

If the local Python certificate store cannot validate the download, fetch the official archive with a trusted system client and pass it with `--archive path/to/eurofxref-hist.zip`; the updater never accepts an insecure TLS mode.

The fixtures intentionally end on 2024-12-31 so benchmark results do not drift when the ECB publishes new observations. The benchmark contract stores each expected row count and SHA-256 checksum separately from the generated metadata, so refreshing both files cannot silently move the frozen baseline.

## Federal Reserve H.15 Treasury yield

`frb_us_treasury_10y_2012_2024.csv` is a frozen slice of the Federal Reserve Board's H.15 daily series `RIFLGFCY10_N.B`: the market yield on U.S. Treasury securities at 10-year constant maturity, quoted as percent per year. H.15 identifies the U.S. Treasury as the source for constant-maturity yields.

Source: [Federal Reserve Board H.15 Data Download Program](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15)

The Board states that, unless otherwise indicated, information on its website is in the public domain and may be copied and distributed with attribution. See the [Federal Reserve Board disclaimer and reuse notice](https://www.federalreserve.gov/disclaimer.htm).

To reproduce the fixture and metadata from the official preformatted Treasury Constant Maturities CSV package:

```bash
python scripts/update_federal_reserve_fixture.py
```

The updater selects the daily 10-year series, removes unavailable observations, applies the frozen date range, preserves the published numeric values, and renames the source columns only for pipeline compatibility. It also accepts `--source path/to/h15.csv` for a previously downloaded official package.
