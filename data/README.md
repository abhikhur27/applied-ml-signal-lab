# Benchmark data

The checked-in `ecb_eur_{usd,gbp,jpy}_2012_2024.csv` files are frozen slices of the European Central Bank's daily euro foreign-exchange reference rates. Their `close` fields are the published quote-currency-per-euro reference rates, not exchange-venue closing trades. They are used only as realistic, non-synthetic financial time series.

Source: [European Central Bank euro reference exchange rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)

The ECB permits free reuse of publicly available ESCB statistics when the source is quoted and the statistics are not modified. The fixtures select, filter, order, and rename columns without changing the published values. See the [ESCB statistics reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html).

To reproduce the CSV and its provenance metadata from the official historical archive:

```bash
python scripts/update_ecb_fixture.py
```

If the local Python certificate store cannot validate the download, fetch the official archive with a trusted system client and pass it with `--archive path/to/eurofxref-hist.zip`; the updater never accepts an insecure TLS mode.

The fixtures intentionally end on 2024-12-31 so benchmark results do not drift when the ECB publishes new observations. The benchmark contract stores each expected row count and SHA-256 checksum separately from the generated metadata, so refreshing both files cannot silently move the frozen baseline.
