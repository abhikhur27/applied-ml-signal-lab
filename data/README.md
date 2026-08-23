# Benchmark data

`ecb_eur_usd_2012_2024.csv` is a frozen slice of the European Central Bank's daily euro foreign-exchange reference rates. The `close` field is the published USD-per-EUR reference rate, not an exchange venue's closing trade. It is used here only as a realistic, non-synthetic financial time series.

Source: [European Central Bank euro reference exchange rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)

The ECB permits free reuse of publicly available ESCB statistics when the source is quoted and the statistics are not modified. The fixture selects, filters, orders, and renames columns without changing the published values. See the [ESCB statistics reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html).

To reproduce the CSV and its provenance metadata from the official historical archive:

```bash
python scripts/update_ecb_fixture.py
```

The fixture intentionally ends on 2024-12-31 so benchmark results do not drift when the ECB publishes new observations.
