# Scalability estimate

The corpus inventory contains 126,539 word-region images in 5,368 page directories. OCR projections report serial time and a four-worker, 75%-utilization practical scenario. Local OCR compute is kept separate from paid GPT-4o validation.

| configuration | observed_regions | median_seconds_per_region | mean_seconds_per_region | p95_seconds_per_region | regions_per_hour_at_mean |
|---|---|---|---|---|---|
| surya/raw | 512 | 19.3931 | 20.3208 | 31.3588 | 177.1582 |
| surya/processed | 961 | 11.2204 | 12.4321 | 23.2505 | 289.5723 |
| tesseract/raw | 961 | 0.9015 | 0.9181 | 1.1736 | 3921.1838 |
| tesseract/processed | 961 | 0.5843 | 0.5946 | 0.7004 | 6054.8661 |

| scenario | api_calls | estimated_input_tokens | estimated_output_tokens | estimated_standard_api_cost_usd | serial_validation_hours | pricing_as_of |
|---|---|---|---|---|---|---|
| sampled_100_pages | 400 | 245015 | 102140 | 1.6339 | 0.3968 | 2026-06-23 |
| full_5368_pages | 21472 | 13152405 | 5482875 | 87.7098 | 21.3002 | 2026-06-23 |

Pricing is a dated, reproducible snapshot and must be rechecked before a future production run.
