# Dataset availability audit

On 2026-11-25 all six configured interaction and metadata endpoints responded HTTP 200 to HEAD requests. Exact URLs, response statuses and advertised compressed byte lengths are in `DATA_AVAILABILITY.json`. A HEAD response verifies endpoint availability, not content integrity. Downloader and preprocessor record actual SHA-256 after retrieval.

The authoritative [Amazon 2014 data page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) provides category 5-core reviews and metadata. Defaults select Beauty, Sports and Outdoors, and Toys and Games. Beauty interaction and metadata archives have now been downloaded and hashed. The verified Beauty bundle has 22,363 users, 12,101 items, 198,502 interactions, 153,776 train events and zero missing metadata items. Test warm-target coverage is 0.9938290926977597. Sports and Toys have availability checks only; neither is downloaded yet. Archive sizes must not be interpreted as processed-memory estimates.

Run `scripts/download_data.py` followed by `scripts/preprocess_data.py` for each category. Each bundle writes actual filtered counts, metadata missingness and validation/test warm coverage to `statistics.json`; these will supply the eventual dataset statistics table. Inspect timestamp ties and the effect of earliest user/item deduplication before freezing experimental splits.


The actual Beauty model-specific training sample counts and split version are in results/reports/beauty_data_audit.json. Raw file provenance is recorded beside the archives as .source.json. docs/PHASE34_RESULTS.md links the pilot execution evidence.
