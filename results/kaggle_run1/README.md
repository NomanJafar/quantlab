Run 1 (2026-10-11): died in the eval-set cell: `HfUriError: Repository id must be 'namespace/name', got 'wikitext'`.
The `datasets` short name no longer resolves on the current Hub client. Fix for run 2: read the parquet files directly
(`Salesforce/wikitext` and `openai/gsm8k` resolve URLs) with pandas; no `datasets` dependency.
