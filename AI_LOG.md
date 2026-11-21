# AI-assistant use log

The course requires three lines per piece: what the assistant wrote, which test proves it,
and what I reviewed. The third column is for **me** to fill in after reading the code.

Assistant: Claude Code (Claude Opus 5.5).

| # | What it wrote | Test that proves it | What I reviewed |
|---|---|---|---|
| 1 | `kv_cache.py`: KV-cache size of Qwen2.5-7B from its Hugging Face `config.json`, plus the 10% answer-key comparison | Built-in asserts (57,344 B/token, 448 MiB at 8k); `--key` checked with a passing and a failing key | |
| 2 | Class 1 wrapper `../SQL-Calculation-Agent/sqlagent/llm.py`: `format` argument, `done_reason`, notional cost per call (slide 20 rates), `total_cost()` | `../SQL-Calculation-Agent/tests/test_llm.py` (`test_format_passed_and_cost_recorded`, `test_slide_cost_example`) | |
| 3 | `schema/notice_v1.schema.json`: schema v1 with `schema_version` "1.0", 8 required fields | `tests/test_contract.py` (`test_v1_is_valid_and_versioned`); every answer-key entry validated against it | |
| 4 | `data/notices.json` and `data/answer_key.json`: 50 domain notices written together with their correct values | Script check: key valid against schema, each domain and registrar appears in its notice, amount and currency both set or both null. Reviewed by me before any model run (`data/answer_key_review.md`) | |
| 5 | `extract.py`: both methods, same prompt and settings, seed = run number, resumable | 300 records in `runs/`; Ollama accepted the full schema in a probe call | |
| 6 | `score.py`: parsing (fence only forgiven), schema validation, truncation, field rules, spread | `tests/test_score.py` (7 tests); the 23 invalid prompt outputs were checked by hand to be real schema errors, not scorer bugs | |
| 7 | `tests/test_contract.py`: compatibility rules and the additive / breaking test | The test itself: additive `registrar_url` accepted; rename, retype and new enum value rejected | |
| 8 | `REPORT.md`, written from `results/summary.json` and `runs/*.jsonl` | No unit test: every number came from `score.py` or a script over the run files | |
