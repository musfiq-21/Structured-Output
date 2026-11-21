# Week 02: inference capacity and structured output as a contract

Agentic Software Engineering, Part 1, Class 2. Model for Task 2: **`qwen2.5:3b`** (Ollama 0.40.17,
id `357c53fb659c`), CPU only. Every call goes through the class 1 wrapper
(`../SQL-Calculation-Agent/sqlagent/llm.py`), whose ledger now also records `done_reason` and a
notional cost.

## How to run

`extract.py` imports the class 1 wrapper from the sibling folder, so clone both repos side by
side, with `SQL-Calculation-Agent` on a commit that includes `b6272ee` (wrapper `format` and cost):

```
AgenticSWE/
  SQL-Calculation-Agent/     class 1 repo (wrapper + ledger)
  Week02-Structured-Output/  this repo
```

Needs Python 3.10+, `jsonschema` (`pip install jsonschema`) and Ollama with `qwen2.5:3b`.

```bash
python3 kv_cache.py                            # Task 1 working
python3 kv_cache.py --key <8k size>,<32k size> # compare with the answer key (10% rule)
python3 extract.py                             # Task 2: 300 calls, ~48 min on CPU, resumable
python3 score.py                               # writes results/summary.{json,md}
python3 -m unittest                            # contract test + scorer tests
```

## Task 1: KV-cache size of Qwen2.5-7B

Config from `Qwen/Qwen2.5-7B-Instruct/config.json`: 28 layers, 28 query heads, **4 KV heads**
(grouped-query attention), hidden size 3584, bf16.

```
head_dim  = 3584 / 28                                   = 128
per token = 2 (K,V) x 28 layers x 4 KV heads x 128 x 2 B = 57,344 B (56 KiB)
 8k: 8,192 x 57,344 B  =   469,762,048 B = 448 MiB  = 0.47 GB
32k: 32,768 x 57,344 B = 1,879,048,192 B = 1.75 GiB = 1.88 GB
```

Batch size 1, cache only (weights and buffers are extra, slide 14). Without GQA (28 KV heads)
the 32k cache would be 12.25 GiB, 7x more.

**Answer key comparison:** _pending. The key was not provided; run
`python3 kv_cache.py --key ...` when it is, and paste the output here._

## Task 2: prompt-only vs grammar-constrained extraction

### Setup
- **Corpus:** 50 domain-name notices (`data/notices.json`), written for this assignment to
  include traps: DD/MM vs MM/DD dates, Bangla text and digits, sent/due/suspension dates next to
  the expiry date, receipts and refunds (nothing due), two registrars in one notice, subdomains.
  Type mix: expiry 14, other 9, renewal 7, payment 7, transfer 7, suspension 6.
- **Schema v1** (`schema/notice_v1.schema.json`): `schema_version` (const `"1.0"`), `domain`,
  `notice_type` (enum of 6), `expires_on` (YYYY-MM-DD or null), `registrar`, `amount_due`,
  `currency` (ISO 4217 or null), `action_required`. All required; unknown fields allowed.
- **Answer key** (`data/answer_key.json`): written with the notices, reviewed before any run.
- **Method A, prompt:** the system prompt describes every field and rule; the raw reply is
  parsed (a surrounding ``` fence would be forgiven and logged; none occurred) and validated.
- **Method B, grammar:** same messages, plus the schema as Ollama `format`.
- **Held equal:** model, prompt text, temperature 0.3, `num_predict` 256, `num_ctx` 4096.
  Run *k* uses seed *k* for **both** methods, so runs differ but each run is a paired comparison.
- **Scoring (slide 33):** invalid = not JSON, fails schema v1, or cut off (`done_reason` =
  `length`). Field accuracy = correct fields / (calls x 7); invalid outputs score 0 on every
  field. No retries.
- **Cost:** teaching rates from slide 20, $2 per million input and $8 per million output tokens.

### Results (3 runs x 50 notices per method)

| Metric (mean, min–max over 3 runs) | Prompt-only | Grammar-constrained |
|---|---|---|
| **Invalid rate** | **46.7%** (46.0%–48.0%) | **0.0%** (0.0%–0.0%) |
| **Field accuracy** | **43.9%** (43.4%–44.3%) | **75.0%** (74.0%–75.7%) |
| Field accuracy, scoring fields even in invalid outputs | 70.5% (69.4%–71.1%) | 75.0% (74.0%–75.7%) |
| **Cost per run** (USD, notional) | **0.0804** (0.0803–0.0805) | **0.0806** (0.0805–0.0807) |
| Total cost, 3 runs (USD) | 0.2411 | 0.2418 |
| Seconds per run (50 calls) | 484 (468–512) | 471 (463–478) |
| Refusals / truncated / fenced | 0 / 0 / 0 | 0 / 0 / 0 |

Per-field accuracy (mean of 3 runs, invalid outputs count as wrong):

| Field | Prompt-only | Grammar |
|---|---|---|
| domain | 49.3% | 88.0% |
| notice_type | 41.3% | 66.0% |
| expires_on | 45.3% | 71.3% |
| registrar | 33.3% | 52.0% |
| amount_due | 46.0% | 82.0% |
| currency | 48.0% | 77.3% |
| action_required | 44.0% | 88.0% |

### What the numbers mean
1. **Why prompt-only fails is almost entirely one thing.** Every prompt-only reply was
   well-formed JSON. Of the 23 invalid outputs in a run, 22 put a currency symbol (`"$"`, `"Tk"`)
   where the schema requires an ISO code, despite the mapping in the prompt. One (n21) dropped
   the required `action_required` key. The same notices failed in all 3 runs, so the spread is
   small: the failures are systematic, not random.
2. **The constraint fixes validity, not knowledge.** On the notices where prompt-only output was
   valid, both methods score *identically* (81.5%, 83.5%, 82.0% per run): the grammar never
   changed those answers. All of the gain comes from the 23 notices that grammar turns valid.
3. **Valid JSON with wrong values (slide 29).** The pattern `^[A-Z]{3}$` forces three capital
   letters, so the model wrote `"TKR"`, which is not a real code. Other grammar outputs that pass
   the schema but are wrong: `"rahmantraders"` (TLD dropped), `"baeckerei-weber_de"`, an invented
   `amount_due: 25` for a notice with no price (n05), and a doubled price (n12).
4. **Errors that are the model's, under both methods.** Expiry reminders are often labelled
   `renewal` (notice_type 66%); registrars are often `null` or written as slugs
   (`"tidewater-registrations"`); suspension and sent dates are sometimes taken as `expires_on`.
   These need a better prompt, few-shot examples or a larger model, not a grammar.
5. **Cost and time are the same.** Grammar costs 0.3% more (a few more output tokens) and runs
   at the same speed. On this setup constrained decoding is effectively free.
6. **Spread is small** (about 2 points) because temperature is 0.3 and only the seed changes.

**Conclusion:** for a 3B model, grammar-constrained decoding removes the invalid outputs
entirely (47% to 0%) at no extra cost, raising field accuracy from 44% to 75%. It does not make
the values correct. The application still has to check values against the source (slide 29),
and an enum or pattern in the schema only guarantees the shape, not the meaning.

## Contract test

`tests/test_contract.py`. Policy (slide 31): consumers ignore unknown fields. A change is
compatible when every v1 field still exists, keeps or narrows its type, gains no enum values,
stays required, new fields are optional and the major version stays 1.

- **Accepts:** v1.1 adding optional `registrar_url`; a v1 consumer still validates a v1.1 document.
- **Fails:** renaming `domain` to `hostname`, changing `expires_on` to an object, adding a
  `notice_type` value. A v1 consumer rejects the renamed document.

## Limitations
- The notices and the answer key were written together for this assignment, not taken from a
  real inbox or the course starter repo (none was provided).
- `qwen2.5:3b` instead of 7B: the machine has about 2 GB of free RAM and no GPU.
- Costs are notional teaching rates; local inference has no bill. Ollama reported full prompt
  token counts (no prefix-cache discount), so there is no cached-input line.
- Prompt-only parsing is strict except for a surrounding code fence; a more forgiving parser
  (for example, mapping `$` to `USD`) would lower its invalid rate, but would be the application
  repairing the model's output.
