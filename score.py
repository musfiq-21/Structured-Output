"""Score runs/*.jsonl against data/answer_key.json (definitions from Lecture 2, slide 33).

Invalid rate   = outputs that do not parse, fail schema v1, or were cut off by num_predict / all calls
Field accuracy = correct fields / all expected fields (7 per notice; invalid outputs score 0)
Spread         = each run, then mean and min-max over the 3 runs
Refusals (no JSON object at all) are counted separately. There are no retries.

    python3 score.py      # prints the table, writes results/summary.json and results/summary.md
"""
import json
import re
import statistics
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SQL-Calculation-Agent"))
from sqlagent.llm import notional_cost  # noqa: E402  (class 1 pricing)

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema" / "notice_v1.schema.json").read_text())
KEY = json.loads((HERE / "data" / "answer_key.json").read_text())
FIELDS = ["domain", "notice_type", "expires_on", "registrar", "amount_due", "currency",
          "action_required"]
VALIDATOR = Draft202012Validator(SCHEMA)
CACHE_FILE = HERE / "runs" / "prompt_cache.jsonl"  # written by cache_from_logs.py
CACHED = ({(c["method"], c["run"], c["id"]): c["cached_tokens"]
           for c in map(json.loads, CACHE_FILE.read_text().splitlines())}
          if CACHE_FILE.exists() else {})
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse(raw):
    """Return (obj or None, fenced?). Only a surrounding ``` fence is forgiven."""
    text = raw.strip()
    m = FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, bool(m)
    return obj, bool(m)


def field_equal(name, got, want):
    if want is None or got is None:
        return got is None and want is None
    if name == "amount_due":
        return (isinstance(got, (int, float)) and not isinstance(got, bool)
                and abs(got - want) < 0.005)
    if name == "action_required":
        return got is want
    if name == "registrar":
        norm = lambda s: " ".join(str(s).casefold().split())  # noqa: E731
        return norm(got) == norm(want)
    if name == "domain":
        return str(got).strip().lower() == want
    return got == want


def score_record(rec):
    obj, fenced = parse(rec["raw"])
    truncated = rec.get("done_reason") == "length"
    schema_ok = isinstance(obj, dict) and not any(VALIDATOR.iter_errors(obj))
    valid = schema_ok and not truncated
    key = KEY[rec["id"]]
    correct = {f: valid and field_equal(f, obj.get(f), key[f]) for f in FIELDS}
    return {"id": rec["id"], "valid": valid, "fenced": fenced, "truncated": truncated,
            "refusal": "{" not in rec["raw"], "parsed": obj is not None,
            "correct": correct}


def score_run(path):
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    scored = [score_record(r) for r in recs]
    n = len(scored)
    valid = [s for s in scored if s["valid"]]
    right = sum(sum(s["correct"].values()) for s in scored)
    return {
        "file": path.name, "calls": n,
        "invalid_rate": 1 - len(valid) / n,
        "field_accuracy": right / (n * len(FIELDS)),
        "accuracy_on_valid": (sum(sum(s["correct"].values()) for s in valid)
                              / (len(valid) * len(FIELDS))) if valid else 0.0,
        "per_field": {f: sum(s["correct"][f] for s in scored) / n for f in FIELDS},
        "refusals": sum(s["refusal"] for s in scored),
        "truncated": sum(s["truncated"] for s in scored),
        "unparseable": sum(not s["parsed"] for s in scored),
        "fenced": sum(s["fenced"] for s in scored),
        "cost_usd": sum(r["cost_usd"] for r in recs),
        "cost_cached_usd": sum(notional_cost(r["prompt_tokens"], r["completion_tokens"],
                                             CACHED.get((r["method"], r["run"], r["id"]), 0))
                               for r in recs),
        "cached_tokens": sum(CACHED.get((r["method"], r["run"], r["id"]), 0) for r in recs),
        "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
        "completion_tokens": sum(r["completion_tokens"] for r in recs),
        "seconds": sum(r["seconds"] for r in recs),
        "invalid_ids": [s["id"] for s in scored if not s["valid"]],
    }


def spread(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "runs": values}


def main():
    out = {}
    for method in ("prompt", "grammar"):
        runs = [score_run(p) for p in sorted((HERE / "runs").glob(f"{method}_run*.jsonl"))]
        if not runs:
            continue
        out[method] = {
            "runs": runs,
            **{m: spread([r[m] for r in runs]) for m in
               ("invalid_rate", "field_accuracy", "accuracy_on_valid", "cost_usd",
                "cost_cached_usd", "seconds")},
            "total_cost_usd": sum(r["cost_usd"] for r in runs),
            "total_cost_cached_usd": sum(r["cost_cached_usd"] for r in runs),
            "per_field_mean": {f: statistics.mean(r["per_field"][f] for r in runs)
                               for f in FIELDS},
        }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    pct = lambda s: f"{s['mean']:.1%} ({s['min']:.1%}–{s['max']:.1%})"  # noqa: E731
    lines = ["| Metric (mean, min–max over runs) | " + " | ".join(out) + " |",
             "|---|" + "---|" * len(out)]
    rows = [("Invalid rate", lambda m: pct(m["invalid_rate"])),
            ("Field accuracy", lambda m: pct(m["field_accuracy"])),
            ("Field accuracy on valid outputs", lambda m: pct(m["accuracy_on_valid"])),
            ("Cost per run (USD, notional)", lambda m: "{mean:.4f} ({min:.4f}–{max:.4f})"
             .format(**m["cost_usd"])),
            ("Total cost, all runs (USD)", lambda m: f"{m['total_cost_usd']:.4f}"),
            ("Cost per run with prefix cache (USD)", lambda m: "{mean:.4f} ({min:.4f}–{max:.4f})"
             .format(**m["cost_cached_usd"])),
            ("Total cost with prefix cache (USD)",
             lambda m: f"{m['total_cost_cached_usd']:.4f}"),
            ("Seconds per run", lambda m: "{mean:.0f} ({min:.0f}–{max:.0f})"
             .format(**m["seconds"])),
            ("Refusals / truncated / fenced (all runs)", lambda m: "{} / {} / {}".format(
                sum(r["refusals"] for r in m["runs"]), sum(r["truncated"] for r in m["runs"]),
                sum(r["fenced"] for r in m["runs"])))]
    rows += [(f"  accuracy: {f}", lambda m, f=f: f"{m['per_field_mean'][f]:.1%}") for f in FIELDS]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(m) for m in out.values()) + " |")
    table = "\n".join(lines)
    (HERE / "results" / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
