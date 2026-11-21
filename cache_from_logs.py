"""Recover per-call prompt-cache hits from the Ollama server log.

Ollama's /api/chat reports prompt_eval_count but not how much of the prompt came from its
prefix cache. The server log does: each call logs "new prompt ... task.n_tokens = N" and then
"cached n_tokens = C". This matches the last 300 calls at the run's context size to the run
records, in the order extract.py made them, and checks every N against prompt_tokens.

    journalctl -u ollama --since today -o short-iso > /tmp/ollama.log
    python3 cache_from_logs.py /tmp/ollama.log      # writes runs/prompt_cache.jsonl
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM_CTX = 4096  # extract.SETTINGS["num_ctx"]
ORDER = [(m, k) for k in (1, 2, 3) for m in ("prompt", "grammar")]  # extract.py loop order


def parse_log(lines):
    calls, cur = [], None
    for line in lines:
        m = re.search(r"new prompt, n_ctx_slot = (\d+).*task.n_tokens = (\d+)", line)
        if m:
            cur = {"n_ctx": int(m.group(1)), "n_tokens": int(m.group(2)), "log_time": line[:25]}
            continue
        m = re.search(r"cached n_tokens = (\d+)", line)
        if m and cur:
            calls.append({**cur, "cached_tokens": int(m.group(1))})
            cur = None
    return [c for c in calls if c["n_ctx"] == NUM_CTX]


def main(log_path):
    calls = parse_log(Path(log_path).read_text().splitlines())
    recs = [json.loads(line) for m, k in ORDER
            for line in (HERE / "runs" / f"{m}_run{k}.jsonl").read_text().splitlines()]
    if len(calls) < len(recs):
        sys.exit(f"log has {len(calls)} calls, runs have {len(recs)}")
    calls = calls[-len(recs):]
    bad = [r["id"] for r, c in zip(recs, calls) if r["prompt_tokens"] != c["n_tokens"]]
    if bad:
        sys.exit(f"log and runs disagree on prompt size for {len(bad)} calls, e.g. {bad[:5]}")
    out = HERE / "runs" / "prompt_cache.jsonl"
    with out.open("w") as f:
        for r, c in zip(recs, calls):
            f.write(json.dumps({"method": r["method"], "run": r["run"], "id": r["id"],
                                "prompt_tokens": r["prompt_tokens"],
                                "cached_tokens": c["cached_tokens"],
                                "log_time": c["log_time"]}) + "\n")
    print(f"wrote {out.name}: {len(recs)} calls matched, all prompt sizes agree")


if __name__ == "__main__":
    main(sys.argv[1])
