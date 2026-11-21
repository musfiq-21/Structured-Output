"""Task 2: extract the 50 notices with two methods, 3 runs each, through the class 1 wrapper.

Method A (prompt): the prompt describes the JSON; we save whatever the model returns.
Method B (grammar): same prompt, plus the schema passed as Ollama `format`.
Both methods use the same model, temperature, token limit and per-run seed (run k uses seed k),
so the runs differ from each other but each run is paired across methods.

Writes runs/<method>_run<k>.jsonl, one line per call (raw output + ledger entry).
Re-running skips notices already done, so an interrupted run can resume.

    python3 extract.py                     # all 300 calls
    python3 extract.py --runs 1 --limit 3  # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "SQL-Calculation-Agent"))
from sqlagent.llm import OllamaChat  # noqa: E402  (class 1 wrapper and ledger)

SCHEMA = json.loads((HERE / "schema" / "notice_v1.schema.json").read_text())
NOTICES = json.loads((HERE / "data" / "notices.json").read_text())
RUNS_DIR = HERE / "runs"
METHODS = ("prompt", "grammar")

SYSTEM_PROMPT = """You extract one domain-name notice into JSON.
Reply with a single JSON object and nothing else, with exactly these keys:
- "schema_version": always the string "1.0".
- "domain": the registered domain name in lowercase, without "www.", subdomains, "http://" or a path.
- "notice_type": one of "expiry", "renewal", "transfer", "suspension", "payment", "other".
    expiry = the domain will expire or has expired; renewal = the domain was renewed;
    transfer = moving the domain between registrars (requests, auth codes, status);
    suspension = the domain is, will be, or was suspended (including reactivation);
    payment = an invoice, failed payment, payment received or refund;
    other = anything else. If the domain is or will be suspended, use "suspension";
    otherwise if a payment failed or an invoice is due, use "payment".
- "expires_on": the domain's registration expiry date as "YYYY-MM-DD", or null if the notice
    does not state a full date. For a renewal or transfer, use the new expiry date. Do not use
    sent dates, due dates, suspension dates or certificate dates. Never invent a date.
- "registrar": the registrar that sent the notice, as named (company name only); if the sender is
    not a registrar, the registrar named in the notice; otherwise null.
- "amount_due": the amount the recipient must pay or will be charged, as a number; null if nothing
    is due or the payment is already complete (receipts, refunds).
- "currency": ISO 4217 code of amount_due ("$" is "USD", "Tk" or "৳" is "BDT", "€" is "EUR",
    "£" is "GBP", "Rs" is "INR"); null when amount_due is null.
- "action_required": true if the recipient is asked to do something; false if the notice is
    informational or the only action is conditional ("if you did not request this...")."""

SETTINGS = {"model": "qwen2.5:3b", "temperature": 0.3, "num_predict": 256, "num_ctx": 4096}


def run_file(method, run):
    return RUNS_DIR / f"{method}_run{run}.jsonl"


def extract_one(llm, notice, method):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": notice["text"]}]
    res = llm.chat(messages, format=SCHEMA if method == "grammar" else None)
    return {"id": notice["id"], "method": method, "raw": res.content, **llm.ledger[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N notices (smoke test)")
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    notices = NOTICES[: args.limit] if args.limit else NOTICES
    RUNS_DIR.mkdir(exist_ok=True)

    for run in range(1, args.runs + 1):
        for method in args.methods.split(","):
            llm = OllamaChat(SETTINGS["model"], temperature=SETTINGS["temperature"], seed=run,
                             num_ctx=SETTINGS["num_ctx"], num_predict=SETTINGS["num_predict"])
            path = run_file(method, run)
            done = set()
            if path.exists():
                done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line}
            with path.open("a") as f:
                for n in notices:
                    if n["id"] in done:
                        continue
                    rec = {"run": run, "seed": run, **extract_one(llm, n, method)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"run {run} {method:7} {n['id']} {rec['seconds']:5.1f}s "
                          f"{rec['done_reason']}", flush=True)
            print(f"run {run} {method}: {len(llm.ledger)} new calls, ${llm.total_cost():.5f}")


if __name__ == "__main__":
    main()
