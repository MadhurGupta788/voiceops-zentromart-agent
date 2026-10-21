"""Step 1: prompt-only baseline. No documents, no tools. Saves after every question and resumes."""
import argparse
import time
import pandas as pd
from .config import DATASET_DIR, RESULTS_DIR
from .llm import generate_json, LLMUnavailable
from .scoring import check_must_include, check_must_not_include, check_decision

DECISION_HELP = {
    "answer": "you can answer the question directly (this includes saying you do not have that information)",
    "clarify": "you need one more detail from the customer before you can help",
    "escalate": "a human colleague must handle this",
    "refuse": "the request breaks policy or cannot be granted, so politely decline",
}

SYSTEM = """You are the customer support assistant for ZentroMart, a fictional Indian online store that sells groceries and electronics.
In your first reply you must tell the customer you are an AI assistant.
You do not have access to any company documents or order data right now, so do not invent policies, dates or amounts.
Reply in the same language the customer used (English, Hindi or Hinglish). Keep replies short, 1 to 3 sentences, written to be spoken aloud.
Respond ONLY with JSON: {{"decision": "<one of: {decisions}>", "reply": "<your reply to the customer>"}}
Decision meanings:
{meanings}
"""

OUT = RESULTS_DIR / "step1_baseline.csv"


def summarize(res, total):
    n = len(res)
    print(f"\nSTEP 1 RESULT on {n}/{total} questions" + ("" if n == total else "  (INCOMPLETE, rerun to resume)"))
    print(f"  content pass (must_include + must_not_include): {res['pass'].sum()}/{n} = {res['pass'].mean():.1%}")
    print(f"  decision correct: {res['decision_ok'].sum()}/{n} = {res['decision_ok'].mean():.1%}")
    print(res.groupby("answerable")["pass"].agg(["sum", "count"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--fresh", action="store_true", help="ignore saved results and start over")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_DIR / "practice" / "questions.csv", dtype=str).fillna("")
    if args.limit:
        df = df.head(args.limit)

    decisions = sorted(set(df["expected_decision"]) | set(DECISION_HELP))
    meanings = "\n".join(f"- {d}: {DECISION_HELP.get(d, d)}" for d in decisions)
    system = SYSTEM.format(decisions=", ".join(decisions), meanings=meanings)

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    if OUT.exists() and not args.fresh:
        old = pd.read_csv(OUT, dtype=str).fillna("")
        for c in ("include_ok", "exclude_ok", "decision_ok", "pass"):
            old[c] = old[c] == "True"
        rows = old.to_dict("records")
    done = {r["id"] for r in rows}
    print(f"{len(done & set(df['id']))} already done, {len(df) - len(done & set(df['id']))} to go\n")

    for _, r in df.iterrows():
        if r["id"] in done:
            continue
        try:
            out = generate_json(system, r[args.field])
        except LLMUnavailable as e:
            print(f"\nSTOPPED: {e}\nProgress is saved. Fix the quota issue and rerun the same command to resume.")
            break
        reply, dec = str(out.get("reply", "")), str(out.get("decision", ""))
        inc = check_must_include(reply, r["must_include"])
        exc = check_must_not_include(reply, r["must_not_include"])
        dok = check_decision(dec, r["expected_decision"], r["acceptable_decisions"])
        rows.append({"id": r["id"], "language": r["language"], "answerable": r["answerable"],
                     "question": r[args.field], "pred_decision": dec, "expected_decision": r["expected_decision"],
                     "reply": reply, "include_ok": inc, "exclude_ok": exc, "decision_ok": dok,
                     "pass": inc and exc})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{r['id']}: pass={inc and exc} decision_ok={dok}")
        time.sleep(args.delay)

    if rows:
        summarize(pd.DataFrame(rows), len(df))


if __name__ == "__main__":
    main()
