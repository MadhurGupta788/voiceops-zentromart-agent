"""Step 3 evaluation: replay practice/conversations.csv turn by turn. Saves after each conversation and resumes."""
import argparse
import re
import time
import pandas as pd
from .config import DATASET_DIR, RESULTS_DIR
from .llm import LLMUnavailable
from .scoring import check_must_include, check_must_not_include

OUT = RESULTS_DIR / "step3_conv.csv"
BOOLS = ("content_ok", "decision_ok", "tools_ok", "evidence_ok", "turn_ok")


def calls_of(cell):
    return sorted(re.sub(r"\s+", "", c).lower() for c in re.findall(r"[A-Za-z_]+\([^)]*\)", cell or ""))


def items(cell):
    return [x.strip() for x in (cell or "").split(" ; ") if x.strip()]


def summarize(df, total_convs):
    n = len(df)
    print(f"\nSTEP 3 RESULT: {df['conversation_id'].nunique()}/{total_convs} conversations, {n} turns")
    for col, label in [("content_ok", "content (must_include / must_not_include)"), ("decision_ok", "decision"),
                       ("tools_ok", "tool calls"), ("evidence_ok", "gold evidence cited"), ("turn_ok", "ALL checks at once")]:
        print(f"  {label}: {df[col].sum()}/{n} = {df[col].mean():.1%}")
    conv_ok = df.groupby("conversation_id")["turn_ok"].all()
    print(f"  conversations with every turn fully correct: {conv_ok.sum()}/{len(conv_ok)}")
    print(df.groupby("scenario")["turn_ok"].agg(["sum", "count"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="", help="comma-separated conversation ids, e.g. PC007,PC021")
    ap.add_argument("--limit", type=int, default=None, help="first N conversations")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--fails", action="store_true", help="print every turn that failed a check")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_DIR / "practice" / "conversations.csv", dtype=str).fillna("")
    order = list(dict.fromkeys(df["conversation_id"]))
    total = len(order)
    if args.conv:
        order = [c for c in order if c in set(args.conv.split(","))]
    if args.limit:
        order = order[:args.limit]

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    if OUT.exists() and not args.fresh:
        old = pd.read_csv(OUT, dtype=str).fillna("")
        for c in BOOLS:
            old[c] = old[c] == "True"
        rows = old.to_dict("records")
    done = {r["conversation_id"] for r in rows}

    from .conv import Conversation
    for cid in order:
        if cid in done:
            continue
        g = df[df["conversation_id"] == cid].copy()
        g["t"] = g["turn"].astype(int)
        g = g.sort_values("t")
        conv, new = Conversation(), []
        try:
            for _, r in g.iterrows():
                out = conv.respond(r["customer_message"])
                inc = check_must_include(out["reply"], r["must_include"])
                exc = check_must_not_include(out["reply"], r["must_not_include"])
                dok = out["decision"] == r["expected_decision"].strip().lower()
                tok = calls_of(r["expected_tool_calls"]) == calls_of(", ".join(out["tool_calls"]))
                gold = items(r["gold_evidence"])
                eok = all(x in out["sources"] for x in gold)
                row = {"conversation_id": cid, "turn": r["turn"], "scenario": r["scenario"],
                       "customer_message": r["customer_message"], "expected_decision": r["expected_decision"],
                       "pred_decision": out["decision"], "reply": out["reply"],
                       "expected_tools": r["expected_tool_calls"], "tools": " ; ".join(out["tool_calls"]),
                       "gold": " ; ".join(gold), "sources": " ; ".join(out["sources"]),
                       "content_ok": inc and exc, "decision_ok": dok, "tools_ok": tok, "evidence_ok": eok,
                       "turn_ok": inc and exc and dok and tok and eok, "event": out["event"]}
                new.append(row)
                flags = f"content={inc and exc} decision={dok}({out['decision']}) tools={tok} evidence={eok}"
                print(f"{cid} t{r['turn']}: {flags}")
                time.sleep(args.delay)
        except LLMUnavailable as e:
            print(f"\nSTOPPED: {e}\nCompleted conversations are saved. Rerun the same command to resume.")
            break
        rows += new
        pd.DataFrame(rows).to_csv(OUT, index=False)

    if rows:
        res = pd.DataFrame(rows)
        summarize(res, total)
        if args.fails:
            for _, x in res[~res["turn_ok"]].iterrows():
                print(f"\n{x['conversation_id']} t{x['turn']} [{x['scenario']}] exp={x['expected_decision']} got={x['pred_decision']}")
                print("  C:", x["customer_message"][:100])
                print("  tools exp:", x["expected_tools"][:80], "| got:", x["tools"][:80])
                print("  gold:", x["gold"], "| cited:", x["sources"])
                print("  A:", x["reply"][:220])


if __name__ == "__main__":
    main()
