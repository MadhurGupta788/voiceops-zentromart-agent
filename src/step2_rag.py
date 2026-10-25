"""Step 2 evaluation: RAG on the 52 practice questions, compared with the Step 1 baseline."""
import argparse
import time
import pandas as pd
from .config import DATASET_DIR, RESULTS_DIR
from .llm import LLMUnavailable
from .scoring import check_must_include, check_must_not_include, check_decision
from . import data

OUT = RESULTS_DIR / "step2_rag.csv"
BOOLS = ("include_ok", "exclude_ok", "decision_ok", "pass", "retrieval_hit", "used_hit", "hardneg_used")


def items(cell):
    return [x.strip() for x in (cell or "").split(" ; ") if x.strip()]


def retrieval_only(df, args):
    from .retrieval import get_index
    idx = get_index()
    rows = []
    for _, r in df.iterrows():
        q = r[args.field]
        hits = idx.search([q], k=args.k)
        ret = []
        for x in [h["section_id"] for h in hits] + [e[0] for e in data.lookup(q)]:
            if x not in ret:
                ret.append(x)
        gold = items(r["gold_evidence"])
        if gold:
            rows.append({"id": r["id"], "gold": gold, "hit": all(g in ret for g in gold),
                         "any": any(g in ret for g in gold), "retrieved": ret})
    n = len(rows)
    print(f"\nRETRIEVAL ONLY (k={args.k}, no query rewrite) on {n} questions with gold evidence")
    print(f"  all gold found: {sum(x['hit'] for x in rows)}/{n} = {sum(x['hit'] for x in rows) / n:.1%}")
    print(f"  any gold found: {sum(x['any'] for x in rows)}/{n} = {sum(x['any'] for x in rows) / n:.1%}")
    print("\nMisses:")
    for x in rows:
        if not x["hit"]:
            print(f"  {x['id']}: gold={x['gold']} retrieved={x['retrieved'][:8]}")


def summarize(res, total):
    n = len(res)
    print(f"\nSTEP 2 RESULT on {n}/{total} questions" + ("" if n == total else "  (INCOMPLETE, rerun to resume)"))
    print(f"  content pass: {res['pass'].sum()}/{n} = {res['pass'].mean():.1%}")
    print(f"  decision correct: {res['decision_ok'].sum()}/{n} = {res['decision_ok'].mean():.1%}")
    g = res[res["gold"] != ""]
    if len(g):
        print(f"  retrieval found all gold evidence: {g['retrieval_hit'].sum()}/{len(g)} = {g['retrieval_hit'].mean():.1%}")
        print(f"  answer cited all gold evidence: {g['used_hit'].sum()}/{len(g)} = {g['used_hit'].mean():.1%}")
    print(f"  answers citing a hard-negative section: {res['hardneg_used'].sum()}")
    print(res.groupby("answerable")["pass"].agg(["sum", "count"]).to_string())
    s1 = RESULTS_DIR / "step1_baseline.csv"
    if s1.exists():
        b = pd.read_csv(s1, dtype=str)
        bp = (b["pass"] == "True")
        print(f"\nCOMPARISON  Step 1: {bp.sum()}/{len(b)} = {bp.mean():.1%}   Step 2: {res['pass'].sum()}/{n} = {res['pass'].mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--delay", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_DIR / "practice" / "questions.csv", dtype=str).fillna("")
    if args.limit:
        df = df.head(args.limit)
    if args.retrieval_only:
        retrieval_only(df, args)
        return

    from .rag import answer
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    if OUT.exists() and not args.fresh:
        old = pd.read_csv(OUT, dtype=str).fillna("")
        for c in BOOLS:
            old[c] = old[c] == "True"
        rows = old.to_dict("records")
    done = {r["id"] for r in rows}
    print(f"{len(done & set(df['id']))} already done, {len(df) - len(done & set(df['id']))} to go\n")

    for _, r in df.iterrows():
        if r["id"] in done:
            continue
        try:
            out = answer(r[args.field], k=args.k, rewrite=args.rewrite)
        except LLMUnavailable as e:
            print(f"\nSTOPPED: {e}\nProgress is saved. Fix the quota issue and rerun the same command to resume.")
            break
        reply, dec = out["reply"], out["decision"]
        inc = check_must_include(reply, r["must_include"])
        exc = check_must_not_include(reply, r["must_not_include"])
        dok = check_decision(dec, r["expected_decision"], r["acceptable_decisions"])
        gold, hard = items(r["gold_evidence"]), items(r["hard_negatives"])
        rows.append({
            "id": r["id"], "language": r["language"], "answerable": r["answerable"],
            "question": r[args.field], "rewrite": out["rewrite"],
            "pred_decision": dec, "expected_decision": r["expected_decision"], "reply": reply,
            "include_ok": inc, "exclude_ok": exc, "decision_ok": dok, "pass": inc and exc,
            "gold": " ; ".join(gold), "retrieved": " ; ".join(out["retrieved"]),
            "used": " ; ".join(out["sources"]),
            "retrieval_hit": all(g in out["retrieved"] for g in gold) if gold else True,
            "used_hit": all(g in out["sources"] for g in gold) if gold else True,
            "hardneg_used": any(h in out["sources"] for h in hard),
        })
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{r['id']}: pass={inc and exc} decision_ok={dok} used={out['sources']}")
        time.sleep(args.delay)

    if rows:
        summarize(pd.DataFrame(rows), len(df))


if __name__ == "__main__":
    main()
