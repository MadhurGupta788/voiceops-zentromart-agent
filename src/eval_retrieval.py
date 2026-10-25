"""Retrieval-only evaluation that mirrors the real pipeline (always-on sections, product rows, optional rewrite)."""
import argparse
import time
import pandas as pd
from .config import DATASET_DIR
from . import data


def items(cell):
    return [x.strip() for x in (cell or "").split(" ; ") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--field", default="question", choices=["question", "asr_text"])
    ap.add_argument("--rewrite", action="store_true", help="use the LLM query rewrite (cached)")
    ap.add_argument("--fusion", default="max", choices=["max", "rrf"])
    ap.add_argument("--extra", default="", help="comma-separated section ids to always include, e.g. ESC-1")
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--quiet", action="store_true", help="hide the list of misses")
    args = ap.parse_args()

    from .retrieval import get_index
    from .rag import ALWAYS, REWRITE_SYSTEM
    idx = get_index()
    extra = [s.strip() for s in args.extra.split(",") if s.strip()]
    always = [c["section_id"] for c in idx.chunks if c["section_id"] in ALWAYS or c["section_id"] in extra]
    search = idx.search_rrf if args.fusion == "rrf" else idx.search

    df = pd.read_csv(DATASET_DIR / "practice" / "questions.csv", dtype=str).fillna("")
    rows = []
    for _, r in df.iterrows():
        gold = items(r["gold_evidence"])
        if not gold:
            continue
        q = r[args.field]
        qs = [q]
        if args.rewrite:
            from .llm import generate_json, _cache_path, LLMUnavailable
            cached = _cache_path(REWRITE_SYSTEM, q).exists()
            try:
                qs.append(str(generate_json(REWRITE_SYSTEM, q).get("query", "")))
            except LLMUnavailable as e:
                print(f"STOPPED: {e}")
                break
            if not cached:
                time.sleep(args.delay)
        ret = [h["section_id"] for h in search(qs, k=args.k)] + always + [e[0] for e in data.lookup(" ".join(qs))]
        ret = list(dict.fromkeys(ret))
        found = [g for g in gold if g in ret]
        rows.append({"id": r["id"], "gold": gold, "ret": ret, "all": len(found) == len(gold),
                     "any": bool(found), "recall": len(found) / len(gold)})

    n = len(rows)
    mode = "with LLM rewrite" if args.rewrite else "no rewrite"
    print(f"\nRETRIEVAL k={args.k}, {mode}, fusion={args.fusion}, extra={extra or 'none'}, {n} questions with gold evidence")
    print(f"  all gold found:  {sum(x['all'] for x in rows)}/{n} = {sum(x['all'] for x in rows) / n:.1%}")
    print(f"  any gold found:  {sum(x['any'] for x in rows)}/{n} = {sum(x['any'] for x in rows) / n:.1%}")
    print(f"  mean recall:     {sum(x['recall'] for x in rows) / n:.1%}")
    if not args.quiet:
        print("\nMisses (gold sections not retrieved):")
        for x in rows:
            if not x["all"]:
                miss = [g for g in x["gold"] if g not in x["ret"]]
                print(f"  {x['id']}: missing={miss}  gold={x['gold']}")


if __name__ == "__main__":
    main()
