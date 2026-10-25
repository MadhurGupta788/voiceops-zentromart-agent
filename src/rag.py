"""Step 2: retrieve KB chunks and answer only from them."""
from .llm import generate_json
from .retrieval import get_index
from . import data
from .calendar_tools import calendar_block

ALWAYS = ("KB-00.1", "KB-00.2", "KB-00.3", "ESC-1")  # precedence, today's date, business days, escalation guide

REWRITE_SYSTEM = """You turn a customer's support message into a short English search query for a company policy knowledge base.
The message may be English, Hindi, Hinglish or another Indian language, and may contain speech-recognition errors.
Use the company's own terms: "send it back" or "nobody came to collect it" -> reverse pickup; "money back" -> refund; "stop my order" -> cancellation; "late" or "not arrived" -> delivery delay; "COD" -> cash on delivery.
Respond ONLY with JSON: {"query": "<english search query, max 20 words>"}"""

ANSWER_SYSTEM = """You are ZentroMart's customer support assistant, an AI assistant for an Indian online store selling groceries and electronics.
Say you are an AI assistant when you first greet or introduce yourself.
Answer ONLY from the CONTEXT below. Never invent a policy, date, amount or product fact. Each context block starts with its [section id].
If documents disagree, follow this precedence: current policy, then agent guide, then help article, then FAQ, then marketing. Blocks marked SUPERSEDED are reference only: use them only for the cases a current document says they still cover.
Return windows count calendar days; refund and logistics timelines count business days. Today's date is in the context.
A CALENDAR block, when present, lists dates already computed with the business-day rules. Never calculate dates yourself. When the customer asks by when or by which date something will happen, take the LONGEST number of business days in the policy's range for the payment method or situation and give the matching date from the CALENDAR block, written with the year.
Rows labelled products.csv or pincodes.csv are catalogue facts for the product or pincode in the question; combine them with the policies.
Never say a service or option is not offered unless a context block says so. If the context does not cover the question, say plainly that you do not have that information and offer to connect the customer with a human colleague. Do not guess.
Reply in the language the customer used (English, Hindi, Hinglish or another Indian language). Keep it to 1 to 3 short sentences written to be spoken aloud: no markdown. Write every number, amount, duration and count as digits (3 attempts, 12 months, Rs 79, 48 hours), never as words, and include the year in dates.
If your reply asks the customer for a detail you need, the decision must be clarify.
Respond ONLY with JSON: {"decision": "answer|clarify|escalate|refuse", "reply": "...", "sources": ["ids"]}
Decisions: answer = you answered, or said you lack the information; clarify = you need one more detail first; escalate = a human colleague must decide; refuse = the request breaks policy, so decline politely.
In sources list the id of every context block you relied on, including blocks that only supplied a condition or exception. Also list KB-00.1 if you resolved a conflict between documents, KB-00.2 if you used today's date, and KB-00.3 if you counted business days or holidays. List only ids that appear in the context."""


def _label(c):
    tag = (f"[{c['section_id']}] {c['heading']} | {c['file']} | {c['doc_type']}, "
           f"status {c['status']}, effective {c['effective_from']}")
    if c["status"] != "current":
        tag += " | SUPERSEDED: reference only"
    return tag


def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def answer(question, k=12, rewrite=False):
    idx = get_index()
    rq = ""
    if rewrite:
        rq = str(generate_json(REWRITE_SYSTEM, question).get("query", ""))
    hits = idx.search_rrf([question, rq], k=k)
    always = [c for c in idx.chunks if c["section_id"] in ALWAYS]

    blocks, ids, seen = [], [], set()
    for c in always + hits:
        if c["key"] in seen:
            continue
        seen.add(c["key"])
        blocks.append(f"{_label(c)}\n{c['text']}")
        ids.append(c["section_id"])
    for ev_key, txt in data.lookup(question + " " + rq):
        blocks.append(f"[{ev_key}] catalogue data\n{txt}")
        ids.append(ev_key)
    cal = calendar_block(question)
    if cal:
        blocks.append(cal)

    prompt = "CONTEXT:\n\n" + "\n\n".join(blocks) + "\n\nCUSTOMER MESSAGE:\n" + question
    out = generate_json(ANSWER_SYSTEM, prompt)
    src = out.get("sources", [])
    if not isinstance(src, list):
        src = []
    valid = set(ids)
    return {
        "decision": str(out.get("decision", "")),
        "reply": str(out.get("reply", "")),
        "sources": [s for s in _dedup(src) if s in valid],
        "retrieved": _dedup(ids),
        "rewrite": rq,
    }
