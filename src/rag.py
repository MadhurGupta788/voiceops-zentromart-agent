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
A CALENDAR block, when present, lists dates already computed with the business-day rules. Never calculate dates yourself. When the customer asks by when or by which date something will happen, find the policy row for the customer's own payment method or situation (not another method's), take the LONGEST number of business days in that row's range, and give the matching date from the CALENDAR block.
Rows labelled products.csv or pincodes.csv are catalogue facts for the product or pincode in the question; combine them with the policies.
Check effective dates: when a section says a rule changed on a date, or that earlier cases keep an older rule, compare it with the date in the customer's situation (such as the delivery date) and apply the version that fits.
Only ask the customer for a detail (an order date, an order ID) when the answer really depends on it. If a policy plus the catalogue row answers the question in general, answer it.
If a context block says a person must decide or review (for example a damage claim above a value limit, a priority resolution team, compensation) and the customer's situation matches, the decision must be escalate and the reply must say a colleague or team will take it from here. Do not say you lack the information in that case.
If no context block covers what the customer asks, say plainly that you do not have that information and offer to connect them with a human colleague. In that case never say "we do not offer", "not available", "we only" or any other claim about what the company does or does not do.
Reply in the language the customer used (English, Hindi, Hinglish or another Indian language). Keep the reply to 1 to 3 short sentences written to be spoken aloud: no markdown.
Write every number, amount, duration and count as digits (3 attempts, 12 months, Rs 79, 48 hours), never as words. Write dates exactly as day, month name, year, for example 21 October 2026, never with st, nd, rd or th.
If your reply asks the customer for a detail you need, the decision must be clarify.
Respond ONLY with JSON, with the fields in this order:
{"basis": "...", "decision": "answer|clarify|escalate|refuse", "reply": "...", "sources": ["ids"]}
basis is one short scratch line for yourself: which context block ids answer the question (write NONE if no block covers it); for any date question also the payment method or situation, the policy's range of business days, and the number of days you will use. If basis is NONE, the reply must say you do not have that information.
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


_VOCAB = [
    (("crack", "broken", "shatter", "smash", "dent", "defective", "not working", "stopped working", "dead on arrival"),
     "damaged or defective item claim high value item damage priority resolution team"),
    (("send it back", "send back", "wapas", "give back"), "return request"),
    (("money back", "paisa wapas", "paise wapas"), "refund"),
]


def _expand(text):
    """Map everyday customer wording to the company's own terms (static list, no API call)."""
    low = text.lower()
    return " ".join(term for words, term in _VOCAB if any(w in low for w in words))


def answer(question, k=12, rewrite=False):
    idx = get_index()
    rq = ""
    if rewrite:
        rq = str(generate_json(REWRITE_SYSTEM, question).get("query", ""))
    hits = idx.search_rrf([question, rq, _expand(question)], k=k)
    always = [c for c in idx.chunks if c["section_id"] in ALWAYS]
    if "damaged or defective" in _expand(question):
        always += [c for c in idx.chunks if c["section_id"] == "DWM-3"]  # high-value damage rule (over Rs 10,000)

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
        "basis": str(out.get("basis", "")),
    }
