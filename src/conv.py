"""Step 3: multi-turn support agent.
Identity verification, tool calls and handover tickets are enforced in code; the LLM only reads the context and writes the reply."""
import json
import re
import sys
from datetime import date, timedelta

from .config import DATASET_DIR
from .llm import generate_json
from .retrieval import get_index
from . import data
from .data import WORD_DIGITS
from .calendar_tools import calendar_block, fmt, after, is_business_day
from .rag import _label, _dedup, _expand

TODAY_D = date(2026, 10, 20)
ALWAYS_CONV = ("KB-00.1", "KB-00.2", "KB-00.3", "ESC-1", "ESC-2", "ESC-3", "ESC-4",
               "IDV-1", "IDV-2", "IDV-3", "IDV-4", "STY-1", "STY-2", "STY-3", "STY-4", "STY-5")
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"}
SKIP_BEFORE = MONTHS | {"₹", "rs", "inr"}
_DEV = str.maketrans("०१२३४५६७८९", "0123456789")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$")
DECISIONS = {"answer", "clarify", "escalate", "refuse"}
HIDE = {"delivery_pincode", "customer_id", "verified", "found", "verification_required", "message"}

CONV_SYSTEM = """You are ZentroMart's customer support assistant, an AI assistant for an Indian online store selling groceries and electronics. You are in a live conversation by chat or voice, so keep every reply to 1 to 3 short sentences written to be spoken aloud, with no markdown.
The CONVERSATION STATE block is maintained by the system's code and is always correct. The code, not you, looks up orders and creates handover tickets. Your only actions are looking up an order and handing over to a human colleague. Never say you did anything else (cancelled, refunded, approved, added a voucher, scheduled, changed an address) and never say a ticket was created.
Answer ONLY from the CONTEXT. Never invent a policy, date, amount or product fact. Each context block starts with its [section id].
If documents disagree, follow this precedence: current policy, then agent guide, then help article, then FAQ, then marketing. SUPERSEDED blocks are reference only.
Return windows count calendar days; refund and logistics timelines count business days. Today's date is in the context. ORDER DATE NOTES and CALENDAR blocks hold dates and day counts already computed: never calculate dates yourself. For "by when" questions, use the LONGEST number of business days in the policy row for the customer's own payment method or situation.
Write dates as day, month name, year (9 October 2026), numbers and amounts as digits (3 attempts, 49 rupees), and order IDs in groups of four digits (8978 2837). Never read out a full phone number, email, street address or pincode, and never mention a verification value.
Verification rules, following the STATE block:
- General policy questions need no verification: answer them directly from the documents.
- Anything about a specific order (status, dates, items, amounts, refund details) needs a VERIFIED order. If there is no order ID, ask for the order ID only (decision clarify). If the order ID is given but not verified, ask for ONE detail: the last 4 digits of the registered phone number OR the delivery pincode (decision clarify), and share nothing about the order.
- When the state says the customer just verified, say so briefly and answer what they asked earlier, or ask how you can help (decision answer).
- When a verification attempt failed once, say it did not match and ask them to try once more (decision clarify). When it failed twice, share no order details, say you cannot share them, suggest contacting support from the registered phone number, and offer general policy help (decision refuse).
Conversation rules:
- Use the conversation so far and the order record to understand follow-ups such as "how long will that take?" without asking again for the order ID or the problem. If the customer switches topic, answer the new question; return to the order record when they come back to it.
- Ask for one detail at a time. Only ask when the answer really depends on it.
- If the customer asks for something the policy does not allow (compensation, a voucher, cancelling an order that has shipped, an exception), decision refuse: apologise briefly if they are unhappy, say what the policy does allow, and promise nothing.
- If the customer asks about cancelling an order that has not shipped, or a return, explain the policy and what they can do; you cannot perform it yourself.
- Hand over to a human (decision escalate) whenever the ESC-1 conditions apply: damaged, defective or wrong item worth more than 10,000 rupees; activity the customer did not make; legal action, consumer court or police; a third or later contact about the same issue within 7 days (use contacts_last_7_days in the ORDER RECORD, which counts earlier contacts only, and escalate when it is 2 or more, or when the customer says it is their third or later call); a refund more than 3 business days past its maximum timeline; the customer asks for a human; a product safety risk. Then say a human colleague will contact them, promise no outcome, amount or time beyond what a policy states, and fill intent, handover_summary and handover_reason.
- If no context block covers the question, say plainly that you do not have that information and offer a human colleague. Never claim the company does not offer something unless a block says so.
- Introduce yourself as ZentroMart's AI assistant only in the very first reply of the conversation (turn 1).
- Reply in the language the customer uses (English, Hindi, Hinglish or another Indian language) and switch if they switch.
Respond ONLY with JSON, with the fields in this order:
{"basis": "...", "decision": "answer|clarify|escalate|refuse", "reply": "...", "sources": ["ids"], "intent": "...", "handover_summary": "...", "handover_reason": "..."}
basis is one short scratch line: which block ids answer the question (NONE if no block covers it) and, for date questions, the policy range and the days you will use. intent, handover_summary and handover_reason are only filled when the decision is escalate; the summary must state the customer's issue, the detected intent, the order ID if any, what the customer has already told you, what you already checked, and the reason for escalation, without any verification value.
In sources list the id of every context block you relied on, including blocks that only supplied a condition or exception. List only ids that appear in the context."""

EVENTS = {
    "": "No verification event this turn.",
    "order_id_received": "The customer just gave an order ID. It is NOT verified. Ask for ONE detail: the last 4 digits of the registered phone number OR the delivery pincode. Say nothing about the order.",
    "verified": "The customer just passed verification. Say so briefly, then answer what they asked earlier or ask how you can help.",
    "verification_failed_retry": "The value did not match. First failed attempt: say it did not match and ask them to try once more. Never reveal the right value.",
    "verification_failed_final": "The value did not match again. Verification failed twice: share NO order details, say you cannot share them, suggest contacting support from the registered phone number, offer general policy help. Decision refuse.",
    "locked_value_ignored": "Verification already failed twice for this order. Share no order details; general policy help only.",
}

_T = None


def _tools():
    global _T
    if _T is None:
        if str(DATASET_DIR) not in sys.path:
            sys.path.insert(0, str(DATASET_DIR))
        from tools import lookup_order, create_ticket
        _T = (lookup_order, create_ticket)
    return _T


def _finish(run, prev):
    """run: list of (digits, is_word). Return the digit string if it is a 4, 6 or 8 digit run."""
    while run:
        d = "".join(x for x, _ in run)
        if len(d) in (4, 6, 8):
            if prev in SKIP_BEFORE:   # a year after a month name, or an amount after Rs
                return None
            return d
        if run[0][1]:
            run = run[1:]
        else:
            return None
    return None


def digit_runs(text):
    """Digit sequences in a message: 8 digits = order ID, 4 = phone suffix, 6 = pincode. Understands spoken digits."""
    text = text.translate(_DEV)
    toks = re.findall(r"[0-9]+|[A-Za-z]+|[^\s\w]|\w", text)
    runs, cur, run_prev, last = [], [], "", ""
    for t in toks + ["\n"]:
        low = t.lower()
        if t.isascii() and t.isdigit():
            if not cur:
                run_prev = last
            cur.append((t, False))
            continue
        if t.isalpha() and low in WORD_DIGITS:
            if not cur:
                run_prev = last
            cur.append((WORD_DIGITS[low], True))
            continue
        if t == "-" and cur:
            continue
        d = _finish(cur, run_prev)
        if d:
            runs.append(d)
        cur = []
        last = low
    return runs


def bd_between(start, end):
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d):
            n += 1
    return n


def order_notes(rec):
    notes = [f"Today is {fmt(TODAY_D)}."]
    for k, v in rec.items():
        if isinstance(v, str) and ISO.match(v):
            try:
                d = date.fromisoformat(v[:10])
            except ValueError:
                continue
            line = f"{k} = {fmt(d)}; calendar days from it to today: {(TODAY_D - d).days}"
            if "refund" in k.lower():
                line += (f"; business days elapsed from it to today: {bd_between(d, TODAY_D)}; "
                         + "; ".join(f"after {n} business days = {fmt(after(d, n))}" for n in range(1, 11)))
            notes.append(line)
    if "contacts_last_7_days" in rec:
        notes.append(f"contacts_last_7_days = {rec['contacts_last_7_days']} (earlier contacts only; ESC-1 escalates at 2 or more)")
    return notes


class Conversation:
    def __init__(self, k=10):
        self.k = k
        self.turn = 0
        self.history = []        # (role, text), verification values redacted
        self.order_id = None
        self.verified = False
        self.record = None
        self.attempts = 0
        self.locked = False
        self.ticket_id = None
        self.secrets = set()

    def _state(self, event):
        if self.locked:
            ver = "FAILED TWICE: locked. Share no order details."
        elif self.verified:
            ver = "VERIFIED: you may use the ORDER RECORD."
        elif self.order_id:
            ver = f"NOT verified yet ({self.attempts} failed attempts)."
        else:
            ver = "No order ID given yet."
        oid = f"{self.order_id[:4]} {self.order_id[4:]}" if self.order_id else "none"
        return (f"turn number: {self.turn} ({'first reply: introduce yourself' if self.turn == 1 else 'not the first reply'})\n"
                f"order ID: {oid}\nverification: {ver}\nthis turn: {EVENTS.get(event, '')}\n"
                f"handover ticket already created in this conversation: {'yes' if self.ticket_id else 'no'}")

    def respond(self, message):
        lookup_order, create_ticket = _tools()
        self.turn += 1
        calls, code_src, event = [], [], ""
        runs = digit_runs(message)
        oids = [r for r in runs if len(r) == 8]
        vals = [r for r in runs if len(r) in (4, 6)]
        if self.turn == 1:
            code_src.append("STY-1")

        if oids and oids[0] != self.order_id:
            self.order_id = oids[0]
            self.verified, self.record, self.attempts, self.locked = False, None, 0, False
            lookup_order(self.order_id)
            calls.append(f"lookup_order(order_id={self.order_id})")
            code_src.append("IDV-2")
            event = "order_id_received"

        if self.order_id and not self.verified and not self.locked and vals:
            v = vals[0]
            self.secrets.add(v)
            res = lookup_order(self.order_id, verification_value=v)
            calls.append(f"lookup_order(order_id={self.order_id}, verification_value={v})")
            if res.get("found") and res.get("verified"):
                self.verified, self.record, event = True, res, "verified"
                code_src.append("IDV-2")
            else:
                self.attempts += 1
                code_src.append("IDV-3")
                if self.attempts >= 2:
                    self.locked, event = True, "verification_failed_final"
                else:
                    event = "verification_failed_retry"
        elif self.locked and vals:
            event = "locked_value_ignored"

        safe = message
        for v in vals:
            safe = safe.replace(v, "[value]")
        prior = " ".join(m for r, m in self.history if r == "customer")[-300:]

        idx = get_index()
        hits = idx.search_rrf([safe, f"{prior} {safe}".strip(), _expand(f"{safe} {prior}")], k=self.k)
        wanted = set(ALWAYS_CONV)
        if "damaged or defective" in _expand(f"{safe} {prior}"):
            wanted.add("DWM-3")
        always = [c for c in idx.chunks if c["section_id"] in wanted]

        blocks, ids, seen = [], [], set()
        for c in always + hits:
            if c["key"] in seen:
                continue
            seen.add(c["key"])
            blocks.append(f"{_label(c)}\n{c['text']}")
            ids.append(c["section_id"])

        rows = data.lookup(f"{safe} {prior}")
        if self.verified and self.record:
            for it in self.record.get("items") or []:
                if it.get("sku"):
                    rows += data.lookup(str(it["sku"]))
        seen_rows = set()
        for key, txt in rows:
            if key in seen_rows or len(seen_rows) >= 6:
                continue
            seen_rows.add(key)
            blocks.append(f"[{key}] catalogue data\n{txt}")
            ids.append(key)

        if self.verified and self.record:
            rec = {k: v for k, v in self.record.items() if v is not None and k not in HIDE}
            blocks.append("[ORDER RECORD] verified order, from the order lookup tool\n" + json.dumps(rec, ensure_ascii=False, default=str))
            blocks.append("[ORDER DATE NOTES] computed by code\n" + "\n".join(order_notes(rec)))
        cal = calendar_block(safe)
        if cal:
            blocks.append(cal)

        hist = "\n".join(f"{'Customer' if r == 'customer' else 'Assistant'}: {m}" for r, m in self.history[-8:]) or "(none yet)"
        prompt = ("CONTEXT:\n\n" + "\n\n".join(blocks) + "\n\nCONVERSATION STATE:\n" + self._state(event)
                  + "\n\nCONVERSATION SO FAR:\n" + hist + f"\n\nCUSTOMER MESSAGE (turn {self.turn}):\n" + safe)
        out = generate_json(CONV_SYSTEM, prompt)

        decision = str(out.get("decision", "")).strip().lower()
        if decision not in DECISIONS:
            decision = "answer"
        reply = str(out.get("reply", "")).strip()
        if event == "verification_failed_retry":
            decision = "clarify"
        elif event == "verification_failed_final":
            decision = "refuse"

        if decision == "escalate" and not self.ticket_id:
            summary = str(out.get("handover_summary", "")).strip()
            if len(summary) < 40:
                summary = (f"Customer issue: {safe[:150]}. Detected intent: {out.get('intent') or 'unknown'}. "
                           f"Order ID: {self.order_id or 'none given'}. Verified: {'yes' if self.verified else 'no'}. "
                           f"Escalation reason: {out.get('handover_reason') or 'a human decision is required'}.")
            for v in self.secrets:
                summary = summary.replace(v, "[withheld]")
            res = create_ticket(summary=summary, order_id=self.order_id,
                                reason=str(out.get("handover_reason") or "escalation required"),
                                intent=str(out.get("intent") or ""))
            calls.append("create_ticket()")
            if res.get("created"):
                self.ticket_id = res.get("ticket_id")
                code_src += ["ESC-1", "ESC-2", "ESC-3"]
            else:
                reply = "I am sorry, I could not pass this to a colleague just now. Please contact support again in a few minutes."

        valid = set(ids)
        src = out.get("sources", [])
        src = [s for s in src if isinstance(s, str)] if isinstance(src, list) else []
        sources = [s for s in _dedup(src + code_src) if s in valid]

        self.history.append(("customer", safe))
        self.history.append(("assistant", reply))
        return {"decision": decision, "reply": reply, "sources": sources, "retrieved": _dedup(ids),
                "tool_calls": calls, "ticket_id": self.ticket_id, "basis": str(out.get("basis", "")), "event": event}
