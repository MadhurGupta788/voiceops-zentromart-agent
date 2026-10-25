"""Load kb/*.md into section chunks that keep their [SECTION-ID] and document metadata."""
import re
from .config import DATASET_DIR

KB_DIR = DATASET_DIR / "kb"
HEAD_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$")


def _front_matter(text):
    meta, body = {}, text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = parts[2]
    return meta, body


def _split(text, limit=1200):
    """Split long sections on blank lines (tables stay whole)."""
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for p in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(p) + 2 > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def load_chunks():
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").lstrip("\ufeff")
        meta, body = _front_matter(text)
        sections, cur = [], None
        for line in body.splitlines():
            m = HEAD_RE.match(line)
            if m:
                cur = {"section_id": m.group(1).strip(), "heading": m.group(2).strip(), "lines": []}
                sections.append(cur)
            elif cur is not None:
                cur["lines"].append(line)
        if not sections:
            sections = [{"section_id": meta.get("doc_id", path.stem),
                         "heading": meta.get("title", path.stem), "lines": body.splitlines()}]
        for s in sections:
            body_text = "\n".join(s["lines"]).strip()
            for i, part in enumerate(_split(body_text)):
                chunks.append({
                    "file": path.name,
                    "doc_id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "status": meta.get("status", "current"),
                    "effective_from": meta.get("effective_from", ""),
                    "language": meta.get("language", ""),
                    "supersedes": meta.get("supersedes", ""),
                    "section_id": s["section_id"],
                    "heading": s["heading"],
                    "text": part,
                    "key": f"{path.name}#{s['section_id']}#{i}",
                    "embed_text": f"{meta.get('title', '')} | {s['heading']}\n{part}",
                })
    return chunks


if __name__ == "__main__":
    from collections import Counter
    ch = load_chunks()
    print(len(ch), "chunks from", len({c["file"] for c in ch}), "files")
    print("status:", dict(Counter(c["status"] for c in ch)))
    print("doc_type:", dict(Counter(c["doc_type"] for c in ch)))
    ids = Counter(c["section_id"] for c in ch)
    print("section ids used more than once:", [i for i, n in ids.items() if n > 1][:20])
    print("longest chunk (chars):", max(len(c["text"]) for c in ch))
    print("example ids:", [c["section_id"] for c in ch[:12]])
