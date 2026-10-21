def _norm(s):
    return " ".join(str(s).lower().split())


def _items(cell):
    """'a / b ; c' -> [['a','b'], ['c']]"""
    cell = (cell or "").strip()
    if not cell:
        return []
    return [[_norm(a) for a in item.split(" / ") if a.strip()]
            for item in cell.split(" ; ") if item.strip()]


def check_must_include(reply, cell):
    r = _norm(reply)
    return all(any(alt in r for alt in alts) for alts in _items(cell))


def check_must_not_include(reply, cell):
    r = _norm(reply)
    return not any(alt in r for alts in _items(cell) for alt in alts)


def check_decision(pred, expected, acceptable):
    ok = {_norm(expected)} | {a[0] for a in _items(acceptable)}
    return _norm(pred) in ok
