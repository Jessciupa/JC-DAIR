"""LTL formula trees with the two string forms used by the datasets.

canonical: prefix, operators as words, APs as space-separated phrases
    globally ( implies ( pedestrian in crosswalk , not ( enter crosswalk ) ) )
raw: fully parenthesised infix, Spot-compatible operators, snake_case APs
    G ( ( pedestrian_in_crosswalk ) -> ( ! ( enter_crosswalk ) ) )

Both forms are printed deterministically from a tree, and both parse back to the same tree,
so a formula is well-formed iff it round-trips.
"""

from dataclasses import dataclass

UNARY = {"globally": "G", "finally": "F", "next": "X", "not": "!"}
BINARY = {"and": "&", "or": "|", "until": "U", "implies": "->"}
OPERATOR_WORDS = set(UNARY) | set(BINARY)
RAW_UNARY = {v: k for k, v in UNARY.items()}
RAW_BINARY = {v: k for k, v in BINARY.items()}


@dataclass(frozen=True)
class Node:
    op: str  # an operator word, or "ap"
    args: tuple = ()
    name: str = ""  # AP phrase in canonical words, e.g. "red light"

    def aps(self):
        if self.op == "ap":
            return {self.name}
        return set().union(*(a.aps() for a in self.args))


def ap(phrase):
    words = phrase.split()
    if not words or any(w in OPERATOR_WORDS or set(w) & set("(),_") for w in words):
        raise ValueError(f"bad AP phrase: {phrase!r}")
    return Node("ap", name=" ".join(words))


def op(name, *args):
    arity = 1 if name in UNARY else 2 if name in BINARY else None
    if arity != len(args):
        raise ValueError(f"{name} takes {arity} arguments, got {len(args)}")
    return Node(name, tuple(args))


def ap_to_raw(phrase):
    return phrase.replace(" ", "_")


def to_canonical(n):
    if n.op == "ap":
        return n.name
    return f"{n.op} ( {' , '.join(to_canonical(a) for a in n.args)} )"


def to_raw(n):
    if n.op == "ap":
        return ap_to_raw(n.name)
    if n.op in UNARY:
        return f"{UNARY[n.op]} ( {to_raw(n.args[0])} )"
    left, right = n.args
    return f"( {to_raw(left)} ) {BINARY[n.op]} ( {to_raw(right)} )"


class _Tokens:
    def __init__(self, s):
        self.toks = s.split()
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("unexpected end of formula")
        self.i += 1
        return tok

    def expect(self, tok):
        got = self.next()
        if got != tok:
            raise ValueError(f"expected {tok!r}, got {got!r} at token {self.i}")

    def done(self):
        if self.peek() is not None:
            raise ValueError(f"trailing tokens from {self.i}: {self.toks[self.i:]}")


def parse_canonical(s):
    t = _Tokens(s)
    n = _canonical_expr(t)
    t.done()
    return n


def _canonical_expr(t):
    tok = t.peek()
    if tok in OPERATOR_WORDS:
        t.next()
        t.expect("(")
        args = [_canonical_expr(t)]
        if tok in BINARY:
            t.expect(",")
            args.append(_canonical_expr(t))
        t.expect(")")
        return op(tok, *args)
    words = []
    while t.peek() not in (None, "(", ")", ","):
        words.append(t.next())
    return ap(" ".join(words))


def parse_raw(s):
    t = _Tokens(s)
    n = _raw_expr(t)
    t.done()
    return n


def _raw_expr(t):
    left = _raw_primary(t)
    if t.peek() in RAW_BINARY:
        return op(RAW_BINARY[t.next()], left, _raw_primary(t))
    return left


def _raw_primary(t):
    tok = t.next()
    if tok in RAW_UNARY:
        t.expect("(")
        n = _raw_expr(t)
        t.expect(")")
        return op(RAW_UNARY[tok], n)
    if tok == "(":
        n = _raw_expr(t)
        t.expect(")")
        return n
    if tok in RAW_BINARY or tok == ")":
        raise ValueError(f"unexpected {tok!r}")
    return ap(tok.replace("_", " "))


def check_roundtrip(n):
    """Raise if either string form of n does not parse back to n."""
    for fmt, parse in ((to_canonical, parse_canonical), (to_raw, parse_raw)):
        s = fmt(n)
        if parse(s) != n:
            raise ValueError(f"{fmt.__name__} does not round-trip: {s}")
