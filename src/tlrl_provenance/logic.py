"""Occurrence-preserving future-time LTLf syntax.

The AST is deliberately immutable.  Formula occurrences are indexed by tree
path rather than structural equality, so repeated text remains distinguishable
throughout provenance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterator


class Op(str, Enum):
    ATOM = "atom"
    TOP = "top"
    BOTTOM = "bottom"
    NOT = "not"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    IFF = "iff"
    NEXT = "next"
    WEAK_NEXT = "weak_next"
    EVENTUALLY = "eventually"
    GLOBALLY = "globally"
    UNTIL = "until"
    RELEASE = "release"
    WEAK_UNTIL = "weak_until"


_ARITY = {
    Op.ATOM: 0,
    Op.TOP: 0,
    Op.BOTTOM: 0,
    Op.NOT: 1,
    Op.NEXT: 1,
    Op.WEAK_NEXT: 1,
    Op.EVENTUALLY: 1,
    Op.GLOBALLY: 1,
    Op.AND: 2,
    Op.OR: 2,
    Op.IMPLIES: 2,
    Op.IFF: 2,
    Op.UNTIL: 2,
    Op.RELEASE: 2,
    Op.WEAK_UNTIL: 2,
}


@dataclass(frozen=True, slots=True)
class Formula:
    op: Op
    args: tuple["Formula", ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        if len(self.args) != _ARITY[self.op]:
            raise ValueError(f"{self.op.value} expects {_ARITY[self.op]} operands")
        if self.op is Op.ATOM:
            if self.name is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
                raise ValueError(f"invalid atomic proposition: {self.name!r}")
        elif self.name is not None:
            raise ValueError(f"{self.op.value} cannot carry an atom name")

    def atoms(self) -> frozenset[str]:
        if self.op is Op.ATOM:
            assert self.name is not None
            return frozenset((self.name,))
        return frozenset(atom for child in self.args for atom in child.atoms())

    def to_source(self) -> str:
        if self.op is Op.ATOM:
            assert self.name is not None
            return self.name
        if self.op is Op.TOP:
            return "top"
        if self.op is Op.BOTTOM:
            return "bottom"
        unary = {
            Op.NOT: "not",
            Op.NEXT: "X",
            Op.WEAK_NEXT: "Xw",
            Op.EVENTUALLY: "F",
            Op.GLOBALLY: "G",
        }
        if self.op in unary:
            return f"{unary[self.op]}({self.args[0].to_source()})"
        binary = {
            Op.AND: "and",
            Op.OR: "or",
            Op.IMPLIES: "->",
            Op.IFF: "<->",
            Op.UNTIL: "U",
            Op.RELEASE: "R",
            Op.WEAK_UNTIL: "W",
        }
        return f"({self.args[0].to_source()} {binary[self.op]} {self.args[1].to_source()})"

    def __str__(self) -> str:
        return self.to_source()


def Atom(name: str) -> Formula:
    return Formula(Op.ATOM, name=name)


def Top() -> Formula:
    return Formula(Op.TOP)


def Bottom() -> Formula:
    return Formula(Op.BOTTOM)


def unary(op: Op, child: Formula) -> Formula:
    return Formula(op, (child,))


def binary(op: Op, left: Formula, right: Formula) -> Formula:
    return Formula(op, (left, right))


@dataclass(frozen=True, slots=True)
class Occurrence:
    id: str
    path: tuple[int, ...]
    formula: Formula
    parent_id: str | None
    child_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IndexedFormula:
    formula: Formula
    occurrences: tuple[Occurrence, ...]

    @classmethod
    def build(cls, formula: Formula) -> "IndexedFormula":
        staged: list[tuple[str, tuple[int, ...], Formula, str | None]] = []

        stack: list[tuple[Formula, tuple[int, ...], str | None]] = [(formula, (), None)]
        while stack:
            node, path, parent = stack.pop()
            occurrence_id = "o" + ("_".join(map(str, path)) if path else "root")
            staged.append((occurrence_id, path, node, parent))
            for index in range(len(node.args) - 1, -1, -1):
                stack.append((node.args[index], path + (index,), occurrence_id))
        id_by_path = {path: occurrence_id for occurrence_id, path, _, _ in staged}
        occurrences = tuple(
            Occurrence(
                id=occurrence_id,
                path=path,
                formula=node,
                parent_id=parent,
                child_ids=tuple(id_by_path[path + (i,)] for i in range(len(node.args))),
            )
            for occurrence_id, path, node, parent in staged
        )
        return cls(formula=formula, occurrences=occurrences)

    @property
    def root(self) -> Occurrence:
        return self.occurrences[0]

    def by_id(self) -> dict[str, Occurrence]:
        return {occurrence.id: occurrence for occurrence in self.occurrences}

    def postorder(self) -> Iterator[Occurrence]:
        yield from sorted(self.occurrences, key=lambda item: (-len(item.path), item.path))


class FormulaParser:
    _token = re.compile(
        r"\s*(<->|->|[()]|!|~|&|\||\b(?:Xw|WX|X|F|G|U|R|W|and|or|not|top|true|bottom|false)\b|[A-Za-z_][A-Za-z0-9_]*)"
    )

    def __init__(self, source: str, *, max_tokens: int = 512):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.tokens = self._tokenize(source)
        if len(self.tokens) > max_tokens:
            raise SyntaxError(f"formula exceeds token limit {max_tokens}")
        self.index = 0

    @classmethod
    def _tokenize(cls, source: str) -> tuple[str, ...]:
        source = source.strip()
        if not source:
            raise SyntaxError("formula cannot be empty")
        result: list[str] = []
        position = 0
        while position < len(source):
            match = cls._token.match(source, position)
            if match is None:
                raise SyntaxError(f"unexpected token near {source[position:]!r}")
            result.append(match.group(1))
            position = match.end()
        return tuple(result)

    def parse(self) -> Formula:
        result = self._parse_iff()
        if self._peek() is not None:
            raise SyntaxError(f"unexpected trailing token {self._peek()!r}")
        return result

    def _parse_iff(self) -> Formula:
        left = self._parse_implies()
        while self._accept("<->"):
            left = binary(Op.IFF, left, self._parse_implies())
        return left

    def _parse_implies(self) -> Formula:
        left = self._parse_or()
        if self._accept("->"):
            return binary(Op.IMPLIES, left, self._parse_implies())
        return left

    def _parse_or(self) -> Formula:
        left = self._parse_and()
        while self._peek() in {"or", "|"}:
            self.index += 1
            left = binary(Op.OR, left, self._parse_and())
        return left

    def _parse_and(self) -> Formula:
        left = self._parse_temporal_binary()
        while self._peek() in {"and", "&"}:
            self.index += 1
            left = binary(Op.AND, left, self._parse_temporal_binary())
        return left

    def _parse_temporal_binary(self) -> Formula:
        left = self._parse_unary()
        operators = {"U": Op.UNTIL, "R": Op.RELEASE, "W": Op.WEAK_UNTIL}
        while self._peek() in operators:
            op = operators[self._peek()]
            self.index += 1
            left = binary(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> Formula:
        token = self._peek()
        operators = {
            "not": Op.NOT,
            "!": Op.NOT,
            "~": Op.NOT,
            "X": Op.NEXT,
            "Xw": Op.WEAK_NEXT,
            "WX": Op.WEAK_NEXT,
            "F": Op.EVENTUALLY,
            "G": Op.GLOBALLY,
        }
        if token in operators:
            self.index += 1
            return unary(operators[token], self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> Formula:
        token = self._peek()
        if token is None:
            raise SyntaxError("unexpected end of formula")
        if self._accept("("):
            result = self._parse_iff()
            self._expect(")")
            return result
        self.index += 1
        if token in {"top", "true"}:
            return Top()
        if token in {"bottom", "false"}:
            return Bottom()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            return Atom(token)
        raise SyntaxError(f"unexpected token {token!r}")

    def _peek(self) -> str | None:
        return None if self.index >= len(self.tokens) else self.tokens[self.index]

    def _accept(self, token: str) -> bool:
        if self._peek() == token:
            self.index += 1
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._accept(token):
            raise SyntaxError(f"expected {token!r}, found {self._peek()!r}")


def parse_formula(source: str, *, max_tokens: int = 512) -> Formula:
    return FormulaParser(source, max_tokens=max_tokens).parse()
