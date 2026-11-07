"""Lines in, numbered units out. No format-specific parsing: each non-empty line is a unit."""

from dataclasses import dataclass


def ntok(s: str) -> int:
    """Cheap token estimate (~4 chars/token)."""
    return max(1, (len(s) + 3) // 4)


@dataclass
class Unit:
    id: int
    text: str

    def render(self) -> str:
        return f"u{self.id} {self.text}"


def units(lines):
    """Yield a Unit for every non-empty line, as lines arrive."""
    n = 0
    for line in lines:
        text = line.strip()
        if text:
            n += 1
            yield Unit(n, text)
