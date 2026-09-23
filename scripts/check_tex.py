r"""Structural check on the paper source, for a machine with no TeX installed.

This does not replace a compile -- it cannot catch a bad macro or a package
clash. It catches the failures that actually happen while a paper is being
written: an environment left open, a \ref to a label that was renamed, a \cite
whose key never made it into the bib, a figure referenced before it was copied
into figures/. Those are the ones that surface as a wall of LaTeX errors on
Overleaf twenty minutes before a deadline.

    python scripts/check_tex.py paper
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path


def check(root: Path) -> list[str]:
    files = [root / "main.tex"] + sorted((root / "sections").glob("*.tex"))
    text = {f: f.read_text() for f in files if f.exists()}
    if not text:
        return [f"no .tex files under {root}"]
    joined = "\n".join(text.values())
    problems: list[str] = []

    for f, s in text.items():
        stack: list[tuple[str, int]] = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
            kind, env = m.groups()
            line = s[: m.start()].count("\n") + 1
            if kind == "begin":
                stack.append((env, line))
            elif not stack:
                problems.append(f"{f}:{line} \\end{{{env}}} with nothing open")
            elif stack[-1][0] != env:
                problems.append(f"{f}:{line} \\end{{{env}}} closes "
                                f"\\begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
            else:
                stack.pop()
        problems += [f"{f}:{line} \\begin{{{env}}} never closed" for env, line in stack]

        # Escaped braces are literal characters, and a comment can legitimately
        # leave one unmatched, so strip both before counting.
        depth = 0
        for ch in re.sub(r"\\[{}]", "", re.sub(r"(?<!\\)%.*", "", s)):
            depth += (ch == "{") - (ch == "}")
            if depth < 0:
                problems.append(f"{f}: closing brace with nothing open")
                break
        if depth > 0:
            problems.append(f"{f}: {depth} unclosed brace(s)")

    labels = re.findall(r"\\label\{([^}]+)\}", joined)
    for ref in sorted(set(re.findall(r"\\ref\{([^}]+)\}", joined)) - set(labels)):
        problems.append(f"\\ref{{{ref}}} has no \\label")
    problems += [f"duplicate \\label{{{k}}}"
                 for k, n in collections.Counter(labels).items() if n > 1]

    bib = root / "references.bib"
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text())) if bib.exists() else set()
    cited = {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", joined)
             for k in g.split(",")}
    problems += [f"\\cite{{{c}}} not in references.bib" for c in sorted(cited - keys)]

    for graphic in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", joined):
        if not (root / graphic).exists():
            problems.append(f"missing graphic: {graphic}")
    for inp in re.findall(r"\\input\{([^}]+)\}", joined):
        if not (root / (inp + ".tex")).exists() and not (root / inp).exists():
            problems.append(f"missing \\input: {inp}")

    words = len(re.findall(r"[A-Za-z][A-Za-z-]+", joined))
    print(f"{len(text)} files | {len(set(labels))} labels | {len(cited)} of "
          f"{len(keys)} bib entries cited | ~{words:,} words")
    unused = sorted(keys - cited)
    if unused:
        print(f"uncited bib entries: {', '.join(unused)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=Path("paper"))
    args = ap.parse_args()
    problems = check(args.root)
    print("\n".join(problems) if problems else "no structural problems found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
