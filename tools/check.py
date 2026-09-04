#!/usr/bin/env python3
"""Structural checks across every chapter of every volume.

Run from the repository root:

    python3 tools/check.py

Checks performed, per chapter file:

  1. problem count == solution count  (the invariant stated in the README)
  2. no straight ASCII double quotes  (LaTeX needs ``...'')
  3. every \\begin{...} in the problems/solutions machinery is closed
  4. the chapter has a \\chapter line
  5. the chapter has exactly one Worked Example section, holding at least
     one worked box

It also reports the per-volume and repository totals, so the README table
can be regenerated from the source rather than remembered.
"""

import glob
import os
import re
import sys

VOLUMES = ["algebra", "functions", "matrices", "foundations",
           "derivatives", "integrals", "optimization", "linprog", "mixed"]


def count_env(src, env):
    """Count \\item at depth 1 of the given environment, ignoring nested lists."""
    n = 0
    inside = False
    nest = 0
    for line in src.splitlines():
        if re.search(r"\\begin\{%s\}" % env, line):
            inside, nest = True, 0
            continue
        if re.search(r"\\end\{%s\}" % env, line):
            inside = False
            continue
        if inside:
            if re.search(r"\\begin\{(itemize|enumerate)\}", line):
                nest += 1
            if re.search(r"\\end\{(itemize|enumerate)\}", line):
                nest -= 1
            if nest == 0 and re.match(r"\s*\\item", line):
                n += 1
    return n


def check_file(path):
    """Return (problems, problems_ok, [messages])."""
    src = open(path, encoding="utf-8").read()
    msgs = []

    p = count_env(src, "problems")
    s = count_env(src, "solutions")
    if p != s:
        msgs.append(f"problem/solution mismatch: {p} vs {s}")

    if '"' in src:
        lines = [i + 1 for i, l in enumerate(src.splitlines()) if '"' in l]
        msgs.append(f"straight quotes on line(s) {lines[:5]} — use ``...''")

    for env in ("problems", "solutions", "rulebox", "worked", "gallery",
                "tikzpicture", "axis"):
        opens = len(re.findall(r"\\begin\{%s\}" % env, src))
        closes = len(re.findall(r"\\end\{%s\}" % env, src))
        if opens != closes:
            msgs.append(f"unbalanced {env}: {opens} begin / {closes} end")

    if not re.search(r"^\\chapter\{", src, re.M):
        msgs.append("no \\chapter line")

    heads = re.findall(r"^\\section\{Worked Examples?\}$", src, re.M)
    boxes = len(re.findall(r"\\begin\{worked\}", src))
    if len(heads) != 1:
        msgs.append(f"expected 1 Worked Example section, found {len(heads)}")
    elif boxes == 0:
        msgs.append("Worked Example section contains no worked box")
    elif (boxes > 1) != heads[0].endswith("Examples}"):
        msgs.append(f"section title does not match {boxes} worked box(es)")

    return p, not msgs, msgs


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    total_problems = total_chapters = 0
    failures = 0

    for vol in VOLUMES:
        files = sorted(glob.glob(f"{vol}/tex/chapters/*.tex"))
        if not files:
            print(f"!! {vol}: no chapter files found")
            failures += 1
            continue
        n = 0
        for f in files:
            p, ok, msgs = check_file(f)
            n += p
            if not ok:
                failures += 1
                for m in msgs:
                    print(f"!! {f}: {m}")
        print(f"   {vol:<13} {len(files):>3} chapters {n:>5} problems")
        total_problems += n
        total_chapters += len(files)

    print(f"   {'TOTAL':<13} {total_chapters:>3} chapters "
          f"{total_problems:>5} problems")

    if failures:
        print(f"\nFAILED: {failures} problem(s) found")
        return 1
    print("\nAll structural checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
