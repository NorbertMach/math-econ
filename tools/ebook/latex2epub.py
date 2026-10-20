#!/usr/bin/env python3
"""Build a reflowable EPUB from one volume's LaTeX sources.

Design notes
------------
Maths is the whole problem.  Roughly 70% of the inline fragments in these
books are simple enough to become real HTML (an italic variable, a
superscript, a Greek letter, a relation symbol); those reflow and scale with
the reader's font size.  The remaining 30%, every display, and every figure
are rendered to SVG by LaTeX itself, so they are typographically identical
to the print edition but fixed in size.

The classifier is deliberately conservative: anything it does not recognise
with certainty becomes an image.  A wrong image is a slightly large image;
a wrong transcription is wrong mathematics.
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

# ----------------------------------------------------------------- maths ---

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
    "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ",
    "Phi": "Φ", "Omega": "Ω",
}
SYMBOL = {
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡",
    "to": "→", "rightarrow": "→", "mapsto": "↦",
    "Rightarrow": "⇒", "Longrightarrow": "⟹", "iff": "⟺",
    "Leftrightarrow": "⇔", "in": "∈", "notin": "∉",
    "cup": "∪", "cap": "∩", "setminus": "∖",
    "subset": "⊂", "subseteq": "⊆", "infty": "∞",
    "pm": "±", "mp": "∓", "times": "×", "cdot": "·",
    "cdots": "⋯", "ldots": "…", "dots": "…",
    "circ": "∘", "deg": "°", "partial": "∂",
    "forall": "∀", "exists": "∃", "emptyset": "∅",
    "R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ",
    "quad": " ", "qquad": "  ", "colon": ":",
    "lvert": "|", "rvert": "|", "mid": "|", "%": "%", "$": "$",
}
UPRIGHT = {"sin", "cos", "tan", "cot", "sec", "csc", "ln", "log", "exp",
           "max", "min", "lim", "sup", "inf", "arcsin", "arccos", "arctan",
           "sinh", "cosh", "tanh", "det", "dim", "ker"}

# Anything structural is never transcribed.
STRUCTURAL = re.compile(
    r"\\(?:d?frac|tfrac|sqrt|int|iint|sum|prod|lim(?![a-z])|binom|over|"
    r"begin|end|left|right|bigl|bigr|Bigl|Bigr|big|Big|"
    r"matrix|pmatrix|bmatrix|vmatrix|smallmatrix|cases|array|"
    r"underbrace|overbrace|hat|bar|vec|tilde|dot|ddot|boxed|substack|"
    r"displaystyle|nabla|grad|Hess|partial\b.*\\|text\b)")

TOKEN = re.compile(r"""
    \\(?P<cmd>[A-Za-z]+)
  | (?P<sup>\^)\{(?P<supb>[^{}]*)\}
  | (?P<sup1>\^)(?P<supc>[A-Za-z0-9])
  | (?P<sub>_)\{(?P<subb>[^{}]*)\}
  | (?P<sub1>_)(?P<subc>[A-Za-z0-9])
  | (?P<brace>[{}])
  | (?P<sp>\s+)
  | (?P<ch>.)
""", re.X)


def try_html_math(src):
    """Transcribe a simple inline fragment to HTML, or return None."""
    if STRUCTURAL.search(src) or "\\\\" in src or len(src) > 60:
        return None
    out, i = [], 0
    for m in TOKEN.finditer(src):
        kind = m.lastgroup
        if m.group("cmd") is not None:
            c = m.group("cmd")
            if c in GREEK:
                out.append(GREEK[c])
            elif c in SYMBOL:
                out.append(SYMBOL[c])
            elif c in UPRIGHT:
                out.append(f"<span class='op'>{c}</span>")
            elif c == "mathrm" or c == "mathbb":
                return None
            else:
                return None
        elif m.group("sup") or m.group("sup1"):
            body = m.group("supb") if m.group("sup") else m.group("supc")
            inner = try_html_math(body) if body else ""
            if inner is None:
                return None
            out.append(f"<sup>{inner}</sup>")
        elif m.group("sub") or m.group("sub1"):
            body = m.group("subb") if m.group("sub") else m.group("subc")
            inner = try_html_math(body) if body else ""
            if inner is None:
                return None
            out.append(f"<sub>{inner}</sub>")
        elif m.group("brace") is not None:
            continue
        elif m.group("sp") is not None:
            out.append(" ")
        else:
            ch = m.group("ch")
            if ch.isalpha():
                out.append(f"<i>{ch}</i>")
            elif ch in "+-=<>()[],;:.|/!'0123456789*":
                repl = {"-": "−", "<": "&lt;", ">": "&gt;", "*": "·"}
                out.append(repl.get(ch, ch))
            else:
                return None
    return "".join(out)


# ------------------------------------------------------- SVG rendering ----

PREAMBLE_MATH = r"""\documentclass[11pt]{article}
\usepackage[active,tightpage]{preview}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{xcolor}
\definecolor{accent}{HTML}{1F4E79}
\newcommand{\dd}[2]{\frac{\mathrm{d}#1}{\mathrm{d}#2}}
\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\pdd}[3]{\frac{\partial^{2} #1}{\partial #2\,\partial #3}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\grad}{\nabla}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\Hess}{H}
\newsavebox\fb
\newcommand\emit[1]{\savebox\fb{#1}%
  \typeout{FRAG \the\wd\fb\space \the\ht\fb\space \the\dp\fb}%
  \begin{preview}\usebox\fb\end{preview}}
\begin{document}
"""


def render_svgs(fragments, workdir, tag):
    """fragments: list of (key, latex_body, is_display). Returns {key: meta}."""
    if not fragments:
        return {}
    os.makedirs(workdir, exist_ok=True)
    src = [PREAMBLE_MATH]
    for _, body, disp in fragments:
        wrapped = f"$\\displaystyle {body}$" if disp else f"${body}$"
        src.append(f"\\emit{{{wrapped}}}\n")
    src.append("\\end{document}\n")
    tex = os.path.join(workdir, f"{tag}.tex")
    open(tex, "w").write("".join(src))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"-jobname={tag}",
                    os.path.basename(tex)],
                   cwd=workdir, capture_output=True)
    log = open(os.path.join(workdir, f"{tag}.log"), errors="ignore").read()
    dims = re.findall(r"^FRAG ([\d.]+)pt ([\d.]+)pt ([\d.]+)pt", log, re.M)
    pdf = os.path.join(workdir, f"{tag}.pdf")
    npages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(
        ["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    if npages != len(fragments) or len(dims) != len(fragments):
        raise SystemExit(f"{tag}: expected {len(fragments)} fragments, "
                         f"got {npages} pages and {len(dims)} measurements")
    outdir = os.path.join(workdir, "svg")
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", "-f", "1", "-l", "1", pdf,
                    os.path.join(outdir, "probe.svg")], capture_output=True)
    meta = {}
    for n, ((key, _, disp), (w, h, d)) in enumerate(zip(fragments, dims), 1):
        name = f"{tag}-{n:05d}.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(n), "-l", str(n), pdf,
                        os.path.join(outdir, name)], capture_output=True)
        meta[key] = {"file": name, "w": float(w), "h": float(h),
                     "d": float(d), "display": disp}
    return meta


def render_figures(figs, workdir, preamble_path):
    """figs: list of (key, tikz_source). Rendered with the book's own preamble."""
    if not figs:
        return {}
    os.makedirs(workdir, exist_ok=True)
    pre = open(preamble_path).read()
    # the standalone run needs no page furniture
    pre = pre.replace(r"\pagestyle{fancy}", "")
    body = ["\\documentclass[11pt]{article}\n",
            "\\usepackage[active,tightpage]{preview}\n", pre,
            "\\PreviewEnvironment{tikzpicture}\n\\begin{document}\n"]
    for _, tikz in figs:
        body.append(tikz + "\n")
    body.append("\\end{document}\n")
    tex = os.path.join(workdir, "figs.tex")
    open(tex, "w").write("".join(body))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-jobname=figs",
                    "figs.tex"], cwd=workdir, capture_output=True)
    pdf = os.path.join(workdir, "figs.pdf")
    if not os.path.exists(pdf):
        raise SystemExit("figure render failed; see " + workdir)
    npages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(
        ["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    if npages != len(figs):
        raise SystemExit(f"figures: expected {len(figs)} pages, got {npages}")
    outdir = os.path.join(workdir, "svg")
    os.makedirs(outdir, exist_ok=True)
    meta = {}
    for n, (key, _) in enumerate(figs, 1):
        name = f"fig-{n:04d}.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(n), "-l", str(n), pdf,
                        os.path.join(outdir, name)], capture_output=True)
        meta[key] = {"file": name}
    return meta


# --------------------------------------------------- LaTeX -> XHTML -------

_SEQ = [0]


def protect(text, store, prefix):
    """Replace a matched construct with an opaque token.

    The counter is global: keys must stay unique across chapters, or the
    per-chapter stores overwrite one another when they are merged.
    """
    _SEQ[0] += 1
    key = f"\x00{prefix}{_SEQ[0]}\x00"
    store[key] = text
    return key


def esc(s):
    return html.escape(s, quote=False)


AMP, LT, GT = "\x02A\x02", "\x02L\x02", "\x02G\x02"


def esc_keep(s):
    """Stash the three XML-significant characters until markup is finished."""
    return s.replace("&", AMP).replace("<", LT).replace(">", GT)


def unstash(s):
    return s.replace(AMP, "&amp;").replace(LT, "&lt;").replace(GT, "&gt;")


TEXT_REPL = [
    # tables and displays are already tokenised out, so \\ here is a
    # forced line break in running text
    (r"\\\\(?:\[[^\]]*\])?", "<br/>"),
    (r"\\checkmark", "✓"), (r"\\textbullet", "•"),
    (r"\\ldots", "…"), (r"\\dots", "…"),
    (r"\\%", "%"), (r"\\&", AMP), (r"\\_", "_"),
    (r"\\#", "#"), (r"\\\$", "$"),
    (r"\\,", "\u2009"), (r"\\;", " "), (r"\\ ", " "),
    (r"~", "\u00a0"),
    (r"---", "—"), (r"--", "–"),
    (r"\\@", ""), (r"\\/", ""),
]


def inline_markup(s):
    """Convert the text-level commands. Math is already tokenised out."""
    s = re.sub(r"\\(?:textbf|textsc)\{([^{}]*)\}", r"<strong>\1</strong>", s)
    s = re.sub(r"\\(?:emph|textit|de)\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    for pat, rep in TEXT_REPL:
        s = re.sub(pat, rep, s)
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = re.sub(r"(?<![\w\\])`", "\u2018", s)
    s = re.sub(r"(?<![\w\\])'", "\u2019", s)
    s = re.sub(r"\\[a-zA-Z]+\b\s*", "", s)          # drop stragglers
    s = s.replace("{", "").replace("}", "")
    return s


def env_blocks(src, name):
    """Yield (start, end, argument, body) for each \\begin{name}[{arg}] ..."""
    out, i = [], 0
    pat = re.compile(r"\\begin\{" + name + r"\}")
    endpat = re.compile(r"\\end\{" + name + r"\}")
    while True:
        m = pat.search(src, i)
        if not m:
            return out
        depth, j = 1, m.end()
        while depth:
            nm, ne = pat.search(src, j), endpat.search(src, j)
            if ne is None:
                return out
            if nm and nm.start() < ne.start():
                depth += 1
                j = nm.end()
            else:
                depth -= 1
                j = ne.end()
        out.append((m.start(), j, m.end()))
        i = j



def split_inline_math(src, take):
    """Replace every top-level $...$ using take(body).

    A regex cannot do this: the corpus contains $...\\text{... $x$ ...}...$,
    where the inner dollars are nested inside braces and must be ignored.
    """
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == "\\" and i + 1 < n:
            out.append(src[i:i + 2]); i += 2; continue
        if c == "$":
            j, depth = i + 1, 0
            while j < n:
                d = src[j]
                if d == "\\" and j + 1 < n:
                    j += 2; continue
                if d == "{":
                    depth += 1
                elif d == "}":
                    depth -= 1
                elif d == "$" and depth == 0:
                    break
                j += 1
            out.append(take(src[i + 1:j]))
            i = j + 1
            continue
        out.append(c); i += 1
    return "".join(out)


def balanced(body):
    d, i, n = 0, 0, len(body)
    while i < n:
        if body[i] == "\\" and i + 1 < n:
            i += 2; continue
        if body[i] == "{":
            d += 1
        elif body[i] == "}":
            d -= 1
        i += 1
    return d == 0


def braced(src, pos):
    """src[pos] must be '{'. Return (content, index_after_closing_brace)."""
    assert src[pos] == "{", src[pos:pos + 20]
    depth, i = 0, pos
    while i < len(src):
        if src[i] == "{" and (i == pos or src[i - 1] != "\\"):
            depth += 1
        elif src[i] == "}" and src[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return src[pos + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced brace at %d" % pos)


def cmd_with_arg(src, name, render):
    """Replace \\name{...} (balanced) using render(content)."""
    out, i = [], 0
    needle = "\\" + name + "{"
    while True:
        j = src.find(needle, i)
        if j < 0:
            out.append(src[i:])
            return "".join(out)
        out.append(src[i:j])
        content, k = braced(src, j + len(needle) - 1)
        out.append(render(content))
        i = k


def tabular_to_html(spec, body):
    body = re.sub(r"\\hline", "", body)
    body = re.sub(r"\\\\\s*\[[^\]]*\]", "@@BR@@", body)
    rows = [r for r in re.split(r"@@BR@@|\\\\", body) if r.strip()]
    cells = [[cellmarkup(c.strip()) for c in re.split(r"(?<!\\)&", r)]
             for r in rows]
    head = ""
    if cells and any("<strong>" in c for c in cells[0]):
        head = "<tr>" + "".join(f"<th>{c}</th>" for c in cells[0]) + "</tr>"
        cells = cells[1:]
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                  for r in cells)
    return f"<table>{head}{trs}</table>"


def cellmarkup(c):
    return inline_markup(esc_keep(c))


def chapter_refs(src, vol):
    """Expand \\chref, \\chrefs, \\chnum and \\volref as the volume edition does.

    The volume numbers come from the preamble's \\@namedef{chref@vol@...}
    table, so this stays in step with the PDF without a copy of its own.
    """
    pre = open(f"{vol}/tex/preamble.tex").read()
    num = dict(re.findall(r"\\@namedef\{chref@vol@(\w+)\}\{(\d+)\}", pre))

    def ref(m):
        cmd, v, ch = m.group(1), m.group(2), m.group(3)
        if v not in num:
            raise SystemExit(f"unknown volume in \\{cmd}{{{v}}}")
        if cmd == "volref":
            return f"Volume~{num[v]}"
        if cmd == "chnum":
            return ch
        word, vword = ("Ch.", "Vol.") if cmd == "chrefs" else ("Chapter", "Volume")
        return (f"{word}~{ch}" if v == vol
                else f"{vword}~{num[v]}, {word}~{ch}")

    src = re.sub(r"\\(chrefs?|chnum)\{(\w+)\}\{(\d+)\}", ref, src)
    return re.sub(r"\\(volref)\{(\w+)\}()", ref, src)


def convert_chapter(path, math_store, fig_store, counters):
    src = open(path).read()
    src = re.sub(r"(?<!\\)%.*", "", src)                 # comments
    src = chapter_refs(src, path.split("/")[0])

    figs, disps, inls = {}, {}, {}

    # --- figures -----------------------------------------------------
    for a, b, _ in reversed(env_blocks(src, "tikzpicture")):
        key = protect(src[a:b], figs, "F")
        src = src[:a] + key + src[b:]
    # --- display maths -----------------------------------------------
    def take_display(m):
        return protect(m.group(1), disps, "D")
    # (?<!\\) matters: a table row ending \\\\[0.8ex] contains the substring
    # \\[ and would otherwise start a spurious display, swallowing the table.
    src = re.sub(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", take_display,
                 src, flags=re.S)
    # --- inline maths -------------------------------------------------
    src = split_inline_math(src, lambda body: protect(body, inls, "M"))

    # register fragments for rendering
    for k, v in figs.items():
        fig_store[k] = v
    for k, v in disps.items():
        math_store[k] = (v.strip(), True)
    for k, v in inls.items():
        body = v.strip()
        if try_html_math(body) is None:
            math_store[k] = (body, False)

    # --- structure ----------------------------------------------------
    title = re.search(r"\\chapter\{", src)
    chap_title, after = braced(src, title.end() - 1)
    src = src[after:]

    src = cmd_with_arg(src, "section", lambda c: f"\n\x01H2\x01{c}\x01\n")
    src = cmd_with_arg(src, "wtrap",
                       lambda c: f"\n\x01TRAP\x01{c}\x01\n")
    src = cmd_with_arg(src, "wstep", lambda c: f"\n\n\x01STEP\x01{c}\x01 ")
    src = cmd_with_arg(src, "gcap", lambda c: f"\n\x01CAP\x01{c}\x01\n")

    for env, cls in (("rulebox", "rulebox"), ("worked", "worked")):
        while True:
            blocks = env_blocks(src, env)
            if not blocks:
                break
            a, b, argstart = blocks[0]
            arg, bodystart = braced(src, argstart)
            body = src[bodystart:b - len(f"\\end{{{env}}}")]
            _SEQ[0] += 1
            bid = _SEQ[0]
            src = (src[:a]
                   + f"\n\x01BOX{cls}:{bid}\x01{arg}\x01{body}\x01ENDBOX{bid}\x01\n"
                   + src[b:])

    for env, tag, cls in (("problems", "ol", "problems"),
                          ("solutions", "ol", "solutions"),
                          ("itemize", "ul", "bullets"),
                          ("enumerate", "ol", "plain")):
        while True:
            blocks = env_blocks(src, env)
            if not blocks:
                break
            a, b, argend = blocks[0]
            body = src[argend:b - len(f"\\end{{{env}}}")]
            body = re.sub(r"^\s*\[resume\]", "", body)
            items = [x.strip() for x in re.split(r"\\item\b", body)[1:]]
            _SEQ[0] += 1
            lid = _SEQ[0]
            lis = "".join(f"\x01LI{lid}\x01{x}" for x in items)
            src = (src[:a]
                   + f"\n\x01LIST{tag}:{cls}:{lid}\x01{lis}\x01ENDLIST{lid}\x01\n"
                   + src[b:])

    while True:
        blocks = env_blocks(src, "tabular")
        if not blocks:
            break
        a, b, argend = blocks[0]
        spec, bodystart = braced(src, src.index("{", argend))
        body = src[bodystart:b - len("\\end{tabular}")]
        src = src[:a] + "\x01TAB\x01" + tabular_to_html(spec, body) + "\x01\n" + src[b:]

    src = re.sub(r"\\begin\{center\}", "\n\x01CTR\x01\n", src)
    src = re.sub(r"\\end\{center\}", "\n\x01ENDCTR\x01\n", src)
    src = re.sub(r"\\(?:begin|end)\{gallery\}", "\n", src)
    src = re.sub(r"\\hfill", " ", src)

    return chap_title, src, figs, disps, inls


# ------------------------------------------------------------- emitter ----

def emit(text, mathmeta, figmeta, inls, disps, figs):
    """Turn the marker stream into XHTML."""

    EM = 10.95   # \documentclass[11pt] sets the base font to 10.95pt

    def mathspan(key):
        body = inls.get(key)
        if body is not None:
            simple = try_html_math(body.strip())
            if simple is not None:
                return f"<span class='m'>{simple}</span>"
        meta = mathmeta.get(key)
        if not meta:
            return ""
        w = meta["w"] / EM
        va = -meta["d"] / EM
        return (f"<img class='mi' src='img/{meta['file']}' alt='math' "
                f"style='width:{w:.3f}em;max-width:100%;"
                f"vertical-align:{va:.3f}em'/>")

    def dispblock(key):
        meta = mathmeta.get(key)
        if not meta:
            return ""
        w = meta["w"] / EM
        return (f"<div class='disp'><img src='img/{meta['file']}' alt='equation' "
                f"style='width:{w:.3f}em;max-width:100%'/></div>")

    def figblock(key):
        meta = figmeta.get(key)
        if not meta:
            return ""
        return (f"<div class='fig'><img src='img/{meta['file']}' "
                f"alt='figure'/></div>")

    def tokens(s):
        s = re.sub(r"\x00F\d+\x00", lambda m: figblock(m.group(0)), s)
        s = re.sub(r"\x00D\d+\x00", lambda m: dispblock(m.group(0)), s)
        s = re.sub(r"\x00M\d+\x00", lambda m: mathspan(m.group(0)), s)
        return s

    def paras(s):
        """Split a plain run into paragraphs, converting text markup."""
        chunks = re.split(r"\n\s*\n", s)
        out = []
        for c in chunks:
            c = c.strip()
            if not c:
                continue
            c = tokens(inline_markup(esc_keep(c)))
            # [ \t\n]+, not \s+: \s also matches the no-break space from ~
            c = re.sub(r"[ \t\n]+", " ", c).strip()
            if c and c != "<br/>":
                out.append(f"<p>{c}</p>")
        return "".join(out)

    def walk(s):
        out = []
        while s:
            m = re.search(
                r"\x01(H2|STEP|TRAP|CAP|TAB|CTR|ENDCTR"
                r"|BOX[A-Za-z]+:\d+|LIST[A-Za-z]+:[A-Za-z]+:\d+)\x01", s)
            if not m:
                out.append(paras(s))
                break
            out.append(paras(s[:m.start()]))
            kind = m.group(1)
            rest = s[m.end():]
            if kind == "H2":
                body, rest = rest.split("\x01", 1)
                out.append(f"<h2>{tokens(inline_markup(esc_keep(body)))}</h2>")
            elif kind == "STEP":
                body, rest = rest.split("\x01", 1)
                out.append(f"<p class='step'><strong>"
                           f"{tokens(inline_markup(esc_keep(body)))}</strong>")
                # the following text belongs to the same paragraph
                nxt = re.search(r"\x01|\x00[DF]\d+\x00|\n\s*\n", rest)
                cut = nxt.start() if nxt else len(rest)
                out.append(tokens(inline_markup(esc_keep(rest[:cut]))) + "</p>")
                rest = rest[cut:]
            elif kind == "TRAP":
                body, rest = rest.split("\x01", 1)
                out.append(f"<div class='trap'><strong>Trap.</strong> <em>"
                           f"{tokens(inline_markup(esc_keep(body)))}</em></div>")
            elif kind == "CAP":
                body, rest = rest.split("\x01", 1)
                out.append(f"<p class='cap'>{tokens(inline_markup(esc_keep(body)))}</p>")
            elif kind == "TAB":
                body, rest = rest.split("\x01", 1)
                out.append("<div class='tabwrap'>" + tokens(body) + "</div>")
            elif kind == "CTR":
                out.append("<div class='center'>")
                rest = rest
            elif kind == "ENDCTR":
                out.append("</div>")
            elif kind.startswith("BOX"):
                cls, bid = kind[3:].split(":")
                arg, rest = rest.split("\x01", 1)
                body, rest = rest.split(f"\x01ENDBOX{bid}\x01", 1)
                out.append(f"<div class='{cls}'><h3>"
                           f"{tokens(inline_markup(esc_keep(arg)))}</h3>"
                           f"{walk(body)}</div>")
            elif kind.startswith("LIST"):
                tag, cls, lid = kind[4:].split(":")
                body, rest = rest.split(f"\x01ENDLIST{lid}\x01", 1)
                items = body.split(f"\x01LI{lid}\x01")[1:]
                lis = "".join(f"<li>{walk(it)}</li>" for it in items)
                out.append(f"<{tag} class='{cls}'>{lis}</{tag}>")
            s = rest
        return "".join(out)

    return unstash(walk(text))


# --------------------------------------------------------------- EPUB -----

CSS = """
html { -webkit-hyphens:auto; hyphens:auto; }
body { margin:0.4em 0.5em; line-height:1.35; text-align:justify; }
h1 { font-size:1.35em; margin:0.6em 0 0.5em; line-height:1.2; }
h2 { font-size:1.12em; margin:1.1em 0 0.4em; }
h3 { font-size:1em; margin:0 0 0.35em; color:#1F4E79; }
p  { margin:0.42em 0; text-indent:0; }
img.mi { vertical-align:baseline; }
.disp { text-align:center; margin:0.6em 0; }
.disp img { max-width:100%; height:auto; }
.fig { text-align:center; margin:0.7em 0; }
.fig img { max-width:100%; height:auto; }
.cap { font-size:0.85em; text-align:center; color:#444; }
.rulebox { border:1px solid #1F4E79; background:#F4F7FA;
           padding:0.45em 0.55em; margin:0.7em 0; border-radius:3px; }
.worked  { border:1px solid #7d9dc0; padding:0.45em 0.55em;
           margin:0.7em 0; border-radius:3px; }
.trap { margin:0.45em 0 0.45em 1em; font-size:0.95em; }
.step { margin-top:0.55em; }
ol.problems, ol.solutions { padding-left:1.4em; }
ol.problems > li, ol.solutions > li { margin:0.4em 0; }
ul.bullets { padding-left:1.2em; }
table { border-collapse:collapse; margin:0.4em auto; font-size:0.9em; }
th, td { padding:0.15em 0.4em; text-align:left; }
th { border-bottom:1px solid #888; }
.tabwrap { overflow-x:auto; }
.center { text-align:center; }
.m { white-space:nowrap; }
.op { font-style:normal; }
"""

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""


def build_epub(out_path, title, chapters, imgdir, images):
    opf_items, spine, nav = [], [], []
    for i, (ctitle, _) in enumerate(chapters, 1):
        opf_items.append(f'<item id="c{i}" href="ch{i:02d}.xhtml" '
                         f'media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{i}"/>')
        nav.append(f'<li><a href="ch{i:02d}.xhtml">{esc(ctitle)}</a></li>')
    for n, img in enumerate(sorted(images)):
        opf_items.append(f'<item id="i{n}" href="img/{img}" '
                         f'media-type="image/svg+xml"/>')
    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bid">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:identifier id="bid">urn:uuid:math-econ-{re.sub(r'[^a-z0-9]+','-',title.lower())}</dc:identifier>
  <dc:title>{esc(title)}</dc:title>
  <dc:language>en</dc:language>
  <dc:creator>Norbert Mach</dc:creator>
  <meta property="dcterms:modified">2026-09-06T00:00:00Z</meta>
 </metadata>
 <manifest>
  <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
  <item id="css" href="style.css" media-type="text/css"/>
  {"".join(opf_items)}
 </manifest>
 <spine>{"".join(spine)}</spine>
</package>
"""
    navdoc = f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title><meta charset="utf-8"/></head>
<body><nav epub:type="toc"><h1>Contents</h1><ol>{"".join(nav)}</ol></nav></body></html>
"""
    with zipfile.ZipFile(out_path, "w") as z:
        z.writestr("mimetype", "application/epub+zip",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/content.opf", opf, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/nav.xhtml", navdoc, zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/style.css", CSS, zipfile.ZIP_DEFLATED)
        for i, (ctitle, body) in enumerate(chapters, 1):
            page = f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><meta charset="utf-8"/>
<title>{esc(ctitle)}</title><link rel="stylesheet" href="style.css"/></head>
<body><h1>{esc(ctitle)}</h1>{body}</body></html>
"""
            z.writestr(f"OEBPS/ch{i:02d}.xhtml", page, zipfile.ZIP_DEFLATED)
        for img in sorted(images):
            z.write(os.path.join(imgdir, img), f"OEBPS/img/{img}",
                    zipfile.ZIP_DEFLATED)


def main():
    vol = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    work = os.path.join("/tmp", f"epub-{vol}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    chapfiles = sorted(__import__("glob").glob(f"{vol}/tex/chapters/*.tex"))
    math_store, fig_store, parsed = {}, {}, []
    for p in chapfiles:
        t, body, figs, disps, inls = convert_chapter(p, math_store, fig_store, None)
        parsed.append((t, body, figs, disps, inls))

    for k, (body, _) in math_store.items():
        if not balanced(body):
            raise SystemExit("unbalanced braces in a maths fragment:\n  "
                             + body[:200])
    frags = [(k, v[0], v[1]) for k, v in math_store.items()]
    print(f"{vol}: {len(chapfiles)} chapters, {len(frags)} rendered fragments, "
          f"{len(fig_store)} figures")
    mathmeta = render_svgs(frags, work, "math")
    figmeta = render_figures(list(fig_store.items()), work,
                             f"{vol}/tex/preamble.tex")

    imgdir = os.path.join(work, "svg")
    chapters = []
    for (t, body, figs, disps, inls) in parsed:
        chapters.append((t, emit(body, mathmeta, figmeta, inls, disps, figs)))
    images = sorted(os.listdir(imgdir))
    outdir = os.path.join(root, "ebook")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{vol}.epub")
    build_epub(out, vol.capitalize(), chapters, imgdir, images)
    print("wrote", out, f"({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
