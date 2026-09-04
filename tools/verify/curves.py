"""Re-check the curve discussions (derivatives ch08) and the trig chapter
(functions ch05): critical points, extrema values, inflections, asymptotes,
exact trigonometric values and equation solutions."""
from sympy import *

x = symbols("x", real=True)


def discuss(label, f, domain=S.Reals):
    f1, f2 = diff(f, x), diff(f, x, 2)
    crit = solveset(Eq(f1, 0), x, domain)
    infl = solveset(Eq(f2, 0), x, domain)
    out = []
    if isinstance(crit, FiniteSet):
        for c in sorted(crit, key=lambda v: float(v)):
            s2 = f2.subs(x, c)
            kind = ("min" if s2 > 0 else "max" if s2 < 0 else "test inconclusive")
            out.append(f"{kind} at x={c}, f={simplify(f.subs(x, c))}")
    else:
        out.append(f"critical set {crit}")
    ipts = []
    if isinstance(infl, FiniteSet):
        for c in sorted(infl, key=lambda v: float(v)):
            ipts.append(f"x={c} (f={simplify(f.subs(x, c))})")
    print(f"{label}\n    extrema: {'; '.join(out)}")
    print(f"    f''=0 at: {'; '.join(ipts) if ipts else infl}")


print("=== curve discussions, derivatives ch08 ===")
discuss("21 x^3-3x^2", x**3 - 3*x**2)
discuss("22 x^4-4x^2", x**4 - 4*x**2)
discuss("23 x/(x^2+1)", x / (x**2 + 1))
discuss("24 x^2/(x-1)", x**2 / (x - 1))
discuss("25 e^(-x^2)", exp(-x**2))
discuss("26 x e^(-x)", x * exp(-x))
discuss("27 ln x / x", log(x) / x, Interval.open(0, oo))
discuss("29 x^2 e^(-x)", x**2 * exp(-x))
discuss("31 sqrt(x)-x", sqrt(x) - x, Interval.open(0, oo))
discuss("32 x^3-3x+2", x**3 - 3*x + 2)
discuss("S1 x^3-12x", x**3 - 12*x)
discuss("S2 x^4-8x^2", x**4 - 8*x**2)
discuss("S5 x-ln x", x - log(x), Interval.open(0, oo))
discuss("A1 2x^3-9x^2+12x", 2*x**3 - 9*x**2 + 12*x)
discuss("A10 x^3-6x^2+9x", x**3 - 6*x**2 + 9*x)
discuss("A12 (x-2)e^(-x)", (x - 2) * exp(-x))
discuss("A13 x^2 ln x", x**2 * log(x), Interval.open(0, oo))
discuss("A14 4x/(x^2+4)", 4*x / (x**2 + 4))
discuss("A19 x^4-4x^3", x**4 - 4*x**3)
discuss("A20 x+1/x", x + 1/x, Union(Interval.open(-oo, 0), Interval.open(0, oo)))

print("\n=== oblique asymptotes ===")
for lab, f, claim in [("x^2/(x-1)", x**2/(x - 1), x + 1),
                      ("x^3/(x^2+1)", x**3/(x**2 + 1), x),
                      ("(2x^2-3)/(x+1)", (2*x**2 - 3)/(x + 1), 2*x - 2),
                      ("x e^(1/x)", x*exp(1/x), x + 1)]:
    d = limit(f - claim, x, oo)
    print(f"  {'ok ' if d == 0 else 'BAD'} {lab}: f-({claim}) -> {d}")

print("\n=== economics, derivatives ch08 ===")
q = symbols("q", positive=True)
C = q**3 - 6*q**2 + 15*q
print("  C''=0 at q =", solveset(Eq(diff(C, q, 2), 0), q, Interval.open(0, oo)),
      "; MC min at", minimum(diff(C, q), q, Interval.open(0, oo)))
AC = simplify(C/q)
print("  AC min at q =", solveset(Eq(diff(AC, q), 0), q, Interval.open(0, oo)),
      "value", AC.subs(q, 3), "; MC(3) =", diff(C, q).subs(q, 3))
profit = 15*q - C
print("  profit max at q =", solveset(Eq(diff(profit, q), 0), q, Interval.open(0, oo)),
      "value", profit.subs(q, 4))
C2 = Rational(1, 2)*q**2 + 2*q + 8
print("  A: AC min at q =", solveset(Eq(diff(C2/q, q), 0), q, Interval.open(0, oo)),
      "AC(4) =", (C2/q).subs(q, 4), "MC(4) =", diff(C2, q).subs(q, 4))
C3 = q**3 - 9*q**2 + 30*q
print("  A21: infl", solveset(Eq(diff(C3, q, 2), 0), q, Interval.open(0, oo)),
      "AC min", solveset(Eq(diff(C3/q, q), 0), q, Interval.open(0, oo)),
      "AC(4.5)", simplify((C3/q).subs(q, Rational(9, 2))),
      "MC(4.5)", diff(C3, q).subs(q, Rational(9, 2)))
Q = 12*symbols("L", positive=True)**2 - symbols("L", positive=True)**3
L = symbols("L", positive=True)
Q = 12*L**2 - L**3
print("  A25: AP max at L =", solveset(Eq(diff(Q/L, L), 0), L, Interval.open(0, 12)),
      "AP(6) =", (Q/L).subs(L, 6), "MP(6) =", diff(Q, L).subs(L, 6))

print("\n=== trigonometry, functions ch05 ===")
vals = [("sin(pi/6)", sin(pi/6), Rational(1, 2)),
        ("cos(pi/4)", cos(pi/4), sqrt(2)/2),
        ("tan(pi/3)", tan(pi/3), sqrt(3)),
        ("sin(5pi/6)", sin(5*pi/6), Rational(1, 2)),
        ("cos(2pi/3)", cos(2*pi/3), Rational(-1, 2)),
        ("tan(pi)", tan(pi), 0),
        ("sin(-pi/3)", sin(-pi/3), -sqrt(3)/2),
        ("cos(7pi/2)", cos(7*pi/2), 0),
        ("sin(9pi/4)", sin(9*pi/4), sqrt(2)/2),
        ("tan(3pi/4)", tan(3*pi/4), -1),
        ("cos(5pi/4)", cos(5*pi/4), -sqrt(2)/2),
        ("sin(4pi/3)", sin(4*pi/3), -sqrt(3)/2),
        ("tan(5pi/6)", tan(5*pi/6), -1/sqrt(3)),
        ("sin(11pi/6)", sin(11*pi/6), Rational(-1, 2)),
        ("cos(-pi/6)", cos(-pi/6), sqrt(3)/2),
        ("tan(7pi/4)", tan(7*pi/4), -1)]
bad = 0
for lab, got, claim in vals:
    ok = simplify(got - claim) == 0
    bad += 0 if ok else 1
    print(f"  {'ok ' if ok else 'BAD'} {lab} = {got}  (claimed {claim})")

print("\n  equation solution sets (principal families):")
for lab, eq, dom in [("sin x = 1/2", Eq(sin(x), Rational(1, 2)), Interval(0, 2*pi)),
                     ("cos x = 0", Eq(cos(x), 0), Interval(0, 2*pi)),
                     ("tan x = 1", Eq(tan(x), 1), Interval(0, pi)),
                     ("2cos x+1=0", Eq(2*cos(x) + 1, 0), Interval(0, 2*pi)),
                     ("sin 2x = 1", Eq(sin(2*x), 1), Interval(0, 2*pi)),
                     ("cos^2 x = 1/4", Eq(cos(x)**2, Rational(1, 4)), Interval(0, 2*pi)),
                     ("sin x = cos x", Eq(sin(x), cos(x)), Interval(0, 2*pi)),
                     ("sin x = -sqrt3/2", Eq(sin(x), -sqrt(3)/2), Interval(0, 2*pi)),
                     ("tan 2x = sqrt3", Eq(tan(2*x), sqrt(3)), Interval(0, pi)),
                     ("2sin^2x - sin x = 0", Eq(2*sin(x)**2 - sin(x), 0), Interval(0, 2*pi)),
                     ("cos 2x = 1/2", Eq(cos(2*x), Rational(1, 2)), Interval(0, 2*pi))]:
    print(f"    {lab:<22} {solveset(eq, x, dom)}")
print(f"\n{'trig values all agree' if bad == 0 else str(bad) + ' TRIG MISMATCHES'}")
