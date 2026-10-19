"""Re-check the shifted-conic material added to Volume 2 (functions):
ch02 rule box example, second worked example, problems 46-57 and the figure
parametrisations; ch04 problems 46-52 (regions bounded by shifted ellipses
and hyperbolas). Every claim is retyped by hand; no LaTeX parsing."""
from sympy import (symbols, Rational, sqrt, expand, simplify, solve, cosh,
                   sinh, cos, sin, factor, Eq, S, hessian,
                   discriminant, Poly)

x, y, t = symbols("x y t", real=True)
results = []


def check(label, cond):
    ok = bool(cond)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


def same(label, a, b):
    check(label, simplify(expand(a - b)) == 0)


def on(expr, pt):
    return simplify(expr.subs({x: pt[0], y: pt[1]})) == 0


def val(expr, pt):
    return simplify(expr.subs({x: pt[0], y: pt[1]}))


def conic_check(label, original, k, std_lhs, centre, ab, kind,
                vertices=(), asymptotes=()):
    """original = 0 must equal k*(std_lhs - 1) = 0, i.e. original == k*(std_lhs-1)."""
    same(f"{label}: original == {k}*(standard form - 1)", original,
         k * (std_lhs - 1))
    u, v = centre
    a, b = S(ab[0]), S(ab[1])
    if kind == "ellipse":
        same(f"{label}: standard form shape", std_lhs,
             (x - u)**2 / a**2 + (y - v)**2 / b**2)
    elif kind == "hyp_lr":
        same(f"{label}: standard form shape", std_lhs,
             (x - u)**2 / a**2 - (y - v)**2 / b**2)
    elif kind == "hyp_ud":
        same(f"{label}: standard form shape", std_lhs,
             -(x - u)**2 / a**2 + (y - v)**2 / b**2)
    for p in vertices:
        check(f"{label}: vertex {p} on original curve", on(original, p))
    for line in asymptotes:
        # asymptote y = line(x) must equal v +- (b/a)(x-u)
        cands = [v + b / a * (x - u), v - b / a * (x - u)]
        check(f"{label}: asymptote y={line}",
              any(simplify(line - c) == 0 for c in cands))


print("=== ch02 rule box example ===")
E = 4*x**2 + 9*y**2 - 16*x + 18*y - 11
same("4(x-2)^2+9(y+1)^2-36", E, 4*(x - 2)**2 + 9*(y + 1)**2 - 36)
same("std form (x-2)^2/9+(y+1)^2/4", E, 36*((x - 2)**2/9 + (y + 1)**2/4 - 1))
same("intermediate 4(x-2)^2-16+9(y+1)^2-9-11", E,
     4*(x - 2)**2 - 16 + 9*(y + 1)**2 - 9 - 11)

print("\n=== ch02 figure parametrisations ===")
same("fig ellipse", ((2 + 3*cos(t)) - 2)**2/9 + ((-1 + 2*sin(t)) + 1)**2/4, 1)
same("fig hyp L/R right", ((-1 + 2*cosh(t)) + 1)**2/4 - ((1 + 3*sinh(t)) - 1)**2/9, 1)
same("fig hyp L/R left", ((-1 - 2*cosh(t)) + 1)**2/4 - ((1 + 3*sinh(t)) - 1)**2/9, 1)
same("fig hyp U/D upper", -((1 + 2*sinh(t)) - 1)**2/4 + ((-1 + cosh(t)) + 1)**2, 1)
same("fig hyp U/D lower", -((1 + 2*sinh(t)) - 1)**2/4 + ((-1 - cosh(t)) + 1)**2, 1)
for (a, b, u, v, box) in [(2, 3, -1, 1, [(-3, -2), (1, -2), (1, 4), (-3, 4)]),
                          (2, 1, 1, -1, [(-1, -2), (3, -2), (3, 0), (-1, 0)])]:
    corners = {(u + sa*a, v + sb*b) for sa in (1, -1) for sb in (1, -1)}
    check(f"fig box corners centre ({u},{v})", corners == set(box))

print("\n=== ch02 worked example 2 ===")
W = 9*x**2 - 4*y**2 + 18*x + 8*y - 31
same("factor step 9(x^2+2x)-4(y^2-2y)-31", W, 9*(x**2 + 2*x) - 4*(y**2 - 2*y) - 31)
same("complete step 9(x+1)^2-9-4(y-1)^2+4-31", W,
     9*(x + 1)**2 - 9 - 4*(y - 1)**2 + 4 - 31)
check("31+9-4 = 36", 31 + 9 - 4 == 36)
conic_check("worked", W, 36, (x + 1)**2/4 - (y - 1)**2/9, (-1, 1), (2, 3),
            "hyp_lr", vertices=[(-3, 1), (1, 1)],
            asymptotes=[Rational(3, 2)*x + Rational(5, 2),
                        -Rational(3, 2)*x - Rational(1, 2)])
check("worked: vertex check 9-4+18+8-31=0", 9 - 4 + 18 + 8 - 31 == 0)

print("\n=== ch02 problems 46-57 ===")
conic_check("46", x**2 + 4*y**2 - 6*x + 16*y + 9, 16,
            (x - 3)**2/16 + (y + 2)**2/4, (3, -2), (4, 2), "ellipse",
            vertices=[(-1, -2), (7, -2), (3, 0), (3, -4)])
check("46: -9+9+16=16", -9 + 9 + 16 == 16)
conic_check("47", 9*x**2 + 4*y**2 + 36*x - 24*y + 36, 36,
            (x + 2)**2/4 + (y - 3)**2/9, (-2, 3), (2, 3), "ellipse",
            vertices=[(-4, 3), (0, 3), (-2, 0), (-2, 6)])
conic_check("48", 4*x**2 - 9*y**2 - 16*x - 18*y - 29, 36,
            (x - 2)**2/9 - (y + 1)**2/4, (2, -1), (3, 2), "hyp_lr",
            vertices=[(-1, -1), (5, -1)],
            asymptotes=[Rational(2, 3)*x - Rational(7, 3),
                        -Rational(2, 3)*x + Rational(1, 3)])
check("48: 29+16-9=36", 29 + 16 - 9 == 36)
conic_check("49", 4*y**2 - x**2 + 2*x + 24*y + 19, 16,
            -(x - 1)**2/16 + (y + 3)**2/4, (1, -3), (4, 2), "hyp_ud",
            vertices=[(1, -1), (1, -5)],
            asymptotes=[x/2 - Rational(7, 2), -x/2 - Rational(5, 2)])
check("49: -19-1+36=16", -19 - 1 + 36 == 16)
P50 = 16*x**2 + 16*y**2 + 32*x - 24*y - 75
same("50: circle centre (-1,3/4) r^2=25/4", P50,
     16*((x + 1)**2 + (y - Rational(3, 4))**2 - Rational(25, 4)))
check("50: (75+16+9)/16 = 25/4", Rational(75 + 16 + 9, 16) == Rational(25, 4))
P51 = 4*x**2 + 4*y**2 - 12*x + 8*y - 3
same("51: circle centre (3/2,-1) r=2", P51,
     4*((x - Rational(3, 2))**2 + (y + 1)**2 - 4))
check("51: 3/4+9/4+1 = 4", Rational(3, 4) + Rational(9, 4) + 1 == 4)
P52 = (x**2 + 4*x + 2*y**2) - (8*y - 4)
conic_check("52", P52, 8, (x + 2)**2/8 + (y - 2)**2/4, (-2, 2),
            (2*sqrt(2), 2), "ellipse",
            vertices=[(-2 + 2*sqrt(2), 2), (-2 - 2*sqrt(2), 2), (-2, 4), (-2, 0)])
P53 = (9*x**2 + 54*x) - (4*y**2 - 8*y - 113)
same("53: 9(x+3)^2-4(y-1)^2+36", P53, 9*(x + 3)**2 - 4*(y - 1)**2 + 36)
check("53: -113+81-4=-36", -113 + 81 - 4 == -36)
conic_check("53", P53, -36, -(x + 3)**2/4 + (y - 1)**2/9, (-3, 1), (2, 3),
            "hyp_ud", vertices=[(-3, -2), (-3, 4)],
            asymptotes=[Rational(3, 2)*x + Rational(11, 2),
                        -Rational(3, 2)*x - Rational(7, 2)])
P54 = 4*x**2 + 9*y**2 - 8*x + 36*y + 40
same("54: 4(x-1)^2+9(y+2)^2", P54, 4*(x - 1)**2 + 9*(y + 2)**2)
check("54: only point (1,-2)", solve([4*(x - 1)**2, 9*(y + 2)**2], [x, y], dict=True)
      == [{x: 1, y: -2}])
P55 = x**2 + 3*y**2 + 2*x - 6*y + 10
same("55: (x+1)^2+3(y-1)^2+6", P55, (x + 1)**2 + 3*(y - 1)**2 + 6)
# Empty set, checked for real: the only stationary point of P55 is (-1,1),
# P55 is 6 there, and its Hessian is positive definite, so P55 >= 6 > 0 on
# all of R^2 and no real point satisfies P55 = 0.
_crit55 = solve([P55.diff(x), P55.diff(y)], [x, y], dict=True)
check("55: unique stationary point (-1,1)", _crit55 == [{x: -1, y: 1}])
check("55: minimum value 6 > 0", val(P55, (-1, 1)) == 6)
_H55 = hessian(P55, (x, y))
check("55: Hessian positive definite", _H55[0, 0] > 0 and _H55.det() > 0)
# Independent second route: as a quadratic in x, P55 has discriminant
# -12((y-1)^2+2), negative for every real y, so no real x exists for any y.
_disc55 = discriminant(Poly(P55, x))
same("55: x-discriminant = -12((y-1)^2+2)", _disc55, -12*((y - 1)**2 + 2))
check("55: x-discriminant < 0 for all real y",
      (-12*((y - 1)**2 + 2)).is_negative is True)
P56 = x**2 - 4*y**2 - 6*x - 8*y + 5
same("56: (x-3)^2-4(y+1)^2", P56, (x - 3)**2 - 4*(y + 1)**2)
same("56: factors (x-2y-5)(x+2y-1)", P56, (x - 2*y - 5)*(x + 2*y - 1))
same("56: line 1 is y+1=(x-3)/2", x/2 - Rational(5, 2), -1 + (x - 3)/2)
same("56: line 2 is y+1=-(x-3)/2", -x/2 + Rational(1, 2), -1 - (x - 3)/2)
check("56: lines cross at (3,-1)", on(x - 2*y - 5, (3, -1)) and on(x + 2*y - 1, (3, -1)))
P57 = 2*y**2 - 8*y - x + 5
same("57: x = 2(y-2)^2-3", solve(P57, x)[0], 2*(y - 2)**2 - 3)
check("57: vertex (-3,2) on curve", on(P57, (-3, 2)))
check("57: (5,0) on curve", on(P57, (5, 0)))
ys = solve(P57.subs(x, 0), y)
check("57: y-axis at 2 +- sqrt(3/2)",
      set(simplify(s - 2)**2 for s in ys) == {Rational(3, 2)})

print("\n=== ch02 solution-figure parametrisations ===")
same("fig 46", ((3 + 4*cos(t)) - 3)**2/16 + ((-2 + 2*sin(t)) + 2)**2/4, 1)
same("fig 48", ((2 + 3*cosh(t)) - 2)**2/9 - ((-1 + 2*sinh(t)) + 1)**2/4, 1)
same("fig 53", -((-3 + 2*sinh(t)) + 3)**2/4 + ((1 + 3*cosh(t)) - 1)**2/9, 1)

print("\n=== ch04 problems 46-52 ===")
R46 = 4*x**2 + 9*y**2 - 16*x - 18*y - 11
same("46: 4(x-2)^2+9(y-1)^2-36", R46, 4*(x - 2)**2 + 9*(y - 1)**2 - 36)
check("46: origin inside (-11<=0)", val(R46, (0, 0)) <= 0)
check("46: extremes x in [-1,5], y in [-1,3]",
      all(on(R46, p) for p in [(-1, 1), (5, 1), (2, -1), (2, 3)]))

R47 = x**2 - 4*y**2 + 4*x + 8*y - 4     # region R47 >= 0
same("47: (x+2)^2-4(y-1)^2-4", R47, (x + 2)**2 - 4*(y - 1)**2 - 4)
check("47: vertices (-4,1),(0,1) on boundary", on(R47, (-4, 1)) and on(R47, (0, 1)))
check("47: centre not in region", val(R47, (-2, 1)) < 0)
check("47: origin not in region", val(R47, (0, 0)) < 0)
xr = solve(R47.subs(y, 0), x)
check("47: right branch at y=0 is x=-2+2sqrt2",
      any(simplify(r - (-2 + 2*sqrt(2))) == 0 for r in xr))

R48 = 9*y**2 - 4*x**2 + 8*x + 18*y - 31   # region R48 <= 0
same("48: 9(y+1)^2-4(x-1)^2-36", R48, 9*(y + 1)**2 - 4*(x - 1)**2 - 36)
check("48: vertices (1,1),(1,-3)", on(R48, (1, 1)) and on(R48, (1, -3)))
for xc in (-2, 4):
    for s in (1, -1):
        check(f"48: corner ({xc}, -1{'+' if s > 0 else '-'}2sqrt2) on boundary",
              on(R48, (xc, -1 + s*2*sqrt(2))))
check("48: centre in region", val(R48, (1, -1)) < 0)
check("48: midpoint (1,-1+2sqrt2) outside",
      simplify(val(R48, (1, -1 + 2*sqrt(2)))) > 0)
check("48: midpoint ratio = 2", simplify((2*sqrt(2))**2/4) == 2)

R49 = x**2 + 4*y**2 - 2*x + 16*y + 1
same("49: (x-1)^2+4(y+2)^2-16", R49, (x - 1)**2 + 4*(y + 2)**2 - 16)
xs = solve(R49.subs(y, (1 - x)/2), x)
check("49: line meets ellipse at x=1,5", set(xs) == {1, 5})
check("49: points (1,0),(5,-2)", on(R49, (1, 0)) and on(R49, (5, -2)))
same("49: quadratic", expand((x - 1)**2 + (x - 5)**2 - 16), 2*(x**2 - 6*x + 5))
check("49: centre side 1+2(-2)=-3<=1", 1 + 2*(-2) == -3)
check("49: cap point (4,-1) in ellipse, excluded by line",
      val(R49, (4, -1)) <= 0 and 4 + 2*(-1) > 1)

R50 = 4*x**2 + 9*y**2 + 8*x - 18*y - 23
same("50: expanded == 36*(std-1)", R50,
     36*((x + 1)**2/9 + (y - 1)**2/4 - 1))
check("50: ellipse through given points",
      all(on(R50, p) for p in [(-4, 1), (2, 1), (-1, 3), (-1, -1)]))
check("50: (-1,2) inside", val(R50, (-1, 2)) <= 0)
check("50: (-1,0) inside ellipse, fails y>=1", val(R50, (-1, 0)) <= 0)
check("50: (-1,3.5) outside ellipse", val(R50, (-1, Rational(7, 2))) > 0)
same("fig (i) ellipse", 4*(-1 + 3*cos(t))**2 + 9*(1 + 2*sin(t))**2
     + 8*(-1 + 3*cos(t)) - 18*(1 + 2*sin(t)) - 23, 0)

H51 = (x - 1)**2/4 - (y + 1)**2/16 - 1
check("51: vertices (-1,-1),(3,-1)", on(H51, (-1, -1)) and on(H51, (3, -1)))
check("51: asymptote slope b/a=2", Rational(4, 2) == 2)
check("51: centre inside (<1)", val(H51, (1, -1)) < 0)
corners = [(Rational(-3, 2), 2), (Rational(7, 2), 2),
           (Rational(-3, 2), -4), (Rational(7, 2), -4)]
check("51: four corners on hyperbola", all(on(H51, p) for p in corners))
check("51: 1+9/16=25/16", 1 + Rational(9, 16) == Rational(25, 16))
same("fig (j) left boundary", ((1 - 2*sqrt(1 + (t + 1)**2/16)) - 1)**2/4
     - (t + 1)**2/16, 1)
same("fig (j) branch", ((1 + 2*cosh(t)) - 1)**2/4 - ((-1 + 4*sinh(t)) + 1)**2/16, 1)

H52 = (x - 2)**2/4 - y**2 - 1
check("52: vertices (0,0),(4,0)", on(H52, (0, 0)) and on(H52, (4, 0)))
check("52: centre (2,0) between, 0<1", val(H52, (2, 0)) + 1 == 0)
check("52: (4.8,+-1) in (a): 0.96<1",
      val(H52 + 1, (Rational(24, 5), 1)) == Rational(96, 100)
      and val(H52 + 1, (Rational(24, 5), -1)) == Rational(96, 100))
check("52: midpoint (4.8,0) gives 1.96",
      val(H52 + 1, (Rational(24, 5), 0)) == Rational(196, 100))
xs52 = solve(Eq((x - 2)**2/4, 1 + y**2), x)
check("52: right branch x = 2+2sqrt(1+y^2)",
      any(simplify(r - (2 + 2*sqrt(1 + y**2))) == 0 for r in xs52))

# ======================================================================
# Round 2: ch03 rewriting drills (Additional 26-36), ch04 rule-box example,
# second worked example, Additional 26-45 and their figures.
# ======================================================================
from sympy import log, exp, oo, limit, Interval, solveset, S as SS
from sympy.calculus.util import function_range

xr = symbols("xr", real=True)


def frange(f, dom):
    return function_range(f, xr, dom)


def ex(e):
    return expand(e.rewrite(exp), force=True)


print("\n=== ch03 rule box ===")
same("3^{2x}/2^x = (9/2)^x", ex(3**(2*x)/2**x), ex(Rational(9, 2)**x))
same("8/(2x-6)^2 = 2/(x-3)^2", 8/(2*x - 6)**2, 2/(x - 3)**2)
same("sqrt(4-2x) inside = -2(x-2)", 4 - 2*x, -2*(x - 2))

print("\n=== ch03 Additional 26-36 ===")
f26 = 3 - 2*5**(-xr)
same("26: 5^-x = (1/5)^x", ex(5**(-x)), ex(Rational(1, 5)**x))
check("26: range (-oo,3)", frange(f26, SS.Reals) == Interval.open(-oo, 3))
check("26: y(0)=1", f26.subs(xr, 0) == 1)
check("26: limit x->oo is 3", limit(f26, xr, oo) == 3)
check("26: limit x->-oo is -oo", limit(f26, xr, -oo) == -oo)
r26 = -log(Rational(3, 2))/log(5)
check("26: root -ln1.5/ln5", abs(float(f26.subs(xr, r26))) < 1e-12)
check("26: root approx -0.25", abs(float(r26) + 0.2519) < 1e-3)

f27 = 1 + 3*Rational(2, 5)**xr
check("27: range (1,oo)", frange(f27, SS.Reals) == Interval.open(1, oo))
check("27: y(0)=4", f27.subs(xr, 0) == 4)
check("27: decreasing", f27.diff(xr).subs(xr, 0) < 0)
same("27: equivalent 3*(5/2)^-x+1", ex(3*Rational(5, 2)**(-x) + 1),
     ex(1 + 3*Rational(2, 5)**x))

f28 = 12**xr/3**xr - 2
same("28: 12^x/3^x = 4^x", ex(12**x/3**x), ex(4**x))
check("28: range (-2,oo)", frange(4**xr - 2, SS.Reals) == Interval.open(-2, oo))
check("28: y(0)=-1, root 1/2", f28.subs(xr, 0) == -1
      and simplify(f28.subs(xr, Rational(1, 2))) == 0)

f29 = 4 - 2**(2*xr)/8**xr
same("29: 2^{2x}/8^x = 2^{-x}", ex(2**(2*x)/8**x), ex(2**(-x)))
check("29: range (-oo,4)", frange(4 - 2**(-xr), SS.Reals) == Interval.open(-oo, 4))
check("29: y(0)=3, root -2", f29.subs(xr, 0) == 3 and simplify(f29.subs(xr, -2)) == 0)
check("29: limit oo is 4", limit(f29, xr, oo) == 4)

f30 = 2*sqrt(6 - 2*xr) + 1
same("30: 6-2x = -2(x-3)", 6 - 2*x, -2*(x - 3))
check("30: domain x<=3", solveset(6 - 2*xr >= 0, xr, SS.Reals) == Interval(-oo, 3))
check("30: range [1,oo)", frange(f30, Interval(-oo, 3)) == Interval(1, oo))
check("30: start (3,1), y(1)=5", f30.subs(xr, 3) == 1 and f30.subs(xr, 1) == 5)

f31 = 2 - sqrt(5 - xr)
check("31: range (-oo,2]", frange(f31, Interval(-oo, 5)) == Interval(-oo, 2))
check("31: start (5,2), root 1", f31.subs(xr, 5) == 2 and f31.subs(xr, 1) == 0)

f32 = 3*log(2*xr + 6) - 1
same("32: second reading", expand(3*log(2*(x + 3)) - 1, force=True),
     expand(3*log(x + 3) + 3*log(2) - 1, force=True))
check("32: domain x>-3", solveset(2*xr + 6 > 0, xr, SS.Reals) == Interval.open(-3, oo))
check("32: shift 3ln2-1 approx 1.08", abs(float(3*log(2) - 1) - 1.0794) < 1e-3)
r32 = (exp(Rational(1, 3)) - 6)/2
check("32: root", simplify(f32.subs(xr, r32)) == 0 and abs(float(r32) + 2.302) < 1e-3)
check("32: range R", limit(f32, xr, -3, "+") == -oo and limit(f32, xr, oo) == oo)

f33 = 1 - 2*log(4 - xr)
check("33: x->4- gives +oo", limit(f33, xr, 4, "-") == oo)
check("33: x->-oo gives -oo", limit(f33, xr, -oo) == -oo)
check("33: increasing, f' = 2/(4-x)", simplify(f33.diff(xr) - 2/(4 - xr)) == 0)
r33 = 4 - sqrt(exp(1))
check("33: root 4-sqrt(e)", simplify(f33.subs(xr, r33)) == 0
      and abs(float(r33) - 2.351) < 1e-3)

f34 = 2/(xr - 1)**2 - 3
check("34: range (-3,oo) on both branches",
      frange(f34, Interval.open(1, oo)) == Interval.open(-3, oo)
      and frange(f34, Interval.open(-oo, 1)) == Interval.open(-3, oo))
for s_ in (1, -1):
    check(f"34: root 1{'+' if s_ > 0 else '-'}sqrt(2/3)",
          simplify(f34.subs(xr, 1 + s_*sqrt(Rational(2, 3)))) == 0)
check("34: sqrt(2/3) approx 0.82", abs(float(sqrt(Rational(2, 3))) - 0.8165) < 1e-3)

f35 = 1 - 1/(xr + 2)**3
check("35: right branch below 1",
      frange(f35, Interval.open(-2, oo)) == Interval.open(-oo, 1))
check("35: left branch above 1",
      frange(f35, Interval.open(-oo, -2)) == Interval.open(1, oo))
check("35: root -1, y(0)=7/8", f35.subs(xr, -1) == 0 and f35.subs(xr, 0) == Rational(7, 8))

f36 = 4/(2*xr - 6)**2 + 1
same("36: = 1/(x-3)^2+1", 4/(2*x - 6)**2 + 1, 1/(x - 3)**2 + 1)
check("36: range (1,oo)", frange(f36, Interval.open(3, oo)) == Interval.open(1, oo))

print("\n=== ch04 rule box 'Where two conics meet' ===")


def realpts(eqs):
    out = set()
    for d in solve(eqs, [x, y], dict=True):
        if d[x].is_real and d[y].is_real:
            out.add((simplify(d[x]), simplify(d[y])))
    return out


check("rb: four points (+-3,+-2)", realpts([x**2 + y**2 - 13, 4*x**2 + 9*y**2 - 72])
      == {(3, 2), (3, -2), (-3, 2), (-3, -2)})
check("rb: 72-52=20, y^2=4", 72 - 4*13 == 20 and Rational(20, 5) == 4)
check("rb: nested pair has no real point",
      realpts([x**2 + y**2 - 16, x**2 + 2*y**2 - 8]) == set())
check("rb: ellipse vertex (2sqrt2,0) inside circle", (2*sqrt(2))**2 < 16)
check("rb: 4x^2+9y^2<=36 std", simplify((4*x**2 + 9*y**2)/36 - (x**2/9 + y**2/4)) == 0)
check("rb: x^2+4y<=12 -> y<=3-x^2/4", simplify(solve(x**2 + 4*y - 12, y)[0] - (3 - x**2/4)) == 0)
check("rb trap: x^2-4y<=12 -> y>=x^2/4-3", simplify(solve(x**2 - 4*y - 12, y)[0] - (x**2/4 - 3)) == 0)

print("\n=== ch04 worked example 2 ===")
P = x**2 + 4*y - 12
E = x**2 + 2*y**2 - 18
same("we2: elimination", expand(E.subs(x**2, 12 - 4*y)), 2*(y**2 - 2*y - 3))
same("we2: factor", y**2 - 2*y - 3, (y - 3)*(y + 1))
check("we2: common points (+-4,-1),(0,3)", realpts([P, E]) == {(4, -1), (-4, -1), (0, 3)})
check("we2: at x=2 ellipse +-sqrt7", set(solve(E.subs(x, 2), y)) == {sqrt(7), -sqrt(7)})
check("we2: x=4.1 parabola -1.2025", 3 - Rational(41, 10)**2/4 == Rational(-12025, 10000))
check("we2: x=4.1 ellipse low approx -0.77",
      abs(float(-sqrt((18 - Rational(41, 10)**2)/2)) + 0.771) < 1e-3)
same("we2: ellipse^2 - parabola^2 = x^2(1-x^2/16)",
     (18 - x**2)/2 - (3 - x**2/4)**2, x**2*(1 - x**2/16))
check("we2: (0,-2) inside", val(P, (0, -2)) <= 0 and val(E, (0, -2)) <= 0)
check("we2: a=3sqrt2", sqrt(18) == 3*sqrt(2))
check("we2: roots 2sqrt3", set(solve(P.subs(y, 0), x)) == {2*sqrt(3), -2*sqrt(3)})

print("\n=== ch04 Additional 26-31 (intersections) ===")
check("26: (+-4,+-3)", realpts([x**2 + y**2 - 25, x**2 + 2*y**2 - 34])
      == {(4, 3), (4, -3), (-4, 3), (-4, -3)})
check("27: no real point", realpts([x**2 + y**2 - 4, 9*x**2 + 16*y**2 - 144]) == set())
check("27: 7y^2=108, x^2=-80/7", 144 - 36 == 108 and 4 - Rational(108, 7) == Rational(-80, 7))
check("27: (2,0) inside ellipse", 9*4 < 144)
check("28: only (+-3,0)", realpts([4*x**2 + 9*y**2 - 36, x**2 + y**2 - 9])
      == {(3, 0), (-3, 0)})
same("28: x^2+y^2 on ellipse = 4+5x^2/9", x**2 + (36 - 4*x**2)/9, 4 + Rational(5, 9)*x**2)
check("29: (0,-2),(+-sqrt3,1)", realpts([y - x**2 + 2, x**2 + y**2 - 4])
      == {(0, -2), (sqrt(3), 1), (-sqrt(3), 1)})
same("29: quadratic", expand((y + 2) + y**2 - 4), (y + 2)*(y - 1))
check("30: (2,+-2sqrt2)", realpts([y**2 - 4*x, x**2 + y**2 - 12])
      == {(2, 2*sqrt(2)), (2, -2*sqrt(2))})
same("30: quadratic", x**2 + 4*x - 12, (x + 6)*(x - 2))
check("31: no real point", realpts([y - x**2 - 2, x**2 + 2*y**2 - 1]) == set())
same("31: quadratic", expand((y - 2) + 2*y**2 - 1), (2*y + 3)*(y - 1))
check("31: x^2 values -1, -7/2", 1 - 2 == -1 and Rational(-3, 2) - 2 == Rational(-7, 2))
check("31: ellipse |y|<=1/sqrt2", set(solve(2*y**2 - 1, y)) == {sqrt(2)/2, -sqrt(2)/2})

print("\n=== ch04 Additional 32-38 (forward) ===")
A1, A2 = x**2 + 6*y - 9, 4*x**2 + 9*y**2 - 36       # both <= 0
same("32: parabola std", solve(A1, y)[0], Rational(3, 2) - x**2/6)
check("32: pts (+-3,0)", realpts([A1, A2]) == {(3, 0), (-3, 0)})
same("32: elimination", expand(A2.subs(x**2, 9 - 6*y)), 9*y**2 - 24*y)
check("32: y=8/3 gives x^2=-7", 9 - 6*Rational(8, 3) == -7)
check("32: (0,1.8) fails only parabola",
      val(A1, (0, Rational(9, 5))) > 0 and val(A2, (0, Rational(9, 5))) <= 0
      and 6*Rational(9, 5) == Rational(54, 5))

B1, B2 = x**2 + 4*y - 16, 9*x**2 + 16*y**2 - 144   # B1<=0, B2>=0
check("33: four points", realpts([B1, B2]) ==
      {(4, 0), (-4, 0), (sqrt(7), Rational(9, 4)), (-sqrt(7), Rational(9, 4))})
same("33: elimination", expand(B2.subs(x**2, 16 - 4*y)), 16*y**2 - 36*y)
check("33: (0,3.5) in", val(B1, (0, Rational(7, 2))) <= 0 and val(B2, (0, Rational(7, 2))) >= 0)
check("33: (0,-4) in", val(B1, (0, -4)) <= 0 and val(B2, (0, -4)) >= 0)
check("33: midpoint (0,-1/4) out", val(B2, (0, Rational(-1, 4))) < 0)
check("33: origin out", val(B2, (0, 0)) < 0)
check("33: x=3 parabola 1.75 < ellipse top 1.98",
      4 - Rational(9, 4) < 3*sqrt(1 - Rational(9, 16))
      and abs(float(3*sqrt(Rational(7, 16))) - 1.984) < 1e-3)
check("33: sqrt7 approx 2.65", abs(float(sqrt(7)) - 2.6458) < 1e-3)

C1, C2 = x**2 + 2*y**2 - 9, (x - 2)**2 + y**2 - 5   # C1>=0, C2<=0, y>=0
check("34: real points (1,+-2)", realpts([C1, C2]) == {(1, 2), (1, -2)})
same("34: y^2 from circle", 5 - (x - 2)**2, 1 + 4*x - x**2)
same("34: elimination", expand(x**2 + 2*(1 + 4*x - x**2) - 9), -(x - 1)*(x - 7))
check("34: x=7 gives y^2=-20", 1 + 28 - 49 == -20)
check("34: (4,0) in, (2,0) out", val(C1, (4, 0)) >= 0 and val(C2, (4, 0)) <= 0
      and val(C1, (2, 0)) < 0)
check("34: midpoint (2,1) fails ellipse", val(C1, (2, 1)) < 0)
check("34: corners on curves", on(C1, (3, 0)) and on(C2, (2 + sqrt(5), 0)))
check("34: x=0 circle top 1 < ellipse top", val(C2, (0, 1)) == 0
      and sqrt(Rational(9, 2)) > 1)
check("34: b=3/sqrt2 approx 2.12", abs(float(3/sqrt(2)) - 2.1213) < 1e-3)

D1, D2 = x**2 + y**2 - 13, x**2 - y**2 - 5
check("35: real points (+-3,+-2)", realpts([D1, D2]) ==
      {(3, 2), (3, -2), (-3, 2), (-3, -2)})
check("35: (3,0) in", val(D1, (3, 0)) <= 0 and val(D2, (3, 0)) >= 0)
check("35: corners", on(D2, (sqrt(5), 0)) and on(D1, (sqrt(13), 0)))
check("35: sqrt(5+y^2) convex",
      simplify(sqrt(5 + y**2).diff(y, 2) - 5/(5 + y**2)**Rational(3, 2)) == 0)

E1 = (x - 3)**2 + (y - 3)**2 - 10    # >= 0
check("36: x-axis 2,4", set(solve(E1.subs(y, 0), x)) == {2, 4})
check("36: y-axis 2,4", set(solve(E1.subs(x, 0), y)) == {2, 4})
check("36: origin in (18>=10)", val(E1 + 10, (0, 0)) == 18)
check("36: (6,6) in, (3,3) out", val(E1, (6, 6)) >= 0 and val(E1, (3, 3)) < 0)
check("36: radius approx 3.16", abs(float(sqrt(10)) - 3.162) < 1e-3)

check("37: x=0 ->1, x=3 ->4", ((0 - 1)**2, (3 - 1)**2) == (1, 4))
check("37: y=5 meets at 1+-sqrt5 outside [0,3]",
      set(solve((x - 1)**2 - 5, x)) == {1 - sqrt(5), 1 + sqrt(5)}
      and float(1 - sqrt(5)) < 0 and float(1 + sqrt(5)) > 3)
check("38: corners on parabola", (-1 - 1)**2 == 4 and (3 - 1)**2 == 4)
check("38: (-1,2),(3,2) in; (1,2) out",
      2 <= (-1 - 1)**2 and 2 <= (3 - 1)**2 and not (2 <= (1 - 1)**2))
check("38: (0,0.5) in", Rational(1, 2) <= (0 - 1)**2)

print("\n=== ch04 Additional 39-45 (reverse) ===")
pk = (x - 1)**2 - 2
check("39: a=1 from (3,2)", Rational(2 - (-2), (3 - 1)**2) == 1)
same("39: expanded", pk, x**2 - 2*x - 1)
check("39: (0,-1),(3,2) on parabola", pk.subs(x, 0) == -1 and pk.subs(x, 3) == 2)
check("39: other root of (x-1)^2=4 is -1", set(solve((x - 1)**2 - 4, x)) == {-1, 3})
check("39: (-0.2,1): parabola -0.56 <= 1",
      pk.subs(x, Rational(-1, 5)) == Rational(-56, 100))
check("39: (1,-2.5) below vertex", Rational(-5, 2) < pk.subs(x, 1))

cl = (x - 1)**2 + (y - 1)**2 - 5
check("40: through (3,2)", on(cl, (3, 2)))
same("40: expanded", cl, x**2 + y**2 - 2*x - 2*y - 3)
check("40: corners (3,0),(0,3)", on(cl, (3, 0)) and on(cl, (0, 3)))
check("40: (2,-0.3),(-0.3,2) inside circle, (3,3): 8>5",
      val(cl, (2, Rational(-3, 10))) <= 0 and val(cl, (Rational(-3, 10), 2)) <= 0
      and val(cl + 5, (3, 3)) == 8)
check("40: rightmost 1+sqrt5", abs(float(1 + sqrt(5)) - 3.236) < 1e-3)

em = x**2/16 + y**2/4 - 1
pm = x**2/4 - 1
check("41: ellipse through vertices",
      all(on(em, p) for p in [(4, 0), (-4, 0), (0, 2), (0, -2)]))
check("41: parabola (0,-1),(2,0)", pm.subs(x, 0) == -1 and pm.subs(x, 2) == 0)
check("41: (0,2.1) fails ellipse only", val(em, (0, Rational(21, 10))) > 0)
check("41: (0,-1.2) fails parabola only", Rational(-6, 5) < pm.subs(x, 0)
      and val(em, (0, Rational(-6, 5))) <= 0)
check("41: (2.2,0.5) values 0.365, 0.21",
      val(em + 1, (Rational(11, 5), Rational(1, 2))) == Rational(365, 1000)
      and pm.subs(x, Rational(11, 5)) == Rational(21, 100))
same("41: expanded parabola", 4*(y - pm), 4*y + 4 - x**2)
xl = -sqrt(2 + 2*sqrt(13))
check("41/fig: left crossing x approx -3.035", abs(float(xl) + 3.035) < 1e-3
      and simplify(em.subs({x: xl, y: pm.subs(x, xl)})) == 0)

check("42: (2,2) in", 2 + 2 <= 5 and 8 >= 4)
check("42: (1,1) fails circle, (3,3) fails line", 1 + 1 < 4 and 6 > 5)
check("42: (-0.2,3),(3,-0.2) satisfy circle & line",
      Rational(1, 25) + 9 >= 4 and Rational(14, 5) <= 5)

po = Rational(1, 2)*(x - 1)**2 + 1
check("43: through (3,3)", po.subs(x, 3) == 3)
same("43: expanded", po, x**2/2 - x + Rational(3, 2))
check("43: corners (-1,3),(4,5.5)", po.subs(x, -1) == 3 and po.subs(x, 4) == Rational(11, 2))
check("43: midpoint (1.5,4.25) above parabola 1.125",
      po.subs(x, Rational(3, 2)) == Rational(9, 8)
      and (Rational(-1 + 4, 2), (3 + Rational(11, 2))/2) == (Rational(3, 2), Rational(17, 4)))
check("43: (-1.2,1) and (4.2,1) below parabola",
      po.subs(x, Rational(-6, 5)) >= 1 and po.subs(x, Rational(21, 5)) >= 1)

hp = x**2/4 - y**2 - 1
check("44: vertices", on(hp, (2, 0)) and on(hp, (-2, 0)))
check("44: corners (+-4,+-sqrt3)",
      all(on(hp, (sx*4, sy*sqrt(3))) for sx in (1, -1) for sy in (1, -1)))
check("44: (3,0) in, centre out", val(hp, (3, 0)) >= 0 and val(hp, (0, 0)) < 0)
check("44: asymptote slope 1/2", Rational(2 - 1, 4 - 2) == Rational(1, 2))

eq = x**2/9 + (y - 1)**2/4 - 1
check("45: vertices", all(on(eq, p) for p in [(-3, 1), (3, 1), (0, 3), (0, -1)]))
check("45: (4,1),(-4,1) in", val(eq, (4, 1)) >= 0 and val(eq, (-4, 1)) >= 0)
xs46 = solve(eq.subs(y, 0), x)
check("45: corners at x=+-3sqrt3/2",
      set(xs46) == {Rational(3, 2)*sqrt(3), -Rational(3, 2)*sqrt(3)}
      and set(solve(eq.subs(y, 2), x)) == set(xs46))
check("45: (2,1) gives 4/9", val(eq + 1, (2, 1)) == Rational(4, 9))
check("45: (4,2.2) ellipse ok", val(eq, (4, Rational(11, 5))) >= 0)
check("45: midpoint gives 3/4", val(eq + 1, (Rational(3, 2)*sqrt(3), 1)) == Rational(3, 4))
check("45: 3sqrt3/2 approx 2.60", abs(float(Rational(3, 2)*sqrt(3)) - 2.598) < 1e-3)

print("\n=== ch04 figure constants ===")
check("fig (l) right end 3.2361", abs(float(1 + sqrt(5)) - 3.2361) < 1e-4)
check("fig (l) circle r 2.2361", abs(float(sqrt(5)) - 2.2361) < 1e-4)
check("fig 33 sqrt7 2.6458", abs(float(sqrt(7)) - 2.6458) < 1e-4)
check("fig 34 2+sqrt5 4.2361", abs(float(2 + sqrt(5)) - 4.2361) < 1e-4)
check("fig 36 3+sqrt10 6.1623", abs(float(3 + sqrt(10)) - 6.1623) < 1e-4)
check("fig we2 sqrt18 4.2426", abs(float(sqrt(18)) - 4.2426) < 1e-4)

print(f"\n{sum(results)}/{len(results)} PASS, "
      f"{len(results) - sum(results)} FAIL")
