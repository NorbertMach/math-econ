"""Re-check the gap-closing additions to Volume 5 (derivatives).

ch09: Additional problems 26-33 (shifts and tilts of the standard surfaces).
ch10: second worked example (tangent plane) and Additional problems 22-46
      (differentiability, tangent planes, sums / row derivative / a^x).

Every claim is retyped by hand from the book; nothing is parsed from LaTeX.
Run with a Python that has sympy installed.
"""
from sympy import (symbols, Rational as R, sqrt, log, exp, diff, simplify,
                   expand, limit, Abs, Min, N, oo, Symbol, solve, Matrix,
                   series, Function, factorial, E as EE, sin, cos, eye, zeros,
                   cbrt, integrate, nsimplify)

x, y, z, h, k, m, t = symbols("x y z h k m t", real=True)
K, L = symbols("K L", positive=True)
bad = 0
count = 0


def ok(label, cond):
    global bad, count
    count += 1
    cond = bool(cond)
    if not cond:
        bad += 1
    print(f"{'PASS' if cond else 'FAIL'}  {label}")


def zero(e):
    return simplify(e) == 0


def close(a, b, tol=R(5, 10**7)):
    return abs(N(a, 20) - N(b, 20)) < tol


# ---------------------------------------------------------------- ch09 ----
print("--- ch09 Additional 26-33: shifts and tilts ---")
f = x**2 - 6*x + y**2 + 4*y + 10
ok("26 square completed", zero(f - ((x - 3)**2 + (y + 2)**2 - 3)))
ok("26 min -3 at (3,-2)", f.subs({x: 3, y: -2}) == -3)
ok("26 level 1 radius 2", zero(f.subs({x: 3 + 2, y: -2}) - 1))

f = (2*x - y)**2
ok("27 c=0 on y=2x", zero(f.subs(y, 2*x)))
ok("27 c=1 on y=2x-1", zero(f.subs(y, 2*x - 1) - 1))
ok("27 c=1 on y=2x+1", zero(f.subs(y, 2*x + 1) - 1))
ok("27 c=4 on y=2x-2", zero(f.subs(y, 2*x - 2) - 4))
ok("27 c=4 on y=2x+2", zero(f.subs(y, 2*x + 2) - 4))

f = 5 - (x + 3*y)**2
ok("28 crest 5 on x+3y=0", zero(f.subs(y, -x/3) - 5))
ok("28 c=1 on y=(-x+2)/3", zero(f.subs(y, (-x + 2)/3) - 1))
ok("28 c=1 on y=(-x-2)/3", zero(f.subs(y, (-x - 2)/3) - 1))
ok("28 B(1) is |x+3y|<=2", zero((5 - t**2 - 1) - (4 - t**2)))

f = (x + 1)**2 - (y - 2)**2
ok("29 saddle grad 0 at (-1,2)",
   all(zero(diff(f, v).subs({x: -1, y: 2})) for v in (x, y)))
ok("29 c=0 on y=x+3", zero(f.subs(y, x + 3)))
ok("29 c=0 on y=-x+1", zero(f.subs(y, -x + 1)))
ok("29 c=4 vertices (1,2),(-3,2)",
   f.subs({x: 1, y: 2}) == 4 and f.subs({x: -3, y: 2}) == 4)
ok("29 c=-4 vertices (-1,4),(-1,0)",
   f.subs({x: -1, y: 4}) == -4 and f.subs({x: -1, y: 0}) == -4)

f = y**2 - 4*y + 7
ok("30 square completed", zero(f - ((y - 2)**2 + 3)))
ok("30 c=7 at y=0 and y=4", f.subs(y, 0) == 7 and f.subs(y, 4) == 7)
ok("30 c=3 only y=2", solve(f - 3, y) == [2])

fc = sqrt((x - 1)**2 + (y - 4)**2)
gb = (x - 1)**2 + (y - 4)**2
ok("31 cone radii 1,2,3",
   all(zero(fc.subs({x: 1 + c, y: 4}) - c) for c in (1, 2, 3)))
ok("31 bowl radii 1,sqrt2,sqrt3",
   all(zero(gb.subs({x: 1 + sqrt(c), y: 4}) - c) for c in (1, 2, 3)))

f = (x - 2)*(y + 3)
ok("32 saddle at (2,-3)",
   all(zero(diff(f, v).subs({x: 2, y: -3})) for v in (x, y)))
ok("32 c=6 on y=-3+6/(x-2)", zero(f.subs(y, -3 + 6/(x - 2)) - 6))

s = symbols("s", real=True)
A = (x - y)**2
ok("33 A: level on y=x+s is s^2 (iv)", zero(A.subs(y, x + s) - s**2))
B = (x - 1)**2 + (y + 1)**2
ok("33 B: centre (1,-1) (vi)",
   all(zero(diff(B, v).subs({x: 1, y: -1})) for v in (x, y)))
C = x + 2*y
ok("33 C: slope -1/2 (iii)", zero(diff(solve(C - t, y)[0], x) + R(1, 2)))
D = (x - 2)*(y - 1)
ok("33 D: asymptotes x=2,y=1 (v)", zero(D.subs(y, 1 + t/(x - 2)) - t))
E = (y + 1)**2
ok("33 E: no x, y=-1+-sqrt(c) (i)",
   diff(E, x) == 0 and zero(E.subs(y, -1 + sqrt(t)) - t))
F = (x - 1)**2 - (y + 1)**2
ok("33 F: level 0 on y=x-2 and y=-x (ii)",
   zero(F.subs(y, x - 2)) and zero(F.subs(y, -x)))

# ---------------------------------------------------------------- ch10 ----
print("--- ch10 worked example 2: tangent plane of x ln(x+2y) ---")
f = x*log(x + 2*y)
P = {x: 1, y: 0}
ok("W f(P)=0", f.subs(P) == 0)
ok("W f_x formula", zero(diff(f, x) - (log(x + 2*y) + x/(x + 2*y))))
ok("W f_y formula", zero(diff(f, y) - 2*x/(x + 2*y)))
ok("W f_x(P)=1, f_y(P)=2",
   diff(f, x).subs(P) == 1 and diff(f, y).subs(P) == 2)
Q = {x: R(11, 10), y: R(5, 100)}
Lp = x + 2*y - 1
ok("W f(Q)=0.200554", close(f.subs(Q), R(200554, 10**6)))
ok("W L(Q)=0.2", Lp.subs(Q) == R(1, 5))
ok("W error ~0.000554", close(f.subs(Q) - Lp.subs(Q), R(554, 10**6)))
ok("W |Q-P|~0.112, square 0.0125",
   close(sqrt(R(1, 100) + R(25, 10**4)), R(112, 1000), R(1, 1000))
   and R(1, 100) + R(25, 10**4) == R(125, 10**4))
ok("W error below the squared step 0.0125",
   N(f.subs(Q) - Lp.subs(Q)) < R(125, 10**4))

print("--- ch10 rule box: xy/(x^2+y^2) ---")
f = x*y/(x**2 + y**2)
ok("box: on y=x equals 1/2", zero(f.subs(y, x) - R(1, 2)))

print("--- ch10 Additional 22-29: differentiability ---")
f = sqrt((x - 1)**2 + (y + 2)**2)
ok("22 f_x", zero(diff(f, x) - (x - 1)/f))
ok("22 f_y", zero(diff(f, y) - (y + 2)/f))
q = (f.subs({x: 1 + h, y: -2}) - 0)/h
ok("22 quotient one-sided +1/-1",
   limit(q, h, 0, "+") == 1 and limit(q, h, 0, "-") == -1)

f = sqrt(4*x**2 + (y - 3)**2)
ok("23 f_x", zero(diff(f, x) - 4*x/f))
ok("23 f_y", zero(diff(f, y) - (y - 3)/f))
q = f.subs({x: h, y: 3})/h
ok("23 x-quotient +-2", limit(q, h, 0, "+") == 2 and limit(q, h, 0, "-") == -2)
q = f.subs({x: 0, y: 3 + h})/h
ok("23 y-quotient +-1", limit(q, h, 0, "+") == 1 and limit(q, h, 0, "-") == -1)
ok("23 ellipse semi-axes c/2, c",
   all(zero(f.subs({x: R(c, 2), y: 3}) - c) and zero(f.subs({x: 0, y: 3 + c}) - c)
       for c in (1, 2, 3)))

f = Abs(x - 2) + Abs(y)
y0 = Symbol("y0", real=True)
q = (f.subs({x: 2 + h, y: y0}) - f.subs({x: 2, y: y0}))/h
ok("24 x-quotient on x=2 is +-1",
   limit(q, h, 0, "+") == 1 and limit(q, h, 0, "-") == -1)
ok("24 diamond corners (2+-c,0),(2,+-c)",
   all(f.subs(p) == 3 for p in ({x: 5, y: 0}, {x: -1, y: 0},
                                 {x: 2, y: 3}, {x: 2, y: -3})))

x0 = Symbol("x0", positive=True)
f = Min(2*x, 5*y)
pt = {x: x0, y: 2*x0/5}
qx = (f.subs({x: x0 + h, y: 2*x0/5}) - f.subs(pt))/h
qy = (f.subs({x: x0, y: 2*x0/5 + h}) - f.subs(pt))/h
ok("25 f_x one-sided 0 and 2",
   limit(qx, h, 0, "+") == 0 and limit(qx, h, 0, "-") == 2)
ok("25 f_y one-sided 0 and 5",
   limit(qy, h, 0, "+") == 0 and limit(qy, h, 0, "-") == 5)
c = Symbol("c", positive=True)
ok("25 corner of level c at (c/2, c/5)",
   2*(c/2) == c and 5*(c/5) == c and zero(R(2, 5)*(c/2) - c/5))

f = 3*x*y/(2*x**2 + y**2)
ok("26 f_x off origin",
   zero(diff(f, x) - 3*y*(y**2 - 2*x**2)/(2*x**2 + y**2)**2))
ok("26 f_y off origin",
   zero(diff(f, y) - 3*x*(2*x**2 - y**2)/(2*x**2 + y**2)**2))
ok("26 zero on both axes", f.subs(y, 0) == 0 and f.subs(x, 0) == 0)
ok("26 on y=mx equals 3m/(2+m^2)", zero(f.subs(y, m*x) - 3*m/(2 + m**2)))
ok("26 on y=x equals 1", zero(f.subs(y, x) - 1))
g = 3*m/(2 + m**2)
crit = solve(diff(g, m), m)
ok("26 extremes +-3sqrt2/4 at m=+-sqrt2",
   set(crit) == {sqrt(2), -sqrt(2)}
   and zero(g.subs(m, sqrt(2)) - 3*sqrt(2)/4)
   and zero(g.subs(m, -sqrt(2)) + 3*sqrt(2)/4)
   and limit(g, m, oo) == 0)

f = sqrt(x**4 + y**4)
ok("27 f_x formula", zero(diff(f, x) - 2*x**3/f))
ok("27 f_x(0,0)=0 via quotient", limit(f.subs({x: h, y: 0})/h, h, 0) == 0)
ok("27 f_y(0,0)=0 via quotient", limit(f.subs({x: 0, y: h})/h, h, 0) == 0)
ok("27 x^4+y^4 <= (x^2+y^2)^2",
   zero(expand((x**2 + y**2)**2 - (x**4 + y**4)) - 2*x**2*y**2))

f = sqrt(9 - x**2 - y**2)
ok("28 f_x, f_y", zero(diff(f, x) + x/f) and zero(diff(f, y) + y/f))
ok("28 top (0,0,3), grad 0",
   f.subs({x: 0, y: 0}) == 3
   and diff(f, x).subs({x: 0, y: 0}) == 0)
ok("28 |grad| -> oo at rim",
   limit(sqrt(diff(f, x)**2 + diff(f, y)**2).subs(y, 0), x, 3, "-") == oo)

f = Abs(x - y)
q = (f.subs({x: x0 + h, y: x0}) - 0)/h
ok("29 x-quotient on y=x is +-1",
   limit(q, h, 0, "+") == 1 and limit(q, h, 0, "-") == -1)
ok("29 contours y=x+-c", f.subs(y, x + 3) == 3 and f.subs(y, x - 3) == 3)

print("--- ch10 Additional 30-37: tangent planes ---")


def tp(label, f, vs, P, grad_claim, fP, plane, Q, fQ6, LQ6):
    sub = dict(zip(vs, P))
    subQ = dict(zip(vs, Q))
    g = [simplify(diff(f, v).subs(sub)) for v in vs]
    ok(f"{label} f(P)={fP}", zero(f.subs(sub) - fP))
    ok(f"{label} grad(P)={grad_claim}",
       all(zero(a - b) for a, b in zip(g, grad_claim)))
    lin = fP + sum(gi*(v - p) for gi, v, p in zip(grad_claim, vs, P))
    ok(f"{label} plane as printed", zero(expand(lin - plane)))
    ok(f"{label} f(Q)~{fQ6}", close(f.subs(subQ), fQ6, R(5, 10**7)))
    ok(f"{label} L(Q)~{LQ6}", close(plane.subs(subQ), LQ6, R(5, 10**7)))


tp("30", log(2*x + y**2), (x, y), (0, 1), (2, 2), 0, 2*x + 2*y - 2,
   (R(5, 100), R(98, 100)), R(58646, 10**6), R(6, 100))
ok("30 9604 and gap 0.001354",
   R(98, 100)**2 == R(9604, 10**4)
   and close(R(6, 100) - log(R(10604, 10**4)), R(1354, 10**6)))
tp("31", exp(x - 2*y), (x, y), (2, 1), (1, -2), 1, x - 2*y + 1,
   (R(21, 10), R(102, 100)), R(1061837, 10**6), R(106, 100))
tp("32", x*y**2*exp(x - 1), (x, y), (1, 2), (8, 4), 4,
   4 + 8*(x - 1) + 4*(y - 2), (R(98, 100), R(203, 100)),
   R(3958515, 10**6), R(396, 100))
ok("32 f_x factored", zero(diff(x*y**2*exp(x - 1), x)
                           - y**2*exp(x - 1)*(1 + x)))
ok("32 2.03^2=4.1209, e^-0.02~0.980199",
   R(203, 100)**2 == R(41209, 10**4)
   and close(exp(R(-2, 100)), R(980199, 10**6)))
tp("33", (x + y)/(x - y), (x, y), (3, 1), (R(-1, 2), R(3, 2)), 2,
   2 - R(1, 2)*(x - 3) + R(3, 2)*(y - 1), (R(31, 10), R(9, 10)),
   R(1818182, 10**6), R(18, 10))
ok("33 partial formulas",
   zero(diff((x + y)/(x - y), x) + 2*y/(x - y)**2)
   and zero(diff((x + y)/(x - y), y) - 2*x/(x - y)**2))
tp("34", K**R(1, 4)*L**R(3, 4), (K, L), (16, 1), (R(1, 32), R(3, 2)), 2,
   2 + R(1, 32)*(K - 16) + R(3, 2)*(L - 1), (R(33, 2), R(11, 10)),
   R(2164789, 10**6), R(2165625, 10**6))
tp("35", log(x + 2*y + 3*z), (x, y, z), (2, -1, R(1, 3)), (1, 2, 3), 0,
   x + 2*y + 3*z - 1, (R(202, 100), R(-99, 100), R(35, 100)),
   R(86178, 10**6), R(9, 100))
tp("36", x*exp(y*z), (x, y, z), (2, 0, 5), (1, 10, 0), 2, x + 10*y,
   (R(205, 100), R(1, 100), R(49, 10)), R(2152952, 10**6), R(215, 100))
tp("37", x*y/z, (x, y, z), (2, 3, 6), (R(1, 2), R(1, 3), R(-1, 6)), 1,
   1 + R(1, 2)*(x - 2) + R(1, 3)*(y - 3) - R(1, 6)*(z - 6),
   (R(21, 10), R(29, 10), R(62, 10)), R(982258, 10**6), R(983333, 10**6))
ok("worked-style gaps 31,32,34,36,37",
   close(exp(R(6, 100)) - R(106, 100), R(1837, 10**6))
   and close(R(396, 100) - R(98, 100)*R(203, 100)**2*exp(R(-2, 100)),
             R(1485, 10**6))
   and close(R(2165625, 10**6) - R(33, 2)**R(1, 4)*R(11, 10)**R(3, 4),
             R(836, 10**6))
   and close(R(205, 100)*exp(R(49, 1000)) - R(215, 100), R(2952, 10**6))
   and close(R(983333, 10**6) - R(609, 620), R(1075, 10**6))
   and close(R(1818182, 10**6) - R(18, 10), R(18182, 10**6))
   and close(R(9, 100) - log(R(109, 100)), R(3822, 10**6)))

print("--- ch10 Additional 38-46: sums, row derivative, a^x ---")
n = 5
X = symbols(f"x1:{n + 1}", real=True)
ones = {v: 1 for v in X}

f = sum(log(1 + xi**2) for xi in X)
ok("38 partials 2x_k/(1+x_k^2), all 1 at ones",
   all(zero(diff(f, X[j]) - 2*X[j]/(1 + X[j]**2)) for j in range(n))
   and all(diff(f, X[j]).subs(ones) == 1 for j in range(n)))

f = sum(X)**3
ok("39 partial 3(sum)^2", all(zero(diff(f, X[j]) - 3*sum(X)**2)
                              for j in range(n)))
X4 = symbols("y1:5", real=True)
f4 = sum(X4)**3
pt4 = dict(zip(X4, (1, 2, 3, 4)))
ok("39 n=4 at (1,2,3,4): 300 each",
   all(diff(f4, v).subs(pt4) == 300 for v in X4))

f = sum(X[j]**(j + 2) for j in range(n))   # index i=j+1 -> x_i^(i+1)
ok("40 partial (k+1)x_k^k",
   all(zero(diff(f, X[j]) - (j + 2)*X[j]**(j + 1)) for j in range(n))
   and [diff(f, X[j]).subs(ones) for j in range(n)] == [2, 3, 4, 5, 6])
X3 = symbols("z1:4", real=True)
f3 = X3[0]**2 + X3[1]**3 + X3[2]**4
pt3 = dict(zip(X3, (2, 1, -1)))
ok("40 n=3 at (2,1,-1): (4,3,-4)",
   [diff(f3, v).subs(pt3) for v in X3] == [4, 3, -4])

f = sum((j + 1)**2/X[j]**3 for j in range(n))
ok("41 partial -3k^2/x_k^4, at ones -3k^2",
   all(zero(diff(f, X[j]) + 3*(j + 1)**2/X[j]**4) for j in range(n))
   and [diff(f, X[j]).subs(ones) for j in range(n)]
   == [-3, -12, -27, -48, -75])

f = sum(xi**2 for xi in X) - sum(X)**2/n
xbar = sum(X)/n
ok("42 partial 2(x_k - mean)",
   all(zero(diff(f, X[j]) - 2*(X[j] - xbar)) for j in range(n)))
ok("42 components sum to 0", zero(sum(diff(f, v) for v in X)))
f3 = sum(v**2 for v in X3) - sum(X3)**2/3
pt3 = dict(zip(X3, (1, 2, 6)))
ok("42 n=3 at (1,2,6): (-4,-2,6)",
   [simplify(diff(f3, v).subs(pt3)) for v in X3] == [-4, -2, 6])
sol = solve([diff(f3, v) for v in X3], list(X3), dict=True)
ok("42 grad=0 iff all equal (n=3)",
   len(sol) == 1 and zero(sol[0].get(X3[0], X3[0]) - sol[0].get(X3[1], X3[1]))
   and zero(sol[0].get(X3[1], X3[1]) - sol[0].get(X3[2], X3[2])))

f = y**2*log(x**2 + 3*y*z)
sP = x**2 + 3*y*z
P = {x: 1, y: 1, z: 1}
ok("43 f_x", zero(diff(f, x) - y**2*2*x/sP))
ok("43 f_y", zero(diff(f, y) - (2*y*log(sP) + y**2*3*z/sP)))
ok("43 f_z", zero(diff(f, z) - 3*y**3/sP))
ok("43 values (1/2, 2ln4+3/4, 3/4)",
   zero(diff(f, x).subs(P) - R(1, 2))
   and zero(diff(f, y).subs(P) - (2*log(4) + R(3, 4)))
   and zero(diff(f, z).subs(P) - R(3, 4))
   and close(2*log(4) + R(3, 4), R(3522589, 10**6)))

E = exp(y**2 - 2*x*z)
f = x**3*E
P = {x: 1, y: 0, z: 1}
ok("44 f_x = x^2(3-2xz)E", zero(diff(f, x) - x**2*(3 - 2*x*z)*E))
ok("44 f_y = 2x^3 y E", zero(diff(f, y) - 2*x**3*y*E))
ok("44 f_z = -2x^4 E", zero(diff(f, z) + 2*x**4*E))
ok("44 grad(P) = e^-2 (1,0,-2)",
   zero(diff(f, x).subs(P) - exp(-2)) and diff(f, y).subs(P) == 0
   and zero(diff(f, z).subs(P) + 2*exp(-2))
   and close(exp(-2), R(135335, 10**6)) and close(2*exp(-2), R(270671, 10**6)))

f = 2**x*3**y + 5**(x*y)
ok("45 f_x", zero(diff(f, x) - (2**x*3**y*log(2) + y*5**(x*y)*log(5))))
ok("45 f_y", zero(diff(f, y) - (2**x*3**y*log(3) + x*5**(x*y)*log(5))))
P = {x: 1, y: 1}
ok("45 at (1,1): 6ln2+5ln5 ~12.206073, 6ln3+5ln5 ~14.638863",
   zero(diff(f, x).subs(P) - (6*log(2) + 5*log(5)))
   and zero(diff(f, y).subs(P) - (6*log(3) + 5*log(5)))
   and close(6*log(2) + 5*log(5), R(12206073, 10**6))
   and close(6*log(3) + 5*log(5), R(14638863, 10**6)))

f = sum(3**((j + 1)*X[j]) for j in range(n))
zeros_ = {v: 0 for v in X}
ok("46 partial k ln3 3^(k x_k), at 0: k ln3",
   all(zero(diff(f, X[j]) - (j + 1)*log(3)*3**((j + 1)*X[j])) for j in range(n))
   and all(zero(diff(f, X[j]).subs(zeros_) - (j + 1)*log(3)) for j in range(n)))

print("--- ch10 rule-box sum patterns ---")
f = sum(X[j]**(j + 1) for j in range(n))
ok("box sum x_i^i -> k x_k^(k-1)",
   all(zero(diff(f, X[j]) - (j + 1)*X[j]**j) for j in range(n)))
f = sum((j + 1)/X[j]**4 for j in range(n))
ok("box sum j/x_j^4 -> -4k/x_k^5",
   all(zero(diff(f, X[j]) + 4*(j + 1)/X[j]**5) for j in range(n)))
f = sum(X)**2
ok("box (sum)^2 -> 2 sum", all(zero(diff(f, v) - 2*sum(X)) for v in X))
ok("box 2^(xy) -> y 2^(xy) ln2", zero(diff(2**(x*y), x) - y*2**(x*y)*log(2)))

# ============================================================ round 2 ====
# Second gap-closing pass: ch01, ch02, ch04, ch08, ch09, ch10-ch14.
a, b, c = symbols("a b c", positive=True)
kk = symbols("k", real=True)
u, w, s = symbols("u w s", real=True)

print("--- ch01 Additional 16-18: letters as constants ---")
f = sqrt((a*x + b)/(c - x))
ok("16 u' = (ac+b)/(c-x)^2",
   zero(diff((a*x + b)/(c - x), x) - (a*c + b)/(c - x)**2))
xv = {x: R(1, 3), a: 2, b: 3, c: 5}
ok("16 f' closed form (numeric spot)",
   close(diff(f, x).subs(xv),
         ((a*c + b)/(2*(c - x)**R(3, 2)*(a*x + b)**R(1, 2))).subs(xv)))
q = (4 - a*x)/(x + b)
ok("17 u' = -(ab+4)/(x+b)^2", zero(diff(q, x) + (a*b + 4)/(x + b)**2))
xv = {x: R(1, 2), a: 1, b: 2}
ok("17 g' closed form (spot, positive quotient)",
   close(diff(q**R(1, 3), x).subs(xv),
         (-(a*b + 4)/(3*(x + b)**2)*((x + b)/(4 - a*x))**R(2, 3)).subs(xv)))
ok("18 h' = x(a-bx)^2(2a-5bx)",
   zero(diff(x**2*(a - b*x)**3, x) - x*(a - b*x)**2*(2*a - 5*b*x)))

print("--- ch02 Additional 26-29 ---")
ok("26", zero(diff(x*log(a*x**2 + c), x)
              - (log(a*x**2 + c) + 2*a*x**2/(a*x**2 + c))))
ok("27", zero(diff(sin(2 - kk*x)/a**(c*x), x)
              + (kk*cos(2 - kk*x) + c*log(a)*sin(2 - kk*x))/a**(c*x)))
tt = symbols("t", positive=True)
ok("28", zero(diff(log(sqrt((tt + a)/(tt - a))), tt).subs(tt, a + 1)
              + (a/((tt)**2 - a**2)).subs(tt, a + 1)))
ok("29", zero(diff(exp(-a*x)*sqrt(x**2 + c), x)
              - exp(-a*x)*(x - a*(x**2 + c))/sqrt(x**2 + c)))

print("--- ch04 rule box and Additional 11-15 ---")
g = Function("g")(x)
F = exp(g)
ok("box f''' = (g'''+3g'g''+g'^3)f",
   zero(diff(F, x, 3) - (diff(g, x, 3) + 3*diff(g, x)*diff(g, x, 2)
                         + diff(g, x)**3)*F))
ok("box e^{x^2-4x} at 2: T3 = e^-4(1+(x-2)^2)",
   zero(series(exp(x**2 - 4*x), x, 2, 4).removeO()
        - exp(-4)*(1 + (x - 2)**2)))


def taylor(f, a0, n):
    return sum(diff(f, x, k).subs(x, a0)/factorial(k)*(x - a0)**k
               for k in range(n + 1))


T = taylor(exp(x**2), 1, 3)
ok("11 T3 e^{x^2} at 1", zero(T - EE*(1 + 2*(x - 1) + 3*(x - 1)**2
                                     + R(10, 3)*(x - 1)**3)))
ok("11 f'''=(12x+8x^3)f", zero(diff(exp(x**2), x, 3) - (12*x + 8*x**3)*exp(x**2)))
ok("11 T3(1.1)=3.352548, true 3.353485",
   close(T.subs(x, R(11, 10)), 3.352548, 1e-6) and close(exp(R(121, 100)), 3.353485, 1e-6))
T = taylor(sqrt(1 + x**2), 1, 3)
ok("12 T3 sqrt(1+x^2) at 1",
   zero(T - sqrt(2)*(1 + (x - 1)/2 + (x - 1)**2/8 - (x - 1)**3/16)))
ok("12 f''=(1+x^2)^-3/2", zero(diff(sqrt(1 + x**2), x, 2) - (1 + x**2)**R(-3, 2)))
ok("12 T3(1.1)=1.486604, true 1.486607",
   close(T.subs(x, R(11, 10)), 1.486604, 1e-6) and close(sqrt(R(221, 100)), 1.486607, 1e-6))
fl = log(1 + x**2)
T = taylor(fl, 1, 3)
ok("13 f''' = (4x^3-12x)/(1+x^2)^3", zero(diff(fl, x, 3) - (4*x**3 - 12*x)/(1 + x**2)**3))
ok("13 T3 = ln2+(x-1)-(x-1)^3/6", zero(T - (log(2) + (x - 1) - (x - 1)**3/6)))
ok("13 4th coefficient 1/8", zero(diff(fl, x, 4).subs(x, 1)/24 - R(1, 8)))
ok("13 T3(1.2)=0.891814, +term 0.892014, true 0.891998",
   close(T.subs(x, R(6, 5)), 0.891814, 1e-6)
   and close(T.subs(x, R(6, 5)) + R(1, 8)*R(1, 5)**4, 0.892014, 1e-6)
   and close(log(R(244, 100)), 0.891998, 1e-6))
ok("14 T3 e^{x^2+x} at -1",
   zero(taylor(exp(x**2 + x), -1, 3)
        - (1 - (x + 1) + R(3, 2)*(x + 1)**2 - R(7, 6)*(x + 1)**3)))
ok("15 T3 = 1-(x-3)^2 and next non-zero terms",
   zero(taylor(exp(6*x - x**2 - 9), 3, 3) - (1 - (x - 3)**2))
   and zero(taylor(exp(6*x - x**2 - 9), 3, 7)
            - (1 - (x - 3)**2 + (x - 3)**4/2 - (x - 3)**6/6)))

print("--- ch08 logistic, production, inflection tangents ---")
G_, A_, k_ = symbols("G A k", positive=True)
fL = G_/(1 + A_*exp(-k_*G_*x))
ok("box f' = k f (G-f)", zero(diff(fL, x) - k_*fL*(G_ - fL)))
ok("box f'' = k f'(G-2f)", zero(diff(fL, x, 2) - k_*diff(fL, x)*(G_ - 2*fL)))
xs = log(A_)/(k_*G_)
ok("box inflection f(x*)=G/2, f'(x*)=kG^2/4",
   zero(fL.subs(x, xs) - G_/2) and zero(diff(fL, x).subs(x, xs) - k_*G_**2/4))
S = 60/(1 + 5*exp(-R(3, 10)*x))
ok("26 S'=0.005 S(60-S)", zero(diff(S, x) - R(5, 1000)*S*(60 - S)))
ok("26 S(0)=10, x*=ln5/0.3~5.365, S'(x*)=4.5",
   S.subs(x, 0) == 10 and close(log(5)/R(3, 10), 5.365, 1e-3)
   and zero(diff(S, x).subs(x, log(5)/R(3, 10)) - R(9, 2)))
p_ = 1/(1 + 9*exp(-R(4, 5)*x))
ok("27 p'=0.8p(1-p), t*=ln9/0.8~2.75, rate 0.2",
   zero(diff(p_, x) - R(4, 5)*p_*(1 - p_))
   and close(log(9)/R(4, 5), 2.7465, 1e-4)
   and zero(diff(p_, x).subs(x, log(9)/R(4, 5)) - R(1, 5)))
yL = 50/(1 + 9*exp(-x))
ok("28 y solves y'=0.02y(50-y), y(0)=5, rate 12.5 at ln9",
   zero(diff(yL, x) - R(2, 100)*yL*(50 - yL)) and yL.subs(x, 0) == 5
   and zero(diff(yL, x).subs(x, log(9)) - R(25, 2)) and close(log(9), 2.197, 1e-3))
ok("29 f''=k^2 f(G-f)(G-2f)",
   zero(diff(fL, x, 2) - k_**2*fL*(G_ - fL)*(G_ - 2*fL)))
x10 = (log(A_) - log(9))/(k_*G_)
ok("29 f(x10)=G/10, duration ln9/(kG)",
   zero(fL.subs(x, x10) - G_/10) and zero(xs - x10 - log(9)/(k_*G_)))
q_ = 2*x**2 - x**3/6
ok("30 MP, f''", zero(diff(q_, x) - x*(8 - x)/2) and zero(diff(q_, x, 2) - (4 - x)))
ok("30 MP(4)=8, AP(6)=MP(6)=6, f(8)=128/3, f(5)=175/6, AP(5)=35/6",
   diff(q_, x).subs(x, 4) == 8 and (q_/x).subs(x, 6) == 6
   and diff(q_, x).subs(x, 6) == 6 and q_.subs(x, 8) == R(128, 3)
   and q_.subs(x, 5) == R(175, 6) and (q_/x).subs(x, 5) == R(35, 6)
   and solve(diff(q_/x, x), x) == [6])
yy = 40*x**2*exp(-x/5)
ok("31 y' = 8x e^{-x/5}(10-x)", zero(diff(yy, x) - 8*x*exp(-x/5)*(10 - x)))
ok("31 y'' = 8/5 e^{-x/5}(x^2-20x+50)",
   zero(diff(yy, x, 2) - R(8, 5)*exp(-x/5)*(x**2 - 20*x + 50)))
ok("31 numbers",
   close(10 - 5*sqrt(2), 2.93, 5e-3) and close(diff(yy, x).subs(x, 10 - 5*sqrt(2)), 92.23, 5e-3)
   and zero(diff(yy/x, x).subs(x, 5)) and close(200/EE, 73.58, 5e-3)
   and zero(diff(yy, x).subs(x, 5) - 200/EE) and close(yy.subs(x, 10), 541.34, 5e-3)
   and close(yy.subs(x, 4), 287.57, 5e-3) and close((yy/x).subs(x, 4), 71.89, 5e-3)
   and close(yy.subs(x, 7), 483.33, 5e-3))
fr = sqrt(x)/(3 - x)
ok("33 f' = (x+3)/(2 sqrt x (3-x)^2)", zero(diff(fr, x) - (x + 3)/(2*sqrt(x)*(3 - x)**2)))
ok("33 f'' = 3(x^2+6x-3)/(4x^{3/2}(3-x)^3)",
   zero(diff(fr, x, 2) - 3*(x**2 + 6*x - 3)/(4*x**R(3, 2)*(3 - x)**3)))
x0 = 2*sqrt(3) - 3
ok("33 tangent numbers",
   close(x0, 0.464102, 1e-6) and close(fr.subs(x, x0), 0.268642, 1e-6)
   and close(diff(fr, x).subs(x, x0), 0.395358, 1e-6)
   and close(fr.subs(x, x0) - diff(fr, x).subs(x, x0)*x0, 0.085156, 1e-6))
ok("34 inflections ±1, slopes ±1, meet at (0,ln2-1)",
   set(solve(diff(fl, x, 2), x)) == {-1, 1}
   and diff(fl, x).subs(x, 1) == 1 and diff(fl, x).subs(x, -1) == -1)
fe = x*exp(-x)
ok("35 inflection 2, tangent e^-2(4-x)",
   solve(diff(fe, x, 2), x) == [2]
   and zero(fe.subs(x, 2) + diff(fe, x).subs(x, 2)*(x - 2) - exp(-2)*(4 - x)))

print("--- ch09 Additional 34-47 ---")
ok("34 level e^4: ellipse semi-axes 2,1",
   zero(exp(x**2 + 4*y**2).subs({x: 2, y: 0}) - exp(4))
   and zero(exp(x**2 + 4*y**2).subs({x: 0, y: 1}) - exp(4)))
f = log(x**2 + 2*y**2 - 1)
ok("35 c=ln3: semi-axes 2, sqrt2", zero(f.subs({x: 2, y: 0}) - log(3))
   and zero(f.subs({x: 0, y: sqrt(2)}) - log(3)))
f = 2 + sqrt(y - x**2 + 1)
ok("36 c=4 on y=x^2+3", zero(f.subs(y, x**2 + 3) - 4))
f = 5 - sqrt(2*x - y + 3)
ok("37 c=3 on y=2x-1", zero(f.subs(y, 2*x - 1) - 3))
f = log(9 - x**2 - 4*y**2)
ok("38 c=ln5: semi-axes 2,1; c=0: 2sqrt2, sqrt2",
   zero(f.subs({x: 2, y: 0}) - log(5)) and zero(f.subs({x: 0, y: 1}) - log(5))
   and zero(f.subs({x: 2*sqrt(2), y: 0})) and zero(f.subs({x: 0, y: sqrt(2)})))
X_, Y_ = symbols("X Y", positive=True)
ok("39 level 4 through (1,4),(2,1)", (X_**2*Y_).subs({X_: 1, Y_: 4}) == 4
   and (X_**2*Y_).subs({X_: 2, Y_: 1}) == 4)
ok("40 y=c^{3/2}x^{-1/2} on level c",
   zero((X_**R(1, 3)*(c**R(3, 2)*X_**R(-1, 2))**R(2, 3)) - c))
ok("41 (b) circles r^2=1/c-1", zero((1/(1 + x**2 + y**2)).subs({x: sqrt(1/c - 1), y: 0}) - c))
f = x**3 - 3*x + y**2
grads = {(0, 0): (-3, 0), (2, 1): (9, 2), (0, 2): (-3, 4), (-2, 1): (9, 2)}
ok("42 gradients at P1,P4,P5,P6",
   all((diff(f, x).subs({x: p[0], y: p[1]}), diff(f, y).subs({x: p[0], y: p[1]})) == gv
       for p, gv in grads.items()))
ok("42 heights f(P6)=-1, f(P5)=4, f(P4)=3",
   f.subs({x: -2, y: 1}) == -1 and f.subs({x: 0, y: 2}) == 4 and f.subs({x: 2, y: 1}) == 3)
ok("43 slope of contour at P5 = 3/4, (-3,4).(4,3)=0",
   (-diff(f, x)/diff(f, y)).subs({x: 0, y: 2}) == R(3, 4) and -3*4 + 4*3 == 0)
f = 4 - y**2 - (x**2 - 1)**2
ok("44 levels: Q4 saddle at 3, Q5 peak 4, Q1,Q3 on 2, Q2 on 0, R on 3.5",
   f.subs({x: 0, y: 0}) == 3 and f.subs({x: 1, y: 0}) == 4
   and f.subs({x: 0, y: 1}) == 2 and zero(f.subs({x: -sqrt(2), y: 1}) - 2)
   and zero(f.subs({x: sqrt(3), y: 0})) and zero(f.subs({x: 1, y: 1/sqrt(2)}) - R(7, 2)))
H4 = Matrix([[diff(f, x, 2), diff(f, x, y)], [diff(f, x, y), diff(f, y, 2)]])
ok("44 Q4 saddle (det<0), Q5 max", H4.subs({x: 0, y: 0}).det() < 0
   and H4.subs({x: 1, y: 0}).det() > 0 and H4.subs({x: 1, y: 0})[0, 0] < 0)
fx, fy = diff(f, x), diff(f, y)
ok("45 signs Q1 (0,-) Q2 (-,0) Q3 (+,-)",
   fx.subs({x: 0, y: 1}) == 0 and fy.subs({x: 0, y: 1}) < 0
   and fx.subs({x: sqrt(3), y: 0}) < 0 and fy.subs({x: sqrt(3), y: 0}) == 0
   and fx.subs({x: -sqrt(2), y: 1}) > 0 and fy.subs({x: -sqrt(2), y: 1}) < 0)
ok("46 f(±1,1.4)=2.04, f(0,1.4)=1.04, f(±2,0)<0",
   zero(f.subs({x: 1, y: R(7, 5)}) - R(204, 100)) and zero(f.subs({x: 0, y: R(7, 5)}) - R(104, 100))
   and f.subs({x: 2, y: 0}) < 0)
ok("47 E level 0 lines y=±sqrt2(x-1)", zero((2*(x - 1)**2 - y**2).subs(y, sqrt(2)*(x - 1))))
ok("47 C floor 2y=x", zero(((2*y - x)**2).subs(y, x/2)))

print("--- ch10 worked example 3 and Additional 47-56 ---")
h1, h2, a0, b0 = symbols("h1 h2 a0 b0", real=True)


def remainder(f, a_, b_):
    fa = f.subs({x: a_, y: b_})
    gx = diff(f, x).subs({x: a_, y: b_})
    gy = diff(f, y).subs({x: a_, y: b_})
    return expand(f.subs({x: a_ + h1, y: b_ + h2}, simultaneous=True)
                  - fa - gx*h1 - gy*h2)


ok("W3 x^2y remainder = 2a h1h2 + b h1^2 + h1^2 h2",
   zero(remainder(x**2*y, a0, b0) - (2*a0*h1*h2 + b0*h1**2 + h1**2*h2)))
ok("box |h1h2| <= |h|^2/2 (AM-GM)", zero(expand((h1**2 + h2**2)/2 - h1*h2
                                                - (h1 - h2)**2/2)))
ok("box e^t remainder <= 2t^2 on |t|<=1 (e/2<2)", float(EE/2) < 2)
ok("box ln(1+t) remainder <= 2t^2 on |t|<=1/2: 1/(2(1/2)^2)=2",
   R(1, 2)/(R(1, 2))**2 == 2)
ok("W3 (x^2+y^2)^{1/3}: |h|^{2/3}/h unbounded",
   limit(Abs(h1)**R(2, 3)/h1, h1, 0, "+") == oo)
ok("W3 (x^2+y^2)^{2/3}: quotient -> 0", limit(Abs(h1)**R(4, 3)/h1, h1, 0, "+") == 0)
ok("47 3xy remainder 3h1h2", zero(remainder(3*x*y, a0, b0) - 3*h1*h2))
ok("48 x^2+y^2 remainder |h|^2",
   zero(remainder(x**2 + y**2, a0, b0) - (h1**2 + h2**2)))
ok("49 xy^2 remainder", zero(remainder(x*y**2, a0, b0)
                             - (a0*h2**2 + 2*b0*h1*h2 + h1*h2**2)))
Rt = exp(3*h1) - 1 - 3*h1
ok("50 y e^{3x} remainder e^{3a}(3h1h2 + (b+h2)R(3h1))",
   zero(simplify(remainder(y*exp(3*x), a0, b0)
                 - exp(3*a0)*(3*h1*h2 + (b0 + h2)*Rt))))
ok("51 (a) f(0,h)=4^{1/3}|h|^{2/3}",
   zero(((x**2 + 4*y**2)**R(1, 3)).subs({x: 0, y: h1}) - 4**R(1, 3)*(h1**2)**R(1, 3)))
ok("51 (b) |h|^{3/2}/h -> 0", limit(Abs(h1)**R(3, 2)/h1, h1, 0, "+") == 0)
ok("52 |xy| at (2,0): 2|h|", zero(Abs((2)*h1) - 2*Abs(h1)))
f = x**2*y/(x**4 + y**2)
ok("53 lines -> 0, parabola 1/2",
   limit(f.subs(y, kk*x), x, 0) == 0 and zero(simplify(f.subs(y, x**2)) - R(1, 2)))
f = 2*x*y**2/(x**2 + y**4)
ok("54 lines -> 0, parabola x=y^2 gives 1",
   limit(f.subs(y, kk*x), x, 0) == 0 and zero(simplify(f.subs(x, y**2)) - 1))
f = sqrt((x - 1)**2 + 4*y**2)
ok("56 level c semi-axes c and c/2", zero(f.subs({x: 1 + c, y: 0}) - c)
   and zero(f.subs({x: 1, y: c/2}) - c))

print("--- ch11 Additional 16-36 ---")


def hess(f, vs):
    return Matrix([[diff(f, p, q) for q in vs] for p in vs])


def crit_check(label, f, pts, kinds, vals=None):
    good = True
    for i, (P, kd) in enumerate(zip(pts, kinds)):
        sub = {x: P[0], y: P[1]}
        g0 = [simplify(diff(f, v).subs(sub)) for v in (x, y)]
        H = hess(f, [x, y]).subs(sub)
        d1, d2 = simplify(H[0, 0]), simplify(H.det())
        k = "saddle" if d2 < 0 else ("min" if d2 > 0 and d1 > 0 else
                                     ("max" if d2 > 0 and d1 < 0 else "?"))
        good = good and g0 == [0, 0] and k == kd
        if vals is not None:
            good = good and zero(f.subs(sub) - vals[i])
    ok(label, good)


crit_check("16 (2,-1) min, 3-6ln2", x**2 + x*y + y**2 - 6*log(x),
           [(2, -1)], ["min"], [3 - 6*log(2)])
ok("16 x=-2 is the discarded root",
   set(solve((2*x + y - 6/x).subs(y, -x/2), x)) == {-2, 2})
crit_check("17 (-sqrt2, sqrt2) saddle", x**2 + 2*x*y + 4*log(y),
           [(-sqrt(2), sqrt(2))], ["saddle"], [2*log(2) - 2])
crit_check("18 (1/2,1/2) max -1/2", log(x + y) - x**2 - y**2,
           [(R(1, 2), R(1, 2))], ["max"], [-R(1, 2)])
f = exp(x)*(y**2 - x**2)
crit_check("19 (0,0) saddle, (-2,0) min -4e^-2", f, [(0, 0), (-2, 0)],
           ["saddle", "min"], [0, -4*exp(-2)])
ok("19 f_xx formula", zero(diff(f, x, 2) - exp(x)*(y**2 - x**2 - 4*x - 2)))
ok("19 only these", len(solve([diff(f, x), diff(f, y)], [x, y], dict=True)) == 2)
f = x*y*exp(-(x + y))
crit_check("20 (0,0) saddle, (1,1) max e^-2", f, [(0, 0), (1, 1)],
           ["saddle", "max"], [0, exp(-2)])
ok("20 f_xy = (1-x)(1-y)e^-(x+y)", zero(diff(f, x, y) - (1 - x)*(1 - y)*exp(-(x + y))))
f = (x + y)*exp(-x**2 - y**2)
crit_check("21 (1/2,1/2) max, (-1/2,-1/2) min", f,
           [(R(1, 2), R(1, 2)), (-R(1, 2), -R(1, 2))], ["max", "min"],
           [exp(-R(1, 2)), -exp(-R(1, 2))])
ok("21 H at (1/2,1/2) = e^{-1/2}[[-3,-1],[-1,-3]]",
   simplify(hess(f, [x, y]).subs({x: R(1, 2), y: R(1, 2)})
        - exp(-R(1, 2))*Matrix([[-3, -1], [-1, -3]])) == zeros(2, 2))
f = 3*exp(-(x + 1)**2 - 4*y**2)
crit_check("22 (-1,0) max 3", f, [(-1, 0)], ["max"], [3])
ok("22 H = diag(-6,-24)", hess(f, [x, y]).subs({x: -1, y: 0}) == Matrix([[-6, 0], [0, -24]]))
Xp, Yp = symbols("Xp Yp", positive=True)
f = (Xp*Yp)**R(2, 5)
ok("23 f_x>0, f_xx<0, f_xy=0.16(xy)^-0.6",
   zero(diff(f, Xp) - R(2, 5)*Xp**R(-3, 5)*Yp**R(2, 5))
   and zero(diff(f, Xp, 2) + R(6, 25)*Xp**R(-8, 5)*Yp**R(2, 5))
   and zero(diff(f, Xp, Yp) - R(4, 25)*(Xp*Yp)**R(-3, 5)))
f = log(Xp + 2)*exp(-Yp)
ok("24 signs", zero(diff(f, Xp) - exp(-Yp)/(Xp + 2))
   and zero(diff(f, Xp, 2) + exp(-Yp)/(Xp + 2)**2)
   and zero(diff(f, Yp) + log(Xp + 2)*exp(-Yp)) and zero(diff(f, Yp, 2) - f))
f = Yp*exp(Xp**2/2)
ok("25 f_xx = y(1+x^2)e^{x^2/2}, f_yy=0",
   zero(diff(f, Xp, 2) - Yp*(1 + Xp**2)*exp(Xp**2/2)) and diff(f, Yp, 2) == 0)
f = Xp**R(3, 5)*Yp**R(2, 3)
H = hess(f, [Xp, Yp])
ok("26 f_xx<0, f_yy<0, det = -(8/75) x^-4/5 y^-2/3",
   zero(H[0, 0] + R(6, 25)*Xp**R(-7, 5)*Yp**R(2, 3))
   and zero(H[1, 1] + R(2, 9)*Xp**R(3, 5)*Yp**R(-4, 3))
   and zero(simplify(H.det() + R(8, 75)*Xp**R(-4, 5)*Yp**R(-2, 3))))
ok("26 f(t,t) = t^{19/15}", zero(simplify(f.subs({Xp: tt, Yp: tt}) - tt**R(19, 15))))
Al, Be, A0 = symbols("alpha beta A0", positive=True)
Yc = A0*L**Al*K**Be
Hc = hess(Yc, [L, K])
ok("box CD ratio forms and det", zero(diff(Yc, L) - Al*Yc/L)
   and zero(Hc[0, 0] - Al*(Al - 1)*Yc/L**2) and zero(Hc[0, 1] - Al*Be*Yc/(L*K))
   and zero(simplify(Hc.det() - Al*Be*(1 - Al - Be)*Yc**2/(L**2*K**2))))
Y5 = 5*L**R(3, 5)*K**R(3, 10)
H5 = hess(Y5, [L, K])
ok("27 det = 0.018 Y^2/(LK)^2, Y_LL=-0.24Y/L^2, Y_KK=-0.21Y/K^2",
   zero(simplify(H5.det() - R(18, 1000)*Y5**2/(L**2*K**2)))
   and zero(H5[0, 0] + R(24, 100)*Y5/L**2) and zero(H5[1, 1] + R(21, 100)*Y5/K**2))
ok("27 5*1.01^0.6 = 5.02994", close(5*R(101, 100)**R(3, 5), 5.02994, 1e-5))
f = x**2 + y**4
ok("28 H(0)=diag(2,0)", hess(f, [x, y]).subs({x: 0, y: 0}) == Matrix([[2, 0], [0, 0]]))
f = x**2 + y**3
ok("29 same H, f(0,y)=y^3", hess(f, [x, y]).subs({x: 0, y: 0}) == Matrix([[2, 0], [0, 0]]))
gt = tt**4 - 4*tt**3
tr = symbols("tr", real=True)
gtr = tr**4 - 4*tr**3
ok("30 g'=4t^2(t-3), g''(3)=36, g(3)=-27",
   zero(diff(gtr, tr) - 4*tr**2*(tr - 3)) and diff(gtr, tr, 2).subs(tr, 3) == 36
   and gtr.subs(tr, 3) == -27)
f = (x - y)**4 - 4*(x - y)**3
ok("30 grad zero on y=x and y=x-3",
   zero(diff(f, x).subs(y, x)) and zero(diff(f, x).subs(y, x - 3))
   and zero(diff(f, y).subs(y, x - 3)))
f = x**3 - 3*x*y**2
r_, th = symbols("r theta", real=True)
ok("31 H(0)=0, polar r^3 cos3theta",
   hess(f, [x, y]).subs({x: 0, y: 0}) == zeros(2, 2)
   and zero(simplify(f.subs({x: r_*cos(th), y: r_*sin(th)}) - r_**3*cos(3*th))))
x1, x2, x3 = symbols("x1 x2 x3", real=True)
A = Matrix([[1, 2], [0, 3]])
xv = Matrix([x1, x2])
f = (xv.T*A*xv)[0]
ok("33 H = A+A^T = [[2,2],[2,6]], PD",
   hess(f, [x1, x2]) == A + A.T == Matrix([[2, 2], [2, 6]]) and (A + A.T).det() == 8)
nn = 4
Xs = symbols("y1:5", positive=True)
f = sum((i + 1)*Xs[i]**2 - log(Xs[i]) for i in range(nn))
ok("34 H diag(2k+1/x_k^2), crit x_k=1/sqrt(2k)",
   hess(f, list(Xs)) == Matrix.diag(*[2*(i + 1) + 1/Xs[i]**2 for i in range(nn)])
   and all(zero(diff(f, Xs[i]).subs(Xs[i], 1/sqrt(2*(i + 1)))) for i in range(nn)))
A = Matrix([[1, 1, 0], [3, 2, 0], [0, 4, 1]])
bv = Matrix([1, 0, -1])
xv = Matrix([x1, x2, x3])
f = (xv.T*A*xv)[0] + (bv.T*xv)[0]
Hm = hess(f, [x1, x2, x3])
sol = solve([diff(f, v) for v in (x1, x2, x3)], [x1, x2, x3])
ok("35 H=[[2,4,0],[4,4,4],[0,4,2]], D2=-8, crit (-1/2,0,1/2)",
   Hm == Matrix([[2, 4, 0], [4, 4, 4], [0, 4, 2]]) and Hm[:2, :2].det() == -8
   and sol == {x1: -R(1, 2), x2: 0, x3: R(1, 2)})
f = sum(xv)**2
ok("36 H(sum^2) = 2J, v=(1,-1,0) null", hess(f, [x1, x2, x3]) == 2*Matrix.ones(3, 3)
   and (Matrix([1, -1, 0]).T*hess(f, [x1, x2, x3])*Matrix([1, -1, 0]))[0] == 0)

print("--- ch12 Additional 16-18 ---")


def total(f, vs, P, d):
    sub = dict(zip(vs, P))
    Q = dict(zip(vs, [p + dd for p, dd in zip(P, d)]))
    df = sum(diff(f, v).subs(sub)*dd for v, dd in zip(vs, d))
    return f.subs(sub), [diff(f, v).subs(sub) for v in vs], df, f.subs(Q) - f.subs(sub)


v0, g_, df, ex = total(4*sqrt(x**2 + y*z), [x, y, z], [3, 2, 8],
                       [R(1, 10), -R(2, 10), R(3, 10)])
ok("16 Q=20, grad (12/5,16/5,4/5), dQ=-0.16, exact -0.180817",
   v0 == 20 and g_ == [R(12, 5), R(16, 5), R(4, 5)] and df == -R(16, 100)
   and close(ex, -0.180817, 1e-6))
v0, g_, df, ex = total(2*sqrt(x*y + z**2), [x, y, z], [2, 8, 3],
                       [R(5, 100), R(1, 10), -R(5, 100)])
ok("17 Y=10, grad (8/5,2/5,6/5), dY=0.06, exact 0.061312",
   v0 == 10 and g_ == [R(8, 5), R(2, 5), R(6, 5)] and df == R(6, 100)
   and close(ex, 0.061312, 1e-6))
f = log(x**2*y) - x*y
v0, g_, df, ex = total(f, [x, y], [1, 2], [R(1, 10), R(1, 10)])
H = hess(f, [x, y]).subs({x: 1, y: 2})
q2 = R(1, 2)*(Matrix([[R(1, 10), R(1, 10)]])*H*Matrix([R(1, 10), R(1, 10)]))[0]
ok("18 grad (0,-1/2), df=-0.05, exact -0.070589, 2nd order ~ -0.021",
   g_ == [0, -R(1, 2)] and df == -R(5, 100) and close(ex, -0.070589, 1e-6)
   and close(q2, -0.02125, 1e-9) and H == Matrix([[-2, -1], [-1, -R(1, 4)]]))

print("--- ch13 Additional 11-42 ---")


def grad2(f, P):
    return [diff(f, v).subs({x: P[0], y: P[1]}) for v in (x, y)]


f = x**2*y + y**3
ok("11 f=10, grad (4,13), D_u along (3,-4) = -8",
   f.subs({x: 1, y: 2}) == 10 and grad2(f, (1, 2)) == [4, 13]
   and R(3, 5)*4 - R(4, 5)*13 == -8)
f = log(x**2 + 3*y)
gg = grad2(f, (2, 0))
ok("12 grad (1,3/4), |grad|=5/4", gg == [1, R(3, 4)] and sqrt(gg[0]**2 + gg[1]**2) == R(5, 4))
f = (x + y)*exp(x - y)
gg = grad2(f, (1, 1))
ok("13 grad (3,-1), D_u = sqrt5, grad.h=5",
   gg == [3, -1] and zero((2*gg[0] + gg[1])/sqrt(5) - sqrt(5)) and 2*gg[0] + gg[1] == 5)
Qf = 4*K**R(1, 2)*L**R(1, 4)
ok("14 Q=32, Q_L=1/2, Q(16,17)~32.4887",
   Qf.subs({K: 16, L: 16}) == 32 and diff(Qf, L).subs({K: 16, L: 16}) == R(1, 2)
   and close(Qf.subs({K: 16, L: 17}), 32.4887, 1e-4))
f = sqrt(2*x**2 + 3*y + 1)
gg = grad2(f, (1, 2))
ok("15 c=3, y=(8-2x^2)/3, grad (2/3,1/2) _|_ (3,-4), |grad|=5/6",
   f.subs({x: 1, y: 2}) == 3 and zero(f.subs(y, (8 - 2*x**2)/3) - 3)
   and gg == [R(2, 3), R(1, 2)] and 3*gg[0] - 4*gg[1] == 0
   and sqrt(gg[0]**2 + gg[1]**2) == R(5, 6))
f = Xp**R(1, 3)*Yp**R(2, 3)
gp = [diff(f, v).subs({Xp: 1, Yp: 8}) for v in (Xp, Yp)]
ok("16 c=4, y=8x^-1/2, grad (4/3,1/3) _|_ (1,-4), |grad|=sqrt17/3",
   f.subs({Xp: 1, Yp: 8}) == 4 and zero(simplify(f.subs(Yp, 8/sqrt(Xp)) - 4))
   and gp == [R(4, 3), R(1, 3)] and gp[0] - 4*gp[1] == 0
   and zero(sqrt(gp[0]**2 + gp[1]**2) - sqrt(17)/3))
f = x*exp(y) + y
gg = grad2(f, (2, 0))
xy_ = (2 - y)*exp(-y)
ok("17 x=(2-y)e^-y, dx/dy=-3, grad (1,3), |grad|=sqrt10",
   zero(f.subs(x, xy_) - 2) and diff(xy_, y).subs(y, 0) == -3 and gg == [1, 3]
   and -3*gg[0] + 1*gg[1] == 0)
Fv = Matrix([x**2*y, x - 3*y**2, x*y])
J = Fv.jacobian([x, y])
rem = expand(Fv.subs({x: 1 + h1, y: 1 + h2}, simultaneous=True) - Fv.subs({x: 1, y: 1})
             - J.subs({x: 1, y: 1})*Matrix([h1, h2]))
ok("18 J(1,1) and remainder components",
   J.subs({x: 1, y: 1}) == Matrix([[2, 1], [1, -6], [1, 1]])
   and zero(rem[0] - (2*h1*h2 + h1**2 + h1**2*h2)) and zero(rem[1] + 3*h2**2)
   and zero(rem[2] - h1*h2))
ok("19 first row", zero(diff(log(3*x**2 + y**4), x) - 6*x/(3*x**2 + y**4))
   and zero(diff(log(3*x**2 + y**4), y) - 4*y**3/(3*x**2 + y**4)))
rv = Matrix([log(1 + t**2), exp(-2*t), cos(3*t), 5])
ok("21 r'(0)=(0,-2,0,0)", rv.diff(t).subs(t, 0) == Matrix([0, -2, 0, 0]))
Fv = Matrix([x*y*z, x**2 - z])
x4 = symbols("x4", real=True)
gq = x1*x2 + x3*exp(x4)
ok("22 J(1,2,3)=[[6,3,2],[2,0,-1]], grad g=(1,1,1,1)",
   Fv.jacobian([x, y, z]).subs({x: 1, y: 2, z: 3}) == Matrix([[6, 3, 2], [2, 0, -1]])
   and [diff(gq, v).subs({x1: 1, x2: 1, x3: 1, x4: 0}) for v in (x1, x2, x3, x4)] == [1, 1, 1, 1])
Fv = Matrix([s + t, s*t, s**2, exp(s - t)])
ok("23 J(1,1)", Fv.jacobian([s, t]).subs({s: 1, t: 1})
   == Matrix([[1, 1], [1, 1], [2, 0], [1, -1]]))
Xn = symbols("z1:6", real=True)
Fn = Matrix([(k + 1)*exp(Xn[k]) for k in range(5)])
ok("24 (b) diagonal", Fn.jacobian(list(Xn)) == Matrix.diag(*[(k + 1)*exp(Xn[k]) for k in range(5)]))
rn = Matrix([t**(k + 1)/(k + 1) for k in range(5)])
ok("24 (c) column t^{k-1}", rn.diff(t) == Matrix([t**k for k in range(5)]))
fn = sum(Xn[k]**(k + 1)/(k + 1) for k in range(5))
ok("24 (d) row x_k^{k-1}", [diff(fn, Xn[k]) for k in range(5)] == [Xn[k]**k for k in range(5)])
A = Matrix([[2, -1, 0], [1, 0, 3], [0, 4, -2]])
xv = Matrix([x1, x2, x3])
av = Matrix(symbols("a1:4", real=True))
ok("25 J(Ax)=A, J(3x-2a)=3I", (A*xv).jacobian([x1, x2, x3]) == A
   and (3*xv - 2*av).jacobian([x1, x2, x3]) == 3*eye(3))
u_, v_ = symbols("u_ v_", real=True)
fI = Matrix([x + y*z, x*z])
gO = Matrix([u_, v_, u_*v_, u_ - v_])
P = {x: 1, y: 2, z: 1}
Jc = gO.jacobian([u_, v_]).subs({u_: fI[0].subs(P), v_: fI[1].subs(P)})*fI.jacobian([x, y, z]).subs(P)
comp = gO.subs({u_: fI[0], v_: fI[1]}, simultaneous=True)
ok("27 chain = compose = [[1,1,2],[1,0,1],[4,1,5],[0,1,1]]",
   Jc == Matrix([[1, 1, 2], [1, 0, 1], [4, 1, 5], [0, 1, 1]])
   and comp.jacobian([x, y, z]).subs(P) == Jc)
gS = x + 2*y + z**2
fC = Matrix([s, s**2, exp(s)])
comp = fC.subs(s, gS)
ok("28 J(f o g)(1,0,0) = [[1,2,0],[2,4,0],[e,2e,0]]",
   comp.jacobian([x, y, z]).subs({x: 1, y: 0, z: 0})
   == Matrix([[1, 2, 0], [2, 4, 0], [EE, 2*EE, 0]]))
ok("28 (g o f)' = 1+4s+2e^{2s}",
   zero(diff(gS.subs({x: fC[0], y: fC[1], z: fC[2]}, simultaneous=True), s)
        - (1 + 4*s + 2*exp(2*s))))
w_ = symbols("w_", real=True)
fA = Matrix([x, y, x*y])
gB = Matrix([u_ + w_, v_*w_])
gf = gB.subs({u_: fA[0], v_: fA[1], w_: fA[2]}, simultaneous=True)
fg = fA.subs({x: gB[0], y: gB[1]}, simultaneous=True)
ok("29 J(g o f)(1,2)=[[3,1],[4,4]], J(f o g)(1,1,1)=[[1,0,1],[0,1,1],[1,2,3]]",
   gf.jacobian([x, y]).subs({x: 1, y: 2}) == Matrix([[3, 1], [4, 4]])
   and fg.jacobian([u_, v_, w_]).subs({u_: 1, v_: 1, w_: 1})
   == Matrix([[1, 0, 1], [0, 1, 1], [1, 2, 3]]))
ok("30 d/dt = 2t", zero(diff((cos(t)**2 + sin(t)**2 + t**2), t) - 2*t))


def sdd(f, vs, P, h):
    sub = dict(zip(vs, P))
    gv = Matrix([diff(f, v) for v in vs]).subs(sub)
    Hm = hess(f, vs).subs(sub)
    hv = Matrix(h)
    phi = f.subs({v: p + t*hh for v, p, hh in zip(vs, P, h)}, simultaneous=True)
    return ((gv.T*hv)[0], (hv.T*Hm*hv)[0], simplify(diff(phi, t).subs(t, 0)),
            simplify(diff(phi, t, 2).subs(t, 0)), Hm)


gh, hHh, p1, p2, Hm = sdd(x*log(y) + y*z**2, [x, y, z], [2, 1, 1], [1, 2, -2])
ok("31 grad.h=2, hHh=-12, profile agrees, H",
   gh == 2 and hHh == -12 and p1 == 2 and p2 == -12
   and Hm == Matrix([[0, 1, 0], [1, -2, 2], [0, 2, 2]]))
gh, hHh, p1, p2, Hm = sdd(x*exp(y*z), [x, y, z], [1, 0, 2], [2, 1, 2])
ok("32 grad.h=4, hHh=16", gh == 4 and hHh == 16 and p1 == 4 and p2 == 16
   and Hm == Matrix([[0, 2, 0], [2, 4, 1], [0, 1, 0]]))
gh, hHh, p1, p2, Hm = sdd(x*y*z + log(x*z), [x, y, z], [1, 1, 1], [1, -1, 1])
ok("33 grad.h=3, hHh=-4", gh == 3 and hHh == -4 and p1 == 3 and p2 == -4
   and Hm == Matrix([[-1, 1, 1], [1, 0, 1], [1, 1, -1]]))
gh, hHh, p1, p2, Hm = sdd(x**2*y, [x, y], [1, 2], [3, 4])
ok("34 16 and 84 = 25*84/25", gh == 16 and hHh == 84 and p1 == 16 and p2 == 84
   and expand((1 + 3*t)**2*(2 + 4*t)) == 2 + 16*t + 42*t**2 + 36*t**3)
gh, hHh, p1, p2, Hm = sdd(x**3 - 3*x + y**2, [x, y], [-1, 1], [3, 4])
ok("35 D_u=8/5, D2=-22/25", gh/5 == R(8, 5) and hHh/25 == -R(22, 25))
f = x**4 - 4*x*y + y**2
ok("36 crit pts and D2 values",
   set(map(tuple, [tuple(s_.values()) for s_ in solve([diff(f, x), diff(f, y)], [x, y], dict=True)]))
   == {(0, 0), (sqrt(2), 2*sqrt(2)), (-sqrt(2), -2*sqrt(2))}
   and hess(f, [x, y]).subs({x: sqrt(2), y: 2*sqrt(2)}).det() == 32
   and zero(f.subs({x: sqrt(2), y: 2*sqrt(2)}) + 4)
   and (Matrix([[1, 1]])*hess(f, [x, y]).subs({x: 0, y: 0})*Matrix([1, 1]))[0]/2 == -3)
f = x*log(y)
Hm = hess(f, [x, y]).subs({x: 1, y: 1})
ok("37 H=[[0,1],[1,-1]], D2 along (1,1)/sqrt2 = 1/2",
   Hm == Matrix([[0, 1], [1, -1]])
   and (Matrix([[1, 1]])*Hm*Matrix([1, 1]))[0]/2 == R(1, 2))
ok("40 (a) 3D counterexample f=z: D_e1=D_e2=0, grad=(0,0,1)",
   [diff(z, v) for v in (x, y, z)] == [0, 0, 1])
YY = 10*K**R(2, 5)*L**R(3, 5)
P = {K: 32768, L: 7776}
YK, YL = diff(YY, K).subs(P), diff(YY, L).subs(P)
nr = sqrt(YK**2 + YL**2)
ok("41 Y=138240, Y_K=27/16, Y_L=32/3, ratio 81:512, |grad|~10.7993",
   YY.subs(P) == 138240 and YK == R(27, 16) and YL == R(32, 3)
   and YK/YL == R(81, 512) and close(nr, 10.7993, 1e-4))
ok("41 amounts 15.63, 98.77; dY 1079.9; exact 1077.5; budget 1066.7/1063.9; sum 114.4",
   close(100*YK/nr, 15.63, 5e-3) and close(100*YL/nr, 98.77, 5e-3)
   and close(100*nr, 1079.9, 5e-2)
   and close(YY.subs({K: 32768 + 100*YK/nr, L: 7776 + 100*YL/nr}) - 138240, 1077.5, 5e-2)
   and close(100*YL, 1066.7, 5e-2) and close(YY.subs({K: 32768, L: 7876}) - 138240, 1063.9, 5e-2)
   and close(100*(YK + YL)/nr, 114.4, 5e-2))
YY = sqrt(K*L)
P = {K: 400, L: 900}
YK, YL = diff(YY, K).subs(P), diff(YY, L).subs(P)
nr = sqrt(YK**2 + YL**2)
ok("42 numbers",
   YK == R(3, 4) and YL == R(1, 3) and close(nr, 0.82074, 1e-5)
   and close(20*YK/nr, 18.28, 5e-3) and close(20*YL/nr, 8.12, 5e-3)
   and close(20*nr, 16.41, 5e-3)
   and close(YY.subs({K: 400 + 20*YK/nr, L: 900 + 20*YL/nr}) - 600, 16.32, 5e-3)
   and close(YY.subs({K: 420, L: 900}) - 600, 14.82, 5e-3))

print("--- ch14 worked examples 3-4 and Additional 26-47 ---")
uu, vv, ww = symbols("uu vv ww", real=True)
Fs = Matrix([x*y - uu, x + y - vv])
My = Fs.jacobian([x, y]).subs({x: 1, y: 2})
Mx = Fs.jacobian([uu, vv])
JG = -My.inv()*Mx
xe = (vv - sqrt(vv**2 - 4*uu))/2
ye = (vv + sqrt(vv**2 - 4*uu))/2
Je = Matrix([xe, ye]).jacobian([uu, vv]).subs({uu: 2, vv: 3})
ok("W3 J_G = [[1,-1],[-1,2]] both ways",
   JG == Matrix([[1, -1], [-1, 2]]) and simplify(Je - JG) == zeros(2, 2)
   and xe.subs({uu: 2, vv: 3}) == 1 and ye.subs({uu: 2, vv: 3}) == 2)
Al_ = Matrix([[1, 2, -1, 1], [2, 4, 1, -1]])
ok("W3 linear minors (x1,x2)=0, (x1,x3)=3, (x3,x4)=0",
   Al_[:, [0, 1]].det() == 0 and Al_[:, [0, 2]].det() == 3 and Al_[:, [2, 3]].det() == 0)
Fm = Matrix([x**2 - y, exp(y)])
Gm = Matrix([-sqrt(uu + log(vv)), log(vv)])
ok("W4 G(1,1)=P, J_G(1,1) = J_F(P)^-1",
   Gm.subs({uu: 1, vv: 1}) == Matrix([-1, 0])
   and Gm.jacobian([uu, vv]).subs({uu: 1, vv: 1}) == Fm.jacobian([x, y]).subs({x: -1, y: 0}).inv()
   and simplify(Fm.subs({x: Gm[0], y: Gm[1]}, simultaneous=True) - Matrix([uu, vv])) == zeros(2, 1))
A26 = Matrix([[1, -1, 2, 1], [2, -2, 1, -1]])
from itertools import combinations
mins = {p: A26[:, list(p)].det() for p in combinations(range(4), 2)}
ok("26 minors", mins == {(0, 1): 0, (0, 2): -3, (0, 3): -3, (1, 2): 3, (1, 3): 3, (2, 3): -3})
JG = -A26[:, [2, 3]].inv()*A26[:, [0, 1]]
sol = solve([(A26*Matrix([x1, x2, x3, x4]))[0] - 1, (A26*Matrix([x1, x2, x3, x4]))[1] - 5], [x3, x4])
ok("26 J_G=[[-1,1],[1,-1]], x3=2-x1+x2, x4=-3+x1-x2",
   JG == Matrix([[-1, 1], [1, -1]]) and zero(sol[x3] - (2 - x1 + x2)) and zero(sol[x4] - (-3 + x1 - x2)))
Fs = Matrix([uu + vv**2 - x, sin(uu) + vv - y])
pt = {x: 1, y: 1, uu: 0, vv: 1}
ok("27 point, det -1, J=[[-1,2],[1,-1]]",
   Fs.subs(pt) == zeros(2, 1) and Fs.jacobian([uu, vv]).subs(pt).det() == -1
   and -Fs.jacobian([uu, vv]).subs(pt).inv()*Fs.jacobian([x, y]).subs(pt)
   == Matrix([[-1, 2], [1, -1]]))
p1_, p2_, aa, bb = symbols("p1 p2 aa bb", real=True)
Es = Matrix([aa - 3*p1_ + p2_, bb + p1_ - 3*p2_])
sol = solve(list(Es), [p1_, p2_])
ok("28 p=(35/8,25/8), J=(1/8)[[3,1],[1,3]]",
   sol[p1_].subs({aa: 10, bb: 5}) == R(35, 8) and sol[p2_].subs({aa: 10, bb: 5}) == R(25, 8)
   and -Es.jacobian([p1_, p2_]).inv()*Es.jacobian([aa, bb]) == Matrix([[3, 1], [1, 3]])/8)
Fs = Matrix([x**2 + y**2 - z, x + y - z*ww])
pt = {x: 1, y: 1, z: 2, ww: 1}
Jp = Fs.jacobian([x, y, z, ww]).subs(pt)
mins = {p: Jp[:, list(p)].det() for p in combinations(range(4), 2)}
JG = -Jp[:, [2, 3]].inv()*Jp[:, [0, 1]]
we = (x + y)/(x**2 + y**2)
ok("29 minors and J for (z,w)",
   Fs.subs(pt) == zeros(2, 1)
   and mins == {(0, 1): 0, (0, 2): -1, (0, 3): -4, (1, 2): -1, (1, 3): -4, (2, 3): 2}
   and JG == Matrix([[2, 2], [-R(1, 2), -R(1, 2)]])
   and [diff(we, v).subs({x: 1, y: 1}) for v in (x, y)] == [-R(1, 2), -R(1, 2)])


def inv_check(label, F, vs, P, G, us):
    sub = dict(zip(vs, P))
    FP = [simplify(c.subs(sub)) for c in F]
    usub = dict(zip(us, FP))
    back = [simplify(c.subs(usub)) for c in G]
    JG = Matrix(G).jacobian(us).subs(usub)
    JF = Matrix(F).jacobian(vs).subs(sub)
    ok(label, back == list(P) and simplify(JG - JF.inv()) == zeros(len(vs), len(vs)))
    return FP, JG


FP, JG = inv_check("30 (3s, s^2+t^2) at (1,-2)", [3*s, s**2 + t**2], [s, t], (1, -2),
                   [uu/3, -sqrt(vv - uu**2/9)], [uu, vv])
ok("30 F(P)=(3,5), J_G=[[1/3,0],[1/6,-1/4]]", FP == [3, 5]
   and JG == Matrix([[R(1, 3), 0], [R(1, 6), -R(1, 4)]]))
FP, JG = inv_check("31 (xy, y^2) at (2,-1)", [x*y, y**2], [x, y], (2, -1),
                   [-uu/sqrt(vv), -sqrt(vv)], [uu, vv])
ok("31 J_G=[[-1,-1],[0,-1/2]]", JG == Matrix([[-1, -1], [0, -R(1, 2)]]))
vp = symbols("vp", positive=True)
FP, JG = inv_check("32 (ln x + y, y^3) at (1,1)", [log(x) + y, y**3], [x, y], (1, 1),
                   [exp(uu - vp**R(1, 3)), vp**R(1, 3)], [uu, vp])
ok("32 J_G=[[1,-1/3],[0,1/3]]", JG == Matrix([[1, -R(1, 3)], [0, R(1, 3)]]))
FP, JG = inv_check("33 3D triangular at (0,2,1)", [x, x + y**2, z*exp(x)], [x, y, z], (0, 2, 1),
                   [uu, sqrt(vv - uu), ww*exp(-uu)], [uu, vv, ww])
ok("33 F(P)=(0,4,1), J_G", FP == [0, 4, 1]
   and JG == Matrix([[1, 0, 0], [-R(1, 4), R(1, 4), 0], [-1, 0, 1]]))
FP, JG = inv_check("34 (x^2-y^2, x+y) at (2,1)", [x**2 - y**2, x + y], [x, y], (2, 1),
                   [(vv + uu/vv)/2, (vv - uu/vv)/2], [uu, vv])
ok("34 J_G=[[1/6,1/3],[-1/6,2/3]]", JG == Matrix([[R(1, 6), R(1, 3)], [-R(1, 6), R(2, 3)]]))
Fv = Matrix([x + y*z, exp(y), x*z**2])
J = Fv.jacobian([x, y, z])
ok("35 det = z e^y (2x - yz), J(Q1)^-1, det(Q2)=0",
   zero(simplify(J.det() - z*exp(y)*(2*x - y*z)))
   and J.subs({x: 1, y: 0, z: 1}).inv() == Matrix([[1, -1, 0], [0, 1, 0], [-R(1, 2), R(1, 2), R(1, 2)]])
   and simplify(J.subs({x: 1, y: 2, z: 1}).det()) == 0)
Fv = Matrix([z**3 + z, exp(2*x), y**3 + 3*y])
J = Fv.jacobian([x, y, z]).subs({x: 0, y: 1, z: 1})
ok("36 F(Q)=(2,1,4), det 48, inverse", Fv.subs({x: 0, y: 1, z: 1}) == Matrix([2, 1, 4])
   and J.det() == 48 and J.inv() == Matrix([[0, R(1, 2), 0], [0, 0, R(1, 6)], [R(1, 4), 0, 0]]))
Fv = Matrix([x + 2*y, 3*x + y**2, z*exp(z)])
J = Fv.jacobian([x, y, z])
ok("37 det, inverse at Q1, det(Q2)=0",
   zero(simplify(J.det() - (2*y - 6)*(1 + z)*exp(z)))
   and J.subs({x: 0, y: 1, z: 0}).inv() == Matrix([[-R(1, 2), R(1, 2), 0], [R(3, 4), -R(1, 4), 0], [0, 0, 1]])
   and J.subs({x: 1, y: 3, z: 0}).det() == 0 and Fv.subs({x: 0, y: 1, z: 0}) == Matrix([2, 1, 0]))
Fe = x**2 - 2*x + y*z + z**2 - 5
ye_ = (5 - x**2 + 2*x - z**2)/z
pt = {x: 1, y: 1, z: 2}
ok("38 partials and explicit y",
   Fe.subs(pt) == 0 and [diff(Fe, v).subs(pt) for v in (x, y, z)] == [0, 2, 5]
   and diff(ye_, x).subs({x: 1, z: 2}) == 0 and diff(ye_, z).subs({x: 1, z: 2}) == -R(5, 2)
   and zero(expand((x - 1)**2 - (6 - y*z - z**2) - Fe)))
Zp = symbols("Zp", positive=True)
fc = Xp**R(1, 2)*Yp**R(1, 4)*Zp**R(1, 4)
ze = 16/(Xp**2*Yp)
ok("39 f(P)=2, IFT grad z = (-4,-4) = explicit",
   zero(fc.subs({Xp: 2, Yp: 1, Zp: 4}) - 2)
   and zero((-diff(fc, Xp)/diff(fc, Zp)).subs({Xp: 2, Yp: 1, Zp: 4}) + 4)
   and zero((-diff(fc, Yp)/diff(fc, Zp)).subs({Xp: 2, Yp: 1, Zp: 4}) + 4)
   and diff(ze, Xp).subs({Xp: 2, Yp: 1}) == -4 and diff(ze, Yp).subs({Xp: 2, Yp: 1}) == -4
   and zero(simplify(fc.subs(Zp, ze) - 2)))
uc = symbols("uc", real=True)
ok("41 inverse of ((x-1)^3, x+y)",
   zero(((1 + Xp**R(1, 3)) - 1)**3 - Xp) and zero(x + (vv - x) - vv)
   and zero(Matrix([(x - 1)**3, x + y]).jacobian([x, y]).det() - 3*(x - 1)**2))
Fv = Matrix([(2*s - t)**3, s + t])
sI = (vv + uu**R(1, 3))/3
tI = (2*vv - uu**R(1, 3))/3
ok("42 det 9(2s-t)^2, inverse checks",
   zero(expand(Fv.jacobian([s, t]).det() - 9*(2*s - t)**2))
   and zero(simplify(2*sI - tI - uu**R(1, 3))) and zero(simplify(sI + tI - vv)))
ok("45 factorisations", zero(expand(2 + 3*x - x**3 - (x + 1)**2*(2 - x)))
   and zero(expand(-2 + 3*x - x**3 + (x - 1)**2*(x + 2)))
   and zero(diff((x + 1)*sqrt(2 - x), x).subs(x, -1) - sqrt(3)))
Fv = Matrix([x*y, y/x, log(z) + x])
Ginv = [sqrt(uu/vv), sqrt(uu*vv), exp(ww - sqrt(uu/vv))]
up, vq = symbols("up vq", positive=True)
Ginv = [sqrt(up/vq), sqrt(up*vq), exp(ww - sqrt(up/vq))]
ok("47 F(G(u,v,w)) = (u,v,w), det = 2y/(xz)",
   simplify(Fv.subs({x: Ginv[0], y: Ginv[1], z: Ginv[2]}, simultaneous=True) - Matrix([up, vq, ww])) == zeros(3, 1)
   and zero(simplify(Fv.jacobian([x, y, z]).det() - 2*y/(x*z))))

print(f"\n{count - bad}/{count} PASS, {bad} FAIL")
raise SystemExit(1 if bad else 0)
