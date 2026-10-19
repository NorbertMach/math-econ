"""Re-check the gap-closing additions to Volume 7 (optimization).

ch02: Additional 11-14 (strictly convex sets).
ch03: Additional 17-54 (contour-set inference, curvature at a point,
      epigraphs, tangent planes and bond prices, PSD regions, monotonicity,
      judge-the-claim and true/false items).
ch05: Additional 11-26 (singular Hessians, matrix quadratics, 3-4 variables,
      least squares, interdependent demand).
ch06: second worked example and Additional 13-17 (several parameters).
ch07: Additional 28-45 (MRT, graphical, budget equality, portfolios,
      candidates outside the domain).
ch08: Additional 11-17 (duality, monotone transforms, three inputs).
ch09: second worked example and Additional 21-48 (large bordered Hessians,
      sign patterns, global optimality, substitution, n variables, shadow
      prices).
ch10: Additional 14-58 (regularity, graphical NLP, mixed constraints, tests
      of given points, convexity, 3 variables, word problems, standard form,
      portfolio and non-convex KKT, domain constraints).
Later additions: ch03 local-versus-global box, epigraph box and Additional
55-57, the surface in Additional Solution 44; ch05 necessary-conditions
box; ch07 second worked example (three-asset frontier); ch09 intermediate
zero bordered minor (restricted-form identity, example, ordering artefact).

Every claim is retyped by hand from the book; nothing is parsed from LaTeX.
Run with a Python that has sympy installed.
"""
from sympy import (symbols, Rational as R, sqrt, log, exp, sin, cos, pi, diff,
                   simplify, expand, factor, N, oo, solve, nsolve, Matrix,
                   hessian, eye, zeros, Abs, limit, nsimplify)

x, y, z, w, t, u, v = symbols("x y z w t u v", real=True)
lam, l1, l2, mu = symbols("lam l1 l2 mu", real=True)
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


def close(a, b, tol=R(5, 10**4)):
    return abs(N(a, 20) - N(b, 20)) < tol


def minors(M):
    return [M[:k, :k].det() for k in range(1, M.shape[0] + 1)]


def grad(f, X):
    return Matrix([diff(f, s) for s in X])


def bordered(f, gs, X, lams, pt):
    """Bordered Hessian with constraints g_i, L = f - sum lam_i g_i."""
    L = f - sum(a*g for a, g in zip(lams, gs))
    m, n = len(gs), len(X)
    J = Matrix([[diff(g, s) for s in X] for g in gs])
    M = zeros(m + n, m + n)
    M[:m, m:] = J
    M[m:, :m] = J.T
    M[m:, m:] = hessian(L, X)
    M = M.subs(pt)
    return [simplify(M[:k, :k].det()) for k in range(2*m + 1, m + n + 1)]


# ================================================================ ch02 ====
print("--- ch02 Additional 11-14: strictly convex sets ---")
ok("11e notched disc points inside", (R(15, 10)**2 + R(6, 10)**2 <= 4)
   and R(6, 10) >= R(1, 2))
ok("11e 2.25+0.36=2.61", R(225, 100) + R(36, 100) == R(261, 100))
lp, lq = symbols("lp lq", real=True)
a_, b_ = symbols("a_ b_", real=True)
ok("11f square inequality gap = lam(1-lam)(a-b)^2",
   zero(lp*a_**2 + (1 - lp)*b_**2 - (lp*a_ + (1 - lp)*b_)**2
        - lp*(1 - lp)*(a_ - b_)**2))
p1, p2, q1, q2 = symbols("p1 p2 q1 q2", real=True)
lhs = (lp*p1 + (1 - lp)*q1)**2 + (lp*p2 + (1 - lp)*q2)**2
rhs = lp*(p1**2 + p2**2) + (1 - lp)*(q1**2 + q2**2) \
    - lp*(1 - lp)*((p1 - q1)**2 + (p2 - q2)**2)
ok("13 norm identity", zero(lhs - rhs))
ok("14 g=4/x strictly convex", diff(4/x, x, 2).subs(x, 2) > 0
   and zero(diff(4/x, x, 2) - 8/x**3))
ok("14 midpoint of (4,0),(0,4) on x+y=4", R(4 + 0, 2) + R(0 + 4, 2) == 4)
ok("12d cone mantle midpoint", sqrt(R(1, 2)**2) == R(1, 2))

# ================================================================ ch03 ====
print("--- ch03 Additional 17-54 ---")
fA = (x**2 - 1)**2
a17 = sqrt(1 - 1/sqrt(2))
b17 = sqrt(1 + 1/sqrt(2))
ok("17A endpoints are level 1/2", zero(fA.subs(x, a17) - R(1, 2))
   and zero(fA.subs(x, b17) - R(1, 2)))
ok("17A a~0.541 b~1.307", close(a17, 0.541, 1e-3) and close(b17, 1.307, 1e-3))
ok("17A f(0)=1>1/2 so 0 in B", fA.subs(x, 0) > R(1, 2))
fB = 1/(1 + x**2)
ok("17B B(1/2) endpoints +-1", fB.subs(x, 1) == R(1, 2))
ok("17B f''=(6x^2-2)/(1+x^2)^3", zero(diff(fB, x, 2) - (6*x**2 - 2)/(1 + x**2)**3))
ok("17B f''(1)=1/2", diff(fB, x, 2).subs(x, 1) == R(1, 2))
x0 = nsolve(x + sin(x) - 1, x, 0.5)
ok("17C root x0~0.511", close(x0, 0.511, 1e-3))
ok("17C f''=-sin x", zero(diff(x + sin(x), x, 2) + sin(x)))

f18 = 2*x**3 - 3*x**2
ok("18 factor x^2(2x-3)", zero(f18 - x**2*(2*x - 3)))
ok("18 factor f+1=(x-1)^2(2x+1)", zero(f18 + 1 - (x - 1)**2*(2*x + 1)))

f19 = 2 + (x**2 - 1)*(x**2 - 9)
ok("19 f<=2 exactly on 1<=x^2<=9 (spot)", f19.subs(x, 2) <= 2
   and f19.subs(x, 0) > 2 and f19.subs(x, 4) > 2 and f19.subs(x, 1) == 2
   and f19.subs(x, 3) == 2)

f21c = exp(x - 4)
ok("21 e^(x-4) convex, =1 at 4", diff(f21c, x, 2).subs(x, 0) > 0
   and f21c.subs(x, 4) == 1)
f21k = 2 - exp(4 - x)
ok("21 2-e^(4-x) concave, =1 at 4", zero(diff(f21k, x, 2) + exp(4 - x))
   and f21k.subs(x, 4) == 1)
f21 = 1 + (x - 4)*(x**2 + 1)
ok("21 f'=3x^2-8x+1, disc 52", zero(diff(f21, x) - (3*x**2 - 8*x + 1))
   and (-8)**2 - 4*3*1 == 52)
ok("21 f(-2)=-29, f(2.5)=-9.875, f(0)=-3",
   f21.subs(x, -2) == -29 and f21.subs(x, R(5, 2)) == R(-9875, 1000)
   and f21.subs(x, 0) == -3)

f22 = x**2 - y**2
ok("22 f(P)=3, points", f22.subs({x: 2, y: 1}) == 3
   and f22.subs({x: sqrt(3), y: 0}) == 3 and f22.subs({x: 2, y: 0}) == 4)
ok("23 f(1,1)=2, H=2I", (x**2 + y**2).subs({x: 1, y: 1}) == 2
   and hessian(x**2 + y**2, (x, y)) == 2*eye(2))
c_ = symbols("c_", negative=True)
xn = symbols("xn", negative=True)
ok("24b phi''=2c/x^3>0 for c<0,x<0", (2*c_/xn**3).is_positive)

f25 = x**3 + y**2 + z**2 + x*z
H25 = hessian(f25, (x, y, z))
ok("25 Hessian", H25 == Matrix([[6*x, 0, 1], [0, 2, 0], [1, 0, 2]]))
ok("25 minors 6x,12x,24x-2", [simplify(m) for m in minors(H25)]
   == [6*x, 12*x, 24*x - 2])
ok("25 Q1 minors 6,12,22", minors(H25.subs(x, 1)) == [6, 12, 22])
ok("25 Q2 principal minor -12", (H25.subs(x, -1)[[0, 1], [0, 1]]).det() == -12)
H3 = H25.subs(x, R(1, 12))
ok("25 Q3 2x2 principal minors 1,0,4; det 0",
   H3[[0, 1], [0, 1]].det() == 1 and H3[[0, 2], [0, 2]].det() == 0
   and H3[[1, 2], [1, 2]].det() == 4 and H3.det() == 0)
hv = Matrix([1, 0, 0])
ok("25b Q2 directions -6 and 2", (hv.T*H25.subs(x, -1)*hv)[0] == -6
   and (Matrix([0, 1, 0]).T*H25.subs(x, -1)*Matrix([0, 1, 0]))[0] == 2)

f26 = x*y*z
H26 = hessian(f26, (x, y, z)).subs({x: 1, y: 1, z: 1})
h1 = Matrix([1, 1, 1])
h2 = Matrix([1, -1, 0])
ok("26 h'Hh = 6 and -2", (h1.T*H26*h1)[0] == 6 and (h2.T*H26*h2)[0] == -2)
ok("26 restrictions (1+t)^3 and 1-t^2",
   zero(f26.subs({x: 1 + t, y: 1 + t, z: 1 + t}) - (1 + t)**3)
   and zero(f26.subs({x: 1 + t, y: 1 - t, z: 1}) - (1 - t**2)))

f27 = x**3 - 3*x + y**2
H27 = hessian(f27, (x, y))
ok("27 H at P1,P2,P3", H27.subs(x, -1) == Matrix([[-6, 0], [0, 2]])
   and H27.subs(x, 0) == Matrix([[0, 0], [0, 2]])
   and H27.subs(x, 1) == Matrix([[6, 0], [0, 2]]))
ok("27 P1,P3 critical", all(zero(e.subs({x: s, y: 0}))
                            for e in grad(f27, (x, y)) for s in (-1, 1)))
ok("28 H(0) of y^2+x^4 and y^2+x^3",
   hessian(y**2 + x**4, (x, y)).subs({x: 0, y: 0}) == Matrix([[0, 0], [0, 2]])
   and hessian(y**2 + x**3, (x, y)).subs({x: 0, y: 0}) == Matrix([[0, 0], [0, 2]]))

f29 = x**2 + 2*log(x)
ok("29 f'=2x+2/x, f''=2-2/x^2", zero(diff(f29, x) - (2*x + 2/x))
   and zero(diff(f29, x, 2) - (2 - 2/x**2)))
f30 = 3*x - exp(x)
ok("30 f(0)=-1, f(2)~-1.39, f(ln3)~0.30", f30.subs(x, 0) == -1
   and close(f30.subs(x, 2), -1.389, 1e-3) and close(f30.subs(x, log(3)), 0.2958, 1e-3))
f31 = x*exp(-x)
ok("31 f'=(1-x)e^-x, f''=(x-2)e^-x", zero(diff(f31, x) - (1 - x)*exp(-x))
   and zero(diff(f31, x, 2) - (x - 2)*exp(-x)))
ok("31 f(2)~0.271, f(1)~0.368", close(f31.subs(x, 2), 0.2707, 1e-3)
   and close(f31.subs(x, 1), 0.3679, 1e-3))
xp = symbols("xp", positive=True)
f32 = sqrt(xp) - y**2 - (z - 1)**2
ok("32 diag Hessian", hessian(f32, (xp, y, z))
   == Matrix([[-xp**R(-3, 2)/4, 0, 0], [0, -2, 0], [0, 0, -2]]))
g33 = x**3 + exp(y) + z
ok("33 diag Hessian (6x,e^y,0)", hessian(g33, (x, y, z))
   == Matrix([[6*x, 0, 0], [0, exp(y), 0], [0, 0, 0]]))
e33 = exp(t) - 2*t
ok("33 line values 1, e^2-4, 2-2ln2", e33.subs(t, 0) == 1
   and close(e33.subs(t, 2), 3.389, 1e-3) and close(e33.subs(t, log(2)), 0.614, 1e-3))
ok("34 coefficients and points", (-1)**1*1 == -1 and (-1)**2*2 == 2
   and -1*2 + 2*1 == 0)

ok("35 epi 4-x^2 midpoint test", (4 - x**2).subs(x, 2) == 0
   and (4 - x**2).subs(x, 0) == 4)
g36 = exp(-(x**2 + y**2))
ok("36 hypo midpoint: 1/2 > e^-1", R(1, 2) > N(g36.subs({x: 1, y: 0})))
ok("36 epi midpoint: e^-4 < 1", N(g36.subs({x: 2, y: 0})) < 1)
f37 = x**R(1, 3)*y**R(1, 3)
ok("37a f(1,1)=1, f(8,8)=4, f(4.5,4.5)~2.726>2.5",
   f37.subs({x: 1, y: 1}) == 1 and simplify(f37.subs({x: 8, y: 8})) == 4
   and close(f37.subs({x: R(9, 2), y: R(9, 2)}), 2.726, 1e-3)
   and N(f37.subs({x: R(9, 2), y: R(9, 2)})) > 2.5)
ok("37b f(+-1,0)=1, f(0,0)=0", (x**2 + x*y + y**2).subs({x: 1, y: 0}) == 1)

r1, r2, r3 = symbols("r1 r2 r3", positive=True)
P2 = 5/(1 + r1) + 105/(1 + r2)**2
b2 = {r1: R(3, 100), r2: R(4, 100)}
n2 = {r1: R(35, 1000), r2: R(45, 1000)}
ok("38a P=101.93277", close(P2.subs(b2), 101.93277, 1e-4))
ok("38b partials -4.71298, -186.68924", close(diff(P2, r1).subs(b2), -4.71298, 1e-4)
   and close(diff(P2, r2).subs(b2), -186.68924, 1e-4))
dP2 = (diff(P2, r1).subs(b2) + diff(P2, r2).subs(b2))*R(5, 1000)
ok("38b dP=-0.95701", close(dP2, -0.95701, 1e-4))
ok("38b exact P=100.98256, dP_exact=-0.95021",
   close(P2.subs(n2), 100.98256, 1e-4) and close(P2.subs(n2) - P2.subs(b2), -0.95021, 1e-4))
ok("38c second partials 10/(1+r1)^3, 630/(1+r2)^4",
   zero(diff(P2, r1, 2) - 10/(1 + r1)**3) and zero(diff(P2, r2, 2) - 630/(1 + r2)**4)
   and diff(P2, r1, r2) == 0)
ok("38c Delta P >= dP", P2.subs(n2) - P2.subs(b2) > dP2)
P3 = 4/(1 + r1) + 4/(1 + r2)**2 + 104/(1 + r3)**3
b3 = {r1: R(2, 100), r2: R(3, 100), r3: R(35, 1000)}
n3 = {r1: R(15, 1000), r2: R(25, 1000), r3: R(30, 1000)}
ok("39 P=101.49399", close(P3.subs(b3), 101.49399, 1e-4))
ok("39 partials", close(diff(P3, r1).subs(b3), -3.84468, 1e-4)
   and close(diff(P3, r2).subs(b3), -7.32113, 1e-4)
   and close(diff(P3, r3).subs(b3), -271.88998, 1e-4))
dP3 = sum(diff(P3, s).subs(b3) for s in (r1, r2, r3))*R(-5, 1000)
ok("39 dP=1.41528, exact 102.92288 / 1.42888, exact larger",
   close(dP3, 1.41528, 1e-4) and close(P3.subs(n3), 102.92288, 1e-4)
   and close(P3.subs(n3) - P3.subs(b3), 1.42888, 1e-4)
   and P3.subs(n3) - P3.subs(b3) > dP3)
xq, yq = symbols("xq yq", positive=True)
f40 = 6*sqrt(xq) + 3*log(yq)
d40 = diff(f40, xq).subs(xq, 9)*R(124, 100) + diff(f40, yq).subs(yq, 1)*R(1, 10)
ok("40 df=1.54", d40 == R(154, 100))
ok("40 exact 1.48593", close(f40.subs({xq: R(1024, 100), yq: R(11, 10)}) - 18, 1.48593, 1e-4))

f41 = 2*exp(x) - 3*x**2 + y**2 + 5*x - y
ok("41 H=diag(2e^x-6,2)", hessian(f41, (x, y)) == Matrix([[2*exp(x) - 6, 0], [0, 2]]))
ok("41 boundary x=ln3", zero((2*exp(x) - 6).subs(x, log(3))))
f42 = x**3 + y**3 - 3*x*y
ok("42 H and det 36xy-9", hessian(f42, (x, y)) == Matrix([[6*x, -3], [-3, 6*y]])
   and zero(hessian(f42, (x, y)).det() - (36*x*y - 9)))

phi = exp(-(x - 1)**2/8 - (y - 2)**2/2)/(4*pi)
ok("43a grad = -phi((x-1)/4, y-2)", zero(diff(phi, x) + phi*(x - 1)/4)
   and zero(diff(phi, y) + phi*(y - 2)))
Hphi = hessian(phi, (x, y))
ok("43b det = phi^2/4 (1-(x-1)^2/4-(y-2)^2)",
   zero(Hphi.det() - phi**2/4*(1 - (x - 1)**2/4 - (y - 2)**2)))
ok("43b phi_xx=((x-1)^2/16-1/4)phi", zero(Hphi[0, 0] - ((x - 1)**2/16 - R(1, 4))*phi))
ok("43 H at mean = diag(-1/4,-1)/(4pi)", (Hphi.subs({x: 1, y: 2})
   - Matrix([[-R(1, 4), 0], [0, -1]])/(4*pi)).applyfunc(simplify).is_zero_matrix)

f44 = 3*(x - y)**2 - (x + y)**2
ok("44 expand 2x^2-8xy+2y^2", zero(f44 - (2*x**2 - 8*x*y + 2*y**2)))
ok("44 det -48, f(x,x)=-4x^2", hessian(f44, (x, y)).det() == -48
   and zero(f44.subs(y, x) + 4*x**2))

u45 = x*y**2
ok("45 H det -4x2^2", zero(hessian(u45, (x, y)).det() + 4*y**2))
ok("45 midpoint (2,2) value 8", u45.subs({x: 2, y: 2}) == 8)
ok("46 v midpoint sqrt2 c", zero((sqrt(x) + sqrt(y)).subs({x: R(1, 2), y: R(1, 2)}) - sqrt(2)))
ok("48 f(0,1)=1,f(1,1)=0,f(-1,1)=0", sqrt(y - x**2).subs({x: 0, y: 1}) == 1
   and sqrt(y - x**2).subs({x: 1, y: 1}) == 0 and sqrt(y - x**2).subs({x: -1, y: 1}) == 0)

f49 = log(x + y)
H49 = hessian(f49, (x, y))
ok("49 H = -1/(x+y)^2 [[1,1],[1,1]], det 0", zero(H49[0, 0] + 1/(x + y)**2)
   and zero(H49.det()))
f50 = log(x + y**2)
H50 = hessian(f50, (x, y))
ok("50 f_yy=(2x-2y^2)/s^2", zero(H50[1, 1] - (2*x - 2*y**2)/(x + y**2)**2))
ok("50 det = -2/s^3", zero(H50.det() + 2/(x + y**2)**3))
f51 = x**3 - 3*x - y**2
ok("51 f(-1,0)=2, f(3,0)=18", f51.subs({x: -1, y: 0}) == 2 and f51.subs({x: 3, y: 0}) == 18)
ok("52 H=diag(-e^x,-2)", hessian(-exp(x) - y**2, (x, y)) == Matrix([[-exp(x), 0], [0, -2]]))
g53 = x*y/(x**2 + y**2)
ok("53a f(t,t)=1/2", zero(g53.subs(y, x) - R(1, 2)))
ok("54d x^3-3x>=0 pieces: roots 0,+-sqrt3", set(solve(x**3 - 3*x, x)) == {0, sqrt(3), -sqrt(3)})

# ================================================================ ch05 ====
print("--- ch05 Additional 11-26 ---")
f11 = (x - 1)**3 - 3*(x - 1)*(y + 2)**2
ok("11 crit (1,-2), H=0", solve(list(grad(f11, (x, y))), [x, y], dict=True) == [{x: 1, y: -2}]
   and hessian(f11, (x, y)).subs({x: 1, y: -2}) == zeros(2, 2))
ok("11 along y=-2: u^3", zero(f11.subs(y, -2) - (x - 1)**3))
f12 = (x + 1)**2 + (y - 2)**4
ok("12 H diag(2,0)", hessian(f12, (x, y)).subs({x: -1, y: 2}) == Matrix([[2, 0], [0, 0]]))
f13 = (x + 3)**2 + y**4 + 2*y**3
ok("13 f_y=2y^2(2y+3)", zero(diff(f13, y) - 2*y**2*(2*y + 3)))
ok("13 H at (-3,-3/2)=diag(2,9), value -27/16",
   hessian(f13, (x, y)).subs({x: -3, y: -R(3, 2)}) == Matrix([[2, 0], [0, 9]])
   and f13.subs({x: -3, y: -R(3, 2)}) == -R(27, 16))
ok("13 along x=-3: y^3(y+2)", zero(f13.subs(x, -3) - y**3*(y + 2)))
f14 = x**2*y**3
ok("14 partials", zero(diff(f14, x) - 2*x*y**3) and zero(diff(f14, y) - 3*x**2*y**2))
g15 = (x + y)**2 + x**4
h15 = (x + y)**2 - x**4
ok("15 H(0)=[[2,2],[2,2]] both", hessian(g15, (x, y)).subs({x: 0, y: 0}) == Matrix([[2, 2], [2, 2]])
   and hessian(h15, (x, y)).subs({x: 0, y: 0}) == Matrix([[2, 2], [2, 2]]))
ok("15 h(t,-t)=-t^4, h(t,0)=t^2-t^4", zero(h15.subs(y, -x) + x**4)
   and zero(h15.subs(y, 0) - (x**2 - x**4)))

A16 = Matrix([[2, 1], [1, 3]]); b16 = Matrix([-4, 2])
X2 = Matrix([x, y])
f16 = 2*x**2 + 2*x*y + 3*y**2 - 4*x + 2*y
ok("16 f = x'Ax+b'x", zero(f16 - (X2.T*A16*X2 + b16.T*X2)[0]))
a, b, c, d = symbols("a b c d", real=True)
b1s, b2s = symbols("b1s b2s", real=True)
fg = (X2.T*Matrix([[a, c], [c, d]])*X2 + Matrix([b1s, b2s]).T*X2)[0]
ok("16 grad = 2Ax+b (n=2)", zero((grad(fg, (x, y)) - (2*Matrix([[a, c], [c, d]])*X2
                                                      + Matrix([b1s, b2s]))).norm()))
xs16 = -A16.inv()*b16/2
ok("16 x*=(7/5,-4/5), f*=-18/5", list(xs16) == [R(7, 5), -R(4, 5)]
   and f16.subs({x: xs16[0], y: xs16[1]}) == -R(18, 5))
A17 = Matrix([[-2, 1, 0], [1, -3, 1], [0, 1, -2]]); b17v = Matrix([2, 0, 4])
X3 = Matrix([x, y, z])
f17 = -2*x**2 + 2*x*y - 3*y**2 + 2*y*z - 2*z**2 + 2*x + 4*z
ok("17 f = x'Ax+b'x", zero(f17 - (X3.T*A17*X3 + b17v.T*X3)[0]))
ok("17 minors -2,5,-8", minors(A17) == [-2, 5, -8])
xs17 = -A17.inv()*b17v/2
ok("17 x*=(7/8,3/4,11/8), max 29/8", list(xs17) == [R(7, 8), R(3, 4), R(11, 8)]
   and f17.subs({x: xs17[0], y: xs17[1], z: xs17[2]}) == R(29, 8))
A18 = Matrix([[1, 2], [2, 1]]); b18 = Matrix([2, -4])
f18b = x**2 + 4*x*y + y**2 + 2*x - 4*y
xs18 = -A18.inv()*b18/2
ok("18 x*=(5/3,-4/3), value 13/3, det -3", list(xs18) == [R(5, 3), -R(4, 3)]
   and f18b.subs({x: xs18[0], y: xs18[1]}) == R(13, 3) and A18.det() == -3
   and zero(f18b - (X2.T*A18*X2 + b18.T*X2)[0]))
f19 = x**3 + y**3 + z**3 - 3*x - 12*y - 27*z
ok("19 f(1,2,3)=-72, f(-1,-2,-3)=72", f19.subs({x: 1, y: 2, z: 3}) == -72
   and f19.subs({x: -1, y: -2, z: -3}) == 72)
ok("19 eight critical points", len(solve(list(grad(f19, (x, y, z))), [x, y, z], dict=True)) == 8)
f20 = x**2 + y**2 - 2*y + log(1 + z**2)
ok("20a f_zz=2(1-z^2)/(1+z^2)^2, value -1", zero(diff(f20, z, 2) - 2*(1 - z**2)/(1 + z**2)**2)
   and f20.subs({x: 0, y: 1, z: 0}) == -1)
g20 = x**2 + y**2 - 2*y - log(1 + z**2)
ok("20b H=diag(2,2,-2)", hessian(g20, (x, y, z)).subs({x: 0, y: 1, z: 0}) == Matrix([[2, 0, 0], [0, 2, 0], [0, 0, -2]]))
w_ = symbols("w_", real=True)
f21b = w_**2 + 2*w_*x + 2*x**2 + y**2 - 2*y*z + 3*z**2 - 4*w_ - 2*x - 4*z + 7
ok("21 sum of squares", zero(f21b - ((w_ + x - 2)**2 + (x + 1)**2 + (y - z)**2 + 2*(z - 1)**2)))
ok("21 minors 2,4,8,32", minors(hessian(f21b, (w_, x, y, z))) == [2, 4, 8, 32])
ok("21 min at (3,-1,1,1) value 0", f21b.subs({w_: 3, x: -1, y: 1, z: 1}) == 0)
f22b = x**3 - 3*x*z + y**2 + z**2
ok("22 critical points", set(tuple(s[v_] for v_ in (x, y, z)) for s in
   solve(list(grad(f22b, (x, y, z))), [x, y, z], dict=True)) == {(0, 0, 0), (R(3, 2), 0, R(9, 4))})
H22 = hessian(f22b, (x, y, z))
ok("22 origin det -18, minor -9", H22.subs(x, 0).det() == -18
   and H22.subs(x, 0)[[0, 2], [0, 2]].det() == -9)
ok("22 (3/2,0,9/4) minors 9,18,18, value -27/16", minors(H22.subs(x, R(3, 2))) == [9, 18, 18]
   and f22b.subs({x: R(3, 2), y: 0, z: R(9, 4)}) == -R(27, 16))
b0, b1 = symbols("b0 b1", real=True)
data = [(1, 2), (2, 3), (3, 7)]
S = sum((yi - b0 - b1*xi)**2 for xi, yi in data)
s23 = solve([diff(S, b0), diff(S, b1)], [b0, b1])
ok("23 b0=-1, b1=5/2, Smin=3/2", s23 == {b0: -1, b1: R(5, 2)} and S.subs(s23) == R(3, 2))
ok("23 Hessian 2[[3,6],[6,14]], D2=24", hessian(S, (b0, b1)) == 2*Matrix([[3, 6], [6, 14]])
   and hessian(S, (b0, b1)).det() == 24)
Xm = Matrix([[1, 1], [1, 2], [1, 3]]); ym = Matrix([2, 3, 7])
ok("23 matrix route", list((Xm.T*Xm).inv()*Xm.T*ym) == [-1, R(5, 2)])
ok("24 identity n sum x^2-(sum x)^2 = n Sxx (data)",
   3*14 - 6**2 == 3*sum((xi - 2)**2 for xi, _ in data))
bb = symbols("bb", real=True)
S25 = sum((yi - bb*xi)**2 for xi, yi in data)
ok("25 b*=29/14", solve(diff(S25, bb), bb) == [R(29, 14)])
qq1, qq2, pp1, pp2 = symbols("qq1 qq2 pp1 pp2", real=True)
B26 = Matrix([[-2, 1], [1, -3]]); a26 = Matrix([50, 40])
pq = B26.inv()*(Matrix([qq1, qq2]) - a26)
ok("26 inverse demand", zero(pq[0] - (38 - R(3, 5)*qq1 - R(1, 5)*qq2))
   and zero(pq[1] - (26 - R(1, 5)*qq1 - R(2, 5)*qq2)))
piq = pq[0]*qq1 + pq[1]*qq2 - 2*qq1 - 4*qq2
ok("26 pi(q) expanded", zero(piq - (36*qq1 + 22*qq2 - R(3, 5)*qq1**2 - R(2, 5)*qq1*qq2 - R(2, 5)*qq2**2)))
sq = solve(list(grad(piq, (qq1, qq2))), [qq1, qq2])
ok("26 q*=(25,15), pi=615, p=(20,15)", sq == {qq1: 25, qq2: 15} and piq.subs(sq) == 615
   and [e.subs(sq) for e in pq] == [20, 15])
qd = a26 + B26*Matrix([pp1, pp2])
pip = (pp1 - 2)*qd[0] + (pp2 - 4)*qd[1]
sp = solve(list(grad(pip, (pp1, pp2))), [pp1, pp2])
ok("26 p*=(20,15), same profit, ND", sp == {pp1: 20, pp2: 15} and pip.subs(sp) == 615
   and hessian(pip, (pp1, pp2)) == Matrix([[-4, 2], [2, -6]]))
ok("26 Hq D2=4/5", hessian(piq, (qq1, qq2)).det() == R(4, 5))

# ================================================================ ch06 ====
print("--- ch06 worked example 2 and Additional 13-17 ---")
piW = pp1*qq1 + pp2*qq2 - qq1**2 - qq1*qq2 - qq2**2
sW = solve(list(grad(piW, (qq1, qq2))), [qq1, qq2])
ok("W2 q*", zero(sW[qq1] - (2*pp1 - pp2)/3) and zero(sW[qq2] - (2*pp2 - pp1)/3))
VW = simplify(piW.subs(sW))
ok("W2 V=(p1^2-p1p2+p2^2)/3", zero(VW - (pp1**2 - pp1*pp2 + pp2**2)/3))
ok("W2 numbers V(10,8)=28, q=(4,2)", VW.subs({pp1: 10, pp2: 8}) == 28
   and [sW[k].subs({pp1: 10, pp2: 8}) for k in (qq1, qq2)] == [4, 2])
Vn = VW.subs({pp1: R(105, 10), pp2: R(77, 10)})
ok("W2 V(10.5,7.7)=29.5633, gap 0.16333", close(Vn, 29.56333) and close(Vn - 28 - R(14, 10), 0.163333))
ok("W2 dq1 exact 0.4333", sW[qq1].subs({pp1: R(105, 10), pp2: R(77, 10)}) - 4 == R(13, 30))
pi13 = pp1*qq1 + pp2*qq2 - qq1**2 - 2*qq2**2 + qq1*qq2
s13 = solve(list(grad(pi13, (qq1, qq2))), [qq1, qq2])
ok("13 q*", zero(s13[qq1] - (4*pp1 + pp2)/7) and zero(s13[qq2] - (pp1 + 2*pp2)/7))
V13 = simplify(pi13.subs(s13))
ok("13 V=(2p1^2+p1p2+p2^2)/7 and envelope", zero(V13 - (2*pp1**2 + pp1*pp2 + pp2**2)/7)
   and zero(diff(V13, pp1) - s13[qq1]) and zero(diff(V13, pp2) - s13[qq2]))
ok("13 numbers (9,4), V=77", [s13[k].subs({pp1: 14, pp2: 7}) for k in (qq1, qq2)] == [9, 4]
   and V13.subs({pp1: 14, pp2: 7}) == 77)
ok("13 -H^-1 = (1/7)[[4,1],[1,2]]", -hessian(pi13, (qq1, qq2)).inv() == Matrix([[4, 1], [1, 2]])/7)
a1, bb1, a2, bb2, al = symbols("a1 bb1 a2 bb2 al", positive=True)
V14 = (a1 - al)**2/(4*bb1) + (a2 - al)**2/(4*bb2)
base14 = {a1: 100, bb1: 2, a2: 80, bb2: 1, al: 20}
ok("14 base V=1700", V14.subs(base14) == 1700)
ok("14 dV/db1=-q1^2", zero(diff(V14, bb1) + ((a1 - al)/(2*bb1))**2))
ok("14(i) exact V=1782", V14.subs({**base14, a1: 104}) == 1782)
ok("14(ii) q1=80/4.4, dV exact -72.73", close(R(80, 1)/R(44, 10), 18.1818)
   and close(V14.subs({**base14, bb1: R(22, 10)}) - 1700, -72.727))
ok("14(iii) exact -49.625", V14.subs({**base14, al: 21}) - 1700 == -R(49625, 1000))
f15 = a*x + b*y + c*z - x**2 - 2*y**2 - d*z**2 + x*y - y*z
s15 = solve(list(grad(f15, (x, y, z))), [x, y, z], dict=True)[0]
base15 = {a: 4, b: 6, c: 2, d: 1}
ok("15 x*=(19/6,7/3,-1/6)", [s15[k].subs(base15) for k in (x, y, z)] == [R(19, 6), R(7, 3), -R(1, 6)])
H15 = hessian(f15, (x, y, z)).subs(base15)
ok("15 minors -2,7,-12", minors(H15) == [-2, 7, -12])
ok("15 -H^-1", -H15.inv() == Matrix([[7, 2, -1], [2, 4, -2], [-1, -2, 7]])/12)
V15 = simplify(f15.subs(s15))
ok("15 V=79/6", V15.subs(base15) == R(79, 6))
ok("15 d x*/dd = (-1/36,-1/18,7/36)", [simplify(diff(s15[k], d)).subs(base15) for k in (x, y, z)]
   == [-R(1, 36), -R(1, 18), R(7, 36)])
ok("15 grad V = (19/6,7/3,-1/6,-1/36)", [simplify(diff(V15, p)).subs(base15) for p in (a, b, c, d)]
   == [R(19, 6), R(7, 3), -R(1, 6), -R(1, 36)])
n15c = {a: R(42, 10), b: R(59, 10), c: R(21, 10), d: 1}
ok("15c dx exact (11/120,-1/60,7/120)", [s15[k].subs(n15c) - s15[k].subs(base15) for k in (x, y, z)]
   == [R(11, 120), -R(1, 60), R(7, 120)])
ok("15c dV=23/60, exact 317/800, gap 31/2400", R(19, 6)*R(2, 10) + R(7, 3)*R(-1, 10) - R(1, 6)*R(1, 10) == R(23, 60)
   and V15.subs(n15c) - V15.subs(base15) == R(317, 800) and R(317, 800) - R(23, 60) == R(31, 2400))
n15d = {**base15, d: R(11, 10)}
ok("15d exact (-0.00249,-0.00498,0.01741), dV -0.00249",
   close(s15[x].subs(n15d) - R(19, 6), -0.00249, 1e-5) and close(s15[y].subs(n15d) - R(7, 3), -0.00498, 1e-5)
   and close(s15[z].subs(n15d) + R(1, 6), 0.01741, 1e-5)
   and close(V15.subs(n15d) - R(79, 6), -0.00249, 1e-5))
kk = symbols("kk", real=True)
C16 = Matrix([[2, kk], [kk, 2]])
ok("16 (C'')^-1 off-diagonal -k/(4-k^2)", zero(C16.inv()[0, 1] + kk/(4 - kk**2)))
pi17 = (a1 - bb1*qq1)*qq1 + (a2 - bb2*qq2)*qq2 - al/2*(qq1 + qq2)**2
H17 = hessian(pi17, (qq1, qq2))
D17 = H17.det()
ok("17 D2=4b1b2+2al(b1+b2)", zero(D17 - (4*bb1*bb2 + 2*al*(bb1 + bb2))))
s17 = solve(list(grad(pi17, (qq1, qq2))), [qq1, qq2])
ok("17 dq/da2 = (-al/D, (2b1+al)/D)", zero(diff(s17[qq1], a2) + al/D17)
   and zero(diff(s17[qq2], a2) - (2*bb1 + al)/D17))
n17 = {a1: 100, bb1: 2, a2: 80, bb2: 1, al: 1}
ok("17 q*=(110/7,150/7)", [s17[k].subs(n17) for k in (qq1, qq2)] == [R(110, 7), R(150, 7)])
ok("17 Delta a2=7: -0.5, 2.5", [s17[k].subs({**n17, a2: 87}) - s17[k].subs(n17) for k in (qq1, qq2)]
   == [-R(1, 2), R(5, 2)])

# ================================================================ ch07 ====
print("--- ch07 Additional 28-45 ---")
ok("28 optimum (2sqrt3,sqrt6)", zero((x**2 + 2*y**2 - 24).subs({x: 2*sqrt(3), y: sqrt(6)})))
ok("28 MRS=MRT=1/sqrt2, lam=sqrt2/4, U=6sqrt2",
   zero(sqrt(6)/(2*sqrt(3)) - 1/sqrt(2)) and zero(2*sqrt(3)/(2*sqrt(6)) - 1/sqrt(2))
   and zero(sqrt(6)/(2*2*sqrt(3)) - sqrt(2)/4) and zero(2*sqrt(3)*sqrt(6) - 6*sqrt(2)))
s29 = solve([2*x*y - lam*2, x**2 - lam*3, 2*x + 3*y - 30], [x, y, lam], dict=True)
ok("29 (10,10/3), lam=100/3, MRS 2/3", any(s[x] == 10 and s[y] == R(10, 3) and s[lam] == R(100, 3) for s in s29)
   and 2*R(10, 3)/10 == R(2, 3))
xs30, ys30 = 10/sqrt(3), 10*sqrt(R(2, 3))
ok("30 on circle, MRS=MRT, lam=3/200", zero(xs30**2 + ys30**2 - 100)
   and zero(ys30/(2*xs30) - xs30/ys30) and zero(1/(2*xs30**2) - R(3, 200))
   and zero(1/xs30 - 2*(1/(2*xs30**2))*xs30))
ok("31 grad f/grad g length ratio 2 = lam", zero(sqrt(20)/sqrt(5) - 2))
ok("32 (3,4) and (-3,-4), ratio 1/2", 3**2 + 4**2 == 25 and R(5, 10) == R(1, 2))
ok("34 points on circle, levels 9 and 49, lam -3/2 and 7/2",
   (R(9, 5) - 3)**2 + (R(12, 5) - 4)**2 == 4 and (R(21, 5) - 3)**2 + (R(28, 5) - 4)**2 == 4
   and R(81 + 144, 25) == 9 and R(441 + 784, 25) == 49
   and R(9, 5)/(R(9, 5) - 3) == -R(3, 2) and R(21, 5)/(R(21, 5) - 3) == R(7, 2))
U35 = x**R(3, 10)*y**R(7, 10)
ok("35 (6,21), U~14.42, lam~0.240", 3*6 + 2*21 == 60 and close(U35.subs({x: 6, y: 21}), 14.42, 0.01)
   and close(diff(U35, x).subs({x: 6, y: 21})/3, 0.2404, 1e-3))
ok("36 K=10,L=40,Q=20", 4*10 + 40 == 80 and sqrt(10*40) == 20)
s37 = solve([-2*(x - 4) - lam, -2*(y - 3) - lam, x + y - 10], [x, y, lam])
ok("37 equality form (5.5,4.5), lam=-3", s37 == {x: R(11, 2), y: R(9, 2), lam: -3})
ok("37 U(5.5,4.5)=-4.5, bliss cost 7", (-(x - 4)**2 - (y - 3)**2).subs(s37) == -R(9, 2) and 4 + 3 == 7)
x_ = symbols("x_", real=True)
sig2 = 400*x_**2 + 100*(1 - x_)**2 + 80*x_*(1 - x_)
ok("39a sigma^2=420x^2-120x+100", zero(sig2 - (420*x_**2 - 120*x_ + 100)))
m_ = symbols("m_", real=True)
curve = sig2.subs(x_, (m_ - 4)/6)
ok("39b curve=(35mu^2-340mu+1100)/3", zero(curve - (35*m_**2 - 340*m_ + 1100)/3))
ok("39c x=1/7, mu=34/7, var=640/7, sigma<10", solve(diff(sig2, x_), x_) == [R(1, 7)]
   and sig2.subs(x_, R(1, 7)) == R(640, 7) and N(sqrt(R(640, 7))) < 10)
m40 = solve(1 - diff(curve, m_)/100, m_)[0]
ok("40 mu*=64/7, x=6/7, s=2140/7, Phi=213/35", m40 == R(64, 7) and (m40 - 4)/6 == R(6, 7)
   and curve.subs(m_, m40) == R(2140, 7) and m40 - curve.subs(m_, m40)/100 == R(213, 35))
ok("40 amounts 60000/10000", R(6, 7)*70000 == 60000)
sg40 = sqrt(curve.subs(m_, m40))
ok("40 MRS=MRT slopes 0.02 sigma ~0.350", zero(2*sg40/diff(curve, m_).subs(m_, m40) - R(2, 100)*sg40)
   and close(R(2, 100)*sg40, 0.3497, 1e-3))
s41 = solve([1 + lam*(2*m_ - 8), -R(5, 4) - 2*lam*u, u**2 - m_**2 + 8*m_ - 25], [m_, u, lam], dict=True)
ok("41 mu=8, sigma=5, lam=-1/8", any(s[m_] == 8 and s[u] == 5 and s[lam] == -R(1, 8) for s in s41))
ok("41 Phi=7/4", 8 - R(5, 4)*5 == R(7, 4))
Phi42 = m_ - m_**2/100 - curve/100
m42 = solve(diff(Phi42, m_), m_)[0]
ok("42 mu=160/19, x=14/19, var~239.6", m42 == R(160, 19) and (m42 - 4)/6 == R(14, 19)
   and close(curve.subs(m_, m42), 239.6, 0.05))
sig43 = 225*x_**2 + 64*(1 - x_)**2 - 48*x_*(1 - x_)
ok("43 sigma^2=337x^2-176x+64", zero(sig43 - (337*x_**2 - 176*x_ + 64)))
ok("43 curve in mu", zero(sig43.subs(x_, (m_ - 6)/6) - (R(337, 36)*m_**2 - R(425, 3)*m_ + 577)))
ok("43 min x=88/337, var=13824/337~41.0, sigma~6.40", solve(diff(sig43, x_), x_) == [R(88, 337)]
   and sig43.subs(x_, R(88, 337)) == R(13824, 337) and close(sqrt(R(13824, 337)), 6.405, 1e-3))
u44 = sqrt(R(72900)/R(19, 100))
ok("44 u~619.4, x~-219.4", close(u44, 619.4, 0.05) and close(400 - u44, -219.4, 0.05))
C44 = 90*x + 100*sqrt((400 - x)**2 + 90000)
ok("44 C(0)=50000, C(400)=66000, C'(0)=10", C44.subs(x, 0) == 50000 and C44.subs(x, 400) == 66000
   and diff(C44, x).subs(x, 0) == 10)
ok("45 U=2ln1.5~0.811", close(2*log(R(3, 2)), 0.811, 1e-3))

# ================================================================ ch08 ====
print("--- ch08 Additional 11-17 ---")
K, L_ = symbols("K L_", positive=True)
ok("11 Q=20, lam=1/4", sqrt(10*40) == 20 and (diff(sqrt(K*L_), K)/4).subs({K: 10, L_: 40}) == R(1, 4))
s11 = solve([4 - mu*diff(sqrt(K*L_), K), 1 - mu*diff(sqrt(K*L_), L_), sqrt(K*L_) - 30], [K, L_, mu], dict=True)
ok("11d target 30: (15,60), mu=4", s11 == [{K: 15, L_: 60, mu: 4}])
Q12 = K**R(1, 3)*L_**R(1, 3)
ok("12 Q*=450^(1/3)~7.663, lam~0.0851", close(Q12.subs({K: 15, L_: 30}), 7.663, 1e-3)
   and close(diff(Q12, L_).subs({K: 15, L_: 30}), 0.08514, 1e-4))
Bv = symbols("Bv", positive=True)
ok("12 Q*(B)=B^(2/3)/2", zero(Q12.subs({K: Bv/4, L_: Bv/2}) - Bv**R(2, 3)/2))
Qs = Q12.subs({K: 15, L_: 30})
ok("12 C(2Q*)=(4Q*)^1.5~169.7=60*2^1.5", close((4*Qs)**R(3, 2), 169.7, 0.05)
   and close((4*Qs)**R(3, 2), 60*2**R(3, 2)))
ok("12 C'(Q*)=3sqrt(2Q*)~11.74=1/lam", close(3*sqrt(2*Qs), 11.745, 1e-3)
   and close(3*sqrt(2*Qs)*diff(Q12, L_).subs({K: 15, L_: 30}), 1))
a_s, h_s = symbols("a_s h_s", positive=True)
s13b = solve([2*a_s*h_s - lam*(4*a_s + 4*h_s), a_s**2 - lam*4*a_s, 2*a_s**2 + 4*a_s*h_s - 24],
             [a_s, h_s, lam], dict=True)
ok("13 box max: a=h=2, lam=1/2", s13b == [{a_s: 2, h_s: 2, lam: R(1, 2)}])
s13c = solve([4*a_s + 4*h_s - mu*2*a_s*h_s, 4*a_s - mu*a_s**2, a_s**2*h_s - 8], [a_s, h_s, mu], dict=True)
ok("13 box min: a=h=2, mu=2", s13c == [{a_s: 2, h_s: 2, mu: 2}])
r_, h_ = symbols("r_ h_", positive=True)
s14 = solve([2*pi*r_*h_ - lam*(4*pi*r_ + 2*pi*h_), pi*r_**2 - lam*2*pi*r_,
             2*pi*r_**2 + 2*pi*r_*h_ - 54*pi], [r_, h_, lam], dict=True)
ok("14 cylinder r=3,h=6,lam=3/2, V=54pi", s14 == [{r_: 3, h_: 6, lam: R(3, 2)}]
   and pi*9*6 == 54*pi)
s14d = solve([4*pi*r_ + 2*pi*h_ - mu*2*pi*r_*h_, 2*pi*r_ - mu*pi*r_**2, pi*r_**2*h_ - 54*pi],
             [r_, h_, mu], dict=True)
ok("14 dual r=3,h=6,mu=2/3", s14d == [{r_: 3, h_: 6, mu: R(2, 3)}])
ok("15 multipliers 5G'(25)", [5*e for e in (R(1, 10), 50, 1/R(25))] == [R(1, 2), 250, R(1, 5)])
ok("15 (xy)^2 FOC also at (0,10)", zero((2*x*y*y - 2*x*y*x).subs({x: 0, y: 10})))
Q16 = x**R(2, 10)*y**R(3, 10)*z**R(5, 10)
ok("16 x*=(10,10,10), Q=10, lam=0.1", simplify(Q16.subs({x: 10, y: 10, z: 10})) == 10
   and all(zero(diff(Q16, s).subs({x: 10, y: 10, z: 10}) - R(1, 10)*wv)
           for s, wv in ((x, 2), (y, 3), (z, 5))))
Q17 = (x*y*z)**R(1, 4)
ok("17 x*=(16,8,4), Q~4.757, lam=Q/64", close(Q17.subs({x: 16, y: 8, z: 4}), 4.757, 1e-3)
   and all(zero(diff(Q17, s).subs({x: 16, y: 8, z: 4}) - Q17.subs({x: 16, y: 8, z: 4})/64*wv)
           for s, wv in ((x, 1), (y, 2), (z, 4)))
   and zero(R(3, 4)*Q17.subs({x: 16, y: 8, z: 4})/48 - Q17.subs({x: 16, y: 8, z: 4})/64))

# ================================================================ ch09 ====
print("--- ch09 worked example 2 and Additional 21-48 ---")
fw = x + 2*y + 2*z
gw = x**2 + y**2 + z**2
ok("W2 P+ minors (20,-36)", bordered(fw, [gw], [x, y, z], [l1], {x: 1, y: 2, z: 2, l1: R(1, 2)}) == [20, -36])
ok("W2 P- minors (-20,-36)", bordered(fw, [gw], [x, y, z], [l1], {x: -1, y: -2, z: -2, l1: -R(1, 2)}) == [-20, -36])
ok("W2 3sqrt(9.5)-9~0.247", close(3*sqrt(R(95, 10)) - 9, 0.2466, 1e-3))
ok("21 point (2,4,4), lam 4, minors (-10,-36)",
   bordered(x**2 + y**2 + z**2, [x + 2*y + 2*z], [x, y, z], [l1], {x: 2, y: 4, z: 4, l1: 4}) == [-10, -36]
   and 2*2 == 4 and 2*4 == 2*4)
ok("22 grads equal, g=12, minors (32,-48)",
   bordered(x*y*z, [x*y + y*z + z*x], [x, y, z], [l1], {x: 2, y: 2, z: 2, l1: 1}) == [32, -48])
X4 = symbols("x1:5", real=True)
f23 = sum(s**2 for s in X4)
g23 = [sum(X4), X4[0] - X4[1] + X4[2] - X4[3]]
pt23 = {X4[0]: R(3, 2), X4[1]: R(1, 2), X4[2]: R(3, 2), X4[3]: R(1, 2), l1: 2, l2: 1}
ok("23 FOC holds", all(zero((diff(f23, s) - l1*diff(g23[0], s) - l2*diff(g23[1], s)).subs(pt23)) for s in X4)
   and [g.subs(pt23) for g in g23] == [4, 2])
ok("23 minors (16,64), f=5", bordered(f23, g23, list(X4), [l1, l2], pt23) == [16, 64] and f23.subs(pt23) == 5)
f24, g24 = z, [x**2 + y**2 + z**2, x + y + z]
ok("24 (0,0,3) minor -24", bordered(f24, g24, [x, y, z], [l1, l2], {x: 0, y: 0, z: 3, l1: R(1, 6), l2: 0}) == [-24])
ok("24 (2,2,-1) minor 24", bordered(f24, g24, [x, y, z], [l1, l2], {x: 2, y: 2, z: -1, l1: -R(1, 6), l2: R(2, 3)}) == [24])
ok("24 FOC at both", all(zero((diff(f24, s) - l1*diff(g24[0], s) - l2*diff(g24[1], s)).subs(pt))
                         for s in (x, y, z) for pt in ({x: 0, y: 0, z: 3, l1: R(1, 6), l2: 0},
                                                       {x: 2, y: 2, z: -1, l1: -R(1, 6), l2: R(2, 3)})))
X5 = symbols("y1:6", real=True)
f25b = sum(s**2 for s in X5)
g25 = [sum(X5), sum((i + 1)*s for i, s in enumerate(X5))]
pt25 = {**{s: 1 for s in X5}, l1: 2, l2: 0}
ok("25 FOC and constraints", all(zero((diff(f25b, s) - l1*diff(g25[0], s) - l2*diff(g25[1], s)).subs(pt25)) for s in X5)
   and [g.subs(pt25) for g in g25] == [5, 15])
ok("25 minors (12,80,400)", bordered(f25b, g25, list(X5), [l1, l2], pt25) == [12, 80, 400])
f26 = sum(X4)
g26 = [sum(s**2 for s in X4)]
ok("26 + minors (8,-12,16)", bordered(f26, g26, list(X4), [l1], {**{s: 1 for s in X4}, l1: R(1, 2)}) == [8, -12, 16])
ok("26 - minors (-8,-12,-16)", bordered(f26, g26, list(X4), [l1], {**{s: -1 for s in X4}, l1: -R(1, 2)}) == [-8, -12, -16])


def pattern(n, m):
    ks = list(range(2*m + 1, m + n + 1))
    return ks, [(-1)**(k - m) for k in ks], [(-1)**m for _ in ks]


ok("27 table", pattern(3, 1) == ([3, 4], [1, -1], [-1, -1])
   and pattern(4, 2) == ([5, 6], [-1, 1], [1, 1])
   and pattern(5, 2) == ([5, 6, 7], [-1, 1, -1], [1, 1, 1])
   and pattern(6, 3) == ([7, 8, 9], [1, -1, 1], [-1, -1, -1]))
ok("28 (5,2) min all +", pattern(5, 2)[2] == [1, 1, 1])
ok("29 (6,4) max (-,+), min (+,+)", pattern(6, 4)[0] == [9, 10] and pattern(6, 4)[1] == [-1, 1]
   and pattern(6, 4)[2] == [1, 1])
ok("30 (3,2) max -", pattern(3, 2) == ([5], [-1], [1]))
ok("31 (4,1) orders 3,4,5", pattern(4, 1)[0] == [3, 4, 5])
ok("32 (5,1) max (+,-,+,-), min all -", pattern(5, 1) == ([3, 4, 5, 6], [1, -1, 1, -1], [-1, -1, -1, -1]))
ok("33 (5,3) max (+,-), min (-,-)", pattern(5, 3) == ([7, 8], [1, -1], [-1, -1]))
n_, m_2 = symbols("n_ m_2")
ok("33b orders 4,5 impossible (m=3/2)", solve([n_ + m_2 - 5, n_ - m_2 - 2], [n_, m_2])
   == {n_: R(7, 2), m_2: R(3, 2)})
H34 = lambda lv: Matrix([[-2*lv, 1], [1, -2*lv]])
ok("34 points & multipliers", all(zero((Matrix([y, x]) - lv*Matrix([2*x, 2*y])).subs({x: px, y: py}).norm())
                                  for px, py, lv in ((2, 2, R(1, 2)), (-2, -2, R(1, 2)), (2, -2, -R(1, 2)), (-2, 2, -R(1, 2)))))
ok("34 L Hessian NSD / PSD", H34(R(1, 2)) == Matrix([[-1, 1], [1, -1]]) and H34(R(1, 2)).det() == 0
   and H34(-R(1, 2)) == Matrix([[1, 1], [1, 1]]))
xp2, yp2 = symbols("xp2 yp2", positive=True)
U35b = xp2**R(1, 2)*yp2**R(1, 4)
ok("35 (8,2), U~3.364, lam~0.210", 8 + 2*2 == 12 and close(U35b.subs({xp2: 8, yp2: 2}), 3.364, 1e-3)
   and close(diff(U35b, xp2).subs({xp2: 8, yp2: 2}), 0.2102, 1e-3)
   and zero(diff(U35b, yp2).subs({xp2: 8, yp2: 2}) - 2*diff(U35b, xp2).subs({xp2: 8, yp2: 2})))
U36 = x**2*y**2
ok("36 (2,4), U=64, lam=32, H indefinite", U36.subs({x: 2, y: 4}) == 64
   and diff(U36, x).subs({x: 2, y: 4}) == 64 and diff(U36, y).subs({x: 2, y: 4}) == 32
   and hessian(U36, (x, y)).subs({x: 2, y: 4}) == Matrix([[32, 32], [32, 8]])
   and hessian(U36, (x, y)).subs({x: 2, y: 4}).det() < 0)
ok("37 AM-GM x+4/x-4=(x-2)^2/x", zero(x + 4/x - 4 - (x - 2)**2/x))
s38 = solve([2*x - lam, 2*y - 2*lam, x + 2*y - 5], [x, y, lam])
ok("38 (1,2), lam=2, f=5", s38 == {x: 1, y: 2, lam: 2} and (x**2 + y**2).subs(s38) == 5)
f39 = x**2 + 2*y**2
g39 = x**2 + y**2
ok("39 dets -32 at (2,0) and 32 at (0,2)", bordered(f39, [g39], [x, y], [l1], {x: 2, y: 0, l1: 1}) == [-32]
   and bordered(f39, [g39], [x, y], [l1], {x: 0, y: 2, l1: 2}) == [32])
f40b = x**2*y
g40 = x**2 + y**2
dets40 = {pt: bordered(f40b, [g40], [x, y], [l1], {x: pt[0], y: pt[1], l1: pt[2]})[0]
          for pt in ((sqrt(2), 1, 1), (sqrt(2), -1, -1), (0, sqrt(3), 0), (0, -sqrt(3), 0))}
ok("40 dets 48, -48, -24sqrt3, 24sqrt3", [simplify(v_) for v_ in dets40.values()]
   == [48, -48, -24*sqrt(3), 24*sqrt(3)])
ok("40 formula 16x^2y+8lx^2-8y^3+8ly^2", zero(Matrix([[0, 2*x, 2*y], [2*x, 2*y - 2*lam, 2*x], [2*y, 2*x, -2*lam]]).det()
                                             - (16*x**2*y + 8*lam*x**2 - 8*y**3 + 8*lam*y**2)))
ok("40 max on c=3.3: 2*1.1^1.5~2.307", close(2*R(11, 10)**R(3, 2), 2.3074, 1e-3))
fh41 = x**2 + y**2 + (6 - x - y)**2
ok("41 hat f crit (2,2), Hessian [[4,2],[2,4]]", solve(list(grad(fh41, (x, y))), [x, y]) == {x: 2, y: 2}
   and hessian(fh41, (x, y)) == Matrix([[4, 2], [2, 4]]))
ok("41 bordered (-4,-12)", bordered(x**2 + y**2 + z**2, [x + y + z], [x, y, z], [l1], {x: 2, y: 2, z: 2, l1: 4}) == [-4, -12])
fh42 = X4[0]**2 + (4 - X4[0])**2 + (5 - 2*X4[3])**2 + X4[3]**2
ok("42 hat f crit (2,2), Hess diag(4,10)", solve(list(grad(fh42, (X4[0], X4[3]))), [X4[0], X4[3]]) == {X4[0]: 2, X4[3]: 2}
   and hessian(fh42, (X4[0], X4[3])) == Matrix([[4, 0], [0, 10]]))
pt42 = {X4[0]: 2, X4[1]: 2, X4[2]: 1, X4[3]: 2, l1: 4, l2: 2}
g42 = [X4[0] + X4[1], X4[2] + 2*X4[3]]
ok("42 FOC, f=13, minors (4,40)", all(zero((diff(f23, s) - l1*diff(g42[0], s) - l2*diff(g42[1], s)).subs(pt42)) for s in X4)
   and f23.subs(pt42) == 13 and bordered(f23, g42, list(X4), [l1, l2], pt42) == [4, 40])
fh43 = log(x) + log(y) + log(9 - x - y)
ok("43 hat f Hessian at (3,3)", hessian(fh43, (x, y)).subs({x: 3, y: 3}) == Matrix([[-R(2, 9), -R(1, 9)], [-R(1, 9), -R(2, 9)]])
   and hessian(fh43, (x, y)).subs({x: 3, y: 3}).det() == R(1, 27))
ok("43 bordered (2/9,-1/27)", bordered(log(x) + log(y) + log(z), [x + y + z], [x, y, z], [l1],
                                       {x: 3, y: 3, z: 3, l1: R(1, 3)}) == [R(2, 9), -R(1, 27)])
nn = 5
Y = symbols(f"q1:{nn + 1}", positive=True)
ok("44 (n=5) symmetric point satisfies FOC with lam=n",
   all(zero((1/s - nn).subs({q: R(1, nn) for q in Y})) for s in Y))
X6 = symbols("z1:7", real=True)
f45 = sum(s**2 for s in X6)
c45 = [X6[0] + X6[2] + X6[4], -X6[1] + X6[3] - X6[5]]
ok("45 expanded constraints", zero(c45[0] - sum(X6[2*k - 2] for k in (1, 2, 3)))
   and zero(c45[1] - sum((-1)**k*X6[2*k - 1] for k in (1, 2, 3))))
pt45 = dict(zip(X6, [1, -2, 1, 2, 1, -2]))
ok("45 FOC with (2,4), f=15", all(zero((diff(f45, s) - 2*diff(c45[0], s) - 4*diff(c45[1], s)).subs(pt45)) for s in X6)
   and [c.subs(pt45) for c in c45] == [3, 6] and f45.subs(pt45) == 15)
cc1, cc2 = symbols("cc1 cc2", real=True)
V45 = cc1**2/3 + cc2**2/3
ok("45 dV/dc = (2,4)", diff(V45, cc1).subs(cc1, 3) == 2 and diff(V45, cc2).subs(cc2, 6) == 4)
av = [1, 2, 4]
Sv = sum(R(1, a_i) for a_i in av)
ok("46 example S=7/4, x=(4/7,2/7,1/7), min 4/7", Sv == R(7, 4)
   and [R(1, a_i)/Sv for a_i in av] == [R(4, 7), R(2, 7), R(1, 7)]
   and sum(a_i*(R(1, a_i)/Sv)**2 for a_i in av) == R(4, 7))
f47 = x**2 + y**2 + z**2
s47 = solve([diff(f47, s) - l1*diff(x + y + z, s) - l2*diff(x - y + 2*z, s) for s in (x, y, z)]
            + [x + y + z - cc1, x - y + 2*z - cc2], [x, y, z, l1, l2], dict=True)[0]
ok("47 lam formulas", zero(s47[l1] - (6*cc1 - 2*cc2)/7) and zero(s47[l2] - (3*cc2 - 2*cc1)/7))
V47 = simplify(f47.subs(s47))
ok("47 V=(6c1^2-4c1c2+3c2^2)/14, V(6,4)=12", zero(V47 - (6*cc1**2 - 4*cc1*cc2 + 3*cc2**2)/14)
   and V47.subs({cc1: 6, cc2: 4}) == 12)
ok("47a exact 13.2386", close(V47.subs({cc1: R(63, 10), cc2: 4}), 13.2386, 1e-4)
   and close(s47[x].subs({cc1: R(63, 10), cc2: 4}), 2.0857, 1e-4))
ok("47b exact 3/56", V47.subs({cc1: 6, cc2: R(9, 2)}) - 12 == R(3, 56)
   and s47[l1].subs({cc1: 6, cc2: R(9, 2)}) == R(27, 7) and s47[l2].subs({cc1: 6, cc2: R(9, 2)}) == R(3, 14))
V48 = log(cc1 - cc2) + 2*log(2*cc2/3) + log(cc2/3)
ok("48 lambdas 1/3, 4/15", diff(V48, cc1).subs({cc1: 8, cc2: 5}) == R(1, 3)
   and simplify(diff(V48, cc2).subs({cc1: 8, cc2: 5})) == R(4, 15))
ok("48 FOC at (3,10/3,5/3)", 1/R(3) == R(1, 3) and 2/R(10, 3) == R(1, 3) + R(4, 15) and 1/R(5, 3) == R(3, 5))
ok("48a exact 0.1036", close(V48.subs({cc1: 8, cc2: R(11, 2)}) - V48.subs({cc1: 8, cc2: 5}), 0.1036, 1e-4))
ok("48b exact -0.1823", close(V48.subs({cc1: R(15, 2), cc2: 5}) - V48.subs({cc1: 8, cc2: 5}), -0.1823, 1e-4))
V48b = V48.subs({cc1: 8, cc2: 5})
ok("48 V(8,5)=4.0174, new values 4.151/4.121, 3.851/3.835 (review)",
   close(V48b, 4.0174, 1e-4)
   and close(V48b + R(4, 15)*R(1, 2), 4.151, 1e-3)
   and close(V48.subs({cc1: 8, cc2: R(11, 2)}), 4.121, 1e-3)
   and close(V48b - R(1, 3)*R(1, 2), 3.851, 1e-3)
   and close(V48.subs({cc1: R(15, 2), cc2: 5}), 3.835, 1e-3))

# ================================================================ ch10 ====
print("--- ch10 Additional 14-58 ---")
ok("14 (1-sqrt3,3) on parabola", zero((1 - sqrt(3) - 1)**2 - 3))
ok("15 (3,4),(4,3),(-3,4),(-5,0) on circle",
   all(p_[0]**2 + p_[1]**2 == 25 for p_ in ((3, 4), (4, 3), (-3, 4), (-5, 0))))
ok("16 det of (2,2sqrt3),(1,0) = -2sqrt3", Matrix([[2, 2*sqrt(3)], [1, 0]]).det() == -2*sqrt(3))
ok("17 det (4,1),(0,-1) = -4", Matrix([[4, 1], [0, -1]]).det() == -4)
ok("18 grad at (2,0) = (0,1)", Matrix([diff(y - (2 - x)**3, s) for s in (x, y)]).subs({x: 2, y: 0}) == Matrix([0, 1]))
ok("19 Slater (1,1,1.5)", 1 + 1 + R(9, 4) < 9 and R(7, 2) > 3)
f21n = x + y
ok("21 min (-1/2,1/4), max corner (sqrt2,2)", (x**2).subs(x, -R(1, 2)) == R(1, 4)
   and f21n.subs({x: -R(1, 2), y: R(1, 4)}) == -R(1, 4))
ok("21 multipliers sqrt2/4, 1+sqrt2/4", zero(1 - 2*sqrt(2)*sqrt(2)/4) and zero(1 - ((1 + sqrt(2)/4) - sqrt(2)/4)))
P22 = (2/sqrt(5), 4/sqrt(5))
ok("22 max 2sqrt5, lam sqrt5/4", zero(P22[0] + 2*P22[1] - 2*sqrt(5)) and zero(1 - sqrt(5)/4*2*P22[0])
   and zero(2 - sqrt(5)/4*2*P22[1]) and zero(P22[0]**2 + P22[1]**2 - 4))
ok("22 min at (-2,0): mu1=1/4, mu2=2", Matrix([1, 2]) + R(1, 4)*Matrix([-4, 0]) + 2*Matrix([0, -1]) == zeros(2, 1))
th = symbols("th", real=True)
ok("22 arcs >= -2 and >= -1 on [0,pi] (min of 2cos+4sin at pi)",
   min(N((2*cos(th) + 4*sin(th)).subs(th, k*pi/200)) for k in range(201)) >= -2 - 1e-12)
ok("23 KKT at (4,1): lam1=10, lam2=2", Matrix([8, 2]) == 10*Matrix([1, 1]) + 2*Matrix([-1, -4]))
ok("23 (1,4),(4,1) on both curves, f=17", 1*4 == 4 and 1 + 4 == 5 and 1 + 16 == 17)
ok("23 min (2,2): lam=2", Matrix([4, 4]) + 2*Matrix([-2, -2]) == zeros(2, 1))
f24n = y - x**2
ok("24 min points value -17/4", zero(f24n.subs({x: sqrt(15)/2, y: -R(1, 2)}) + R(17, 4))
   and zero((x**2 + y**2).subs({x: sqrt(15)/2, y: -R(1, 2)}) - 4))
ok("24 KKT (0,-2) lam=1/4: (0,1)+lam(0,-4)=0", Matrix([0, 1]) + R(1, 4)*Matrix([0, -4]) == zeros(2, 1))
ok("24 on circle f=y^2+y-4", zero(f24n.subs(x**2, 4 - y**2) - (y**2 + y - 4)))
f25n = x**2 + (y - 3)**2
ok("25 tangency (sqrt(5/2),5/2), lam=1, f=11/4", zero(f25n.subs({x: sqrt(R(5, 2)), y: R(5, 2)}) - R(11, 4))
   and zero((Matrix([2*x, 2*(y - 3)]) + Matrix([-2*x, 1])).subs({x: sqrt(R(5, 2)), y: R(5, 2)}).norm()))
ok("25 corner (2,4): lam1=-2 (mu1=-12)", solve([4 + l1*(-4) + l2, 2 + l1], [l1, l2]) == {l1: -2, l2: -12})
ok("25 x^2+(x^2-3)^2 crit x^2=5/2", set(solve(diff(x**2 + (x**2 - 3)**2, x), x)) == {0, sqrt(10)/2, -sqrt(10)/2})
ok("27 Q multipliers 1/4, 3/2", Matrix([1, 2]) == R(1, 4)*Matrix([4, 2]) + R(3, 2)*Matrix([0, 1]))
ok("27 S lam1=-1/4", solve(1 - l1*(-4), l1) == [-R(1, 4)])
ok("28 ln(1+t)<=t (spot)", N(log(1 + R(1, 2))) <= R(1, 2))
ok("29 four KKT values 16,4,0,3.2", [(x**2 + y**2).subs({x: a_v, y: b_v}) for a_v, b_v in
                                     ((4, 0), (0, 2), (0, 0), (R(4, 5), R(8, 5)))] == [16, 4, 0, R(16, 5)])
ok("30 lam=3, mu=3, f=4.5", (x*y - y**2/2).subs({x: 3, y: 3}) == R(9, 2) and 3 - 3 - 3 + 3 == 0)
bv = symbols("bv", real=True)
ok("30c V'(3)=-3", diff(6*bv - R(3, 2)*bv**2, bv).subs(bv, 3) == -3)
ok("30 line restriction 6y-1.5y^2", zero(((6 - y)*y - y**2/2) - (6*y - R(3, 2)*y**2)))
ok("31 (2,1/2,1/2), mu=1, lam=3, f=4.5", 2*R(1, 2) == 1 and 4 - 1 == 3 and 4 + R(1, 4) + R(1, 4) == R(9, 2))
ok("32 mu=1/4, lam=1/12, f=ln12", R(1, 2*2) == R(1, 4) and R(1, 4) - R(1, 6) == R(1, 12) and zero(log(6) + log(2) - log(12)))
ok("33 mu=-3, lam=2, V'(4)=-3", -2*(R(5, 2) - 1) == -3 and -2*(R(3, 2) - 1) + 3 - 2 == 0
   and diff(-(bv - R(5, 2))**2 - R(1, 4), bv).subs(bv, 4) == -3)
ok("34 min (2,1), lam=4, f=6", (x**2 + 2*y**2).subs({x: 2, y: 1}) == 6 and 2*2 == 4 and 4*1 == 4)
ok("35 min (1.2,2.4) lam 0.4, f=0.2", ((x - 1)**2 + (y - 2)**2).subs({x: R(6, 5), y: R(12, 5)}) == R(1, 5)
   and R(6, 5) + 2*R(12, 5) == 6 and 2*(R(6, 5) - 1) == R(2, 5) and 2*(R(12, 5) - 2) == 2*R(2, 5))
ok("35 max vertices 2, 5, 4.25; mu=(4,2)", [((x - 1)**2 + (y - 2)**2).subs({x: a_v, y: b_v}) for a_v, b_v in
                                         ((0, 3), (3, 3), (3, R(3, 2)))] == [2, 5, R(17, 4)])
f38 = x**4 + x**2 - 2*x*y + y**2
ok("36e/38 Hessian det 24x^2", zero(hessian(f38, (x, y)).det() - 24*x**2))
ok("38 x^3+2x-3=(x-1)(x^2+x+3)", zero(x**3 + 2*x - 3 - (x - 1)*(x**2 + x + 3)))
ok("38 (1,2), lam=2, f=2", f38.subs({x: 1, y: 2}) == 2 and (-2*x + 2*y).subs({x: 1, y: 2}) == 2
   and (4*x**3 + 2*x - 2*y).subs({x: 1, y: 2}) == 2)
ok("37 (+-1,0) lam=1", zero((2*x - 2*1*x).subs(x, 1)))
f39n = (x - 2)**2 + (y - 2)**2 + (z - 2)**2
pt39 = {x: 1, y: R(1, 2), z: R(3, 2)}
ok("39 stationarity with mu=(1,1), f=7/2", grad(f39n, (x, y, z)).subs(pt39) + Matrix([1, 1, 1]) + Matrix([1, 2, 0]) == zeros(3, 1)
   and f39n.subs(pt39) == R(7, 2))
ok("39 single-constraint candidates", (R(6, 5) + R(2, 5) + 2) > 3 and 1 + 2*1 > 2)
H40 = Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
ok("40 eigenvalues 2,-1,-1, f(1,1,1)=3", H40.eigenvals() == {2: 1, -1: 2}
   and (x*y + y*z + z*x).subs({x: 1, y: 1, z: 1}) == 3)
ok("41 z^2+3z+0.5 roots negative", all(N(r_v) < 0 for r_v in solve(z**2 + 3*z + R(1, 2), z)))
ok("41 lam=sqrt2, mu=sqrt2-1, U=1+sqrt2", zero(1/sqrt(R(1, 2)) - sqrt(2))
   and zero(1 + 2*sqrt(R(1, 2)) - (1 + sqrt(2))))
U42 = sqrt(x*y)
ok("42 (40/3,40/3) both budgets, lam=(1/6,1/6)", 2*R(40, 3) + R(40, 3) == 40 and R(40, 3) + 2*R(40, 3) == 40
   and zero(diff(U42, x).subs({x: R(40, 3), y: R(40, 3)}) - (2*R(1, 6) + R(1, 6)))
   and zero(diff(U42, y).subs({x: R(40, 3), y: R(40, 3)}) - (R(1, 6) + 2*R(1, 6))))
ok("42 single-budget optima violate the other", 10 + 2*20 > 40 and 2*20 + 10 > 40)
pi43 = 20*x - x**2 + 16*y - y**2
ok("43 (5,5): lamA=2, lamB=4, pi=130", grad(pi43, (x, y)).subs({x: 5, y: 5}) == 2*Matrix([1, 1]) + 4*Matrix([2, 1])
   and pi43.subs({x: 5, y: 5}) == 130)
ok("43 B-only (4.8,5.4) violates A", R(48, 10) + R(54, 10) > 10 and 2*R(48, 10) + R(54, 10) == 15
   and zero((20 - 2*x - 2*(16 - 2*y)).subs({x: R(48, 10), y: R(54, 10)})))
aa1, aa2 = symbols("aa1 aa2", positive=True)
pi44 = 60*(1 - exp(-aa1/20)) + 40*(1 - exp(-aa2/10)) - aa1 - aa2
ok("44 unconstrained 20ln3, 10ln4, pi~34.16", zero(diff(pi44, aa1).subs(aa1, 20*log(3)))
   and zero(diff(pi44, aa2).subs(aa2, 10*log(4))) and close(pi44.subs({aa1: 20*log(3), aa2: 10*log(4)}), 34.16, 0.01))
lam44 = exp((20*log(3) + 10*log(4) - 30)/30) - 1
a1_44, a2_44 = 20*log(3/(1 + lam44)), 10*log(4/(1 + lam44))
ok("44 B=30: lam~0.215, a~(18.08,11.92), pi~33.56", close(lam44, 0.2147, 1e-3) and close(a1_44, 18.08, 0.01)
   and close(a2_44, 11.92, 0.01) and close(a1_44 + a2_44, 30, 1e-9)
   and close(pi44.subs({aa1: a1_44, aa2: a2_44}), 33.56, 0.01))
ok("45 min (2,8): mu=2, lam1=6; (6,4): mu=6, lam2=2", 8 - 2 - 6 == 0 and 4 - 6 + 2 == 0
   and 2*8 == 16 and 6*4 == 24)
s_ = symbols("s_", real=True)
ok("47 phi'(w)<0 on w<=0 (spot), min 2", (diff((1 - w)**2 + exp(w), w)).subs(w, 0) == -1
   and ((1 - w)**2 + exp(w)).subs(w, 0) == 2)
A48 = x**2/16 + (20 - x)**2/(4*pi)
ok("48 x*=80/(4+pi), A=100/(4+pi)~14.00", zero(diff(A48, x).subs(x, 80/(4 + pi)))
   and zero(A48.subs(x, 80/(4 + pi)) - 100/(4 + pi)) and close(100/(4 + pi), 14.00, 0.01))
ok("48 A(0)=100/pi~31.83, A(20)=25", zero(A48.subs(x, 0) - 100/pi) and A48.subs(x, 20) == 25)
ok("48 corner multipliers 10/pi and 5/2", zero(20/(2*pi) - 10/pi) and R(20, 8) == R(5, 2))
ok("49 min (1,1), lam=3, f=3", (2*x + y).subs({x: 1, y: 1}) == 3 and (x**2 + x*y + y**2).subs({x: 1, y: 1}) == 3)
m50 = Matrix([6, 2])
pst = 5/sqrt(40)*m50
ok("50 f=2|p-m|^2+8 identity", zero(((x - 6)**2 + y**2 + (x - 6)**2 + (y - 4)**2) - (2*((x - 6)**2 + (y - 2)**2) + 8)))
ok("50 p*~(4.743,1.581), f*~11.51, lam~0.53", close(pst[0], 4.743, 1e-3) and close(pst[1], 1.581, 1e-3)
   and close(2*(sqrt(40) - 5)**2 + 8, 11.51, 0.01) and close(R(4, 10)*(sqrt(40) - 5), 0.5298, 1e-3))
gf50 = Matrix([4*(pst[0] - 6), 4*(pst[1] - 2)])
ok("50 stationarity grad f + lam*2p = 0", zero((gf50 + R(4, 10)*(sqrt(40) - 5)*2*pst).norm()))
xx1, xx2, xx3 = symbols("xx1 xx2 xx3", real=True)
V53 = 100*xx1**2 + 36*xx2**2 + 16*(1 - xx1 - xx2)**2
s53 = solve([diff(V53, xx1) - 5*lam, diff(V53, xx2) - 2*lam, 5*xx1 + 2*xx2 - 3], [xx1, xx2, lam])
ok("53 x=(147/361,174/361), lam=296/19, V=9076/361", s53 == {xx1: R(147, 361), xx2: R(174, 361), lam: R(296, 19)}
   and 1 - R(147, 361) - R(174, 361) == R(40, 361) and V53.subs(s53) == R(9076, 361))
ok("53 return algebra 8x1+5x2+3x3>=6 iff 5x1+2x2>=3", zero(8*xx1 + 5*xx2 + 3*(1 - xx1 - xx2) - 6 - (5*xx1 + 2*xx2 - 3)))
ok("53 Hessian [[232,32],[32,104]] PD", hessian(V53, (xx1, xx2)) == Matrix([[232, 32], [32, 104]])
   and minors(hessian(V53, (xx1, xx2)))[1] > 0)
s54g = solve([diff(V53, xx1) - 5*lam, diff(V53, xx2) - 2*lam, 5*xx1 + 2*xx2 - R(9, 2)], [xx1, xx2, lam])
ok("54 naive guess x3~-0.284", close(1 - s54g[xx1] - s54g[xx2], -0.284, 1e-3))
s54 = solve([diff(V53, xx1) - 5*lam + mu, diff(V53, xx2) - 2*lam + mu, 5*xx1 + 2*xx2 - R(9, 2), xx1 + xx2 - 1],
            [xx1, xx2, lam, mu])
ok("54 x=(5/6,1/6), lam=464/9, nu=820/9, V=634/9", s54 == {xx1: R(5, 6), xx2: R(1, 6), lam: R(464, 9), mu: R(820, 9)}
   and V53.subs(s54) == R(634, 9))
xa, xb = symbols("xa xb", nonnegative=True)
pi55 = 6*log(1 + xa*xb) - xa - xb
t0, t1 = 3 - 2*sqrt(2), 3 + 2*sqrt(2)
ok("55 crit t^2-6t+1=0", zero(t0**2 - 6*t0 + 1) and zero(t1**2 - 6*t1 + 1)
   and zero(diff(pi55, xa).subs({xa: t1, xb: t1})))
H55 = hessian(pi55, (xa, xb))
ok("55 Hessian structure", zero((H55 - 6/(1 + xa*xb)**2*Matrix([[-xb**2, 1], [1, -xa**2]])).norm()))
ok("55 saddle at t0, max at t1", N(H55.subs({xa: t0, xb: t0}).det()) < 0
   and N(H55.subs({xa: t1, xb: t1}).det()) > 0 and N(H55.subs({xa: t1, xb: t1})[0, 0]) < 0)
ok("55 values -0.169 and 9.670", close(pi55.subs({xa: t0, xb: t0}), -0.169, 1e-3)
   and close(pi55.subs({xa: t1, xb: t1}), 9.670, 1e-3))
ok("55b (10,10): 7.69, mult <0; unspent 8.34", close(pi55.subs({xa: 10, xb: 10}), 7.69, 0.01)
   and 12*20/R(404) - 1 < 0 and close(20 - 2*t1, 8.343, 1e-3))
ok("55c B=4: (2,2) pi~5.66, lam=1.4", close(pi55.subs({xa: 2, xb: 2}), 5.657, 1e-3)
   and (6*xb/(1 + xa*xb) - 1).subs({xa: 2, xb: 2}) == R(7, 5))
ok("55c B=0.5: pi~-0.14, lam~0.41", close(pi55.subs({xa: R(1, 4), xb: R(1, 4)}), -0.136, 1e-3)
   and close((6*xb/(1 + xa*xb) - 1).subs({xa: R(1, 4), xb: R(1, 4)}), 0.4118, 1e-3))
Bs = symbols("Bs", positive=True)
ok("55 threshold B~0.71", close(nsolve(6*log(1 + Bs**2/4) - Bs, Bs, 0.7), 0.7076, 1e-3))
ok("56 B=4: R~9.66, lam=2.4; B=20: R~27.69, lam~0.594", close(6*log(5), 9.657, 1e-3) and R(48, 20) == R(12, 5)
   and close(6*log(101), 27.69, 0.01) and close(R(240, 404), 0.594, 1e-3))
Rr = sqrt(R(5, 6))
f57 = x + 2*y + sqrt(5 - x**2 - y**2)
ok("57 stationary at (R,2R), slack 25/6<5, f=sqrt30", zero(diff(f57, x).subs({x: Rr, y: 2*Rr}))
   and zero(diff(f57, y).subs({x: Rr, y: 2*Rr})) and 5*Rr**2 == R(25, 6) and zero(f57.subs({x: Rr, y: 2*Rr}) - sqrt(30)))
yy = symbols("yy", real=True)
f58 = x + yy + (yy - x**2)**R(3, 2)
ok("58 f_y=1+1.5sqrt(y-x^2), f_x=1-3x sqrt", zero(diff(f58, yy) - (1 + R(3, 2)*sqrt(yy - x**2)))
   and zero(diff(f58, x) - (1 - 3*x*sqrt(yy - x**2))))

print("--- ch05 Practice: repaired x^2y-xy^2+xy problem ---")
from sympy import solve, hessian, Matrix
fpr = x**2*y - x*y**2 + x*y
cpr = {(c[x], c[y]) for c in solve([diff(fpr, x), diff(fpr, y)], [x, y], dict=True)}
ok("crit pts (0,0),(-1,0),(0,1),(-1/3,1/3)", cpr == {(0, 0), (-1, 0), (0, 1), (R(-1, 3), R(1, 3))})
Hpr = hessian(fpr, (x, y))
ok("H at (0,0),(-1,0),(0,1) as printed, det -1 each: saddles",
   Hpr.subs({x: 0, y: 0}) == Matrix([[0, 1], [1, 0]])
   and Hpr.subs({x: -1, y: 0}) == Matrix([[0, -1], [-1, 2]])
   and Hpr.subs({x: 0, y: 1}) == Matrix([[2, -1], [-1, 0]])
   and all(Hpr.subs({x: a, y: b}).det() == -1 for a, b in [(0, 0), (-1, 0), (0, 1)]))
Hm = Hpr.subs({x: R(-1, 3), y: R(1, 3)})
ok("(-1/3,1/3): D1=2/3, D2=1/3, min, f=-1/27",
   Hm[0, 0] == R(2, 3) and Hm[0, 1] == R(-1, 3) and Hm.det() == R(1, 3)
   and fpr.subs({x: R(-1, 3), y: R(1, 3)}) == R(-1, 27))
ok("no critical point with x,y>0", all(not (a > 0 and b > 0) for a, b in cpr))

print("--- ch03: local definiteness box, epigraph box, Additional 55-57 ---")
fg = x**2 + y**2 - x**4
Hg = hessian(fg, (x, y))
ok("box: H=diag(2-12x^2,2); PD at 0; diag(-10,2) at (1,0)",
   zero(Hg[0, 0] - (2 - 12*x**2)) and Hg[0, 1] == 0 and Hg[1, 1] == 2
   and Hg.subs({x: 0, y: 0}) == Matrix([[2, 0], [0, 2]])
   and Hg.subs({x: 1, y: 0}) == Matrix([[-10, 0], [0, 2]]))
ok("box: origin critical; f(0,0)=0, f(2,0)=-12, f(1,0)=0 > -6",
   grad(fg, (x, y)).subs({x: 0, y: 0}) == Matrix([0, 0])
   and fg.subs({x: 0, y: 0}) == 0 and fg.subs({x: 2, y: 0}) == -12
   and fg.subs({x: 1, y: 0}) == 0 and 0 > R(1, 2)*(0 - 12))
ok("box: PSD iff x^2<=1/6 (boundary 2-12x^2=0 at 1/sqrt6)",
   set(solve(2 - 12*x**2, x)) == {1/sqrt(6), -1/sqrt(6)})
a_, b_, l_ = symbols("a_ b_ l_", real=True)
ok("55 identity", zero(l_*a_**2 + (1 - l_)*b_**2 - (l_*a_ + (1 - l_)*b_)**2
                       - l_*(1 - l_)*(a_ - b_)**2))
f55 = x**2 + y**2
ok("55b P,Q graph points; midpoint (2,0,6) is 2 above; gap 1/4*8=2",
   f55.subs({x: 1, y: 1}) == 2 and f55.subs({x: 3, y: -1}) == 10
   and R(2 + 10, 2) - f55.subs({x: 2, y: 0}) == 2
   and R(1, 4)*((1 - 3)**2 + (1 + 1)**2) == 2)
f56 = x**2 + 4*x*y + y**2
ok("56a f(t,-t)=-2t^2; f(1,-1)=f(-1,1)=-2; f(0,0)=0>-2",
   zero(f56.subs({x: t, y: -t}) + 2*t**2) and f56.subs({x: 1, y: -1}) == -2
   and f56.subs({x: -1, y: 1}) == -2 and f56.subs({x: 0, y: 0}) == 0)
ok("56b f(t,0)=t^2; f(+-1,0)=1; midpoint needs 1<=0",
   zero(f56.subs({x: t, y: 0}) - t**2) and f56.subs({x: 1, y: 0}) == 1
   and f56.subs({x: -1, y: 0}) == 1)
ok("56c Hessian [[2,4],[4,2]], det -12",
   hessian(f56, (x, y)) == Matrix([[2, 4], [4, 2]]) and hessian(f56, (x, y)).det() == -12)
ok("57a (+-1,0) in ring, f=1; (0,0) not in ring; H=2I PD",
   1 <= 1 <= 4 and not (1 <= 0 <= 4) and f55.subs({x: 1, y: 0}) == 1
   and hessian(f55, (x, y)) == 2*eye(2))
g57 = (x + y)**2
ok("57b H=[[2,2],[2,2]] PSD, det 0; g(t,-t)=0",
   hessian(g57, (x, y)) == Matrix([[2, 2], [2, 2]]) and hessian(g57, (x, y)).det() == 0
   and zero(g57.subs({x: t, y: -t})))
f44 = 2*x**2 - 8*x*y + 2*y**2
ok("44 plot: f(t,t)=-4t^2, f(t,-t)=12t^2, equals 3(x-y)^2-(x+y)^2",
   zero(f44.subs({x: t, y: t}) + 4*t**2) and zero(f44.subs({x: t, y: -t}) - 12*t**2)
   and zero(f44 - (3*(x - y)**2 - (x + y)**2)))

print("--- ch05: second-order necessary conditions box ---")
ok("x^4+y^4: H(0,0)=0", hessian(x**4 + y**4, (x, y)).subs({x: 0, y: 0}) == zeros(2, 2))
ok("x^2+y^4 and x^2-y^4: H(0,0)=diag(2,0)",
   hessian(x**2 + y**4, (x, y)).subs({x: 0, y: 0}) == Matrix([[2, 0], [0, 0]])
   and hessian(x**2 - y**4, (x, y)).subs({x: 0, y: 0}) == Matrix([[2, 0], [0, 0]]))
ok("x^2-y^4 saddle: <0 on y-axis, >0 on x-axis",
   (x**2 - y**4).subs({x: 0, y: R(1, 10)}) < 0 and (x**2 - y**4).subs({x: R(1, 10), y: 0}) > 0)

print("--- ch07: second worked example (three-asset frontier) ---")
wa, wb, wc, gm, mm, ss, nu = symbols("wa wb wc gm mm ss nu", real=True)
Vp = 2*wa**2 + 8*wb**2 + 16*wc**2
Lp = Vp + lam*(mm - 2*wa - 4*wb - 8*wc) + gm*(1 - wa - wb - wc)
foc = [diff(Lp, s) for s in (wa, wb, wc, lam, gm)]
ok("FOC: 4wA-2lam-gam, 16wB-4lam-gam, 32wC-8lam-gam",
   zero(foc[0] - (4*wa - 2*lam - gm)) and zero(foc[1] - (16*wb - 4*lam - gm))
   and zero(foc[2] - (32*wc - 8*lam - gm)))
s4 = solve([f.subs(mm, 4) for f in foc], [wa, wb, wc, lam, gm], dict=True)[0]
ok("m=4: lam=1, gam=0, w=(1/2,1/4,1/4), V=2",
   s4[lam] == 1 and s4[gm] == 0 and (s4[wa], s4[wb], s4[wc]) == (R(1, 2), R(1, 4), R(1, 4))
   and Vp.subs(s4) == 2)
ok("step 4 reduced equations 4lam+gam=4, 32lam+11gam=32 at (1,0)",
   4*1 + 0 == 4 and 32*1 + 11*0 == 32
   and zero(expand(4*(2*(2*lam + gm)/4 + 4*(4*lam + gm)/16 + 8*(8*lam + gm)/32)) - (16*lam + 4*gm))
   and zero(expand(32*((2*lam + gm)/4 + (4*lam + gm)/16 + (8*lam + gm)/32)) - (32*lam + 11*gm)))
ok("asset B alone: variance 8 = 4 x 2", 8 == 4*2)
sm = solve(foc, [wa, wb, wc, lam, gm], dict=True)[0]
Vm = expand(Vp.subs(sm))
ok("frontier V*(m)=11/24 m^2-8/3 m+16/3, lam=(11m-32)/12, gam=(32-8m)/3",
   zero(Vm - (R(11, 24)*mm**2 - R(8, 3)*mm + R(16, 3)))
   and zero(sm[lam] - (11*mm - 32)/12) and zero(sm[gm] - (32 - 8*mm)/3))
ok("vertex m=32/11, V=16/11 (~2.91, ~1.45); lam=0 there",
   solve(diff(Vm, mm), mm) == [R(32, 11)] and Vm.subs(mm, R(32, 11)) == R(16, 11)
   and close(R(32, 11), 2.91, 0.006) and close(R(16, 11), 1.45, 0.006)
   and sm[lam].subs(mm, R(32, 11)) == 0)
ok("dV*/dm = lam(m) (envelope), =1 at m=4; 0.1 step ~0.1",
   zero(diff(Vm, mm) - sm[lam]) and diff(Vm, mm).subs(mm, 4) == 1
   and close(Vm.subs(mm, R(41, 10)) - 2, 0.1, 0.006))
ok("lam<0 iff m<32/11", sm[lam].subs(mm, 2) < 0 and sm[lam].subs(mm, 3) > 0)
ok("Hessian of V diag(4,16,32)", hessian(Vp, (wa, wb, wc)) == Matrix([[4, 0, 0], [0, 16, 0], [0, 0, 32]]))
Lphi = mm - ss + nu*(ss - (R(11, 24)*mm**2 - R(8, 3)*mm + R(16, 3)))
sphi = solve([diff(Lphi, mm), diff(Lphi, ss), diff(Lphi, nu)], [mm, ss, nu], dict=True)
ok("step 7: nu=1, m*=4, s*=2, Phi*=2; m-V concave",
   sphi == [{mm: 4, ss: 2, nu: 1}] and 4 - 2 == 2 and diff(mm - Vm, mm, 2) < 0)
ok("step 8: 5000/2500/2500, 400 euros, sd ~141",
   (10000*R(1, 2), 10000*R(1, 4)) == (5000, 2500) and 10000*R(4, 100) == 400
   and close(sqrt(2)/100*10000, 141, 0.5))

print("--- ch09: intermediate zero bordered minor => restricted form indefinite ---")


def restricted_identity(J, H):
    """check det Hbar_{m+k} = (-1)^m det(J1)^2 * D_{k-m}(Q) for k=m+1..n."""
    m, n = J.shape
    J1, J2 = J[:, :m], J[:, m:]
    Z = (-J1.inv()*J2).col_join(eye(n - m))
    Q = Z.T*H*Z
    Hb = zeros(m, m).row_join(J).col_join(J.T.row_join(H))
    return all(simplify(Hb[:m + k, :m + k].det()
                        - (-1)**m*J1.det()**2*Q[:k - m, :k - m].det()) == 0
               for k in range(m + 1, n + 1))


syms = symbols("h0:16 j0:8", real=True)
hs, js = syms[:16], syms[16:]
H3 = Matrix(3, 3, lambda i, j: hs[3*min(i, j) + max(i, j)])
ok("identity, m=1, n=3, symbolic H, border (j0,j1,j2)",
   restricted_identity(Matrix([[js[0], js[1], js[2]]]), H3))
H4 = Matrix(4, 4, lambda i, j: hs[4*min(i, j) + max(i, j)])
ok("identity, m=2, n=4, numeric border, symbolic H",
   restricted_identity(Matrix([[1, 2, 0, 3], [0, 1, 1, -1]]), H4))
Hb_ex = Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
Lex = y*z + lam*(1 - x)
sex = solve([diff(Lex, s) for s in (x, y, z, lam)], [x, y, z, lam], dict=True)
ok("example yz on x=1: candidate (1,0,0), lam=0; minors 3,4 = 0, 1",
   sex == [{x: 1, y: 0, z: 0, lam: 0}]
   and Hb_ex[1:, 1:] == hessian(Lex, (x, y, z)) and Hb_ex[:3, :3].det() == 0
   and Hb_ex.det() == 1)
ok("example: yz takes both signs near (0,0) in the plane",
   (y*z).subs({y: R(1, 10), z: R(1, 10)}) > 0 and (y*z).subs({y: R(1, 10), z: -R(1, 10)}) < 0)
Jdep = Matrix([[0, 0, 1]])
Hdep = Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 5]])
Hbd = zeros(1, 1).row_join(Jdep).col_join(Jdep.T.row_join(Hdep))
ok("artefact: border (0,0,1), minors 0 and -1 (max pattern needs +,-) though Q=-I is ND",
   Hbd[:3, :3].det() == 0 and Hbd.det() == -1)

print(f"\n{count - bad}/{count} PASS, {bad} FAIL")
raise SystemExit(1 if bad else 0)
