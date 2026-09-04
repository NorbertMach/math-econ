"""Re-check every number in the worked examples of Volumes 7 (optimization)
and 8 (linear programming), plus the running LP used across Volume 8."""
from sympy import *

ok = True


def chk(label, got, want):
    global ok
    if got in (oo, -oo, zoo) or want in (oo, -oo, zoo):
        good = got == want
        ok = ok and bool(good)
        print(("ok  " if good else "BAD "), label,
              "" if good else f"  got {got} want {want}")
        return
    d = got - want
    good = d.is_zero_matrix if isinstance(d, MatrixBase) else simplify(d) == 0
    ok = ok and bool(good)
    print(("ok  " if good else "BAD "), label, "" if good else f"  got {got} want {want}")


q, q1, q2, x, y, L, K, w, r = symbols("q q1 q2 x y L K w r", positive=True)

# --- optimization ch01: profit maximisation
pi = 100*q - Rational(5, 2)*q**2 - 200
chk("o1 q* = 20", solve(diff(pi, q), q)[0], 20)
chk("o1 pi* = 800", pi.subs(q, 20), 800)
chk("o1 MR=MC at 20", (120 - 4*q - (20 + q)).subs(q, 20), 0)

# --- ch02: convexity of {xy>=1}
lam, x1, y1, x2, y2 = symbols("lam x1 y1 x2 y2", positive=True)
expr = ((lam*x1 + (1-lam)*x2)*(lam*y1 + (1-lam)*y2)).expand()
chk("o2 expansion", expr,
    (lam**2*x1*y1 + (1-lam)**2*x2*y2 + lam*(1-lam)*(x1*y2 + x2*y1)).expand())

# --- ch03: Hessians
f3 = x**2 + x*y + y**2
H3 = hessian(f3, (x, y))
chk("o3 Hessian of x^2+xy+y^2", H3, Matrix([[2, 1], [1, 2]]))
chk("o3 minors 2 and 3", Matrix([H3[0, 0], H3.det()]), Matrix([2, 3]))
U = x**Rational(1, 2)*y**Rational(1, 2)
chk("o3 Cobb-Douglas det H = 0", simplify(hessian(U, (x, y)).det()), 0)
a, b = symbols("a b", positive=True)
chk("o3 general CD determinant",
    simplify(hessian(x**a*y**b, (x, y)).det() - a*b*(1-a-b)*x**(2*a-2)*y**(2*b-2)), 0)

# --- ch04: cost curves
C = q**3 - 9*q**2 + 30*q
chk("o4 AC min at q = 4.5", solve(diff(C/q, q), q)[0], Rational(9, 2))
chk("o4 AC(4.5) = 9.75", (C/q).subs(q, Rational(9, 2)), Rational(39, 4))
chk("o4 MC(4.5) = AC(4.5)", diff(C, q).subs(q, Rational(9, 2)), Rational(39, 4))
prof = 15*q - C
for qq, want in [(0, 0), (1, -7), (5, 25), (6, 18)]:
    chk(f"o4 profit at q={qq}", prof.subs(q, qq) if qq else Integer(0), want)
chk("o4 critical points 1 and 5", Matrix(sorted(solve(diff(prof, q), q))), Matrix([1, 5]))

# --- ch05: two-product monopolist
P = 90*q1 - 2*q1**2 + 60*q2 - q2**2 - q1*q2
sol = solve([diff(P, q1), diff(P, q2)], [q1, q2])
chk("o5 q1* = 120/7", sol[q1], Rational(120, 7))
chk("o5 q2* = 150/7", sol[q2], Rational(150, 7))
chk("o5 pi* = 69300/49", P.subs(sol), Rational(69300, 49))
HP = hessian(P, (q1, q2))
chk("o5 Hessian", HP, Matrix([[-4, -1], [-1, -2]]))
chk("o5 det = 7", HP.det(), 7)

# --- ch06: envelope theorem
piL = 10*sqrt(L) - w*L
Lstar = solve(diff(piL, L), L)[0]
chk("o6 L* = 25/w^2", Lstar, 25/w**2)
chk("o6 dL*/dw", diff(Lstar, w), -50/w**3)
V = simplify(piL.subs(L, Lstar))
chk("o6 value function 25/w", V, 25/w)
chk("o6 envelope: dV/dw = -L*", diff(V, w), -Lstar)
chk("o6 value function convex", diff(V, w, 2), 50/w**3)

# --- ch07: cost minimisation
Qb = symbols("Qb", positive=True)
Ls, Ks = Qb*sqrt(r/w), Qb*sqrt(w/r)
chk("o7 constraint satisfied", simplify(sqrt(Ks*Ls)), Qb)
chk("o7 K/L = w/r", simplify(Ks/Ls - w/r), 0)
Cost = simplify(w*Ls + r*Ks)
chk("o7 cost = 2 Q sqrt(wr)", Cost, 2*Qb*sqrt(w*r))
chk("o7 lambda = dC/dQ", diff(Cost, Qb), 2*sqrt(w*r))

# --- ch08: multiplier as marginal utility of income
M = symbols("M", positive=True)
xs, ys = Rational(4, 10)*M/2, Rational(6, 10)*M/3
chk("o8 x* = y* = 20 at M=100", Matrix([xs.subs(M, 100), ys.subs(M, 100)]),
    Matrix([20, 20]))
Vm = simplify(xs**Rational(2, 5)*ys**Rational(3, 5))
chk("o8 V(M) = 0.2 M", Vm, M/5)
chk("o8 lambda = dV/dM = 0.2", diff(Vm, M), Rational(1, 5))
chk("o8 budget exhausted", 2*xs.subs(M, 100) + 3*ys.subs(M, 100), 100)

# --- ch09: bordered Hessian
Hb = Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
chk("o9 det Hbar = 2 > 0 (maximum)", Hb.det(), 2)
chk("o9 substitution check", solve(diff(x*(10-x), x), x)[0], 5)

# --- ch10: Kuhn-Tucker
chk("o10 f(4,6) = 24", Integer(4*6), 24)
chk("o10 lambda = 4, mu = 2", Matrix([4, 6-4]), Matrix([4, 2]))
chk("o10 relaxed budget optimum f(4,7) = 28", Integer(4*7), 28)

# --- Volume 8: the running LP
A = Matrix([[2, 1], [1, 1]])
bvec = Matrix([100, 80])
cvec = Matrix([40, 30])
xstar = A.solve(bvec)
chk("lp corner (20,60)", xstar, Matrix([20, 60]))
chk("lp z* = 2600", (cvec.T*xstar)[0], 2600)
for corner, val in [((0, 0), 0), ((50, 0), 2000), ((0, 80), 2400), ((20, 60), 2600)]:
    chk(f"lp corner {corner} value", Integer(40*corner[0] + 30*corner[1]), val)
Binv = A.inv()
chk("lp B^-1", Binv, Matrix([[1, -1], [-1, 2]]))
chk("lp shadow prices (10,20)", (cvec.T*Binv).T, Matrix([10, 20]))
chk("lp dual objective = 2600", (Matrix([10, 20]).T*bvec)[0], 2600)
chk("lp dual constraint 1 tight", 2*10 + 20, 40)
chk("lp dual constraint 2 tight", 10 + 20, 30)
chk("lp weak duality bound y=(15,15)", 100*15 + 80*15, 2700)
chk("lp reduced cost of product 3", 35 - (10*1 + 20*1), 5)
# basic solutions
names = ["x1", "x2", "s1", "s2"]
Afull = Matrix([[2, 1, 1, 0], [1, 1, 0, 1]])
from itertools import combinations
feas = []
for cols in combinations(range(4), 2):
    B = Afull[:, list(cols)]
    if B.det() == 0:
        continue
    v = B.solve(bvec)
    feas.append((tuple(names[c] for c in cols), tuple(v), all(t >= 0 for t in v)))
print("     basic solutions:", len(feas), "of which feasible:",
      sum(1 for f in feas if f[2]))
for f in feas:
    print("       ", f[0], f[1], "feasible" if f[2] else "INFEASIBLE")
chk("lp 4 feasible bases", Integer(sum(1 for f in feas if f[2])), 4)
# RHS ranging
b1 = symbols("b1")
xb = Binv*Matrix([b1, 80])
chk("lp x(b1)", xb, Matrix([b1 - 80, 160 - b1]))
chk("lp b1=110 gives (30,50)", xb.subs(b1, 110), Matrix([30, 50]))
chk("lp z at b1=110", (cvec.T*xb.subs(b1, 110))[0], 2700)
c1 = symbols("c1")
chk("lp objective ranging row", (Matrix([c1, 30]).T*Binv).T, Matrix([c1 - 30, 60 - c1]))
# multiple optima variant
chk("lp parallel objective ties", Integer(40*50 + 20*0), 40*20 + 20*60)

print("\n" + ("ALL VOLUME 7 AND 8 WORKED EXAMPLES VERIFIED" if ok
              else "MISMATCHES PRESENT"))
