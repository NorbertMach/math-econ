"""Re-check every number stated in the worked examples of Volumes 1, 2, 4,
5, 6 and 9.  Volumes 3, 7 and 8 have their own scripts (worked_matrices.py,
worked_opt_lp.py)."""
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


x, t, u = symbols("x t u", positive=True)
xr, yr = symbols("xr yr", real=True)
n = symbols("n", positive=True, integer=True)
k = symbols("k", positive=True, integer=True)

# ================= Volume 1: algebra =================
y_, a_, b_ = symbols("y_ a_ b_", positive=True)
chk("v1 ch01 exponent simplification",
    ((8*x**-3*y_**6)/(27*x**3*y_**-3))**Rational(-2, 3), 9*x**4/(4*y_**6))
chk("v1 ch02 log equation root",
    Matrix(solve(Eq(log(x, 3) + log(x - 2, 3), 1), x)), Matrix([3]))
print("     v1 ch02 r = %.4f (claimed 0.0718)" % float(2**Rational(1, 10) - 1))
chk("v1 ch03 complex fraction",
    simplify((1/(x-1) - 1/(x+1))/(1/(x-1) + 1/(x+1))), 1/x)
h = symbols("h", nonzero=True)
chk("v1 ch03 difference quotient", simplify((1/(x+h) - 1/x)/h + 1/(x*(x+h))), 0)
chk("v1 ch04 biquadratic", expand(x**4 - 5*x**2 + 4 - (x-1)*(x+1)*(x-2)*(x+2)), 0)
chk("v1 ch04 x(x-4)=12", Matrix(sorted(solve(Eq(xr*(xr-4), 12), xr))), Matrix([-2, 6]))
chk("v1 ch04 -2P^2+16P-30", Matrix(sorted(solve(-2*xr**2 + 16*xr - 30, xr))), Matrix([3, 5]))
print("     v1 ch05 3/(x+1)<=2 :",
      solveset(3/(xr+1) <= 2, xr, S.Reals))
print("     v1 ch05 |2x-1|<=|x+4| :",
      solveset(Abs(2*xr-1) <= Abs(xr+4), xr, S.Reals))
al, I, px, py = symbols("al I px py", positive=True)
chk("v1 ch06 Cobb-Douglas demand budget", px*(al*I/px) + py*((1-al)*I/py), I)
avec = [3, -1, 2, 4]
abar = Rational(sum(avec), 4)
chk("v1 ch07 mean = 2", abar, 2)
chk("v1 ch07 identity", sum((v - abar)**2 for v in avec),
    sum(v**2 for v in avec) - 4*abar**2)
chk("v1 ch07 value 14", sum((v - abar)**2 for v in avec), 14)
chk("v1 ch08 partial fractions",
    together(-5/x + 6/(x+1) + 4/(x+1)**2) - (x**2-5)/(x*(x+1)**2), 0)
chk("v1 ch08 improper case",
    simplify(1 - 1/(x-1) + 4/(x-2) - x**2/((x-1)*(x-2))), 0)
alpha, M_, K_, L_, A_, be, rr = symbols("alpha M_ K_ L_ A_ be rr", positive=True)
chk("v1 ch09 MRS collapses",
    simplify((alpha*x**(alpha-1)*y_**(1-alpha)) /
             ((1-alpha)*x**alpha*y_**(-alpha)) - alpha*y_/((1-alpha)*x)), 0)
chk("v1 ch09 isolate K",
    simplify((alpha*A_*L_**be/rr)**(1/(1-alpha)) -
             (rr/(alpha*A_*L_**be))**(1/(alpha-1))), 0)

# ================= Volume 2: functions =================
print("     v2 ch01 domain of ln(x+3)/sqrt(4-x^2):",
      Intersection(solveset(4 - xr**2 > 0, xr, S.Reals),
                   solveset(xr + 3 > 0, xr, S.Reals)))
chk("v2 ch01 g(x) = 1 + 5/(x-2)", simplify((x+3)/(x-2) - (1 + 5/(x-2))), 0)
chk("v2 ch02 division 2x+1 over x-3", simplify((2*x+1)/(x-3) - (2 + 7/(x-3))), 0)
chk("v2 ch02 removable hole", simplify((x**2-1)/(x-1) - (x+1)), 0)
chk("v2 ch03 image of (4,2)", (-3*sqrt(2*4-4)+1), -5)
chk("v2 ch03 g(2)=3", 2*log(2-1) + 3, 3)
chk("v2 ch04 corner (2, 2sqrt3)", sqrt(16 - 4), 2*sqrt(3))
print("     v2 ch04 sqrt(2) =", float(sqrt(2)), "> 1, so (2,1) is outside")
print("     v2 ch05 solutions of 2sin^2x - sinx = 0 on [0,2pi]:",
      solveset(Eq(2*sin(xr)**2 - sin(xr), 0), xr, Interval(0, 2*pi)))
chk("v2 ch05 period of sin(2(x-pi/4))", 2*pi/2, pi)

# ================= Volume 4: foundations =================
chk("v4 ch01 polynomial ratio", limit((2*n**3-5*n)/(3*n**3+n**2+1), n, oo), Rational(2, 3))
chk("v4 ch01 n^2/2^n", limit(n**2/2**n, n, oo), 0)
for i, want in [(1, 1), (2, Rational(7, 2)), (3, Rational(19, 4)), (4, Rational(43, 8))]:
    chk(f"v4 ch01 a_{i}", (6 - 5*Rational(1, 2)**(n-1)).subs(n, i), want)
chk("v4 ch01 fixed point 6", solve(Eq(x, x/2 + 3), x)[0], 6)
kk = symbols("kk", integer=True, nonnegative=True)
chk("v4 ch02 geometric sum 5", summation(3*Rational(2, 5)**kk, (kk, 0, oo)), 5)
chk("v4 ch02 partial sum S4", summation(3*Rational(2, 5)**kk, (kk, 0, 4)), Rational(3093, 625))
print("     v4 ch02 S4 = %.4f (claimed 4.9488)" % float(Rational(3093, 625)))
chk("v4 ch02 telescoping = 1", summation(1/(k*(k+1)), (k, 1, oo)), 1)
chk("v4 ch02 harmonic diverges", summation(1/k, (k, 1, oo)), oo)
i_ = Rational(4, 100)/12
chk("v4 ch03 monthly payment",
    Float(round(float(200000*i_/(1 - (1+i_)**-240)), 2)), Float(1211.96))
print("     v4 ch03 EAR = %.6f (claimed 0.040742)" % float((1+i_)**12 - 1))
print("     v4 ch03 total paid = %.0f" % float(240*200000*i_/(1-(1+i_)**-240)))
npv = -1000 + sum(400/Rational(11, 10)**j for j in (1, 2, 3))
print("     v4 ch03 NPV = %.2f (claimed -5.26)" % float(npv))
print("     v4 ch03 IRR = %.4f (claimed 0.0970)"
      % float(nsolve(-1000 + sum(400/(1+x)**j for j in (1, 2, 3)), x, 0.1)))
chk("v4 ch04 conjugate limit", limit((sqrt(xr+4)-2)/xr, xr, 0), Rational(1, 4))
chk("v4 ch04 piecewise a=2,b=-1", Matrix([2, 1-2]), Matrix([2, -1]))
chk("v4 ch04 IVT signs", Matrix([(xr**3+xr-1).subs(xr, 0), (xr**3+xr-1).subs(xr, 1)]),
    Matrix([-1, 1]))

# ================= Volume 5: derivatives =================
chk("v5 ch01 quotient+chain",
    simplify(diff((3*x**2+1)**4/(2*x-5), x)
             - 2*(3*x**2+1)**3*(21*x**2-60*x-1)/(2*x-5)**2), 0)
chk("v5 ch01 f'(0) = -0.08", diff((3*x**2+1)**4/(2*x-5), x).subs(x, 0), Rational(-2, 25))
chk("v5 ch02 product+chain",
    simplify(diff(exp(2*x)*log(3*x**2+1), x)
             - exp(2*x)*(2*log(3*x**2+1) + 6*x/(3*x**2+1))), 0)
chk("v5 ch02 a^u rule", simplify(diff(5**(x**2), x) - 2*x*log(5)*5**(x**2)), 0)
chk("v5 ch02 x^x", simplify(diff(x**x, x) - x**x*(log(x)+1)), 0)
f3 = xr**4 - 4*xr**3
chk("v5 ch03 f' factored", factor(diff(f3, xr)), 4*xr**2*(xr-3))
chk("v5 ch03 f'' factored", factor(diff(f3, xr, 2)), 12*xr*(xr-2))
chk("v5 ch03 f(3) = -27", f3.subs(xr, 3), -27)
chk("v5 ch03 f(2) = -16", f3.subs(xr, 2), -16)
chk("v5 ch04 P3", series(log(1+xr), xr, 0, 4).removeO(), xr - xr**2/2 + xr**3/3)
err = float(log(Rational(3, 2)) - Rational(1, 2) + Rational(1, 8) - Rational(1, 24))
print("     v5 ch04 error at x=.5: %.6f  bound 0.015625" % abs(err))
yy = Function("yy")
folium = xr**3 + yy(xr)**3 - 6*xr*yy(xr)
sl = solve(Eq(diff(folium, xr), 0), Derivative(yy(xr), xr))[0]
chk("v5 ch05 slope at (3,3) = -1", sl.subs({yy(xr): 3, xr: 3}), -1)
print("     v5 ch05 vertical tangents at y =",
      solve(Eq((y_**2/2)**3 + y_**3 - 3*y_**3, 0), y_))
chk("v5 ch06 elasticity at p=20", -2*Rational(20, 60), Rational(-2, 3))
chk("v5 ch06 revenue max p=25", solve(diff(100*x - 2*x**2, x), x)[0], 25)
chk("v5 ch06 elasticity at 25 = -1", -2*Rational(25, 50), -1)
print("     v5 ch06 sqrt(101) approx 10.05, true %.6f" % float(sqrt(101)))
chk("v5 ch07 (cos x)^(1/x^2)", limit(cos(xr)**(1/xr**2), xr, 0), exp(Rational(-1, 2)))
chk("v5 ch07 (x+sin x)/x", limit((xr+sin(xr))/xr, xr, oo), 1)
f8 = xr**2/(xr-1)
chk("v5 ch08 oblique asymptote", simplify(f8 - (xr + 1 + 1/(xr-1))), 0)
chk("v5 ch08 f'", simplify(diff(f8, xr) - xr*(xr-2)/(xr-1)**2), 0)
chk("v5 ch08 f''", simplify(diff(f8, xr, 2) - 2/(xr-1)**3), 0)
chk("v5 ch08 f(2) = 4", f8.subs(xr, 2), 4)
chk("v5 ch09 ellipse semi-axes", Matrix([sqrt(4), sqrt(4)/2]), Matrix([2, 1]))
yv = symbols("yv", positive=True)
f10 = x**3*yv**2 + exp(2*x)*log(yv)
chk("v5 ch10 f_x(1,1) = 3", diff(f10, x).subs({x: 1, yv: 1}), 3)
chk("v5 ch10 f_y(1,1) = 2+e^2", diff(f10, yv).subs({x: 1, yv: 1}), 2 + exp(2))
print("     v5 ch10 |grad| = %.3f, D_u f = %.3f"
      % (float(sqrt(9 + (2+exp(2))**2)), float((3*3 + 4*(2+exp(2)))/5)))
KK, LL = symbols("KK LL", positive=True)
Q = 10*KK**Rational(3, 10)*LL**Rational(7, 10)
print("     v5 ch10 MP_K = %.3f, MP_L = %.3f, MRTS = %.3f"
      % (float(diff(Q, KK).subs({KK: 100, LL: 50})),
         float(diff(Q, LL).subs({KK: 100, LL: 50})),
         float((diff(Q, LL)/diff(Q, KK)).subs({KK: 100, LL: 50}))))
f11 = xr**3 - 3*xr*yr + yr**3
crit = sorted(solve([diff(f11, xr), diff(f11, yr)], [xr, yr]),
              key=lambda p: (re(p[0]), re(p[1])))
real_crit = [c for c in crit if all(im(v) == 0 for v in c)]
chk("v5 ch11 critical points", Matrix(real_crit), Matrix([[0, 0], [1, 1]]))
H11 = hessian(f11, (xr, yr))
chk("v5 ch11 det H(0,0) = -9", H11.subs({xr: 0, yr: 0}).det(), -9)
chk("v5 ch11 det H(1,1) = 27", H11.subs({xr: 1, yr: 1}).det(), 27)
chk("v5 ch11 f(1,1) = -1", f11.subs({xr: 1, yr: 1}), -1)
chk("v5 ch12 chain rule", diff((t**2)**2*log(t), t), 4*t**3*log(t) + t**3)
f13 = xr**2 + 3*xr*yr - yr**2
chk("v5 ch13 gradient (8,-1)",
    Matrix([diff(f13, v).subs({xr: 1, yr: 2}) for v in (xr, yr)]), Matrix([8, -1]))
chk("v5 ch13 D_u f = 4", (8*Rational(3, 5) + (-1)*Rational(4, 5)), 4)
chk("v5 ch13 Jacobian det at (1,2) = 4",
    Matrix([[2*xr, -1], [yr, xr]]).subs({xr: 1, yr: 2}).det(), 4)
F14 = xr**2*yr + yr**3 - 10
chk("v5 ch14 point on curve", F14.subs({xr: 1, yr: 2}), 0)
chk("v5 ch14 F_y = 13", diff(F14, yr).subs({xr: 1, yr: 2}), 13)
chk("v5 ch14 y\' = -4/13", (-diff(F14, xr)/diff(F14, yr)).subs({xr: 1, yr: 2}),
    Rational(-4, 13))
print("     v5 ch14 y(1.01) = %.6f, linear prediction %.6f"
      % (float(nsolve(F14.subs(xr, Rational(101, 100)), yr, 2)),
         float(2 - Rational(4, 13)/100)))
cd = KK**Rational(3, 10)*LL**Rational(7, 10)
chk("v5 ch14 Euler", simplify(KK*diff(cd, KK) + LL*diff(cd, LL) - cd), 0)

# ================= Volume 6: integrals =================
chk("v6 ch01 antiderivative",
    simplify(diff(3*x**5/5 - 2*log(x) + Rational(8, 3)*x**Rational(3, 2)
                  + Rational(5, 2)/x**2, x)
             - (3*x**4 - 2/x + 4*sqrt(x) - 5/x**3)), 0)
chk("v6 ch02 log pattern", simplify(diff(log(x**2+3*x+7), x) - (2*x+3)/(x**2+3*x+7)), 0)
chk("v6 ch02 root substitution",
    simplify(diff(Rational(2, 9)*(x**3+1)**Rational(3, 2), x) - x**2*sqrt(x**3+1)), 0)
chk("v6 ch02 definite", integrate(exp(x)/(1+exp(x)), (x, 0, log(2))), log(Rational(3, 2)))
chk("v6 ch03 parts twice", integrate(x**2*exp(-x), x), -exp(-x)*(x**2+2*x+2))
chk("v6 ch03 ln x", integrate(log(x), x), x*log(x) - x)
chk("v6 ch04 distinct linear",
    simplify(together(Rational(8, 3)/(x-1) + Rational(1, 3)/(x+2))
             - (3*x+5)/((x-1)*(x+2))), 0)
chk("v6 ch04 repeated factor",
    simplify(diff(log(x) - log(x+1) + 1/(x+1), x) - 1/(x*(x+1)**2)), 0)
chk("v6 ch05 definite = 9", integrate(2*x - 3/sqrt(x), (x, 1, 4)), 9)
chk("v6 ch05 average = 3", integrate(2*x - 3/sqrt(x), (x, 1, 4))/3, 3)
chk("v6 ch05 FTC I", diff(integrate(exp(-t**2), (t, 0, x**2)), x), 2*x*exp(-x**4))
chk("v6 ch05 divergent through a pole", integrate(1/xr**2, (xr, -1, 1)), oo)
rp = symbols("rp", positive=True)
chk("v6 ch06 perpetuity", integrate(exp(-rp*t), (t, 0, oo)), 1/rp)
chk("v6 ch06 growing flow", integrate(t*exp(-rp*t), (t, 0, oo)), 1/rp**2)
q_ = symbols("q_", positive=True)
chk("v6 ch07 equilibrium (20,60)", Matrix([20, 100-2*20]), Matrix([20, 60]))
chk("v6 ch07 CS = 400", integrate(100-2*q_, (q_, 0, 20)) - 60*20, 400)
chk("v6 ch07 PS = 400", 60*20 - integrate(20+2*q_, (q_, 0, 20)), 400)
print("     v6 ch07 PV = %.0f (claimed 41210)"
      % float(5000*integrate(exp(-Rational(4, 100)*t), (t, 0, 10))))
chk("v6 ch08 rectangle = 8", integrate(integrate(x*yv, (yv, 1, 3)), (x, 0, 2)), 8)
chk("v6 ch08 type I", integrate(integrate(x+yv, (yv, x**2, 2*x)), (x, 0, 2)),
    Rational(52, 15))
chk("v6 ch08 type II", integrate(integrate(x+yv, (x, yv/2, sqrt(yv))), (yv, 0, 4)),
    Rational(52, 15))
chk("v6 ch08 order reversal",
    integrate(integrate(exp(-yv**2), (x, 0, yv)), (yv, 0, 1)), (1-exp(-1))/2)

# ================= Volume 9: mixed =================
fm = log(x)/x
chk("v9 A left edge", limit(fm, x, 0, "+"), -oo)
chk("v9 A right edge", limit(fm, x, oo), 0)
chk("v9 A maximum at e", solve(diff(fm, x), x)[0], E)
chk("v9 A inflection at e^{3/2}", solve(diff(fm, x, 2), x)[0], exp(Rational(3, 2)))
chk("v9 A integral diverges", integrate(fm, (x, 1, oo)), oo)
lam_ = symbols("lam_", positive=True)
solB = solve([Eq(1/x, lam_), Eq(2/yv, lam_), Eq(x+yv, 12)], [x, yv, lam_], dict=True)[0]
chk("v9 B x* = 4", solB[x], 4)
chk("v9 B y* = 8", solB[yv], 8)
chk("v9 B lambda = 1/4", solB[lam_], Rational(1, 4))
chk("v9 B bordered Hessian = 3/32",
    Matrix([[0, 1, 1], [1, Rational(-1, 16), 0], [1, 0, Rational(-1, 32)]]).det(),
    Rational(3, 32))
MM = symbols("MM", positive=True)
chk("v9 B dV/dM = lambda",
    diff(log(MM/3) + 2*log(2*MM/3), MM).subs(MM, 12), Rational(1, 4))
PV10 = 8000*integrate(exp(-Rational(15, 100)*t), (t, 0, 10))
print("     v9 C PV(10) = %.0f (claimed 41433)" % float(PV10))
chk("v9 C PV(inf) = 160000/3", 8000/Rational(15, 100), Rational(160000, 3))
print("     v9 C half-life = %.3f years (claimed 4.62)"
      % float(log(2)/Rational(15, 100)))
print("     v9 C ten-year share = %.4f (claimed 0.777)"
      % float(PV10/(8000/Rational(15, 100))))

print("\n" + ("ALL WORKED EXAMPLES VERIFIED" if ok else "MISMATCHES PRESENT"))
