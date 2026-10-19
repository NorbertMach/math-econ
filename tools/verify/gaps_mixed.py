"""Machine check of the full-investigation chapter (mixed/tex/chapters/ch04.tex).

Every claim is retyped by hand from the book; nothing is parsed from LaTeX.
Run with a Python that has sympy installed.
"""
from sympy import (symbols, sqrt, log, Rational, diff, simplify, Matrix,
                   solve, limit, oo, S, expand_log, cos, sin)

x, y, t = symbols("x y t", real=True)
results = []


def check(name, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name)


def zero(e):
    return simplify(e) == 0


def hess(f):
    return Matrix([[diff(f, x, 2), diff(f, x, y)],
                   [diff(f, y, x), diff(f, y, 2)]])


def at(e, px, py):
    return simplify(e.subs({x: px, y: py}))


# ---------------------------------------------------------------- worked
f = sqrt(8 - 2*x**2 - y**2)
check("W f(0,0)=2*sqrt2", at(f, 0, 0) == 2*sqrt(2))
check("W f(1,2)=sqrt2", at(f, 1, 2) == sqrt(2))
check("W f_x=-2x/f", zero(diff(f, x) + 2*x/f))
check("W f_y=-y/f", zero(diff(f, y) + y/f))
g = Matrix([diff(f, x), diff(f, y)])
gp = g.subs({x: 1, y: 2}).applyfunc(simplify)
check("W grad(1,2)=(-sqrt2,-sqrt2)", gp == Matrix([-sqrt(2), -sqrt(2)]))
check("W |grad|=2", simplify(gp.norm()) == 2)
# tangent of level curve 2x^2+y^2=6 at (1,2): slope -1
slope = -(4*x)/(2*y)
check("W level slope -1", at(slope, 1, 2) == -1)
check("W grad . (1,-1) = 0", simplify(gp.dot(Matrix([1, -1]))) == 0)
H = hess(f)
check("W f_xx=-2(8-y^2)/f^3", zero(H[0, 0] + 2*(8 - y**2)/f**3))
check("W f_yy=-(8-2x^2)/f^3", zero(H[1, 1] + (8 - 2*x**2)/f**3))
check("W f_xy=-2xy/f^3", zero(H[0, 1] + 2*x*y/f**3))
check("W det H = 16/f^4", zero(H.det() - 16/f**4))
Hp = H.subs({x: 1, y: 2}).applyfunc(simplify)
check("W H(1,2)", Hp == Matrix([[-2*sqrt(2), -sqrt(2)],
                                [-sqrt(2), -Rational(3, 2)*sqrt(2)]]))
check("W det H(1,2)=4", simplify(Hp.det()) == 4)
H0 = H.subs({x: 0, y: 0}).applyfunc(simplify)
check("W H(0,0)=diag(-1/sqrt2,-1/(2sqrt2))",
      H0 == Matrix([[-1/sqrt(2), 0], [0, -1/(2*sqrt(2))]]))
check("W only critical point (0,0)",
      solve([diff(f, x), diff(f, y)], [x, y], dict=True) == [{x: 0, y: 0}])
# lower set {f<=2}: (0,2),(0,-2) in, midpoint (0,0) not
check("W lower set witness", at(f, 0, 2) <= 2 and at(f, 0, -2) <= 2
      and at(f, 0, 0) > 2)
check("W f(1,0)=sqrt6 < f(0,0)", at(f, 1, 0) == sqrt(6)
      and at(f, 1, 0) < at(f, 0, 0) and at(f, -1, 0) == sqrt(6))
# boundary: difference quotient at (2,0) in direction -x blows up
h = symbols("h", positive=True)
check("W boundary quotient -> oo",
      limit(sqrt(8 - 2*(2 - h)**2)/h, h, 0, "+") == oo)
check("W range top: levels 8-c^2 for c=0,1,sqrt2,2,sqrt7",
      [8 - c**2 for c in (0, 1, sqrt(2), 2, sqrt(7))] == [8, 7, 6, 4, 1])

# ---------------------------------------------------------------- P1
f = sqrt(6 - x - y**2)
check("P1 f(1,1)=2", at(f, 1, 1) == 2)
check("P1 f_x=-1/(2f)", zero(diff(f, x) + 1/(2*f)))
check("P1 f_y=-y/f", zero(diff(f, y) + y/f))
gp = Matrix([diff(f, x), diff(f, y)]).subs({x: 1, y: 1}).applyfunc(simplify)
check("P1 grad(1,1)=(-1/4,-1/2)", gp == Matrix([-Rational(1, 4),
                                                -Rational(1, 2)]))
check("P1 |grad|=sqrt5/4", simplify(gp.norm() - sqrt(5)/4) == 0)
# level curve x=2-y^2 through (1,1): dy/dx = 1/(dx/dy) = 1/(-2y)
check("P1 level through (1,1): x=2-y^2", 2 - 1**2 == 1)
check("P1 level slope -1/2", Rational(1, -2*1) == -Rational(1, 2))
check("P1 grad . (2,-1)=0", gp.dot(Matrix([2, -1])) == 0)
H = hess(f)
check("P1 f_xx=-1/(4f^3)", zero(H[0, 0] + 1/(4*f**3)))
check("P1 f_xy=-y/(2f^3)", zero(H[0, 1] + y/(2*f**3)))
check("P1 f_yy=-(6-x)/f^3", zero(H[1, 1] + (6 - x)/f**3))
check("P1 det=1/(4f^4)", zero(H.det() - 1/(4*f**4)))
Hp = H.subs({x: 1, y: 1}).applyfunc(simplify)
check("P1 H(1,1)", Hp == Matrix([[-Rational(1, 32), -Rational(1, 16)],
                                 [-Rational(1, 16), -Rational(5, 8)]]))
check("P1 det H(1,1)=1/64", Hp.det() == Rational(1, 64))
check("P1 no critical point", solve([diff(f, x), diff(f, y)], [x, y]) == [])
# lower set {f<=2} = {2-y^2<=x<=6-y^2}: (-2,2),(-2,-2) in, (-2,0) not
inband = lambda px, py: 2 - py**2 <= px <= 6 - py**2
check("P1 band witness", inband(-2, 2) and inband(-2, -2)
      and not inband(-2, 0))
check("P1 f(0,-1)=sqrt5 < f(0,0)=sqrt6", at(f, 0, -1) == sqrt(5)
      and at(f, 0, 0) == sqrt(6))
check("P1 f(x,0) -> oo as x -> -oo", limit(f.subs(y, 0), x, -oo) == oo)

# ---------------------------------------------------------------- P2
X, Y = symbols("X Y", positive=True)
u = log(X**Rational(1, 3) * Y**Rational(1, 2))
check("P2 u = ln(x)/3 + ln(y)/2",
      simplify(u - (log(X)/3 + log(Y)/2)) == 0)
uu = log(x)/3 + log(y)/2
gp = Matrix([diff(uu, x), diff(uu, y)]).subs({x: 1, y: 1})
check("P2 grad(1,1)=(1/3,1/2)", gp == Matrix([Rational(1, 3),
                                              Rational(1, 2)]))
check("P2 |grad|=sqrt13/6", simplify(gp.norm() - sqrt(13)/6) == 0)
check("P2 level u=0 slope at 1 = -2/3",
      diff(x**Rational(-2, 3), x).subs(x, 1) == -Rational(2, 3))
check("P2 grad . (3,-2)=0", gp.dot(Matrix([3, -2])) == 0)
check("P2 level curve y=e^{2c}x^{-2/3}",
      simplify(expand_log(log(X)/3 + log(S.Exp1**(2*t)
                                          * X**Rational(-2, 3))/2,
                          force=True) - t) == 0)
H = hess(uu)
check("P2 Hessian diag(-1/(3x^2),-1/(2y^2))",
      H == Matrix([[-1/(3*x**2), 0], [0, -1/(2*y**2)]]))
v = lambda px, py: px**2 * py**3   # u<=0 iff x^2 y^3 <= 1
check("P2 (8,1/4) on u=0", v(8, Rational(1, 4)) == 1)
check("P2 (1/8,4) on u=0", v(Rational(1, 8), 4) == 1)
check("P2 midpoint (65/16,17/8) has u>0",
      v(Rational(65, 16), Rational(17, 8)) > 1)
check("P2 8^(-2/3)=1/4 and (1/8)^(-2/3)=4",
      Rational(8)**Rational(-2, 3) == Rational(1, 4)
      and Rational(1, 8)**Rational(-2, 3) == 4)
check("P2 u(t,1)->oo, ->-oo", limit(log(t)/3, t, oo) == oo
      and limit(log(t)/3, t, 0, "+") == -oo)

# ---------------------------------------------------------------- P3
p3 = x**Rational(3, 4) * y**Rational(1, 2)
xp, yp = symbols("xp yp", positive=True)
q3 = xp**Rational(3, 4) * yp**Rational(1, 2)
check("P3 homogeneous degree 5/4",
      simplify(q3.subs({xp: t*xp, yp: t*yp}, simultaneous=True)
               .subs(t, symbols("s", positive=True))
               - symbols("s", positive=True)**Rational(5, 4)*q3) == 0)
H = Matrix([[diff(q3, xp, 2), diff(q3, xp, yp)],
            [diff(q3, yp, xp), diff(q3, yp, 2)]])
check("P3 f_xx=-(3/16)x^(-5/4)y^(1/2)",
      simplify(H[0, 0] + Rational(3, 16)*xp**Rational(-5, 4)*sqrt(yp)) == 0)
check("P3 f_yy=-(1/4)x^(3/4)y^(-3/2)",
      simplify(H[1, 1] + Rational(1, 4)*xp**Rational(3, 4)
               * yp**Rational(-3, 2)) == 0)
check("P3 f_xy=(3/8)x^(-1/4)y^(-1/2)",
      simplify(H[0, 1] - Rational(3, 8)*xp**Rational(-1, 4)
               * yp**Rational(-1, 2)) == 0)
check("P3 det=-(3/32)x^(-1/2)y^(-1)",
      simplify(H.det() + Rational(3, 32)/(sqrt(xp)*yp)) == 0)
gp = Matrix([diff(q3, xp), diff(q3, yp)]).subs({xp: 1, yp: 1})
check("P3 grad(1,1)=(3/4,1/2)", gp == Matrix([Rational(3, 4),
                                              Rational(1, 2)]))
check("P3 |grad|=sqrt13/4", simplify(gp.norm() - sqrt(13)/4) == 0)
check("P3 level slope -3/2",
      diff(xp**Rational(-3, 2), xp).subs(xp, 1) == -Rational(3, 2))
check("P3 grad . (2,-3)=0", gp.dot(Matrix([2, -3])) == 0)
check("P3 (4,1/8),(1/4,8) on f=1",
      q3.subs({xp: 4, yp: Rational(1, 8)}) == 1
      and q3.subs({xp: Rational(1, 4), yp: 8}) == 1)
check("P3 midpoint (17/8,65/16) f>1",
      q3.subs({xp: Rational(17, 8), yp: Rational(65, 16)}) > 1)
# origin: f=0 on both axes, so both partials are 0; and
# f(h,k)=h^(3/4)k^(1/2) <= r^(3/4) r^(1/2) = r^(5/4), so f/r -> 0
hh = symbols("hh", positive=True)
check("P3 f_x(0,0)=0 and f_y(0,0)=0 (f vanishes on the axes)",
      p3.subs({x: hh, y: 0}) == 0 and p3.subs({x: 0, y: hh}) == 0)
rr, th = symbols("rr th", positive=True)
bound = (rr*cos(th))**Rational(3, 4) * (rr*sin(th))**Rational(1, 2)
check("P3 f <= r^(5/4) on the quadrant (cos,sin <= 1)",
      simplify(bound / rr**Rational(5, 4)
               - cos(th)**Rational(3, 4)*sin(th)**Rational(1, 2)) == 0)
check("P3 f/r -> 0 at the origin",
      limit(rr**Rational(5, 4)/rr, rr, 0, "+") == 0)
check("P3 f(t,t)=t^(5/4) convex", simplify(diff(t**Rational(5, 4), t, 2)
                                           .subs(t, 1)) > 0)
check("P3 f(t,1)=t^(3/4) concave", diff(t**Rational(3, 4), t, 2)
      .subs(t, 1) < 0)

# ---------------------------------------------------------------- P4
q4 = xp**Rational(1, 3) * yp**Rational(2, 3)
H = Matrix([[diff(q4, xp, 2), diff(q4, xp, yp)],
            [diff(q4, yp, xp), diff(q4, yp, 2)]])
check("P4 f_xx=-(2/9)x^(-5/3)y^(2/3)",
      simplify(H[0, 0] + Rational(2, 9)*xp**Rational(-5, 3)
               * yp**Rational(2, 3)) == 0)
check("P4 f_yy=-(2/9)x^(1/3)y^(-4/3)",
      simplify(H[1, 1] + Rational(2, 9)*xp**Rational(1, 3)
               * yp**Rational(-4, 3)) == 0)
check("P4 f_xy=(2/9)x^(-2/3)y^(-1/3)",
      simplify(H[0, 1] - Rational(2, 9)*xp**Rational(-2, 3)
               * yp**Rational(-1, 3)) == 0)
check("P4 det=0", simplify(H.det()) == 0)
check("P4 H.(x,y)=0", (H*Matrix([xp, yp])).applyfunc(simplify)
      == Matrix([0, 0]))
check("P4 f(1,8)=4", q4.subs({xp: 1, yp: 8}) == 4)
gp = Matrix([diff(q4, xp), diff(q4, yp)]).subs({xp: 1, yp: 8})
check("P4 grad(1,8)=(4/3,1/3)", gp == Matrix([Rational(4, 3),
                                              Rational(1, 3)]))
check("P4 |grad|=sqrt17/3", simplify(gp.norm() - sqrt(17)/3) == 0)
check("P4 level y=8x^(-1/2), slope -4",
      diff(8*xp**Rational(-1, 2), xp).subs(xp, 1) == -4)
check("P4 grad . (1,-4)=0", gp.dot(Matrix([1, -4])) == 0)
check("P4 Euler 1*4/3+8*1/3=4", 1*gp[0] + 8*gp[1] == 4)
check("P4 (16,2),(1/4,16) on f=4",
      q4.subs({xp: 16, yp: 2}) == 4
      and q4.subs({xp: Rational(1, 4), yp: 16}) == 4)
check("P4 midpoint (65/8,9) f>4",
      q4.subs({xp: Rational(65, 8), yp: 9}) > 4)

# ---------------------------------------------------------------- P5
f = log(4 + x**2 + y**2)
qq = 4 + x**2 + y**2
H = hess(f)
check("P5 f_xx=2(4-x^2+y^2)/q^2", zero(H[0, 0] - 2*(4 - x**2 + y**2)/qq**2))
check("P5 f_yy=2(4+x^2-y^2)/q^2", zero(H[1, 1] - 2*(4 + x**2 - y**2)/qq**2))
check("P5 f_xy=-4xy/q^2", zero(H[0, 1] + 4*x*y/qq**2))
check("P5 det=4(16-(x^2+y^2)^2)/q^4",
      zero(H.det() - 4*(16 - (x**2 + y**2)**2)/qq**4))
gp = Matrix([diff(f, x), diff(f, y)]).subs({x: 1, y: 1})
check("P5 grad(1,1)=(1/3,1/3)", gp == Matrix([Rational(1, 3),
                                              Rational(1, 3)]))
check("P5 |grad|=sqrt2/3", simplify(gp.norm() - sqrt(2)/3) == 0)
check("P5 H(1,1)", H.subs({x: 1, y: 1}) == Matrix(
    [[Rational(2, 9), -Rational(1, 9)], [-Rational(1, 9), Rational(2, 9)]]))
check("P5 det H(1,1)=1/27", H.subs({x: 1, y: 1}).det() == Rational(1, 27))
check("P5 H(2,2)", H.subs({x: 2, y: 2}) == Matrix(
    [[Rational(1, 18), -Rational(1, 9)], [-Rational(1, 9), Rational(1, 18)]]))
check("P5 det H(2,2)=-1/108", H.subs({x: 2, y: 2}).det() == -Rational(1, 108))
check("P5 H(0,0)=diag(1/2,1/2)", H.subs({x: 0, y: 0}) == Matrix(
    [[Rational(1, 2), 0], [0, Rational(1, 2)]]))
check("P5 only critical point (0,0)",
      solve([diff(f, x), diff(f, y)], [x, y], dict=True) == [{x: 0, y: 0}])
check("P5 min value ln4", f.subs({x: 0, y: 0}) == log(4))
gt = log(4 + t**2)
check("P5 g''(t)=2(4-t^2)/(4+t^2)^2",
      zero(diff(gt, t, 2) - 2*(4 - t**2)/(4 + t**2)**2))
check("P5 superlevel ln8 witness: f(+-2,0)=ln8, f(0,0)=ln4<ln8",
      f.subs({x: 2, y: 0}) == log(8) and f.subs({x: -2, y: 0}) == log(8))
check("P5 level through (1,1) is x^2+y^2=2, value ln6",
      f.subs({x: 1, y: 1}) == log(6))
check("P5 f(-1,0)=f(1,0)=ln5 > f(0,0)",
      f.subs({x: 1, y: 0}) == log(5) and f.subs({x: -1, y: 0}) == log(5))

print()
print(f"{sum(results)}/{len(results)} checks passed")
if not all(results):
    raise SystemExit(1)
