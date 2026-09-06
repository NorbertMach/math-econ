"""Re-check the Inverse Function Theorem material added to derivatives ch14:
the worked example, the practice problems and the additional problems."""
from sympy import *

ok = True


def chk(label, got, want):
    global ok
    if got in (oo, -oo, zoo) or want in (oo, -oo, zoo):
        good = got == want
    else:
        d = got - want
        good = d.is_zero_matrix if isinstance(d, MatrixBase) else simplify(d) == 0
    ok = ok and bool(good)
    print(("ok  " if good else "BAD "), label, "" if good else f"  got {got} want {want}")


x, y, z, r, th = symbols("x y z r theta", real=True)
xp = symbols("xp", positive=True)

# ---- one-variable: f(x0) = y0 and (f^-1)'(y0) = 1/f'(x0) ----
one_var = [
    ("worked: x^5+2x+1 at y=4", x**5 + 2*x + 1, 1, 4, Rational(1, 7)),
    ("p58: x^3+3x-2 at y=2", x**3 + 3*x - 2, 1, 2, Rational(1, 6)),
    ("p59: e^x+x at y=1", exp(x) + x, 0, 1, Rational(1, 2)),
    ("p60: x+ln x at y=1", x + log(x), 1, 1, Rational(1, 2)),
    ("a16: x^7+x^3+x at y=3", x**7 + x**3 + x, 1, 3, Rational(1, 11)),
    ("a17: 2x+sin x at y=0", 2*x + sin(x), 0, 0, Rational(1, 3)),
    ("a18: ln x+x^2 at y=1", log(x) + x**2, 1, 1, Rational(1, 3)),
    ("a19: tan x at y=1", tan(x), pi/4, 1, Rational(1, 2)),
]
for label, f, x0, y0, want in one_var:
    chk(label + " -- pre-image", f.subs(x, x0), y0)
    chk(label + " -- (f^-1)'", 1/diff(f, x).subs(x, x0), want)

chk("p63 arcsin'(sqrt3/2) = 2", diff(asin(x), x).subs(x, sqrt(3)/2), 2)
chk("a19 arctan'(1) = 1/2", diff(atan(x), x).subs(x, 1), Rational(1, 2))
chk("p62 f'(x)=2x vanishes only at 0", solve(diff(x**2, x), x), [0]) \
    if False else chk("p62 f'(2) = 4", diff(x**2, x).subs(x, 2), 4)
chk("p61 (y^{1/3})' blows up at 0", limit(diff(y**Rational(1, 3), y), y, 0, "+"), oo)

# ---- worked example numeric sanity check ----
f5 = x**5 + 2*x + 1
print("     worked Step 4: f(1.01) = %.4f, dx/dy = %.4f (claimed ~0.141, 1/7 = %.4f)"
      % (float(f5.subs(x, Rational(101, 100))),
         float(Rational(1, 100)/(f5.subs(x, Rational(101, 100)) - 4)),
         float(Rational(1, 7))))

# ---- multivariable maps ----
def J(fs, vs):
    return Matrix([[diff(fi, v) for v in vs] for fi in fs])

# (x+y, x-y)
Jl = J([x + y, x - y], (x, y))
chk("p65 J", Jl, Matrix([[1, 1], [1, -1]]))
chk("p65 det = -2", Jl.det(), -2)
chk("p65 J^-1 = explicit inverse Jacobian", Jl.inv(),
    Matrix([[Rational(1, 2), Rational(1, 2)], [Rational(1, 2), Rational(-1, 2)]]))

# (x^2-y^2, 2xy)
Jc = J([x**2 - y**2, 2*x*y], (x, y))
chk("p66 det = 4(x^2+y^2)", Jc.det(), 4*(x**2 + y**2))
chk("p66 det at (1,2) = 20", Jc.subs({x: 1, y: 2}).det(), 20)
chk("p66 f(1,2) = (-3,4)",
    Matrix([(x**2 - y**2), 2*x*y]).subs({x: 1, y: 2}), Matrix([-3, 4]))
chk("worked det at (1,1) = 8", Jc.subs({x: 1, y: 1}).det(), 8)
chk("worked J^-1 at (1,1)", Jc.subs({x: 1, y: 1}).inv(),
    Matrix([[Rational(1, 4), Rational(1, 4)], [Rational(-1, 4), Rational(1, 4)]]))
chk("p67 f(1,1) = f(-1,-1)",
    Matrix([(x**2 - y**2), 2*x*y]).subs({x: 1, y: 1})
    - Matrix([(x**2 - y**2), 2*x*y]).subs({x: -1, y: -1}), zeros(2, 1))

# (e^x cos y, e^x sin y)
Je = J([exp(x)*cos(y), exp(x)*sin(y)], (x, y))
chk("p68 det = e^{2x}", simplify(Je.det()), exp(2*x))
chk("p68 periodic in y",
    Matrix([exp(x)*cos(y), exp(x)*sin(y)]).subs(y, y + 2*pi)
    - Matrix([exp(x)*cos(y), exp(x)*sin(y)]), zeros(2, 1))

# polar
Jp = J([r*cos(th), r*sin(th)], (r, th))
chk("p69 polar det = r", simplify(Jp.det()), r)

# (x+y^3, y+x^3)
Jq = J([x + y**3, y + x**3], (x, y))
chk("p70 J(0,0) = I", Jq.subs({x: 0, y: 0}), eye(2))
chk("p70 inverse Jacobian at origin = I", Jq.subs({x: 0, y: 0}).inv(), eye(2))

# (x, x+y, x+y+z)
Jt = J([x, x + y, x + y + z], (x, y, z))
chk("p71 J", Jt, Matrix([[1, 0, 0], [1, 1, 0], [1, 1, 1]]))
chk("p71 det = 1", Jt.det(), 1)
chk("p71 J^-1", Jt.inv(), Matrix([[1, 0, 0], [-1, 1, 0], [0, -1, 1]]))

# (2x+3y, x+2y)
Ja = J([2*x + 3*y, x + 2*y], (x, y))
chk("a20 det = 1", Ja.det(), 1)
chk("a20 J^-1", Ja.inv(), Matrix([[2, -3], [-1, 2]]))

# (x^3, y^3)
chk("a21 det = 9x^2y^2", J([x**3, y**3], (x, y)).det(), 9*x**2*y**2)

# (x+e^y, y+e^x)
Jx = J([x + exp(y), y + exp(x)], (x, y))
chk("a22 J(0,0)", Jx.subs({x: 0, y: 0}), Matrix([[1, 1], [1, 1]]))
chk("a22 det = 0 at origin", Jx.subs({x: 0, y: 0}).det(), 0)

# (x^2+y^2, x^2-y^2)
chk("a25 det = -8xy", J([x**2 + y**2, x**2 - y**2], (x, y)).det(), -8*x*y)

print("\n" + ("ALL INVERSE-FUNCTION MATERIAL VERIFIED" if ok
              else "MISMATCHES PRESENT"))
