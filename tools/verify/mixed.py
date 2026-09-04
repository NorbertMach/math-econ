from sympy import *

x, y, z, p, q, M, n, t, L, K = symbols("x y z p q M n t L K", positive=True)
xr, yr = symbols("xr yr", real=True)

print("--- Set A ---")
print("A2 3^2x-4*3^x+3=0:", solve(3**(2*x) - 4*3**x + 3, x))
f = (2*x - 1) / (x + 3)
print("A3 inverse check f((3u+1)/(2-u)):",
      simplify(f.subs(x, (3*symbols('u') + 1) / (2 - symbols('u')))))
print("A6 limit:", limit((exp(3*x) - 1 - 3*x) / x**2, x, 0))
print("A7 limit:", limit(log(x)**3 / x, x, oo))
print("A9 system:", linsolve((Matrix([[1, 2, -1], [2, -1, 1], [3, 1, 2]]),
                              Matrix([3, 5, 14]))))
print("A10 rank:", Matrix([[1, 2, 3], [2, 4, 6], [1, 0, 1]]).rank())
print("A11 eigs:", Matrix([[2, 1], [1, 2]]).eigenvals())
print("A12 det H:", Matrix([[2, 4], [4, 2]]).det())
fA = xr**3 - 3*xr*yr + yr**3
crit = solve([diff(fA, xr), diff(fA, yr)], [xr, yr], dict=True)
print("A13 critical points:", [c for c in crit])
print("A14 max xy s.t. 3x+4y=24:", solve([Eq(3*x + 4*y, 24), Eq(4*y, 3*x)], [x, y]))
print("A15 exact:", Rational(25**2, 48) - Rational(24**2, 48))
print("A19 elasticity:", simplify(diff(100 / p**2, p) * p / (100 / p**2)))
print("A20 doubling:", float(log(2) / 0.05))

print("--- Set B ---")
print("B2:", solve(Eq(log(x, 2) + log(x - 2, 2), 3), x))
print("B3 min of x+1/x:", minimum(x + 1 / x, x, Interval.open(0, oo)))
print("B6 limit:", limit(x**sin(x), x, 0, "+"))
print("B7 limit:", limit((x**3 - 8) / (x**2 - 4), x, 2))
print("B10 det:", Matrix([[1, 2, 0], [0, 3, 1], [2, 0, 4]]).det())
print("B11 solve:", linsolve((Matrix([[2, 1], [1, 3]]), Matrix([5, 10]))))
fB = x**2 * y**3
print("B12 grad at (1,2):", [diff(fB, v).subs({x: 1, y: 2}) for v in (x, y)])
print("B13 dir deriv:", 16 * Rational(3, 5) + 12 * Rational(4, 5))
print("B14 exact f(1.1,1.9):", float(1.1**2 * 1.9**3))
print("B15 implicit:", solve(Eq(2*1 + 2 + 1*symbols('m') + 2*2*symbols('m'), 0),
                             symbols('m')))
print("B16 min 2x+3y st xy=54:", solve([Eq(x*y, 54), Eq(2*y, 3*x)], [x, y]))
print("B19 PV:", float(1000 / 1.06**5))

print("--- Set C ---")
print("C1:", solve(Abs(2*xr - 5) <= 3, xr))
print("C6 limit:", limit(x*(sqrt(x**2 + 1) - x), x, oo))
print("C7 limit:", limit((x**n - 1) / (x - 1), x, 1))
print("C9 det:", factor(Matrix([[1, 1, t], [1, t, 1], [t, 1, 1]]).det()))
print("C10 eigs:", Matrix([[4, 2], [2, 1]]).eigenvals())
print("C11 det H:", Matrix([[2, -2], [-2, 2]]).det())
print("C15 x*:", solve([Eq(p*x + q*y, M), Eq(p*x, q*y)], [x, y]))
print("C17 integral:", integrate(x**2 * exp(x**3), (x, 0, 1)),
      "=", float(integrate(x**2 * exp(x**3), (x, 0, 1))))
print("C18 improper:", integrate(1 / x**2, (x, 1, oo)))
print("C19 CS:", integrate(20 - 2*q, (q, 0, 5)) - 50)
print("C20 annuity:", float(500 * (1 - 1.04**-10) / 0.04))
