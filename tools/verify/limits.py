"""Re-check every limit stated in the L'Hopital chapter (derivatives ch07)."""
from sympy import *

x, r, n = symbols("x r n", real=True)
a = symbols("a", positive=True)
bad = 0


def chk(label, expr, claimed, pt=0, direction="+-"):
    global bad
    got = limit(expr, x, pt, direction) if direction != "+-" else limit(expr, x, pt)
    ok = simplify(got - claimed) == 0
    if not ok:
        bad += 1
    print(f"{'ok ' if ok else 'BAD'} {label:<34} got {got}   claimed {claimed}")


print("--- practice, 0/0 ---")
chk("1  sin x / x", sin(x) / x, 1)
chk("2  (e^x-1)/x", (exp(x) - 1) / x, 1)
chk("3  ln x/(x-1) at 1", log(x) / (x - 1), 1, 1)
chk("4  (1-cos x)/x^2", (1 - cos(x)) / x**2, Rational(1, 2))
chk("5  (x^2-4)/(x^2-3x+2) at 2", (x**2 - 4) / (x**2 - 3*x + 2), 4, 2)
chk("6  (tan x - x)/x^3", (tan(x) - x) / x**3, Rational(1, 3))
chk("7  x/(e^x-1)", x / (exp(x) - 1), 1)
chk("8  (sqrt(1+x)-1)/x", (sqrt(1 + x) - 1) / x, Rational(1, 2))
chk("9  sin x/(x-pi) at pi", sin(x) / (x - pi), -1, pi)
chk("10 (e^x-1-x-x^2/2)/x^3", (exp(x) - 1 - x - x**2/2) / x**3, Rational(1, 6))

print("--- practice, inf/inf ---")
chk("11 ln x / x", log(x) / x, 0, oo)
chk("12 x/e^x", x / exp(x), 0, oo)
chk("13 x^2/e^x", x**2 / exp(x), 0, oo)
chk("14 ln x/sqrt x", log(x) / sqrt(x), 0, oo)
chk("15 (3x^2+2x)/(x^2-1)", (3*x**2 + 2*x) / (x**2 - 1), 3, oo)
chk("16 ln x/(1/x) at 0+", log(x) / (1/x), 0, 0, "+")
chk("17 e^x/x^10", exp(x) / x**10, oo, oo)
chk("18 ln(x^2+1)/ln x", log(x**2 + 1) / log(x), 2, oo)

print("--- practice, 0*inf and inf-inf ---")
chk("19 x ln x at 0+", x * log(x), 0, 0, "+")
chk("20 x^a ln x at 0+", x**a * log(x), 0, 0, "+")
chk("21 x(e^(1/x)-1)", x * (exp(1/x) - 1), 1, oo)
chk("22 x ln(1+3/x)", x * log(1 + 3/x), 3, oo)
chk("23 sqrt(x^2+x)-x", sqrt(x**2 + x) - x, Rational(1, 2), oo)
chk("24 1/x - 1/sin x", 1/x - 1/sin(x), 0)
chk("25 1/ln x - 1/(x-1) at 1", 1/log(x) - 1/(x - 1), Rational(1, 2), 1)
chk("26 x(pi/2 - atan x)", x * (pi/2 - atan(x)), 1, oo)

print("--- practice, exponential forms ---")
chk("27 x^x at 0+", x**x, 1, 0, "+")
chk("28 (1+1/x)^x", (1 + 1/x)**x, E, oo)
chk("29 (1+r/x)^x", (1 + r/x)**x, exp(r), oo)
chk("30 (1+x)^(1/x)", (1 + x)**(1/x), E)
chk("31 (sin x)^x at 0+", sin(x)**x, 1, 0, "+")
chk("32 x^(1/x)", x**(1/x), 1, oo)
chk("33 (cos x)^(1/x^2)", cos(x)**(1/x**2), exp(Rational(-1, 2)))
chk("34 (1-1/x)^x", (1 - 1/x)**x, exp(-1), oo)

print("--- conditions ---")
chk("36 (x+sin x)/x", (x + sin(x)) / x, 1, oo)
chk("37 (e^x-e^-x)/(e^x+e^-x)", (exp(x) - exp(-x)) / (exp(x) + exp(-x)), 1, oo)
print(f"    40 sin x/x^2 one-sided: right {limit(sin(x)/x**2, x, 0, '+')}, "
      f"left {limit(sin(x)/x**2, x, 0, '-')}  (claimed +oo / -oo, no 2-sided limit)")

print("--- additional ---")
chk("A1 sin3x/x", sin(3*x) / x, 3)
chk("A2 sin3x/sin5x", sin(3*x) / sin(5*x), Rational(3, 5))
chk("A3 (e^3x-1)/(e^2x-1)", (exp(3*x) - 1) / (exp(2*x) - 1), Rational(3, 2))
chk("A4 (x^5-1)/(x^3-1) at 1", (x**5 - 1) / (x**3 - 1), Rational(5, 3), 1)
chk("A5 ln(1+x)/x", log(1 + x) / x, 1)
chk("A6 (x-sin x)/x^3", (x - sin(x)) / x**3, Rational(1, 6))
chk("A7 (cos x-1+x^2/2)/x^4", (cos(x) - 1 + x**2/2) / x**4, Rational(1, 24))
chk("A8 (ln x)^2/x", log(x)**2 / x, 0, oo)
chk("A9 x^3/e^(2x)", x**3 / exp(2*x), 0, oo)
chk("A10 ln(1+e^x)/x", log(1 + exp(x)) / x, 1, oo)
chk("A11 sqrt(x) ln x at 0+", sqrt(x) * log(x), 0, 0, "+")
chk("A12 x e^(1/x) at 0+", x * exp(1/x), oo, 0, "+")
chk("A13 sqrt(x^2+3x)-sqrt(x^2-x)", sqrt(x**2 + 3*x) - sqrt(x**2 - x), 2, oo)
chk("A14 1/x^2 - 1/(x^2 cos x)", 1/x**2 - 1/(x**2*cos(x)), Rational(-1, 2))
chk("A15 (1/x)^x at 0+", (1/x)**x, 1, 0, "+")
chk("A16 (1+2x)^(1/x) at 0+", (1 + 2*x)**(1/x), exp(2), 0, "+")
chk("A17 ((x+2)/(x-1))^x", ((x + 2)/(x - 1))**x, exp(3), oo)
chk("A18 (ln x)^(1/x)", log(x)**(1/x), 1, oo)
chk("A19 x^(1/ln x) at 0+", x**(1/log(x)), E, 0, "+")
chk("A22 (x-cos x)/(x+cos x)", (x - cos(x)) / (x + cos(x)), 1, oo)

print(f"\n{'ALL LIMITS AGREE' if bad == 0 else str(bad) + ' MISMATCHES'}")
