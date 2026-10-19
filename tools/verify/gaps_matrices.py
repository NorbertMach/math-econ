"""Re-check the 3x3 definiteness material added to Volume 3 (matrices), ch09:
the second worked example, Practice Problems 26-32 and Additional Problems
26-31. Every matrix, minor and verdict is retyped by hand from the book; no
LaTeX parsing. The verdict is cross-checked against the eigenvalues."""
from itertools import combinations
from sympy import Matrix, symbols, simplify, expand, solve, Abs, S

x, y, z, t = symbols("x y z t", real=True)
results = []


def check(label, cond):
    ok = bool(cond)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


def leading(A):
    return [A[:k, :k].det() for k in range(1, A.rows + 1)]


def principal(A):
    """Dict: index tuple -> principal minor."""
    n = A.rows
    out = {}
    for k in range(1, n + 1):
        for idx in combinations(range(n), k):
            out[idx] = A.extract(list(idx), list(idx)).det()
    return out


def classify(A):
    # exact zero test via the rank, signs via high-precision numerics
    # (cubic roots may come back as CRootOf, whose sign sympy cannot decide)
    ev = [complex(e.evalf(50)).real for e in A.eigenvals(multiple=True)]
    zero = A.rank() < A.rows
    pos = any(e > 1e-30 for e in ev)
    neg = any(e < -1e-30 for e in ev)
    if pos and neg:
        return "indefinite"
    if pos and not zero:
        return "pd"
    if neg and not zero:
        return "nd"
    if pos or not neg:
        return "psd"
    return "nsd"


def quad(A, v):
    v = Matrix(v)
    return (v.T * A * v)[0]


def full(label, A, D, P, verdict):
    """D: claimed leading minors; P: claimed 7 principal minors in the order
    a11,a22,a33, M12,M13,M23, det (or None to skip)."""
    check(f"{label}: symmetric", A == A.T)
    check(f"{label}: leading minors {D}", leading(A) == [S(d) for d in D])
    if P is not None:
        pm = principal(A)
        got = [pm[(0,)], pm[(1,)], pm[(2,)], pm[(0, 1)], pm[(0, 2)],
               pm[(1, 2)], pm[(0, 1, 2)]]
        check(f"{label}: principal minors {P}", got == [S(p) for p in P])
    check(f"{label}: verdict {verdict}", classify(A) == verdict)


# ---- second worked example ------------------------------------------------
W = Matrix([[-2, 1, 1], [1, -2, 1], [1, 1, -2]])
full("worked", W, [-2, 3, 0], [-2, -2, -2, 3, 3, 3, 0], "nsd")
check("worked: W(1,1,1) = 0", W * Matrix([1, 1, 1]) == Matrix([0, 0, 0]))
check("worked: Q = -[(x-y)^2+(y-z)^2+(z-x)^2]",
      expand(quad(W, [x, y, z]) + (x - y)**2 + (y - z)**2 + (z - x)**2) == 0)
check("worked: eigenvalues 0,-3,-3",
      sorted(W.eigenvals(multiple=True)) == [-3, -3, 0])
check("worked: det expansion -2*3 - 1*(-3) + 1*3 = 0", -2*3 - 1*(-3) + 1*3 == 0)

# ---- Practice 26-32 -------------------------------------------------------
A26 = Matrix([[3, 1, -1], [1, 2, 0], [-1, 0, 4]])
full("P26", A26, [3, 5, 18], None, "pd")
check("P26: expansion 3*8 - 1*4 + (-1)*2 = 18", 3*8 - 1*4 + (-1)*2 == 18)

A27 = Matrix([[-3, 1, 1], [1, -2, 0], [1, 0, -4]])
full("P27", A27, [-3, 5, -18], None, "nd")
check("P27: expansion -3*8 - 1*(-4) + 1*2 = -18", -3*8 - 1*(-4) + 1*2 == -18)

A28 = Matrix([[-1, 0, 2], [0, -1, 2], [2, 2, -1]])
full("P28", A28, [-1, 1, 7], None, "indefinite")
check("P28: expansion -1*(1-4) - 0 + 2*(0+2) = 7", -1*(1 - 4) - 0 + 2*(0 + 2) == 7)
check("P28: Q(1,0,0) = -1", quad(A28, [1, 0, 0]) == -1)
check("P28: Q(1,1,1) = 5", quad(A28, [1, 1, 1]) == 5)

A29 = Matrix([[1, 3, 0], [3, 2, 1], [0, 1, 5]])
full("P29", A29, [1, -7, -36], None, "indefinite")
check("P29: Q(1,0,0)=1", quad(A29, [1, 0, 0]) == 1)
check("P29: Q(3,-2,0) = 9-36+8 = -19", quad(A29, [3, -2, 0]) == -19)

A30 = Matrix([[4, 1, 2], [1, -1, 0], [2, 0, 3]])
check("P30: symmetric", A30 == A30.T)
check("P30: Q(e1)=4, Q(e2)=-1",
      quad(A30, [1, 0, 0]) == 4 and quad(A30, [0, 1, 0]) == -1)
check("P30: verdict indefinite", classify(A30) == "indefinite")

A31 = Matrix([[1, -1, 2], [-1, 5, -2], [2, -2, 4]])
full("P31", A31, [1, 4, 0], [1, 5, 4, 4, 0, 16, 0], "psd")
check("P31: expansion 1*16 + 1*0 + 2*(-8) = 0", 1*16 + 1*0 + 2*(-8) == 0)
check("P31: A(2,0,-1) = 0", A31 * Matrix([2, 0, -1]) == Matrix([0, 0, 0]))
check("P31: Q = (x - y + 2z)^2 + 4y^2",
      expand(quad(A31, [x, y, z]) - (x - y + 2*z)**2 - 4*y**2) == 0)

A32 = Matrix([[-1, 2, -1], [2, -5, 1], [-1, 1, -2]])
full("P32", A32, [-1, 1, 0], [-1, -5, -2, 1, 1, 9, 0], "nsd")
check("P32: expansion -1*9 - 2*(-3) + (-1)*(-3) = 0",
      -1*9 - 2*(-3) + (-1)*(-3) == 0)
check("P32: A(3,1,-1) = 0", A32 * Matrix([3, 1, -1]) == Matrix([0, 0, 0]))
check("P32: Q = -(x - 2y + z)^2 - (y + z)^2",
      expand(quad(A32, [x, y, z]) + (x - 2*y + z)**2 + (y + z)**2) == 0)

# ---- Additional 26-31 -----------------------------------------------------
B26 = Matrix([[2, 1, 3], [1, 0, 0], [3, 0, 4]])
check("A26: symmetric", B26 == B26.T)
check("A26: M12 = -1", principal(B26)[(0, 1)] == -1)
check("A26: Q(1,0,0)=2, Q(1,-2,0)=-2",
      quad(B26, [1, 0, 0]) == 2 and quad(B26, [1, -2, 0]) == -2)
check("A26: verdict indefinite", classify(B26) == "indefinite")

B27 = Matrix([[1, 2, -1], [2, 4, -2], [-1, -2, 1]])
full("A27", B27, [1, 0, 0], [1, 4, 1, 0, 0, 0, 0], "psd")
check("A27: Q = (x+2y-z)^2", expand(quad(B27, [x, y, z]) - (x + 2*y - z)**2) == 0)
check("A27: Q(2,-1,0) = 0", quad(B27, [2, -1, 0]) == 0)

B28 = Matrix([[-2, 0, 1], [0, 0, 0], [1, 0, -3]])
full("A28", B28, [-2, 0, 0], [-2, 0, -3, 0, 5, 0, 0], "nsd")
check("A28: Q(0,1,0)=0", quad(B28, [0, 1, 0]) == 0)

B29 = Matrix([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
full("A29", B29, [1, 0, -1], [1, 1, 1, 0, 1, 0, -1], "indefinite")
check("A29: expansion 1*0 - 1*1 + 0 = -1", 1*0 - 1*1 + 0 == -1)
check("A29: Q(1,-1,1) = -1", quad(B29, [1, -1, 1]) == -1)
check("A29: Q(1,0,0) = 1", quad(B29, [1, 0, 0]) == 1)

B30 = Matrix([[2, 0, 3], [0, 1, 0], [3, 0, 2]])
check("A30: symmetric", B30 == B30.T)
check("A30: M13 = -5", principal(B30)[(0, 2)] == -5)
check("A30: Q(1,0,-1) = -2", quad(B30, [1, 0, -1]) == -2)
check("A30: verdict indefinite", classify(B30) == "indefinite")

B31 = Matrix([[1, 0, t], [0, 2, 0], [t, 0, 4]])
pm = principal(B31)
check("A31: 1x1 minors 1,2,4", [pm[(0,)], pm[(1,)], pm[(2,)]] == [1, 2, 4])
check("A31: M12=2, M13=4-t^2, M23=8",
      pm[(0, 1)] == 2 and expand(pm[(0, 2)] - (4 - t**2)) == 0 and pm[(1, 2)] == 8)
check("A31: det = 2(4-t^2)", expand(pm[(0, 1, 2)] - 2*(4 - t**2)) == 0)
check("A31: leading minors 1, 2, 2(4-t^2)",
      [expand(d) for d in leading(B31)] == [1, 2, expand(2*(4 - t**2))])
check("A31: t=2 psd", classify(B31.subs(t, 2)) == "psd")
check("A31: t=-2 psd", classify(B31.subs(t, -2)) == "psd")
check("A31: t=1 pd", classify(B31.subs(t, 1)) == "pd")
check("A31: t=3 indefinite", classify(B31.subs(t, 3)) == "indefinite")
check("A31: 4-t^2 = 0 at t = +-2", set(solve(4 - t**2, t)) == {-2, 2})
check("A31: Q at t=2 equals (x+2z)^2 + 2y^2",
      expand(quad(B31.subs(t, 2), [x, y, z]) - (x + 2*z)**2 - 2*y**2) == 0)

n = len(results)
p = sum(results)
print(f"\n{p}/{n} checks passed" + ("" if p == n else f", {n - p} FAILED"))
