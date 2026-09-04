"""Re-check every number stated in the worked examples of Volume 3 (matrices)."""
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
    print(("ok  " if good else "BAD "), label)


# ch01 -- vectors and the inner product
a, b = Matrix([3, -1, 2]), Matrix([1, 2, -1])
chk("ch01 a.b = -1", a.dot(b), -1)
chk("ch01 |a| = sqrt14", a.norm(), sqrt(14))
chk("ch01 |b| = sqrt6", b.norm(), sqrt(6))
proj = (a.dot(b) / b.dot(b)) * b
chk("ch01 proj_b a", proj, Matrix([Rational(-1, 6), Rational(-1, 3), Rational(1, 6)]))
chk("ch01 residual orthogonal", (a - proj).dot(b), 0)
print("     angle = %.1f deg (claimed 96.3)"
      % float(acos(a.dot(b) / (a.norm() * b.norm())) * 180 / pi))

# ch02 -- transpose, scalar multiples, symmetric split
A = Matrix([[1, 2, 3], [4, 5, 6]])
B = Matrix([[1, 0], [2, -1], [0, 3]])
chk("ch02 2A-3B^T", 2 * A - 3 * B.T, Matrix([[-1, -2, 6], [8, 13, 3]]))
M = Matrix([[2, 5], [1, 4]])
chk("ch02 symmetric part", (M + M.T) / 2, Matrix([[2, 3], [3, 4]]))
chk("ch02 skew part", (M - M.T) / 2, Matrix([[0, 2], [-2, 0]]))

# ch03 -- multiplication, non-commutativity, trace identity
A3 = Matrix([[1, 2, 3], [0, -1, 4]])
B3 = Matrix([[2, 1], [0, -1], [1, 3]])
chk("ch03 AB", A3 * B3, Matrix([[5, 8], [4, 13]]))
chk("ch03 BA", B3 * A3, Matrix([[2, 3, 10], [0, 1, -4], [1, -1, 15]]))
chk("ch03 tr(AB)=tr(BA)=18", (A3 * B3).trace(), 18)
chk("ch03 N^2 = 0", Matrix([[0, 1], [0, 0]]) ** 2, zeros(2))
chk("ch03 A(2,0,1) = first column of AB", A3 * Matrix([2, 0, 1]), Matrix([5, 4]))

# ch04 -- determinants
M4 = Matrix([[2, -1, 3], [0, 4, 1], [5, 2, -2]])
chk("ch04 det = -85", M4.det(), -85)
chk("ch04 det(2M) = -680", (2 * M4).det(), -680)

# ch05 -- Gaussian elimination, and the two broken variants
A5 = Matrix([[1, 1, 1], [2, -1, 1], [1, 2, -1]])
chk("ch05 solution (1,2,3)", A5.solve(Matrix([6, 3, 2])), Matrix([1, 2, 3]))
chk("ch05 det = 7", A5.det(), 7)
t = symbols("t")
dep = Matrix([[1, 1, 1], [2, -1, 1], [3, 3, 3]])
chk("ch05 dependent family solves rows 1-2",
    dep[:2, :] * Matrix([3 - 2 * t / 3, 3 - t / 3, t]), Matrix([6, 3]))
chk("ch05 inconsistent variant has higher augmented rank",
    dep.row_join(Matrix([6, 3, 17])).rank() - dep.rank(), 1)

# ch06 -- inverse, solve, Cramer
A6 = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
chk("ch06 det = 8", A6.det(), 8)
chk("ch06 inverse", A6.inv(),
    Matrix([[5, -2, 1], [-2, 4, -2], [1, -2, 5]]) / 8)
chk("ch06 x = (1/2,0,3/2)", A6.solve(Matrix([1, 2, 3])),
    Matrix([Rational(1, 2), 0, Rational(3, 2)]))
chk("ch06 Cramer numerator = 4",
    Matrix([[1, 1, 0], [2, 3, 1], [3, 1, 2]]).det(), 4)

# ch07 -- rank, null space, solvability
A7 = Matrix([[1, 2, -1, 3], [2, 4, 1, 0], [1, 2, 2, -3]])
chk("ch07 rank = 2", A7.rank(), 2)
chk("ch07 nullity = 2", len(A7.nullspace()), 2)
chk("ch07 v1 in null space", A7 * Matrix([-2, 1, 0, 0]), zeros(3, 1))
chk("ch07 v2 in null space", A7 * Matrix([-1, 0, 2, 1]), zeros(3, 1))
chk("ch07 b=(3,0,-3) consistent",
    A7.row_join(Matrix([3, 0, -3])).rank(), 2)
chk("ch07 b=(1,0,0) inconsistent",
    A7.row_join(Matrix([1, 0, 0])).rank(), 3)

# ch08 -- eigenvalues, diagonalisation, powers
A8 = Matrix([[4, 1], [2, 3]])
chk("ch08 eigenvalues {5,2}", Matrix(sorted(A8.eigenvals())), Matrix([2, 5]))
chk("ch08 v1 = (1,1) for lambda=5", A8 * Matrix([1, 1]) - 5 * Matrix([1, 1]),
    zeros(2, 1))
chk("ch08 v2 = (1,-2) for lambda=2", A8 * Matrix([1, -2]) - 2 * Matrix([1, -2]),
    zeros(2, 1))
P, D = Matrix([[1, 1], [1, -2]]), diag(5, 2)
chk("ch08 P D P^-1 = A", P * D * P.inv(), A8)
chk("ch08 P^-1", P.inv(), Matrix([[2, 1], [1, -1]]) / 3)
chk("ch08 A^5", A8 ** 5, Matrix([[2094, 1031], [2062, 1063]]))
chk("ch08 tr(A^5) = 3157", (A8 ** 5).trace(), 3157)

# ch09 -- quadratic forms and definiteness
x, y = symbols("x y")
Q = Matrix([[2, -1], [-1, 2]])
chk("ch09 matrix reproduces the form",
    (Matrix([[x, y]]) * Q * Matrix([x, y]))[0], 2*x**2 - 2*x*y + 2*y**2)
chk("ch09 eigenvalues {1,3}", Matrix(sorted(Q.eigenvals())), Matrix([1, 3]))
chk("ch09 minors 2 and 3", Matrix([Q[0, 0], Q.det()]), Matrix([2, 3]))
chk("ch09 completed square",
    2*(x - y/2)**2 + Rational(3, 2)*y**2, 2*x**2 - 2*x*y + 2*y**2)
chk("ch09 indefinite det = -3", Matrix([[1, 2], [2, 1]]).det(), -3)
chk("ch09 indefinite witness Q(1,-1) = -2",
    (x**2 + 4*x*y + y**2).subs({x: 1, y: -1}), -2)
chk("ch09 semidefinite det = 0", Matrix([[1, 1], [1, 1]]).det(), 0)
chk("ch09 (x+y)^2 form", x**2 + 2*x*y + y**2, (x + y)**2)

# ch10 -- Leontief
A10 = Matrix([[Rational(1, 5), Rational(3, 10)],
              [Rational(2, 5), Rational(1, 10)]])
d = Matrix([10, 20])
xs = (eye(2) - A10).inv() * d
chk("ch10 gross output (25, 100/3)", xs, Matrix([25, Rational(100, 3)]))
chk("ch10 balance A x + d = x", A10 * xs + d, xs)
chk("ch10 det(I-A) = 0.6", (eye(2) - A10).det(), Rational(3, 5))
chk("ch10 Leontief inverse", (eye(2) - A10).inv(),
    Matrix([[Rational(3, 2), Rational(1, 2)],
            [Rational(2, 3), Rational(4, 3)]]))
chk("ch10 column sums of A are 0.6 and 0.4",
    Matrix([sum(A10[:, 0]), sum(A10[:, 1])]),
    Matrix([Rational(3, 5), Rational(2, 5)]))

print("\n" + ("ALL VOLUME 3 WORKED EXAMPLES VERIFIED"
               if ok else "MISMATCHES PRESENT"))
