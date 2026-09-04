from sympy import *

def show(label, A, b):
    print(f"  {label}: {linsolve((Matrix(A), Matrix(b)))}")

print("--- unique systems ---")
show("P9  x+y=5,2x-y=1", [[1, 1], [2, -1]], [5, 1])
show("P10 2x+3y=12,x-y=1", [[2, 3], [1, -1]], [12, 1])
show("P11 worked", [[1, 1, 1], [2, -1, 1], [1, 2, -1]], [6, 3, 2])
show("P12", [[1, 2, 3], [2, -1, 1], [3, 1, -1]], [14, 3, 2])
show("P18 overdetermined", [[1, 2], [3, -1], [2, 3]], [4, 5, 7])
show("P13 inconsistent", [[1, 1], [2, 2]], [2, 5])
show("A4", [[1, 1, 1], [1, -1, 2], [2, 1, -1]], [1, 4, -1])
show("A6", [[1, 1, 0], [0, 1, 1], [1, 0, 1]], [1, 2, 3])
show("A7 inconsistent", [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4])
show("A9", [[1, -1, 2], [2, 0, 1]], [1, 3])

print("--- parametric ---")
show("P15", [[1, 1, -1], [2, 3, 1], [3, 4, 0]], [1, 4, 5])
show("P16 homog", [[1, 2, -1], [2, -1, 3]], [0, 0])
show("P17", [[2, 1, -1], [1, -1, 2], [3, 3, -4]], [1, 5, -3])
show("P20", [[1, 1, 2], [2, 3, 5], [1, 2, 3]], [1, 3, 2])
show("P23", [[1, 1, -1], [2, 3, 1]], [1, 4])
show("P25", [[1, 2, -3], [2, 5, -8]], [0, 0])
show("A10", [[1, 1, 1], [1, -1, 1]], [6, 2])

print("--- ranks ---")
for lbl, M in [("P27", [[1, 2, 3], [2, 4, 6], [1, 1, 1]]),
               ("P28", [[1, 2], [3, 4], [5, 6]]),
               ("P29", [[1, 1, 1], [1, 2, 3], [1, 3, 6]]),
               ("P30", [[2, -1, 3], [4, -2, 6], [-2, 1, -3]]),
               ("A11", [[1, 2, 1], [2, 4, 2], [3, 6, 3]]),
               ("A12", [[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
               ("A13", [[1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 0, 1]])]:
    print(f"  {lbl}: rank {Matrix(M).rank()}")

print("--- inverses ---")
print("  P31", Matrix([[1, 2], [3, 4]]).inv().tolist())
print("  P32 (x2)", (Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]]).inv() * 2).tolist())
print("  A14", Matrix([[2, 1], [1, 1]]).inv().tolist())
print("  A15", Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]]).inv().tolist())

print("--- determinants ---")
print("  P39", Matrix([[1, 2, 3], [2, 5, 3], [1, 0, 8]]).det())
print("  A19", Matrix([[0, 1, 2], [1, 0, 3], [2, 3, 0]]).det())

print("--- economics ---")
IA = eye(2) - Matrix([[Rational(2, 10), Rational(4, 10)],
                      [Rational(3, 10), Rational(1, 10)]])
show("P43 Leontief", IA.tolist(), [12, 9])
show("P44 prices", [[2, -1, 0], [-1, 3, -1], [0, -1, 2]], [5, 12, 15])
A3 = Matrix([[0, Rational(1, 2), 0], [Rational(1, 2), 0, Rational(1, 2)],
             [0, Rational(1, 2), 0]])
show("A23 Leontief3", (eye(3) - A3).tolist(), [10, 20, 10])
show("A25a", [[1, 2], [2, 1]], [100, 110])
show("A25b", [[1, 2], [2, 1]], [100, 120])
Y, C, T = symbols("Y C T")
print("  P45 income:", solve([Eq(Y, C + 150 + 200),
                              Eq(C, 100 + Rational(8, 10) * (Y - T)),
                              Eq(T, Rational(1, 4) * Y)], [Y, C, T]))
a1, a2, a3 = symbols("a1 a2 a3")
print("  A24 portfolio:", solve([Eq(a1 + a2 + a3, 100),
                                 Eq(Rational(2,100)*a1 + Rational(5,100)*a2
                                    + Rational(9,100)*a3, 5),
                                 Eq(a1, 2*a3)], [a1, a2, a3]))
