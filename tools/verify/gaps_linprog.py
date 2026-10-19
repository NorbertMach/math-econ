"""Re-check the material added to Volume 8 (linprog) when closing the gaps:

ch01  Additional 9-12 (acquired inputs, judging a fractional optimum)
ch04  Additional 7-11 (origin test, relaxation, simplex paths)
ch07  Additional 9-16 (dual solved instead of primal, sign-restricted
      duals, Kuhn-Tucker, dual slacks)
ch08  second worked example, Practice 15-16, Additional 7-13
      (classic report layout, filling gaps, rebuilding the end tableau,
      bounds beyond the allowable range)

Every claim is retyped by hand from the book; an exact basis-enumeration LP
solver (sympy Rationals) recomputes optima, duals, reduced costs and
ranges independently. No LaTeX parsing.
"""
from itertools import combinations, product
from sympy import Matrix, Rational as Q, oo, zeros, eye, symbols, diff, expand

results = []


def check(label, cond):
    ok = bool(cond)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


# ---------------------------------------------------------------- LP kit
def to_le(rows):
    A, b = [], []
    for a, s, bb in rows:
        sg = 1 if s == '<=' else -1
        A.append([sg * Q(t) for t in a])
        b.append(sg * Q(bb))
    return Matrix(A), Matrix(b)


def solve(c, rows, sense='max'):
    """Optimal basis of max c'x (or min) s.t. rows, x>=0, via enumeration.
    Returns None if no optimal basis exists."""
    c = Matrix([Q(t) for t in c])
    if sense == 'min':
        c = -c
    A, b = to_le(rows)
    m, n = A.shape
    Af = A.row_join(eye(m))
    cf = c.col_join(zeros(m, 1))
    for basis in combinations(range(n + m), m):
        B = Af[:, list(basis)]
        if B.det() == 0:
            continue
        Bi = B.inv()
        xB = Bi * b
        if any(v < 0 for v in xB):
            continue
        y = (cf[list(basis), :].T * Bi).T
        zrow = (y.T * Af - cf.T).T
        if any(v < 0 for v in zrow):
            continue
        x = zeros(n + m, 1)
        for k, j in enumerate(basis):
            x[j] = xB[k]
        z = (cf.T * x)[0]
        return dict(basis=basis, Bi=Bi, xB=xB, y=y, zrow=zrow, x=x,
                    z=z if sense == 'max' else -z, n=n, m=m, Af=Af,
                    T=Bi * Af, sense=sense, rows=rows)
    return None


def xs(S):
    return [S['x'][j] for j in range(S['n'])]


def dual_prices(S):
    """Classic-layout dual prices: improvement per unit increase of b_i."""
    return [S['y'][i] if s == '<=' else -S['y'][i]
            for i, (a, s, bb) in enumerate(S['rows'])]


def slack_surplus(S):
    out = []
    for a, s, bb in S['rows']:
        lhs = sum(Q(t) * v for t, v in zip(a, xs(S)))
        out.append(Q(bb) - lhs if s == '<=' else lhs - Q(bb))
    return out


def rhs_ranges(S):
    """(allowable increase, allowable decrease) for each ORIGINAL b_i."""
    out = []
    Bi, xB, m = S['Bi'], S['xB'], S['m']
    for i, (a, s, bb) in enumerate(S['rows']):
        col = Bi[:, i]
        inc, dec = oo, oo
        for k in range(m):
            if col[k] < 0:
                inc = min(inc, xB[k] / -col[k])
            elif col[k] > 0:
                dec = min(dec, xB[k] / col[k])
        out.append((inc, dec) if s == '<=' else (dec, inc))
    return out


def obj_ranges(S):
    """(allowable increase, allowable decrease) for each ORIGINAL c_j."""
    out = []
    n, m, basis, T, zr = S['n'], S['m'], S['basis'], S['T'], S['zrow']
    for j in range(n):
        if j not in basis:
            r = (zr[j], oo)
        else:
            k = basis.index(j)
            inc, dec = oo, oo
            for q in range(n + m):
                if q in basis:
                    continue
                a = T[k, q]
                if a < 0:
                    inc = min(inc, zr[q] / -a)
                elif a > 0:
                    dec = min(dec, zr[q] / a)
            r = (inc, dec)
        out.append(r if S['sense'] == 'max' else (r[1], r[0]))
    return out


def pivot(T, r, c):
    """T: list of rows (constraint rows then objective row last)."""
    T = [list(map(Q, row)) for row in T]
    p = T[r][c]
    T[r] = [v / p for v in T[r]]
    for i in range(len(T)):
        if i != r and T[i][c] != 0:
            f = T[i][c]
            T[i] = [a - f * b for a, b in zip(T[i], T[r])]
    return T


def same_tab(T, expected):
    return all(Q(a) == Q(b) for ra, rb in zip(T, expected) for a, b in zip(ra, rb))


def feasible(rows, p):
    for a, s, bb in rows:
        lhs = sum(Q(t) * Q(v) for t, v in zip(a, p))
        if s == '<=' and lhs > bb:
            return False
        if s == '>=' and lhs < bb:
            return False
        if s == '=' and lhs != bb:
            return False
    return all(Q(v) >= 0 for v in p)


# ================================================================ ch01
print("=== ch01 Additional 9-12 ===")
pump = [([1, -1, -1], '<=', 0), ([0, 2, 0], '<=', 80), ([1, 0, 0], '<=', 60),
        ([1, 0, 0], '<=', 55)]
check("A9 plan (55,40,15) feasible", feasible(pump, (55, 40, 15)))
check("A9 plan value 1970", 50 * 55 - 12 * 40 - 20 * 15 == 1970)
check("A10 integer form 20*(0.4,0.25,-1,16) = (8,5,-20,320)",
      [20 * Q(2, 5), 20 * Q(1, 4), -20, 20 * 16] == [8, 5, -20, 320])
shed = [([5, 3], '<=', 18), ([4, 6], '<=', 27)]
check("A12 (1.5,3.5) binds both rows",
      5 * Q(3, 2) + 3 * Q(7, 2) == 18 and 4 * Q(3, 2) + 6 * Q(7, 2) == 27)
S = solve([7, 8], shed)
check("A12 LP optimum (3/2,7/2), z=77/2", xs(S) == [Q(3, 2), Q(7, 2)] and S['z'] == Q(77, 2))
nb = {(2, 4): False, (2, 3): False, (1, 4): False, (1, 3): True}
check("A12 rounding neighbours feasibility",
      all(feasible(shed, p) == v for p, v in nb.items()))
check("A12 table lhs values", (5*2+3*4, 4*2+6*4, 5*2+3*3, 4*2+6*3, 5+12, 4+24, 5+9, 4+18)
      == (22, 32, 19, 26, 17, 28, 14, 22))
check("A12 (1,3) earns 31", 7 + 24 == 31)
best = max((7 * i + 8 * j, i, j) for i in range(10) for j in range(10) if feasible(shed, (i, j)))
check("A12 best integer plan (0,4) z=32", best == (32, 0, 4))
rowsx = {0: 4, 1: 3, 2: 2, 3: 1}
check("A12 best x2 per x1",
      all(max(j for j in range(10) if feasible(shed, (i, j))) == v for i, v in rowsx.items()))
check("A12 x1<=3 from carpentry", not feasible(shed, (4, 0)) and feasible(shed, (3, 0)))

# ================================================================ ch04
print("=== ch04 Additional 7-11 ===")
origin_ok = {
    'a': [([1, 1], '<=', 6), ([1, 0], '>=', 2)],
    'b': [([2, 1], '<=', 10), ([1, -2], '>=', -6)],
    'c': [([1, 1], '<=', 8), ([1, -1], '<=', -2)],
    'd': [([1, 3], '<=', 9), ([2, -1], '>=', -4)],
    'e': [([1, 1], '=', 5), ([1, 0], '<=', 4)],
    'f': [([1, 2], '<=', 8), ([-1, 1], '>=', 0)],
}
verdict = {'a': False, 'b': True, 'c': False, 'd': True, 'e': False, 'f': True}
for k, rows in origin_ok.items():
    check(f"A7({k}) origin test verdict {verdict[k]}", feasible(rows, (0, 0)) == verdict[k])
check("A8 LP feasible at (0,1)", feasible([([1, -1], '<=', -1), ([1, 1], '<=', 5)], (0, 1)))
check("A8 origin violates x1-x2<=-1", not feasible([([1, -1], '<=', -1)], (0, 0)))

# A9 relaxation
T0 = [[1, 1, 1, 0, 6], [2, 1, 0, 1, 10], [-3, -2, 0, 0, 0]]
T1 = pivot(T0, 1, 0)
check("A9 tableau 1", same_tab(T1, [[0, Q(1, 2), 1, Q(-1, 2), 1], [1, Q(1, 2), 0, Q(1, 2), 5],
                                    [0, Q(-1, 2), 0, Q(3, 2), 15]]))
T2 = pivot(T1, 0, 1)
check("A9 tableau 2", same_tab(T2, [[0, 1, 2, -1, 2], [1, 0, -1, 1, 4], [0, 0, 1, 1, 16]]))
check("A9 dropped row holds at (4,2)", 4 + 2 >= 2)
S = solve([3, 2], [([1, 1], '<=', 6), ([2, 1], '<=', 10), ([1, 1], '>=', 2)])
check("A9 original LP optimum (4,2), 16", xs(S) == [4, 2] and S['z'] == 16)

# A10
T0 = [[1, 1, 1, 0, 0, 6], [1, 2, 0, 1, 0, 8], [1, 0, 0, 0, 1, 5], [-3, -4, 0, 0, 0, 0]]
T1 = pivot(T0, 1, 1)
check("A10 tableau 1", same_tab(T1, [[Q(1, 2), 0, 1, Q(-1, 2), 0, 2], [Q(1, 2), 1, 0, Q(1, 2), 0, 4],
                                     [1, 0, 0, 0, 1, 5], [-1, 0, 0, 2, 0, 16]]))
check("A10 ratios iteration 2 = 4,8,5", [Q(2) / Q(1, 2), Q(4) / Q(1, 2), Q(5)] == [4, 8, 5])
T2 = pivot(T1, 0, 0)
check("A10 tableau 2", same_tab(T2, [[1, 0, 2, -1, 0, 4], [0, 1, -1, 1, 0, 2],
                                     [0, 0, -2, 1, 1, 1], [0, 0, 2, 1, 0, 20]]))
p1 = [([1, 1], '<=', 6), ([1, 2], '<=', 8), ([1, 0], '<=', 5)]
check("A10 corners feasible", all(feasible(p1, p) for p in [(0, 0), (5, 0), (5, 1), (4, 2), (0, 4)]))
# alternative path x1 first
U1 = pivot(T0, 2, 0)
U2 = pivot(U1, 0, 1)
U3 = pivot(U2, 1, 4)
check("A10 alt path corners (5,0),(5,1),(4,2)",
      (U1[2][5], U2[0][5], U2[2][5]) == (5, 1, 5) and U3[3][5] == 20
      and U3[2][5] == 4 and U3[0][5] == 2)

# A11
T0 = [[1, 1, 1, 0, 0, 5], [1, 0, 0, 1, 0, 3], [0, 1, 0, 0, 1, 4], [-2, -2, 0, 0, 0, 0]]
T1 = pivot(T0, 1, 0)
check("A11 tableau 1", same_tab(T1, [[0, 1, 1, -1, 0, 2], [1, 0, 0, 1, 0, 3],
                                     [0, 1, 0, 0, 1, 4], [0, -2, 0, 2, 0, 6]]))
T2 = pivot(T1, 0, 1)
check("A11 tableau 2", same_tab(T2, [[0, 1, 1, -1, 0, 2], [1, 0, 0, 1, 0, 3],
                                     [0, 0, -1, 1, 1, 2], [0, 0, 2, 0, 0, 10]]))
T3 = pivot(T2, 2, 3)
check("A11 extra pivot tableau", same_tab(T3, [[0, 1, 0, 0, 1, 4], [1, 0, 1, 0, -1, 1],
                                               [0, 0, -1, 1, 1, 2], [0, 0, 2, 0, 0, 10]]))
check("A11 both corners give 10", 2 * 3 + 2 * 2 == 10 and 2 * 1 + 2 * 4 == 10)

# ================================================================ ch07
print("=== ch07 Additional 9-16 ===")
D1 = [([3, 2, 4], '<=', 60), ([1, 4, 1], '<=', 60)]
S = solve([11, 14, 13], D1)
check("A9 primal optimum (12,12,0), Z=300", xs(S) == [12, 12, 0] and S['z'] == 300)
check("A9 dual y*=(3,2)", list(S['y']) == [3, 2])
dual1 = [([3, 1], '>=', 11), ([2, 4], '>=', 14), ([4, 1], '>=', 13)]
SD = solve([60, 60], dual1, 'min')
check("A9 dual optimum (3,2), W=300", xs(SD) == [3, 2] and SD['z'] == 300)
verts = [(0, 13), (2, 5), (3, 2), (7, 0)]
check("A9 dual vertices feasible", all(feasible(dual1, v) for v in verts))
check("A9 dual vertex values", [60 * (a + b) for a, b in verts] == [780, 420, 300, 420])
check("A9 dual slacks (0,0,1)", list(S['zrow'][:3]) == [0, 0, 1])
check("A9 product 3 opportunity cost 14", 4 * 3 + 1 * 2 == 14)

D2 = [([1, 2, 1], '<=', 20), ([3, 1, 2], '<=', 90)]
S = solve([4, 6, 5], D2)
check("A10 primal (0,0,20), Z=100", xs(S) == [0, 0, 20] and S['z'] == 100)
check("A10 y*=(5,0), slack B 50", list(S['y']) == [5, 0] and slack_surplus(S)[1] == 50)
check("A10 dual slacks (1,4,0)", list(S['zrow'][:3]) == [1, 4, 0])
dual2 = [([1, 3], '>=', 4), ([2, 1], '>=', 6), ([1, 2], '>=', 5)]
check("A10 dual vertices feasible", all(feasible(dual2, v) for v in [(0, 6), (Q(7, 3), Q(4, 3)), (5, 0)]))
check("A10 (14/5,2/5) infeasible", not feasible(dual2, (Q(14, 5), Q(2, 5))))
check("A10 dual values", [20 * a + 90 * b for a, b in [(0, 6), (Q(7, 3), Q(4, 3)), (5, 0)]]
      == [540, Q(500, 3), 100])
SD = solve([20, 90], dual2, 'min')
check("A10 dual optimum (5,0)", xs(SD) == [5, 0] and SD['z'] == 100)

# A11 dual simplex
T0 = [[1, 1, 2, 1, 0, 3], [1, 3, 1, 0, 1, 2], [-4, -4, -5, 0, 0, 0]]
T1 = pivot(T0, 0, 2)
check("A11 dual tableau 1", same_tab(T1, [[Q(1, 2), Q(1, 2), 1, Q(1, 2), 0, Q(3, 2)],
                                          [Q(1, 2), Q(5, 2), 0, Q(-1, 2), 1, Q(1, 2)],
                                          [Q(-3, 2), Q(-3, 2), 0, Q(5, 2), 0, Q(15, 2)]]))
T2 = pivot(T1, 1, 0)
check("A11 dual tableau 2", same_tab(T2, [[0, -2, 1, 1, -1, 1], [1, 5, 0, -1, 2, 1],
                                          [0, 6, 0, 1, 3, 9]]))
feed = [([1, 1], '>=', 4), ([1, 3], '>=', 4), ([2, 1], '>=', 5)]
S = solve([3, 2], feed, 'min')
check("A11 primal (1,3), C=9", xs(S) == [1, 3] and S['z'] == 9)
check("A11 surpluses (0,6,0)", slack_surplus(S) == [0, 6, 0])
check("A11 u*=(1,0,1)", list(S['y']) == [1, 0, 1])

D4a = [([1, 1], '<=', 10), ([2, 1], '<=', 16), ([0, 1], '>=', 6)]
S = solve([5, 4], D4a)
check("A12 optimum (4,6), Z=44", xs(S) == [4, 6] and S['z'] == 44)
check("A12 dual y=(5,0,-1) (sign rule y3<=0)", dual_prices(S) == [5, 0, -1])
check("A12 W=44", 10 * 5 + 16 * 0 + 6 * (-1) == 44)
S7 = solve([5, 4], [([1, 1], '<=', 10), ([2, 1], '<=', 16), ([0, 1], '>=', 7)])
check("A12 contract 7: (3,7), Z=43", xs(S7) == [3, 7] and S7['z'] == 43)
D4b = [([1, 2], '>=', 8), ([1, 0], '<=', 4), ([1, -3], '>=', -10)]
S = solve([2, 5], D4b, 'min')
check("A13 optimum (4,2), C=18", xs(S) == [4, 2] and S['z'] == 18)
u = [Q(5, 2), Q(-1, 2), 0]
check("A13 u = dC/db = (5/2,-1/2,0)", [-d for d in dual_prices(S)] == u)
check("A13 dual constraints bind", u[0] + u[1] + u[2] == 2 and 2 * u[0] - 3 * u[2] == 5)
check("A13 W=18", 8 * u[0] + 4 * u[1] - 10 * u[2] == 18)
S5 = solve([2, 5], [([1, 2], '>=', 8), ([1, 0], '<=', 5), ([1, -3], '>=', -10)], 'min')
check("A13 capacity 5: (5,3/2), C=35/2", xs(S5) == [5, Q(3, 2)] and S5['z'] == Q(35, 2))

x1, x2, x3, l1, l2, m1, m2, m3 = symbols('x1 x2 x3 l1 l2 m1 m2 m3', real=True)
L = 3*x1 + 2*x2 + l1*(4 - x1 - x2) + l2*(6 - 2*x1 - x2)
pt = {x1: 2, x2: 2, l1: 1, l2: 1}
check("A14 dL/dx = 0 at (2,2),(1,1)", diff(L, x1).subs(pt) == 0 and diff(L, x2).subs(pt) == 0)
check("A14 dL/dlambda = 0", diff(L, l1).subs(pt) == 0 and diff(L, l2).subs(pt) == 0)
check("A14 dL/dx1 = 3-l1-2l2", expand(diff(L, x1) - (3 - l1 - 2*l2)) == 0)
L2 = (11*x1 + 14*x2 + 13*x3 + l1*(60 - 3*x1 - 2*x2 - 4*x3) + l2*(60 - x1 - 4*x2 - x3)
      + m1*x1 + m2*x2 + m3*x3)
pt = {x1: 12, x2: 12, x3: 0, l1: 3, l2: 2, m1: 0, m2: 0, m3: 1}
check("A15 stationarity", all(diff(L2, v).subs(pt) == 0 for v in (x1, x2, x3)))
check("A15 rows bind", diff(L2, l1).subs(pt) == 0 and diff(L2, l2).subs(pt) == 0)
check("A15 mu*x = 0", all((mv * xv).subs(pt) == 0 for mv, xv in ((m1, x1), (m2, x2), (m3, x3))))
# A16
check("A16(a) margin 9 < 10 -> not produced", 9 < 2 * 5)
S = solve([Q(11, 2), 6, 5], D2)
check("A16(b) margin 5.5 -> product 1 enters", S['x'][0] > 0)
S = solve([4, 6, Q(39, 10)], D2)
check("A16(c) c3=3.9 -> plan switches to product 1", S['x'][2] == 0 and S['x'][0] > 0)
S = solve([4, 6, 5], D2)
check("A16(c) allowable decrease of c3 is 1", obj_ranges(S)[2][1] == 1)

# ================================================================ ch08
print("=== ch08 worked example 2 ===")
W2 = [([1, -1, 0], '>=', -3), ([1, 2, 2], '<=', 12), ([0, 2, -1], '>=', -3), ([0, 0, 1], '<=', 10)]
S = solve([4, -2, -5], W2, 'min')
check("W2 x*=(0,1,5), C*=-27", xs(S) == [0, 1, 5] and S['z'] == -27)
check("W2 slack/surplus (2,0,0,5)", slack_surplus(S) == [2, 0, 0, 5])
check("W2 dual prices (0,2,-1,0)", dual_prices(S) == [0, 2, -1, 0])
check("W2 reduced cost X1 = 6", S['zrow'][0] == 6)
check("W2 obj ranges", obj_ranges(S) == [(oo, 6), (12, 3), (3, oo)])
check("W2 rhs ranges", rhs_ranges(S) == [(2, oo), (12, 6), (6, 3), (oo, 5)])
uu = [0, -2, 1, 0]
check("W2 dual constraints", uu[0] + uu[1] == -2 and -uu[0] + 2*uu[1] + 2*uu[2] == -2
      and 2*uu[1] - uu[2] + uu[3] == -5)
check("W2 W = -27", -3*uu[0] + 12*uu[1] - 3*uu[2] + 10*uu[3] == -27)
check("W2 origin feasible", feasible(W2, (0, 0, 0)))
check("W2 basis {x2,x3,s1,s4}", sorted(S['basis']) == [1, 2, 3, 6])
check("W2 z-row (6,0,0,0,2,1,0 | 27)", list(S['zrow']) == [6, 0, 0, 0, 2, 1, 0] and -S['z'] == 27)
check("W2 27 = 2*12 + 1*3", 2 * 12 + 1 * 3 == 27)
check("W2 row-4 range -6..3", (-3 - 3, -3 + 6) == (-6, 3))

print("=== ch08 Practice 15-16 ===")
def sol_lp(b1=100, b2=80, b3=40, c=(30, 20)):
    return solve(list(c), [([2, 1], '<=', b1), ([1, 1], '<=', b2), ([1, 0], '<=', b3)])
check("P15 z(120)=2000", sol_lp(120)['z'] == 2000)
check("P15 z(130)=2000 at (40,40)", sol_lp(130)['z'] == 2000 and xs(sol_lp(130)) == [40, 40])
check("P15 bounds 2000<=z<=2100", 2000 <= sol_lp(130)['z'] <= 1800 + 10 * 30)
check("P15 corners at 130", [30*a + 20*b for a, b in [(0, 0), (40, 0), (40, 40), (0, 80)]] == [0, 1200, 2000, 1600])
check("P15 z(80)=1600", sol_lp(80)['z'] == 1600)
check("P15 z(70)=1400 at (0,70)", sol_lp(70)['z'] == 1400 and xs(sol_lp(70)) == [0, 70])
check("P15 shadow price at 70 is 20", sol_lp(70)['y'][0] == 20)
check("P15 1400<=1500", sol_lp(70)['z'] <= 1500)
def ws(b1=10, b2=7, c=(4, 3)):
    return solve(list(c), [([2, 1], '<=', b1), ([1, 1], '<=', b2)])
check("P16 y=(1,2), z=24", list(ws()['y']) == [1, 2] and ws()['z'] == 24)
check("P16 ranges", rhs_ranges(ws()) == [(4, 3), (3, 2)])
check("P16(a) z(5)=20, z(3)=12<=16", ws(b2=5)['z'] == 20 and ws(b2=3)['z'] == 12)
check("P16(b) z(10)=30, z(12)=30", ws(b2=10)['z'] == 30 and ws(b2=12)['z'] == 30)
check("P16(c) z(14)=28, z(16)=28", ws(b1=14)['z'] == 28 and ws(b1=16)['z'] == 28)
check("P16(d) c1 range [3,6]", obj_ranges(ws())[0] == (2, 1))
check("P16(d) old plan 33, new optimum 35 at (5,0)",
      7*3 + 3*4 == 33 and ws(c=(7, 3))['z'] == 35 and xs(ws(c=(7, 3))) == [5, 0])

print("=== ch08 Additional 7-13 ===")
A7 = [([2, 2, 1], '<=', 20), ([0, 1, 0], '>=', 7), ([2, 1, 2], '<=', 25)]
S = solve([6, 9, 7], A7)
check("A7 x=(0,7,6), Z=105", xs(S) == [0, 7, 6] and S['z'] == 105)
check("A7 slack (0,0,6)", slack_surplus(S) == [0, 0, 6])
check("A7 dual prices (7,-5,0)", dual_prices(S) == [7, -5, 0])
check("A7 reduced cost X1 = 8", S['zrow'][0] == 8)
check("A7 obj ranges", obj_ranges(S) == [(8, oo), (5, oo), (oo, Q(5, 2))])
check("A7 rhs ranges", rhs_ranges(S) == [(3, 6), (3, 2), (oo, 6)])
check("A7 W=105", 20*7 + 7*(-5) + 25*0 == 105)
check("A9 end tableau body", S['basis'] == (1, 2, 5) and
      [list(S['T'].row(r)) for r in range(3)] ==
      [[0, 1, 0, 0, -1, 0], [2, 0, 1, 1, 2, 0], [-2, 0, 0, -2, -3, 1]] and list(S['xB']) == [7, 6, 6])
check("A9 z-row (8,0,0,7,5,0)", list(S['zrow']) == [8, 0, 0, 7, 5, 0])

A8 = [([-2, 1, 0], '>=', -4), ([1, 2, 1], '<=', 28), ([2, 0, 2], '>=', 8)]
S = solve([-6, 4, 5], A8, 'min')
check("A8 x=(4,4,0), C=-8", xs(S) == [4, 4, 0] and S['z'] == -8)
check("A8 slack (0,16,0)", slack_surplus(S) == [0, 16, 0])
check("A8 dual prices (-4,0,-1)", dual_prices(S) == [-4, 0, -1])
check("A8 reduced cost X3 = 3", S['zrow'][2] == 3)
check("A8 rhs ranges", rhs_ranges(S) == [(8, 4), (oo, 16), (Q(32, 5), 4)])
check("A8 origin infeasible", not feasible(A8, (0, 0, 0)))
check("A8 W=-8", -4*4 + 28*0 + 8*1 == -8)
S2 = solve([-6, 4, 5], [([-2, 1, 0], '>=', 2)] + A8[1:], 'min')
check("A8(e) row2 -> 2: C=16 at (4,10,0)", S2['z'] == 16 and xs(S2) == [4, 10, 0])
S18 = solve([-6, 4, 5], A8[:2] + [([2, 0, 2], '>=', 18)], 'min')
check("A8(e) row4 -> 18: C=35/4 >= 2", S18['z'] == Q(35, 4) and S18['z'] >= 2)
check("A8(e) -8+6.4 = -1.6, +3.6 = 2", -8 + Q(32, 5) == Q(-8, 5) and Q(-8, 5) + Q(18, 5) == 2)

S = sol_lp()
check("A10 basis {xA,xB,s3}", sorted(S['basis']) == [0, 1, 4])
k = {j: S['basis'].index(j) for j in S['basis']}
check("A10 rows", list(S['T'].row(k[0])) == [1, 0, 1, -1, 0] and
      list(S['T'].row(k[1])) == [0, 1, -1, 2, 0] and list(S['T'].row(k[4])) == [0, 0, -1, 1, 1])
check("A10 z-row and RHS", list(S['zrow']) == [0, 0, 10, 10, 0] and S['z'] == 1800
      and [S['xB'][k[0]], S['xB'][k[1]], S['xB'][k[4]]] == [20, 60, 20])

farm = [([1, 1, 0], '<=', 60), ([8, 12, -1], '<=', 400), ([0, 0, 1], '<=', 240)]
S = solve([300, 420, -9], farm)
check("A11 x=(20,40,240), Z=20640", xs(S) == [20, 40, 240] and S['z'] == 20640)
check("A11 dual prices (60,30,21)", dual_prices(S) == [60, 30, 21])
check("A11 obj ranges", obj_ranges(S) == [(84, 20), (30, 84), (oo, 21)])
check("A11 rhs ranges", rhs_ranges(S) == [(20, Q(20, 3)), (80, 160), (80, 160)])
check("A11 30 = 9 + 21 and quarter-hectare argument", 9 + 21 == 30 and Q(420 - 300, 4) == 30)
check("A11 3600+12000+5040", 60*60 + 30*400 + 21*240 == 20640)
own = solve([300, 420, -9], [farm[0], ([8, 12, -1], '<=', 401), farm[2]])
check("A11 one more own hour +30", own['z'] - 20640 == 30)
cap = solve([300, 420, -9], [farm[0], farm[1], ([0, 0, 1], '<=', 241)])
check("A11 one more hireable hour +21", cap['z'] - 20640 == 21)
o80 = solve([300, 420, -9], [farm[0], ([8, 12, -1], '<=', 480), farm[2]])
check("A11(c) 80 hours at 25: net 400", o80['z'] - 20640 - 25 * 80 == 400)
w30 = solve([300, 420, -30], farm)
check("A11(d) at wage 30 hiring 240 still optimal", w30['z'] == 300*20 + 420*40 - 30*240)

def w2(rhs3=12, rhs4=-3, c=(4, -2, -5)):
    return solve(list(c), [W2[0], ([1, 2, 2], '<=', rhs3), ([0, 2, -1], '>=', rhs4), W2[3]], 'min')
check("A12(a) C(24)=-51, C(30)=-55 in [-63,-51]",
      w2(24)['z'] == -51 and w2(30)['z'] == -55 and -63 <= w2(30)['z'] <= -51)
check("A12(b) C(3)=-21, C(6)=-90/7 >= -18",
      w2(rhs4=3)['z'] == -21 and w2(rhs4=6)['z'] == Q(-90, 7) and w2(rhs4=6)['z'] >= -18)
check("A12(c) old plan -13, new min -15 <= -13",
      4*0 + 12*1 - 5*5 == -13 and w2(c=(4, 12, -5))['z'] == -15)
# A13(e): a >= row with surplus 4 has increase 4, decrease oo (W2 row 2 has surplus 2: (2, oo))
check("A13(e) pattern on W2 row 2", rhs_ranges(w2())[0] == (2, oo))
# A13(d): min C and max -C give the same dual prices
Smax = solve([-4, 2, 5], W2, 'max')
check("A13(d) max(-C) same dual prices", dual_prices(Smax) == dual_prices(w2()) and Smax['z'] == 27)

# Nondegeneracy statements (W2 Step 4/5, rulebox, A13(e))
S = w2()
rr = rhs_ranges(S)
check("W2 nondegenerate: basic values (1,5,2,5) all > 0",
      sorted(S['xB']) == [1, 2, 5, 5] and all(v > 0 for v in S['xB']))
check("W2 no printed rhs allowable change is 0", all(v != 0 for pair in rr for v in pair))
check("W2 slacks of rows 3,4 nonbasic, of rows 2,5 basic",
      3 in S['basis'] and 6 in S['basis'] and 4 not in S['basis'] and 5 not in S['basis'])
# a degenerate variant (row 5 tightened to x3 <= 5) shows a zero range entry
Sd = solve([4, -2, -5], [W2[0], W2[1], W2[2], ([0, 0, 1], '<=', 5)], 'min')
check("degenerate variant: a basic variable is 0",
      xs(Sd) == [0, 1, 5] and any(v == 0 for v in Sd['xB']))
check("degenerate variant: some rhs range entry is 0",
      any(v == 0 for pair in rhs_ranges(Sd) for v in pair))
# lemma: zero basic variable k and B^-1 row k nonzero -> zero ratio on one side
k0 = [k for k in range(Sd['m']) if Sd['xB'][k] == 0][0]
check("degenerate variant: B^-1 row of the zero basic variable is nonzero",
      any(Sd['Bi'][k0, i] != 0 for i in range(Sd['m'])))
# A13(e) holds at a degenerate optimum too: row 2 surplus 2 -> (2, oo)
check("A13(e) at degenerate optimum: row 2 range (surplus, oo)",
      slack_surplus(Sd)[0] == 2 and rhs_ranges(Sd)[0] == (2, oo))
kk = list(Sd['basis']).index(3)
check("A13(e) column of B^-1 for the slack-basic row is a unit vector",
      Sd['Bi'][:, 0] == eye(Sd['m'])[:, kk])

print()
n, p = len(results), sum(results)
print(f"SUMMARY: {p}/{n} passed" + ("" if p == n else f", {n - p} FAILED"))
