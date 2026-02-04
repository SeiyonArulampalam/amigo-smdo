import sympy as sp

u1, v1, u2, v2, u3, v3 = sp.symbols("u1 v1 u2 v2 u3 v3")

N1x, N1y, N2x, N2y, N3x, N3y = sp.symbols("N1x N1y N2x N2y N3x N3y")

# alpha = 0.5*(1-nu)
E, nu, alpha = sp.symbols("E nu alpha")

B = sp.Matrix(
    [
        [N1x, 0, N2x, 0, N3x, 0],
        [0, N1y, 0, N2y, 0, N3y],
        [N1y, N1x, N2y, N2x, N3y, N3x],
    ]
)

D = sp.Matrix(
    [
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, alpha],
    ]
)

u = sp.Matrix(
    [
        [u1],
        [v1],
        [u2],
        [v2],
        [u3],
        [v3],
    ]
)

# Ku
res = B.T @ D @ B @ u
res = sp.simplify(res)
print("res1 = ", res[0])
print("res2 = ", res[1])
print("res3 = ", res[2])
print("res4 = ", res[3])
print("res5 = ", res[4])
print("res6 = ", res[5])
