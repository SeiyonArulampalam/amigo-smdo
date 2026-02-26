import numpy as np
import matplotlib.pyplot as plt

# Create mesh
L = 1.0
nnodes = 20
nelems = nnodes - 1
y = np.linspace(0, L, nnodes)
x = np.zeros(nnodes)
nodes = np.arange(nnodes)
conn = np.array([nodes[:-1], nodes[1:]], dtype=int).transpose()

# Define the displacement for each endpoint
dy1 = 0.7  # dy for left node
dy2 = -0.2  # dy for right node
dx = 0.5  # dx is constant for each node

# Assemble K
K = np.zeros((nnodes, nnodes))
A = 1.0
E = 1.0
for e in range(nelems):
    n1_tag = conn[e][0]
    n2_tag = conn[e][1]
    n1y = y[n1_tag]
    n2y = y[n2_tag]
    Le = n2y - n1y
    coeff = E * A / Le
    Ke = coeff * np.array(
        [
            [1, -1],
            [-1, 1],
        ]
    )
    K[n1_tag : n2_tag + 1, n1_tag : n2_tag + 1] += Ke

# RHS vector
f = np.zeros(nnodes)

# Apply BC to endpoints
K[0, :] = 0.0
K[0, 0] = 1.0
f[0] = dy1

K[-1, :] = 0.0
K[-1, -1] = 1.0
f[-1] = dy2

# Solve
u = np.linalg.solve(K, f)
print(u)

# Plot results
fig, ax = plt.subplots()
ax.plot(x, y, "ko-", label="baseline")
ax.plot(x + dx, y + u, "bo-", label="morph")
ax.legend()
plt.savefig("vert_morph.png", dpi=500)
plt.show()
