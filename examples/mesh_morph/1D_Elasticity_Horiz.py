import numpy as np
import matplotlib.pyplot as plt

# Create mesh
L = 1.0
nnodes = 4
nelems = nnodes - 1
x = np.linspace(0, L, nnodes)
y = np.zeros(nnodes)
nodes = np.arange(nnodes)
conn = np.array([nodes[:-1], nodes[1:]], dtype=int).transpose()

# Define the displacement for each endpoint
dx1 = 0.1  # dx for left node
dx2 = -0.2  # dx for right node
dy = 0.5  # dy is constant for each node

# Assemble K
K = np.zeros((nnodes, nnodes))
A = 1.0
E = 1.0
for e in range(nelems):
    n1_tag = conn[e][0]
    n2_tag = conn[e][1]
    n1x = x[n1_tag]
    n2x = x[n2_tag]
    Le = n2x - n1x
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
f[0] = dx1

K[-1, :] = 0.0
K[-1, -1] = 1.0
f[-1] = dx2

# Solve
u = np.linalg.solve(K, f)
print(u)

# Plot results
fig, ax = plt.subplots()
ax.plot(x, y, "ko-", label="baseline")
ax.plot(x + u, y + dy, "bo-", label="morph")
ax.legend()
plt.savefig("horiz_morph.png", dpi=500)
