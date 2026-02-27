import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def plot_matrix(dense_mat):
    plt.figure(figsize=(6, 5))
    plt.imshow(dense_mat, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label="value")
    plt.tight_layout()
    return


def plot_region(ax, X, conn_list, color, label=None):
    """
    Plot a region made of triangular elements.

    Parameters:
      ax         : matplotlib axis
      X          : (Nnodes, 3) array of coordinates
      conn_list  : list of connectivity arrays (each is (Ne, 3))
      color      : facecolor for the region
      label      : optional label for legend
    """
    X = X[:, :2]  # Reduce to 2d (x,y coords only)
    polys = []
    for conn in conn_list:
        # Loop through each connectivity in the list
        for tri in conn:
            # Extract the node coordinates for the triangle
            polys.append(X[tri])

    coll = PolyCollection(
        polys,
        facecolors=color,
        edgecolors="k",
        # edgecolors=None,
        linewidths=0.01,
        label=label,
        antialiaseds=False,  # ← disables smoothing between triangles
    )
    ax.add_collection(coll)
    return


def plot_morph(X, conn):
    fig, ax = plt.subplots()
    plot_region(ax, X, conn, color="#7BE7FF", label="Region 1")
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.autoscale()
    return
