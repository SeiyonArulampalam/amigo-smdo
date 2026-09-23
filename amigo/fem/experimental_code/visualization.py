import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_tri_mesh(X, conn, edge_conn, edge_signs):
    ref = [(0, 1), (1, 2), (2, 0)]
    triang = mtri.Triangulation(X[:, 0], X[:, 1], conn)
    fig, ax = plt.subplots()
    ax.triplot(triang, "k-")
    for nodes, edges, signs in zip(conn, edge_conn, edge_signs):
        centroid = X[nodes, :2].mean(axis=0)
        for c, (i, j) in enumerate(ref):
            n0 = nodes[i]
            n1 = nodes[j]
            x0 = X[n0, :2]
            x1 = X[n1, :2]
            mdpt = 0.5 * (x0 + x1)
            s = signs[c]
            shrink = 0.40
            pos = mdpt + shrink * (centroid - mdpt)
            s = "+" if signs[c] > 0 else "-"
            ax.text(
                pos[0],
                pos[1],
                f"{s}{edges[c]}",
                ha="center",
                va="center",
                fontsize=8,
            )
    plt.show()
    return
