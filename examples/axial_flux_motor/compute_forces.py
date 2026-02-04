import numpy as np
import matplotlib.tri as mtri


def compute(
    z0,
    z1,
    Bphi,
    Bz,
    conn,
    nodeCoords,
    L=1.0,
    mu0=4 * np.pi * 1e-7,
    ax=None,
    fig=None,
):
    if len(Bphi) != len(conn):
        print(len(Bphi))
        print(conn.shape)
        raise Exception("ERROR: Bphi and conn lengths do not match")

    if len(Bz) != len(conn):
        raise Exception("ERROR: Bz and conn lengths do not match")

    # Compute dz
    dz = z1 - z0
    if dz <= 0:
        print(f"z1:{z1}, z0:{z0}, dz:{dz}")
        raise Exception("ERROR: dz<=0")

    force = 0.0
    y_set = []
    x_set = []
    computed_vals = []
    Bx = []
    By = []
    # Loop through each element
    for e in range(len(conn)):
        # Extract the node tags for the element e
        n1 = conn[e][0]
        n2 = conn[e][1]
        n3 = conn[e][2]

        # Determine the x-y coordinate of each node
        n1_x = nodeCoords[n1][0]
        n1_y = nodeCoords[n1][1]
        n2_x = nodeCoords[n2][0]
        n2_y = nodeCoords[n2][1]
        n3_x = nodeCoords[n3][0]
        n3_y = nodeCoords[n3][1]

        # Compute the area of each element
        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

        if a_e <= 0.0:
            raise Exception("ERROR: element area <=0 ")

        # Compute the centroid of each element
        x_c = (n1_x + n2_x + n3_x) / 3.0
        y_c = (n1_y + n2_y + n3_y) / 3.0

        # Integrate for region in z0 and z1
        # 2*dz because the airgap slics is only half of the full airgap
        force += Bphi[e] * Bz[e] * a_e * (1 / mu0) * (L / (2 * dz))

        # (Data for debugging)
        if ax is not None and fig is not None:
            # Store the coordinates and computed valuse
            y_set.append(y_c)
            x_set.append(x_c)
            computed_vals.append(Bphi[e] * Bz[e] * a_e * (1 / mu0))
            Bx.append(Bphi[e])
            By.append(Bz[e])

    if ax is not None and fig is not None:
        # Plot the region in which the forces are computed
        # fig, ax = plt.subplots()
        x_coords = nodeCoords[:, 0]
        y_coords = nodeCoords[:, 1]
        tri = mtri.Triangulation(x_coords, y_coords, conn)
        ax.triplot(tri, color="k", lw=0.05)  # lighter grey
        ax.quiver(
            x_set,
            y_set,
            Bx,
            By,
            color="r",  # vector color (e.g. red)
            scale=50,  # adjust scaling of arrows
            width=0.001,  # thickness of arrows
            headwidth=2,  # size of arrowhead
        )
    return force
