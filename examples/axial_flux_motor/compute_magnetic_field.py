import numpy as np


def compute(conn, u, nodeCoords):
    # Initialize vectors
    Bx = np.zeros(conn.shape[0])
    By = np.zeros(conn.shape[0])

    # Loop through each element in the connectivity
    for e in range(len(conn)):
        # Extract the node tags for the element e
        n1 = conn[e][0]
        n2 = conn[e][1]
        n3 = conn[e][2]

        # Determine the x-y coordinate of each node
        n1 = nodeCoords[n1]
        n2 = nodeCoords[n2]
        n3 = nodeCoords[n3]
        n1_x = n1[0]
        n1_y = n1[1]
        n2_x = n2[0]
        n2_y = n2[1]
        n3_x = n3[0]
        n3_y = n3[1]

        # Compute the area of each element
        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

        # Compute the elemental coefficients
        b1_e = n2_y - n3_y
        b2_e = n3_y - n1_y
        b3_e = n1_y - n2_y

        c1_e = n3_x - n2_x
        c2_e = n1_x - n3_x
        c3_e = n2_x - n1_x

        b_e = [b1_e, b2_e, b3_e]
        c_e = [c1_e, c2_e, c3_e]

        # Extract local solution vector
        u_e = u[conn[e]]

        # Compute Bx and By
        for k in range(3):
            By[e] -= b_e[k] * u_e[k]
            Bx[e] += c_e[k] * u_e[k]

        Bx[e] *= 1.0 / (2.0 * a_e)
        By[e] *= 1.0 / (2.0 * a_e)

    return Bx, By
