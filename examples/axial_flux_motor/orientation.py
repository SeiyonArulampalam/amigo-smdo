def check_areas(X, conn, nelems):
    # Reshape arrays
    conn = conn.reshape(-1, 3)
    X = X.reshape(-1, 3)

    # Check element area
    for i in range(nelems):
        # Zero indexing required
        n1 = conn[i, 0] - 1
        n2 = conn[i, 1] - 1
        n3 = conn[i, 2] - 1

        n1_x = X[n1, 0]
        n1_y = X[n1, 1]

        n2_x = X[n2, 0]
        n2_y = X[n2, 1]

        n3_x = X[n3, 0]
        n3_y = X[n3, 1]

        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

        if a_e <= 0.0:
            raise Exception(f"Element area for element {i} = {a_e}")
