import numpy as np


def mass_per_unit_length(nodeCoords, conn, rho):
    """Compute mass per unit length"""
    mass_per_length = np.zeros(len(conn))
    for i, node_tags in enumerate(conn):
        n1_x = nodeCoords[node_tags[0], 0]
        n1_y = nodeCoords[node_tags[0], 1]
        n2_x = nodeCoords[node_tags[1], 0]
        n2_y = nodeCoords[node_tags[1], 1]
        n3_x = nodeCoords[node_tags[2], 0]
        n3_y = nodeCoords[node_tags[2], 1]

        # Compute the area of each element
        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

        if a_e <= 0.0:
            raise Exception("ERROR: element area <=0 ")

        mass_per_length[i] = rho * a_e  # kg/m
    return mass_per_length


def iron(flux, fund_freq, alpha=2.0, beta=0.0):
    """Compute iron loss density W/kg"""
    # Coefficenct models
    # Kh = (
    #     5.978e-2
    #     - flux * (6.586e-2)
    #     + np.power(flux, 2.0) * (3.521e-2)
    #     - np.power(flux, 3.0) * (6.548e-3)
    # )
    Kh = 1e-3

    # Ke = (
    #     3.831e-5
    #     - flux * (4.2e-5)
    #     + np.power(flux, 2.0) * (2.098e-5)
    #     - np.power(flux, 3.0) * (3.886e-6)
    # )
    Ke = 1e-5

    # print(min(Kh), min(Ke))

    # Hysteresis loss
    hyst_loss = Kh * fund_freq * np.power(flux, (alpha + beta * flux))

    # Eddy loss
    eddy_loss = 2 * np.pi**2 * Ke * fund_freq**2 + np.power(flux, 2)

    # Store the loss
    loss_density = hyst_loss + eddy_loss
    return loss_density


def copper(Jrms, rho_cu, A_cu):
    """Compute the copper loss W/m"""
    return Jrms**2 * rho_cu * A_cu
