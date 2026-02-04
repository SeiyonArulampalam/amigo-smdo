import numpy as np


def phase_currents(alpha, num_mag, Jz_peak):
    """
    Compute the current in phase 1 of the electrical winding.

    Parameters
    ----------
    alpha : Electrical rotation of shaft (rad)
    num_mag : Total number of magnets
    Jz_peak : Maximum current density of a single strand (A/m2)

    Returns
    -------
    phase1 : Current in phase 1
    phase2 : Current in phase 2
    phase3 : Current in phase 3
    """
    k = np.pi / (2 * np.pi / num_mag)
    offset120 = 120 * (np.pi / 180)
    offset240 = 240 * (np.pi / 180)
    phase1 = Jz_peak * np.cos(k * alpha)
    phase2 = Jz_peak * np.cos(k * alpha + offset120)
    phase3 = Jz_peak * np.cos(k * alpha + offset240)
    return phase1, phase2, phase3
