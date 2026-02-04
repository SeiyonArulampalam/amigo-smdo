import numpy as np


def mass(
    Ro,
    Ri,
    tooth_tip_thickness,
    copper_slot_height,
    magnet_thickness,
    back_iron_thickness,
    theta_mi,
    theta_mo,
    theta_bi,
    theta_bo,
    theta_ti,
    theta_to,
    A_strand,
    rho_iron=7874,
    rho_steel=7850,
    rho_magnet=7500,
    rho_copper=8960,
    num_magnets=10,
    num_teeth=12,
):
    """Estimate the total mass of the electric motor"""
    # Single back iron mass
    back_iron_area = np.pi * (Ro**2 - Ri**2)  # Top down area
    back_iron_vol = back_iron_area * back_iron_thickness  # Extrude
    back_iron_mass = back_iron_vol * rho_iron  # Conver to mass

    # Single magnet mass
    magnet_area = 0.5 * (theta_mo * Ro**2 - theta_mi * Ri**2)
    magnet_vol = magnet_area * magnet_thickness
    magnet_mass = magnet_vol * rho_magnet

    # Single tooth tip (bell) mass
    bell_area = 0.5 * (theta_bo * Ro**2 - theta_bi * Ri**2)
    bell_vol = bell_area * tooth_tip_thickness
    bell_mass = bell_vol * rho_steel

    # Single tooth stem mass
    stem_area = 0.5 * (theta_to * Ro**2 - theta_ti * Ri**2)
    stem_vol = stem_area * copper_slot_height
    stem_mass = stem_vol * rho_steel

    # Armature windings
    strand_vol = A_strand * (Ro - Ri)
    strand_vol_out = A_strand * (Ro * theta_to)
    strand_vol_in = A_strand * (Ri * theta_to)
    strand_end_winding_mass = (strand_vol_out + strand_vol_in) * rho_copper
    strand_mass = strand_vol * rho_copper

    # Accumlate mass
    mass = (
        2 * back_iron_mass
        + 2 * num_magnets * magnet_mass
        + num_teeth * (2 * bell_mass + stem_mass)
        + 2 * num_teeth * strand_mass
        + num_teeth * strand_end_winding_mass
    )
    return mass
