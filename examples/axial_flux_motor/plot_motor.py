import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


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


def plot_edge(ax, X, node_tags, color, linestyle="-"):
    coords = X[node_tags, :]  # Extract x, y, z coordinates
    # ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.0, linestyle=linestyle)
    return


def plot(
    slide_number,
    total_length,
    copper_slot_height,
    tooth_tip_thickness,
    airgap,
    magnet_thickness,
    back_iron_thickness,
    X_stator,
    X_inner_rotor,
    X_outter_rotor,
    stator_conn_s1,
    stator_conn_s2,
    stator_conn_s3,
    stator_conn_s4,
    stator_conn_s5,
    stator_conn_s6,
    stator_conn_s7,
    stator_conn_s8,
    stator_conn_s9,
    stator_conn_s10,
    stator_conn_s11,
    stator_conn_s12,
    stator_conn_s13,
    stator_conn_s14,
    stator_conn_s15,
    stator_conn_s16,
    stator_conn_s17,
    stator_conn_s18,
    stator_conn_s19,
    stator_conn_s20,
    stator_conn_s21,
    stator_conn_s22,
    stator_conn_s23,
    stator_conn_s24,
    stator_conn_t1,
    stator_conn_t2,
    stator_conn_t3,
    stator_conn_t4,
    stator_conn_t5,
    stator_conn_t6,
    stator_conn_t7,
    stator_conn_t8,
    stator_conn_t9,
    stator_conn_t10,
    stator_conn_t11,
    stator_conn_t12,
    stator_conn_t13,
    stator_conn_ag_inner,
    stator_conn_ag_outter,
    stator_pbc_nodes_left,
    stator_pbc_nodes_right,
    stator_pbc_nodes_bottom,
    stator_pbc_nodes_top,
    stator_conn_ag_1,
    stator_conn_ag_2,
    stator_conn_ag_3,
    stator_conn_ag_4,
    stator_conn_ag_5,
    stator_conn_ag_6,
    stator_conn_ag_7,
    stator_conn_ag_8,
    stator_conn_ag_9,
    stator_conn_ag_10,
    stator_conn_ag_11,
    stator_conn_ag_12,
    stator_conn_ag_13,
    stator_conn_ag_14,
    stator_conn_ag_15,
    stator_conn_ag_16,
    stator_conn_ag_17,
    stator_conn_ag_18,
    stator_conn_ag_19,
    stator_conn_ag_20,
    stator_conn_ag_21,
    stator_conn_ag_22,
    stator_conn_ag_23,
    stator_conn_ag_24,
    outter_rotor_conn_m1,
    outter_rotor_conn_m2,
    outter_rotor_conn_m3,
    outter_rotor_conn_m4,
    outter_rotor_conn_m5,
    outter_rotor_conn_m6,
    outter_rotor_conn_m7,
    outter_rotor_conn_m8,
    outter_rotor_conn_m9,
    outter_rotor_conn_m10,
    outter_rotor_conn_back_iron,
    outter_rotor_conn_airgap,
    outter_rotor_pbc_nodes_left,
    outter_rotor_pbc_nodes_right,
    outter_rotor_dirichlet_nodes,
    outter_rotor_pbc_nodes_bottom,
    outter_rotor_conn_ag_mag_1,
    outter_rotor_conn_ag_mag_2,
    outter_rotor_conn_ag_mag_3,
    outter_rotor_conn_ag_mag_4,
    outter_rotor_conn_ag_mag_5,
    outter_rotor_conn_ag_mag_6,
    outter_rotor_conn_ag_mag_7,
    outter_rotor_conn_ag_mag_8,
    outter_rotor_conn_ag_mag_9,
    outter_rotor_conn_ag_mag_10,
    outter_rotor_conn_ag_mag_11,
    inner_rotor_conn_m1,
    inner_rotor_conn_m2,
    inner_rotor_conn_m3,
    inner_rotor_conn_m4,
    inner_rotor_conn_m5,
    inner_rotor_conn_m6,
    inner_rotor_conn_m7,
    inner_rotor_conn_m8,
    inner_rotor_conn_m9,
    inner_rotor_conn_m10,
    inner_rotor_conn_back_iron,
    inner_rotor_conn_airgap,
    inner_rotor_pbc_nodes_left,
    inner_rotor_pbc_nodes_right,
    inner_rotor_dirichlet_nodes,
    inner_rotor_pbc_nodes_top,
    inner_rotor_conn_ag_mag_1,
    inner_rotor_conn_ag_mag_2,
    inner_rotor_conn_ag_mag_3,
    inner_rotor_conn_ag_mag_4,
    inner_rotor_conn_ag_mag_5,
    inner_rotor_conn_ag_mag_6,
    inner_rotor_conn_ag_mag_7,
    inner_rotor_conn_ag_mag_8,
    inner_rotor_conn_ag_mag_9,
    inner_rotor_conn_ag_mag_10,
    inner_rotor_conn_ag_mag_11,
    fig=None,
    ax=None,
):
    if fig == None and ax == None:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 2))

    # #########################
    # # Plot the Stator Regions
    # #########################
    # # Plot slots (left half) (orange)
    # slot_conns_left_half = [
    #     stator_conn_s1,
    #     stator_conn_s3,
    #     stator_conn_s5,
    #     stator_conn_s7,
    #     stator_conn_s9,
    #     stator_conn_s11,
    #     stator_conn_s13,
    #     stator_conn_s15,
    #     stator_conn_s17,
    #     stator_conn_s19,
    #     stator_conn_s21,
    #     stator_conn_s23,
    # ]
    # plot_region(ax, X_stator, slot_conns_left_half, color="#FF7B00", label="Copper")

    # # Plot slots (Right half) (orange)
    # slot_conns_right_half = [
    #     stator_conn_s2,
    #     stator_conn_s4,
    #     stator_conn_s6,
    #     stator_conn_s8,
    #     stator_conn_s10,
    #     stator_conn_s12,
    #     stator_conn_s14,
    #     stator_conn_s16,
    #     stator_conn_s18,
    #     stator_conn_s20,
    #     stator_conn_s22,
    #     stator_conn_s24,
    # ]
    # plot_region(ax, X_stator, slot_conns_right_half, color="#FFBB00", label="Copper")

    # # Plot stator iron (gray = teeth surfaces)
    # iron_conns = [
    #     stator_conn_t1,
    #     stator_conn_t2,
    #     stator_conn_t3,
    #     stator_conn_t4,
    #     stator_conn_t5,
    #     stator_conn_t6,
    #     stator_conn_t7,
    #     stator_conn_t8,
    #     stator_conn_t9,
    #     stator_conn_t10,
    #     stator_conn_t11,
    #     stator_conn_t12,
    #     stator_conn_t13,
    # ]
    # plot_region(ax, X_stator, iron_conns, color="#939393", label="Iron")

    # # Plot stator airgap (green)
    # airgap_conns = [stator_conn_ag_inner, stator_conn_ag_outter]
    # plot_region(ax, X_stator, airgap_conns, color="#7BE7FF", label="Airgap")

    # airgap_conn_between_teeth = [
    #     stator_conn_ag_1,
    #     stator_conn_ag_2,
    #     stator_conn_ag_3,
    #     stator_conn_ag_4,
    #     stator_conn_ag_5,
    #     stator_conn_ag_6,
    #     stator_conn_ag_7,
    #     stator_conn_ag_8,
    #     stator_conn_ag_9,
    #     stator_conn_ag_10,
    #     stator_conn_ag_11,
    #     stator_conn_ag_12,
    #     stator_conn_ag_13,
    #     stator_conn_ag_14,
    #     stator_conn_ag_15,
    #     stator_conn_ag_16,
    #     stator_conn_ag_17,
    #     stator_conn_ag_18,
    #     stator_conn_ag_19,
    #     stator_conn_ag_20,
    #     stator_conn_ag_21,
    #     stator_conn_ag_22,
    #     stator_conn_ag_23,
    #     stator_conn_ag_24,
    # ]
    # plot_region(
    #     ax, X_stator, airgap_conn_between_teeth, color="#7BE7FF", label="Airgap"
    # )

    # # Plot the PBC edges
    # plot_edge(ax, X_stator, stator_pbc_nodes_left, color="#FF00F2")
    # plot_edge(ax, X_stator, stator_pbc_nodes_right, color="#7BFF7B")
    # plot_edge(
    #     ax,
    #     X_stator,
    #     stator_pbc_nodes_bottom[-slide_number:-1] if slide_number > 0 else [],
    #     color="#7BFF7B",
    # )
    # plot_edge(
    #     ax,
    #     X_stator,
    #     stator_pbc_nodes_top[-slide_number:-1] if slide_number > 0 else [],
    #     color="#7BFF7B",
    # )

    ###############################
    # Plot the Outter Rotor Regions
    ###############################
    # Plot the magnets for the outter rotor (set1)
    outter_rotor_magnets_conn_set1 = [
        outter_rotor_conn_m1,
        outter_rotor_conn_m3,
        outter_rotor_conn_m5,
        outter_rotor_conn_m7,
        outter_rotor_conn_m9,
    ]
    plot_region(
        ax,
        X_outter_rotor,
        outter_rotor_magnets_conn_set1,
        color="#0026FF",
        label="Magnet",
    )

    # Plot the magnets for the outter rotor (set2)
    outter_rotor_magnets_conn_set2 = [
        outter_rotor_conn_m2,
        outter_rotor_conn_m4,
        outter_rotor_conn_m6,
        outter_rotor_conn_m8,
        outter_rotor_conn_m10,
    ]
    plot_region(
        ax,
        X_outter_rotor,
        outter_rotor_magnets_conn_set2,
        color="#FF0000",
        label="Magnet",
    )

    # Plot the back iron region for the outter rotor
    plot_region(
        ax,
        X_outter_rotor,
        [outter_rotor_conn_back_iron],
        color="#4F4F4F",
        label="Back Iron",
    )

    # Plot the airgap region for the outter rotor
    plot_region(
        ax, X_outter_rotor, [outter_rotor_conn_airgap], color="#7BE7FF", label="Airgap"
    )

    # Plot the airgap region between the magnets
    outter_rotor_conn_airgap_between_magnets = [
        outter_rotor_conn_ag_mag_1,
        outter_rotor_conn_ag_mag_2,
        outter_rotor_conn_ag_mag_3,
        outter_rotor_conn_ag_mag_4,
        outter_rotor_conn_ag_mag_5,
        outter_rotor_conn_ag_mag_6,
        outter_rotor_conn_ag_mag_7,
        outter_rotor_conn_ag_mag_8,
        outter_rotor_conn_ag_mag_9,
        outter_rotor_conn_ag_mag_10,
        outter_rotor_conn_ag_mag_11,
    ]
    plot_region(
        ax,
        X_outter_rotor,
        outter_rotor_conn_airgap_between_magnets,
        color="#7BE7FF",
        label="Airgap",
    )

    # Plot the left and right edge pbc
    plot_edge(
        ax,
        X_outter_rotor,
        (
            outter_rotor_pbc_nodes_left[:-1]
            if slide_number > 0
            else outter_rotor_pbc_nodes_left[1:-1]
        ),
        color="#FF00F2",
    )
    plot_edge(
        ax,
        X_outter_rotor,
        (
            outter_rotor_pbc_nodes_right[:-1]
            if slide_number > 0
            else outter_rotor_pbc_nodes_right[1:-1]
        ),
        color="#7BFF7B",
    )

    # Plot the dirichlet bc edge
    plot_edge(ax, X_outter_rotor, outter_rotor_dirichlet_nodes, color="#2CB0FC")

    # Plot the airgap pbc
    plot_edge(
        ax,
        X_outter_rotor,
        outter_rotor_pbc_nodes_bottom[1:slide_number] if slide_number > 0 else [],
        color="#FF00F2",
    )

    ##############################
    # Plot the Inner Rotor Regions
    ##############################
    # # Plot the magnets for the inner rotor (set1)
    # inner_rotor_magnets_conn_set1 = [
    #     inner_rotor_conn_m1,
    #     inner_rotor_conn_m3,
    #     inner_rotor_conn_m5,
    #     inner_rotor_conn_m7,
    #     inner_rotor_conn_m9,
    # ]
    # plot_region(
    #     ax,
    #     X_inner_rotor,
    #     inner_rotor_magnets_conn_set1,
    #     color="#0026FF",
    #     label="Magnet",
    # )

    # # Plot the magnets for the inner rotor (set2)
    # inner_rotor_magnets_conn_set2 = [
    #     inner_rotor_conn_m2,
    #     inner_rotor_conn_m4,
    #     inner_rotor_conn_m6,
    #     inner_rotor_conn_m8,
    #     inner_rotor_conn_m10,
    # ]
    # plot_region(
    #     ax,
    #     X_inner_rotor,
    #     inner_rotor_magnets_conn_set2,
    #     color="#FF0000",
    #     label="Magnet",
    # )

    # # Plot the back iron region for the inner rotor
    # plot_region(
    #     ax,
    #     X_inner_rotor,
    #     [inner_rotor_conn_back_iron],
    #     color="#4F4F4F",
    #     label="Back Iron",
    # )

    # # Plot the airgap region for the inner rotor
    # plot_region(
    #     ax,
    #     X_inner_rotor,
    #     [inner_rotor_conn_airgap],
    #     color="#7BE7FF",
    #     label="Airgap",
    # )

    # # Plot the airga region between the magnets
    # inner_rotor_conn_airgap_between_magnets = [
    #     inner_rotor_conn_ag_mag_1,
    #     inner_rotor_conn_ag_mag_2,
    #     inner_rotor_conn_ag_mag_3,
    #     inner_rotor_conn_ag_mag_4,
    #     inner_rotor_conn_ag_mag_5,
    #     inner_rotor_conn_ag_mag_6,
    #     inner_rotor_conn_ag_mag_7,
    #     inner_rotor_conn_ag_mag_8,
    #     inner_rotor_conn_ag_mag_9,
    #     inner_rotor_conn_ag_mag_10,
    #     inner_rotor_conn_ag_mag_11,
    # ]
    # plot_region(
    #     ax,
    #     X_inner_rotor,
    #     inner_rotor_conn_airgap_between_magnets,
    #     color="#7BE7FF",
    #     label="Airgap",
    # )

    # # Plot the left and right edge pbc
    # plot_edge(
    #     ax,
    #     X_inner_rotor,
    #     (
    #         inner_rotor_pbc_nodes_left[1:]
    #         if slide_number > 0
    #         else inner_rotor_pbc_nodes_left[1:-1]
    #     ),
    #     color="#FF00F2",
    # )
    # plot_edge(
    #     ax,
    #     X_inner_rotor,
    #     (
    #         inner_rotor_pbc_nodes_right[1:]
    #         if slide_number > 0
    #         else inner_rotor_pbc_nodes_right[1:-1]
    #     ),
    #     color="#7BFF7B",
    # )

    # # Plot the dirichlet bc edge
    # plot_edge(ax, X_inner_rotor, inner_rotor_dirichlet_nodes, color="#2CB0FC")

    # # Plot the airgap pbc edge
    # plot_edge(
    #     ax,
    #     X_inner_rotor,
    #     inner_rotor_pbc_nodes_top[1:slide_number] if slide_number > 0 else [],
    #     color="#FF00F2",
    # )

    ###################################
    # Plot the overlapped airgap region
    ###################################
    # plot_edge(
    #     ax,
    #     X_stator,
    #     stator_pbc_nodes_top[1 : -slide_number - 1],
    #     color="#246317",
    #     linestyle="-",
    # )
    # plot_edge(
    #     ax,
    #     X_stator,
    #     stator_pbc_nodes_bottom[1 : -slide_number - 1],
    #     color="#63175F",
    #     linestyle="-",
    # )

    # plot_edge(
    #     ax,
    #     X_outter_rotor,
    #     outter_rotor_pbc_nodes_bottom[slide_number + 1 : -1],
    #     color="#246317",
    #     linestyle=":",
    # )
    # plot_edge(
    #     ax,
    #     X_inner_rotor,
    #     inner_rotor_pbc_nodes_top[slide_number + 1 : -1],
    #     color="#63175F",
    #     linestyle=":",
    # )

    # Formatting
    # ax.set_xlim(-1e-3, 2 * total_length)
    ax.set_xlim(-5e-3, total_length + 5e-3)
    ylim = (
        0.5 * copper_slot_height
        + tooth_tip_thickness
        + airgap
        + magnet_thickness
        + back_iron_thickness
    ) + 1e-3
    # ax.set_ylim(-ylim, ylim)
    ax.set_ylim(0, ylim + 10e-3)
    ax.set_aspect("equal")
    ax.axis("off")
    # plt.savefig(f"Figures/afpm_geom.png", dpi=800)
    return
