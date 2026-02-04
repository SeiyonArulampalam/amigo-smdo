import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import imageio
import niceplots


def build_global_mapping(surface_conns: dict):
    """
    Build global element indexing and index lists from surface connectivity dict.

    Inputs:
    -------
    surface_conns : dict[str, np.ndarray]
        Dictionary {surface_name: (NELEMS, 3) array}

    Returns:
    --------
    global_conn : np.ndarray
        Concatenated connectivity (NELEMS_total, 3).
    surface_indices : dict[str, np.ndarray]
        Mapping {surface_name: np.ndarray of global element indices}.
    """
    surface_indices = {}
    global_conn = []

    current_idx = 0
    for name, conn in surface_conns.items():
        nele = conn.shape[0]
        indices = np.arange(current_idx, current_idx + nele)
        surface_indices[name] = indices
        global_conn.append(conn)
        current_idx += nele

    global_conn = np.vstack(global_conn)
    return surface_indices, global_conn


def plot_solution(ax, xyz_nodeCoords, conn, z, levels):
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # create a Delaunay triangulation
    # tri = mtri.Triangulation(x, y)
    tri = mtri.Triangulation(x, y, conn)

    # Define colormap
    cmap = "coolwarm"

    # Plot solution
    cntr = ax.tricontourf(tri, z, levels=levels, cmap=cmap)
    ax.tricontour(
        tri, z, levels=levels, colors="k", linewidths=0.2, alpha=0.8
    )  # optional contour lines

    # Overlay mesh
    ax.triplot(tri, color="0.7", lw=0.05, alpha=0.8)  # lighter grey
    return cntr


def create_gif(folder, gif_name, duration=0.5, slice_start=None, slice_end=None):
    """
    Create a GIF from numbered PNG images in a folder.

    Args:
        folder (str): Path to folder containing PNG images.
        gif_name (str): Name of output GIF file.
        duration (float): Time per frame in seconds.
    """
    from PIL import Image

    # Only keep PNGs named as numbers
    images = [f for f in os.listdir(folder) if f.endswith(".png")]

    # Sort numerically by the filename (strip extension and convert to int)
    images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))
    print("Images to include in GIF:", images)

    # Update
    if slice_start != None:
        images = images[slice_start:slice_end]
        print("Sliced images to include in GIF:", images)

    image_paths = [os.path.join(folder, img) for img in images]
    output_gif = os.path.join(folder, gif_name)

    # Read the first image to set target size
    first_img = Image.open(image_paths[0]).convert("RGB")
    target_size = first_img.size  # (width, height)

    # Load all images, convert to RGB and resize to target_size
    frames = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")  # ensures 3 channels
        if img.size != target_size:
            img = img.resize(target_size)  # resize if needed
        frames.append(img)

    # Save as GIF
    # imageio.mimsave(output_gif, frames, duration=duration)
    imageio.mimsave(output_gif, frames, duration=duration)
    print(f"GIF saved at: {output_gif}")


def plot_airgap_flux(
    ax,
    xyz_nodeCoords,
    conn,
    elem_indices_map,
    z,
    min_val,
    max_val,
):
    """Plots the flux in the airgap region of the motor"""
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # Create triangulation for full mesh
    tri = mtri.Triangulation(x, y, conn)

    # Initialize facecolors to NaN (blank for all elements)
    facecolors = np.full(len(conn), np.nan)

    # Assign z values only to elements in elem_indices_map
    facecolors[elem_indices_map] = z

    # Plot element-based values
    pc = ax.tripcolor(
        tri,
        facecolors=facecolors,
        # edgecolors="k",
        cmap="jet",
        shading="flat",
        vmin=min_val,
        vmax=max_val,
    )

    # Overlay full mesh lightly
    ax.triplot(tri, color="0.7", lw=0.05, alpha=0.8)

    return


def plot_flux(
    ax,
    xyz_nodeCoords,
    conn,
    z,
    min_val,
    max_val,
    cmap="jet",
):
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # Create triangulation for full mesh
    tri = mtri.Triangulation(x, y, conn)

    pc = ax.tripcolor(
        tri,
        facecolors=z,
        # edgecolors="k",
        cmap=cmap,
        shading="flat",
        vmin=min_val,
        vmax=max_val,
    )
    return pc


def plot_data(x, y, title, xlabel, ylabel, color, save_loc):
    fig, ax = plt.subplots()
    plt.style.use(niceplots.get_style())
    plt.rcParams["font.family"] = "DejaVu Sans"
    fontsize = 12
    fontweight = "normal"
    ms = 8
    ax.plot(x, y, "-o", color=color, markersize=ms)
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
    ax.set_title(title, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(save_loc, dpi=800)
    return


def plot_losses(
    ax,
    xyz_nodeCoords,
    conn,
    elem_indices_map,
    z,
    min_val,
    max_val,
):
    """Plots the flux in the airgap region of the motor"""
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # Create triangulation for full mesh
    tri = mtri.Triangulation(x, y, conn)

    # Initialize facecolors to NaN (blank for all elements)
    facecolors = np.full(len(conn), np.nan)

    # Assign z values only to elements in elem_indices_map
    facecolors[elem_indices_map] = z

    # Plot element-based values
    pc = ax.tripcolor(
        tri,
        facecolors=facecolors,
        # edgecolors="k",
        cmap="jet",
        shading="flat",
        vmin=min_val,
        vmax=max_val,
    )

    # Overlay full mesh lightly
    ax.triplot(tri, color="0.7", lw=0.05, alpha=0.8)

    return
