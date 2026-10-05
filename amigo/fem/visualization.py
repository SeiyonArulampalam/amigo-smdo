from .fem import Problem
from .cell_types import CellType, DofLayout
from .dof_handler import DofHandler
from .fem_space import FunctionSpace
from .fem import Problem
from dataclasses import dataclass
import numpy as np
import pyvista as pv

# VTK_CELL_TYPES = {
#     CellType.SEGMENT: pv.CellType.LINE,
#     CellType.TRIANGLE: pv.CellType.TRIANGLE,
#     CellType.QUADRILATERAL: pv.CellType.QUAD,
#     CellType.TETRAHEDRON: pv.CellType.TETRA,
#     CellType.HEXAHEDRON: pv.CellType.HEXAHEDRON,
#     CellType.PRISM: pv.CellType.WEDGE,
#     CellType.PYRAMID: pv.CellType.PYRAMID,
# }


# class VisualizationAdapter:
#     def __init__(self, problem: Problem):
#         self.proble = problem

#     def create_mesh(self) -> VisualizationMesh:
#         pass

#     def add_field(
#         self,
#         name: str,
#         dofs: DegreesOfFreedom,
#         values: np.ndarray,
#     ):
#         pass

#     def add_expression(
#         self,
#         name: str,
#         expression,
#     ):
#         pass

#     def to_pyvista(self):
#         pass

#     def write(self, filename):
#         pass


# # DOF Handler


@dataclass
class VizMeshBlock:
    cell_type: pv.CellType
    coords: np.ndarray
    x: np.ndarray
    conn: np.ndarray


def _project_block(
    xcoord: np.ndarray,
    x: np.ndarray,
    geo_layout: DofLayout,
    soln_layout: DofLayout,
    geo_conn: np.ndarray,
    soln_conn: np.ndarray,
    soln_signs: np.ndarray | None = None,
):

    return VizMeshBlock(cell_type=geo_layout.cell_type, coords=coords, x=xmapped)


def _project_to_visual_mesh(
    block_ids: list[int],
    cell_types: list[CellType],
    geo_handler: DofHandler,
    geo_space: FunctionSpace,
    soln_handler: DofHandler,
    soln_space: FunctionSpace,
    xcoord: np.ndarray,
    x: np.ndarray,
):
    # Project problem onto a visualization mesh
    blocks = []

    for block_id, cell_type in zip(block_ids, cell_types):
        # Get the connectivity and signs for the block
        geo_conn = geo_handler.get_dof_conn(geo_space, block_id)
        soln_conn = soln_handler.get_dof_conn(soln_space, block_id)
        soln_signs = soln_handler.get_dof_signs(soln_space, block_id)

        # Get the layout
        geo_layout = DofLayout.make_layout(geo_space, cell_type)
        soln_layout = DofLayout.make_layout(soln_space, cell_type)

        block = _project_block(
            xcoord, x, geo_layout, soln_layout, geo_conn, soln_conn, soln_signs
        )

        blocks.append(block)

    return blocks


def build_pv_grid(
    problem: Problem, name: str, x: Vector, domains: str | list[str] | None = None
):

    block_ids = problem.mesh.get_block_ids(domains)
    cell_types = problem.mesh.get_cell_types(domains)

    geo_space = problem.geo_space.get_spaces()[0]
    geo_handler = problem.geo_dofs.get_dof_handler()

    dof_space = problem.soln_space.get_space(name)
    dof_handler = problem.soln_dofs.get_dof_handler()

    # Build the visualization mesh based on default values
    blocks = _project_to_visual_mesh(
        block_ids, cell_types, geo_handler, geo_space, dof_handler, dof_space, xcoord, x
    )

    # Convert the blocks to a pyvista grid object

    grid = pv.UnstructuredGrid(cells, vtk_types, xyz)
    grid.point_data[name] = xfield

    return grid


def save_vtu(self, x, field, domain):
    grid = self._build_visualization_grid(x, field=field, domain=domain)[0]
    grid.save("output_field.vtu")
    return


def _build_visualization_grid(self, x, field, domain):
    """
    Build a PyVista grid for an H1 triangle solution field of degree 1 or 2
    on a mesh whose geometry is only P1.
    """
    ctype = CellType.TRIANGLE

    soln_space = self.soln_dof.solution_space.get_spaces()[0]
    dof_handler = self.soln_dof.get_dof_handler()
    layout = DofLayout.make_layout(soln_space, ctype)
    ndof_local = len(layout.pts)

    # Map the number of local DOFs to the matching PyVista cell type.
    pv_cell_type = {
        3: pv.CellType.TRIANGLE,  # degree 1
        6: pv.CellType.QUADRATIC_TRIANGLE,  # degree 2
    }.get(ndof_local)

    if pv_cell_type is None:
        raise NotImplementedError(
            "build_visualization_grid supports degree-1 and degree-2 H1 "
            f"triangles (got degree={soln_space.degree}, "
            f"ndof/elem={ndof_local})."
        )

    block_ids = self.mesh.get_block_ids(domain)

    grids = []
    for block_id in block_ids:
        # Global DOF connectivity for the solution space: (nelem, ndof_local),
        soln_conn = dof_handler.get_dof_conn(soln_space, block_id)

        # P1 vertex connectivity and coordinates for the geometry.
        vertex_conn = self.mesh.get_vertex_conn(block_id)
        Xv = np.asarray(self.mesh.X)  # (nnodes, dim)
        dim = Xv.shape[1]

        # Reference elment node layout
        param = np.asarray(layout.pts)  # (ndof_local, 2), (xi, eta)
        xi, eta = param[:, 0], param[:, 1]
        N = np.stack([1.0 - xi - eta, xi, eta], axis=1)  # (ndof_local, 3)

        ndof_global = dof_handler.get_num_dof(soln_space)
        points = np.zeros((ndof_global, 3))
        for e in range(soln_conn.shape[0]):
            vcoords = Xv[vertex_conn[e]]  # (3, dim)
            phys = N @ vcoords  # (ndof_local, dim)
            for a in range(ndof_local):
                points[soln_conn[e, a], :dim] = phys[a]

        grid = pv.UnstructuredGrid({pv_cell_type: soln_conn}, points)
        grid.point_data[field] = np.asarray(x[field])
        grids.append(grid)

    return grids


def visualize_vec_field(self, x, field, domain):
    coords, vecs = self._build_vec_grid(x, field, domain)
    cloud = pv.PolyData(coords)
    cloud["vectors"] = vecs

    arrows = cloud.glyph(orient="vectors", scale="vectors", factor=0.0003)
    pl = pv.Plotter(theme=pv.themes.DarkTheme())
    pl.add_mesh(arrows, cmap="plasma")
    pl.view_xy()
    pl.enable_parallel_projection()
    pl.enable_2d_style()
    # pl.show()
    return pl


def _build_vec_grid(self, x, field, domain):
    ctype = CellType.TRIANGLE
    soln_space = self.soln_dof.solution_space.get_spaces()[0]
    dof_handler = self.soln_dof.get_dof_handler()
    layout = DofLayout.make_layout(soln_space, ctype)
    ndof_local = len(layout.pts)

    # Map the number of local DOFs to the matching PyVista cell type.
    pv_cell_type = {
        3: pv.CellType.TRIANGLE,  # degree 1
        6: pv.CellType.QUADRATIC_TRIANGLE,  # degree 2
    }.get(ndof_local)

    if pv_cell_type is None:
        raise NotImplementedError(
            "build_visualization_grid supports degree-1 and degree-2 H1 "
            f"triangles (got degree={soln_space.degree}, "
            f"ndof/elem={ndof_local})."
        )

    # Global DOF connectivity for the solution space: (nelem, ndof_local),
    soln_conn = dof_handler.get_dof_conn(soln_space, domain, ctype)

    # DOF signs for the H(div) space: (nelem, ndof_local)
    signs = dof_handler.get_dof_signs(soln_space, domain, ctype)

    # Global solution flux values indexed by global DOF number
    u = np.asarray(x[field])

    # P1 vertex connectivity and coordinates for the geometry
    vertex_conn = self.mesh.get_vertex_conn(domain, ctype)  # (nelem, 3)
    Xv = np.asarray(self.mesh.X)  # (nnodes, dim)
    dim = Xv.shape[1]

    # Vector Vandermonde for the RT triangle basis
    vand = TriangleRTBasis([field], soln_space, kind="input").vand

    # Use vec vandermonde at the center of the triangle (1/3, 1/3)
    xi, eta = 1.0 / 3.0, 1.0 / 3.0

    # Compute the basis functions at the center: shape (2, ndof_local)
    N = vand.eval_basis(xi, eta)

    nelem = soln_conn.shape[0]
    centers = np.zeros((nelem, 3))
    vectors = np.zeros((nelem, 3))

    for e in range(nelem):
        vcoords = Xv[vertex_conn[e]]  # (3, dim)

        # Compute J for the affine H1 triangle map (constant over the cell)
        n0_coords = vcoords[0]
        n1_coords = vcoords[1]
        n2_coords = vcoords[2]
        x0 = n0_coords[0]
        y0 = n0_coords[1]
        x1 = n1_coords[0]
        y1 = n1_coords[1]
        x2 = n2_coords[0]
        y2 = n2_coords[1]
        J = np.array(
            [
                [x1 - x0, x2 - x0],
                [y1 - y0, y2 - y0],
            ]
        )

        # Compute detJ
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        # Compute the dot(N, soln): reference-space vector at the center
        vref = np.dot(N, signs[e, :] * u[soln_conn[e, :]])

        # Compute the piola transform (contravariant): v_phys = J @ vref / detJ
        vphys = (J @ vref) / detJ

        # Store the vector
        vectors[e, : dim - 1] = vphys

        # Compute the physical point using an affine transform for H1 triangle
        centers[e, :dim] = vcoords.mean(axis=0)

    return centers, vectors


# def visualize(problem: Problem, soln):

#     cells = []

#     for conn in connectivity:
#         cells.extend([len(conn), *conn])

#     cells = np.asarray(cells, dtype=np.int64)

#     vtk_types = np.asarray(
#         [VTK_CELL_TYPES[t] for t in cell_types],
#         dtype=np.uint8,
#     )

#     grid = pv.UnstructuredGrid(cells, vtk_types, xyz)
