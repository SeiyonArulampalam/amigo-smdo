from parser import InpParser
from pathlib import Path
from reference_cells import CellType
from visualization import plot_tri_mesh


class Mesh:
    def __init__(self, filename: str):
        ext = Path(filename).suffix
        if ext == ".inp" or ext == ".INP":
            self.parser = InpParser()
            self.parser.parse_inp(filename)
        else:
            raise ValueError(f"Unrecognized file extension {ext}")
        self.X = self.parser.get_nodes()

        return

    def get_domains(self):
        """Get the names of all domains within the mesh"""
        return list(self.parser.get_domains().keys())

    def get_cell_types(self, domain):
        """Get the cell types within a domain"""
        elsets = self.parser.get_domains()[domain]
        return [self.parser.elem_type_map[e] for e in elsets]

    def get_vertex_conn(self, domain, cell_type):
        """Get the connectivity of the vertices for a component"""
        return self.parser.get_conn(domain, cell_type)

    def get_edge_conn(self, domain, cell_type):
        """Get the edge connectivity"""
        conn, signs = self.parser.get_conn_edges(domain, cell_type)
        return conn, signs


if __name__ == "__main__":
    mesh = Mesh("mesh.inp")
    domains = mesh.get_domains()
    print("Domains:", domains)
    X = mesh.X
    conn = mesh.get_vertex_conn("SURFACE1", CellType.TRIANGLE)
    edge_conn, edge_signs = mesh.get_edge_conn("SURFACE1", CellType.TRIANGLE)
    cell_types = mesh.get_cell_types("SURFACE1")
    print("Cell Types:", cell_types)
    plot_tri_mesh(X, conn, edge_conn, edge_signs)
