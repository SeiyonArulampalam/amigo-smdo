from pathlib import Path
from .parser import InpParser
from .cell_types import CellType

# # from visualization import plot_tri_mesh


# class Mesh:
#     def __init__(self, filename: str):
#         ext = Path(filename).suffix
#         if ext == ".inp" or ext == ".INP":
#             self.parser = InpParser()
#             self.parser.parse_inp(filename)
#         elif ext == ".bdf" or ext == ".BDF":
#             self.parser = BdfParser(filename)
#         else:
#             raise ValueError(f"Unrecognized file extension {ext}")

#         self.X = self.parser.get_nodes()

#     def get_num_nodes(self):
#         return self.X.shape[0]

#     def get_domains(self):
#         """
#         Get a dictionary of the element types for each domain, indexed by the name
#         of each domain
#         """
#         return self.parser.get_domains()

#     def get_conn(self, name, etype):
#         """
#         Get the connectivity for the given domain name with the given element type
#         """
#         return self.parser.get_conn(name, etype)

#     def get_basis(self, space, etype, kind):
#         """
#         Get an instance of a Basis class that is matched to the solution space, element
#         type and kind of quantity
#         """
#         return self.parser.get_basis(space, etype, kind=kind)

#     def get_quadrature(self, etype):
#         """
#         Get an instance of Quadrature that is matched to the element type
#         """
#         return self.parser.get_quadrature(etype)

#     def get_nodes_in_domain(self, name):
#         """
#         Get nodes in the specified domain for all element types (if etype == None),
#         or a specific element type.
#         """
#         return self.parser.get_nodes_in_domain(name)

#     def get_num_elements(self, name, etype):
#         return self.parser.get_conn(name, etype).shape[0]

#     def get_edge_conn(self, name, etype):
#         conn, signs = self.parser.get_conn_edges(name, etype)
#         return conn, signs

#     def plot(self, u, **kwargs):
#         """Plot the finite element solution on the mesh"""
#         plot(self, u, **kwargs)


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
        return self.parser.get_domains()

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

    def get_num_elements(self, name, cell_type):
        return self.parser.get_conn(name, cell_type).shape[0]


# if __name__ == "__main__":
#     mesh = Mesh("mesh.inp")
#     domains = mesh.get_domains()
#     print("Domains:", domains)
#     X = mesh.X
#     conn = mesh.get_vertex_conn("SURFACE1", CellType.TRIANGLE)
#     edge_conn, edge_signs = mesh.get_edge_conn("SURFACE1", CellType.TRIANGLE)
#     cell_types = mesh.get_cell_types("SURFACE1")
#     print("Cell Types:", cell_types)
#     plot_tri_mesh(X, conn, edge_conn, edge_signs)
