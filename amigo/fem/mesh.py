from pathlib import Path
from .parser import InpParser


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
        return self.parser.get_domains()[domain]

    def get_vertex_conn(self, domain, cell_type):
        """Get the connectivity of the vertices for a component"""
        return self.parser.get_conn(domain, cell_type)

    def get_edge_conn(self, domain, cell_type):
        """Get the edge connectivity"""
        conn, signs = self.parser.get_edge_conn(domain, cell_type)
        return conn, signs

    def get_face_conn(self, domain, cell_type):
        conn, orientation = self.parser.get_face_conn(domain, cell_type)
        return conn, orientation

    def get_num_elements(self, name, cell_type):
        return self.parser.get_conn(name, cell_type).shape[0]

    def get_nodes_in_domain(self, elset):
        return self.parser.get_nodes_in_domain(elset)
