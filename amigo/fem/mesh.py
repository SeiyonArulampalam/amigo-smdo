from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .parser import InpParser, BdfParser, ElementBlock
from .cell_types import REFERENCE_CELLS


@dataclass
class ElementBlockTopology:
    block: ElementBlock
    vertex_conn: np.ndarray | None = None
    edge_conn: np.ndarray | None = None
    edge_orientation: np.ndarray | None = None
    face_conn: np.ndarray | None = None
    face_orientation: np.ndarray | None = None


class Mesh:
    def __init__(self, filename: str):
        ext = Path(filename).suffix.lower()
        if ext == ".inp":
            parser = InpParser()

        elif ext in (".bdf", ".nas", ".dat"):
            parser = BdfParser()

        else:
            raise ValueError(f"Unrecognized mesh file extension {ext}")

        parser.parse(filename)

        self.X = parser.get_nodes()
        self.blocks = parser.get_element_blocks()
        self.node_sets = parser.get_node_sets()

        # Set the block topology
        self.block_topo = [ElementBlockTopology(block=block) for block in self.blocks]

        # Build the block topology for the edges
        self._build_edges()

    def _get_block_vertex_conn(self, block):
        vertex_local = block.element_type.vertices
        return block.connectivity[:, vertex_local]

    def _build_edges(self):
        edge_map = {}
        next_edge = 0

        self.edge_conn = []
        self.edge_orientation = []

        for block_id, block in enumerate(self.blocks):
            ref_cell = REFERENCE_CELLS[block.cell_type]
            vertex_conn = self._get_block_vertex_conn(block)

            nelem = vertex_conn.shape[0]
            nedge = len(ref_cell.edges)
            edge_conn = np.empty((nelem, nedge), dtype=np.int64)
            edge_orientation = np.empty((nelem, nedge), dtype=np.int8)

            for elem in range(nelem):
                for local_edge, (i0, i1) in enumerate(ref_cell.edges):
                    n0 = int(vertex_conn[elem, i0])
                    n1 = int(vertex_conn[elem, i1])
                    key = (min(n0, n1), max(n0, n1))

                    if key not in edge_map:
                        edge_map[key] = next_edge
                        next_edge += 1

                    edge_conn[elem, local_edge] = edge_map[key]
                    edge_orientation[elem, local_edge] = 1 if n0 < n1 else -1

            self.block_topo[block_id].vertex_conn = vertex_conn
            self.block_topo[block_id].edge_conn = edge_conn
            self.block_topo[block_id].edge_orientation = edge_orientation

    def get_num_blocks(self):
        """Get the names of all domains within the mesh"""
        return len(self.blocks)

    def get_domain_name(self, block_id: int):
        return self.blocks[block_id].domain

    def get_cell_type(self, block_id: int):
        """Get the cell type for the given block id"""
        return self.blocks[block_id].cell_type

    def get_cell_types(self, domains: str | list[str] | None = None):
        """Get a list of the cell types corresponding to a domain"""
        if domains is None:
            return dict.fromkeys(block.cell_type for block in self.blocks)
        elif isinstance(domains, str):
            return dict.fromkeys(
                block.cell_type for block in self.blocks if block.domain == domains
            )
        elif isinstance(domains, list):
            d = {}
            for domain in domains:
                for block in self.blocks:
                    if block.domain == domain:
                        d[block.cell_type] = block.cell_type
            return list(d)
        else:
            raise TypeError("Expected string or list of strings")

    def get_block_ids(self, domains: str | list[str]):
        """Get a list of the block ids in a domain"""
        if isinstance(domains, str):
            return [i for i, block in enumerate(self.blocks) if block.domain == domains]
        elif isinstance(domains, list):
            d = {}
            for domain in domains:
                for block_id, block in enumerate(self.blocks):
                    if block.domain == domain:
                        d[block_id] = block_id
            return list(d)
        else:
            raise TypeError("Expected string or list of strings")

    def get_num_elements(self, block_id: int):
        """Get the number of elements"""
        return self.block_topo[block_id].vertex_conn.shape[0]

    def get_vertex_conn(self, block_id: int):
        """Get the connectivity of the vertices for a component"""
        return self.block_topo[block_id].vertex_conn

    def get_edge_conn(self, block_id: int):
        """Get the edge connectivity"""
        return (
            self.block_topo[block_id].edge_conn,
            self.block_topo[block_id].edge_orientation,
        )

    def get_face_conn(self, block_id: int):
        """Get the face connectivity"""
        return (
            self.block_topo[block_id].face_conn,
            self.block_topo[block_id].face_orientation,
        )

    #     return self.parser.get_face_conn(domain, cell_type)


# What's needed:
#
# 1. Element block maps to cell type and domain names


# class Mesh:
#     def __init__(self, filename: str):
#         ext = Path(filename).suffix
#         if ext == ".inp" or ext == ".INP":
#             self.parser = InpParser()
#             self.parser.parse_inp(filename)
#         else:
#             raise ValueError(f"Unrecognized file extension {ext}")
#         self.X = self.parser.get_nodes()

#         return

#     def get_domains(self):
#         """Get the names of all domains within the mesh"""
#         return self.parser.get_domains()

#     def get_cell_types(self, domain):
#         """Get the cell types within a domain"""
#         return self.parser.get_domains()[domain]

#     def get_vertex_conn(self, domain, cell_type):
#         """Get the connectivity of the vertices for a component"""
#         return self.parser.get_conn(domain, cell_type)

#     def get_edge_conn(self, domain, cell_type):
#         """Get the edge connectivity"""
#         return self.parser.get_edge_conn(domain, cell_type)

#     def get_face_conn(self, domain, cell_type):
#         return self.parser.get_face_conn(domain, cell_type)

#     def get_num_elements(self, name, cell_type):
#         return self.parser.get_conn(name, cell_type).shape[0]
