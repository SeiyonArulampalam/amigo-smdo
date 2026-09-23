import numpy as np
import re
from reference_cells import CellType, REFERENCE_CELLS
from visualization import plot_tri_mesh


class InpParser:
    def __init__(self):
        self.X = {}
        self.elem_conn = {}
        self.node_sets = {}
        self.surfaces = []
        self.cell_type_map = {CellType.TRIANGLE: "CPS3", CellType.SEGMENT: "T3D2"}
        self.elem_type_map = {v: k for k, v in self.cell_type_map.items()}

    def _read_file(self, filename):
        with open(filename, "r", errors="ignore") as fp:
            return [line.rstrip("\n") for line in fp]

    def _split_csv_line(self, s):
        # ABAQUS lines are simple CSV-like, no quotes typically
        return [p.strip() for p in s.split(",") if p.strip()]

    def _find_kw(self, header, key):
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,\s]+)", header, flags=re.I)
        return m.group(1) if m else None

    def parse_inp(self, filename):
        self.__init__()
        section = elset = elem_type = nset_name = None

        for line in self._read_file(filename):
            raw = line.strip()
            if not raw or raw.startswith("**"):
                continue

            if raw.startswith("*"):
                header = raw.upper()
                # Match "*NODE" / "*ELEMENT" but not "*NODE OUTPUT" / "*ELEMENT OUTPUT"
                if re.match(r"\*NODE\s*(,|$)", header):
                    section = "NODE"
                elif re.match(r"\*ELEMENT\s*(,|$)", header):
                    section = "ELEMENT"
                    elem_type = self._find_kw(header, "TYPE")
                    elset = self._find_kw(header, "ELSET")
                    self.elem_conn.setdefault(elset, {}).setdefault(elem_type, {})
                    if elset and "SURFACE" in elset and elset not in self.surfaces:
                        self.surfaces.append(elset)
                elif header.startswith("*NSET"):
                    section = "NSET"
                    nset_name = self._find_kw(header, "NSET")
                    self.node_sets.setdefault(nset_name, [])
                else:
                    section = None
                continue

            parts = self._split_csv_line(raw)
            if section == "NODE":
                z = float(parts[3]) if len(parts) > 3 else 0.0
                self.X[int(parts[0]) - 1] = (float(parts[1]), float(parts[2]), z)
            elif section == "ELEMENT":
                self.elem_conn[elset][elem_type][int(parts[0]) - 1] = [
                    int(p) - 1 for p in parts[1:]
                ]
            elif section == "NSET":
                self.node_sets[nset_name].extend(int(p) - 1 for p in parts)

    def get_nodes(self):
        return np.array([self.X[k] for k in sorted(self.X.keys())])

    def get_domains(self):
        return {elset: list(types) for elset, types in self.elem_conn.items()}

    def get_num_surfaces(self):
        return len(self.surfaces)

    def get_conn(self, elset, cell_type):
        elem_type = self.cell_type_map[cell_type]
        conn = self.elem_conn[elset.upper()][elem_type.upper()]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)

    def get_nodes_in_domain(self, elset):
        elset = elset.upper()
        if elset in self.node_sets:
            return np.array(list(dict.fromkeys(self.node_sets[elset])))

        conn = []
        for elem_type in self.elem_conn[elset]:
            cell_type = self.elem_type_map[elem_type]
            conn.extend(self.get_conn(elset, cell_type).flatten())

        # Single unique list of nodes preserving GMSH ordering
        return np.array(list(dict.fromkeys(conn)))

    def get_conn_edges(self, elset, cell_type: CellType):
        conn = self.get_conn(elset, cell_type)
        local_edges = REFERENCE_CELLS[cell_type].edges

        seen = {}
        edge_conn = np.zeros((len(conn), len(local_edges)), dtype=int)
        elem_signs = np.zeros_like(edge_conn)

        for e, nodes in enumerate(conn):
            for i, local in enumerate(local_edges):
                n0, n1 = int(nodes[local[0]]), int(nodes[local[1]])
                key = (min(n0, n1), max(n0, n1))
                if key not in seen:
                    seen[key] = len(seen)
                edge_conn[e, i] = seen[key]
                elem_signs[e, i] = 1 if n0 < n1 else -1

        return edge_conn, elem_signs


if __name__ == "__main__":
    p = InpParser()
    p.parse_inp("mesh.inp")
    X = p.get_nodes()
    conn = p.get_conn("SURFACE1", CellType.TRIANGLE)
    edge_conn, edge_signs = p.get_conn_edges("SURFACE1", CellType.TRIANGLE)
    domains = p.get_domains()
    print(list(domains.keys()))
    print(domains["SURFACE1"])
    plot_tri_mesh(X, conn, edge_conn, edge_signs)
