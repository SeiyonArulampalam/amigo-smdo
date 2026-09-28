import numpy as np
import re
from .cell_types import CellType, REFERENCE_CELLS


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
        # Mapping from element name to cell type
        elem_mapping = {"T3D2": CellType.SEGMENT, "CPS3": CellType.TRIANGLE}

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
                    elem_type = elem_mapping[self._find_kw(header, "TYPE")]
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
        conn = self.elem_conn[elset.upper()][cell_type]
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


class BdfParser:
    def __init__(self, filename: str, debug: bool = False):
        try:
            import pyNastran.bdf.bdf as pn

            self.pn = pn
        except:
            raise RuntimeError("Must install pyNastran to use .bdf files")

        self.scan_bdf(filename)

    def scan_bdf(self, filename: str, debug: bool = False):
        info = self.pn.read_bdf(filename, validate=False, xref=False, debug=debug)

        info.missing_properties = False
        for element_id in info.elements:
            element = info.elements[element_id]
            if element.pid not in info.property_ids:
                # If no material properties were found,
                # add dummy properties and materials
                matID = 1
                E = 70.0
                G = 35.0
                nu = 0.3
                info.add_mat1(matID, E, G, nu)
                info.add_pbar(element.pid, matID)
                # Warn the user that the property card is missing
                # and should not be read in using pytacs elemCallBackFromBDF method
                info.missing_properties = True

        self.property_to_elements = info.get_property_id_to_element_ids_map()

        pid_list = list(self.property_to_elements.keys())
        for pid in pid_list:
            # If there are no elements referencing this property card, remove it
            if len(self.property_to_elements[pid]) == 0:
                info.properties.pop(pid)
                self.property_to_elements.pop(pid)

        # Map to contiguous ordering of nodes, components and elements
        self.node_map = dict(zip(info.node_ids, range(info.nnodes)))
        self.property_map = dict(zip(info.property_ids, range(info.nproperties)))
        self.elem_map = dict(zip(info.element_ids, range(info.nelements)))

        # Try to get the node x,y,z locations from bdf file
        try:
            self.Xpts = info.get_xyz_in_coord(fdtype=float, sort_ids=False)

        # If this fails, the file may reference multiple coordinate systems
        # and will have to be cross-referenced to work
        except:
            info.cross_reference()
            info.is_xrefed = True
            self.Xpts = info.get_xyz_in_coord(fdtype=float, sort_ids=False)

        # Create the element connectivity
        self.elem_conn = {}
        for pid in self.property_to_elements:
            element_ids = self.property_to_elements[pid]

            elset = str(self.property_map[pid])
            if not elset in self.elem_conn:
                self.elem_conn[elset] = {}

            for id in element_ids:
                # Get the element
                element = info.elements[id]
                element_type = element.type.upper()

                # Map the id to the contiguous ordering
                elem_id = self.elem_map[id]

                # Check if this type of element been used
                if not element_type in self.elem_conn[elset]:
                    self.elem_conn[elset][element_type] = {}

                nodes = [self.node_map[n] for n in element.nodes]
                self.elem_conn[elset][element_type][elem_id] = nodes

        self.info = info

    def get_nodes(self):
        print(self.Xpts.shape)
        return self.Xpts

    def get_domains(self):
        names = {}
        for elset in self.elem_conn:
            names[elset] = []
            for elem_type in self.elem_conn[elset]:
                names[elset].append(elem_type)

        return names

    def get_conn(self, elset, elem_type):
        conn = self.elem_conn[elset.upper()][elem_type.upper()]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)

    def get_nodes_in_domain(self, elset):
        elset = elset.upper()
        if elset in self.node_sets:
            return np.array(list(dict.fromkeys(self.node_sets[elset])))

        conn = []
        for elem_type in self.elem_conn[elset]:
            conn.extend(self.get_conn(elset, elem_type).flatten())

        # Turn into a single unique list of nodes preserving GMSH ordering
        node_list = np.array(list(dict.fromkeys(conn)))

        return node_list

    def get_basis(self, space, etype, kind):
        basis_list = []

        for sp in ["H1", "const"]:
            names = space.get_names(sp)

            if len(names) == 0:
                continue

            basis_list.append(self._get_basis(etype, sp, names, kind))

        return basis.BasisCollection(basis_list)

    def _get_basis(self, etype, space, names=[], kind="input"):
        if (etype == "CQUAD4" or etype == "CQUADR") and space == "H1":
            return basis.QuadLagrangeBasis(1, names, kind=kind)

        raise NotImplementedError(
            f"Basis for element {etype} with space {space} not implemented"
        )

    def get_quadrature(self, etype):
        if etype == "CQUAD4" or etype == "CQUADR":
            return basis.QuadQuadrature(2)

        raise NotImplementedError(f"Quadrature for element {etype} not implemented")
