from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import re
from .cell_types import CellType
from .element_mapping import (
    ElementFamily,
    MeshElementType,
    get_abaqus_element_type,
    get_nastran_element_type,
)


@dataclass
class ElementBlock:
    domain: str
    cell_type: CellType
    degree: int
    family: ElementFamily
    connectivity: np.ndarray
    element_type: MeshElementType
    element_ids: np.ndarray | None = None
    source_type: str | None = None
    property_id: int | None = None


class InpParser:
    def __init__(self):
        # Raw ABAQUS data during parsing
        self._nodes = {}
        self._elements = {}

        # File-format-independent output
        self.X = None
        self.element_blocks = []

        # Mappings from ABAQUS IDs -> internal contiguous IDs
        self.node_id_map = {}
        self.element_id_map = {}

        # Node sets from the INP
        self.node_sets = {}

        return

    def _read_file(self, filename):
        with open(filename, "r", errors="ignore") as fp:
            return [line.rstrip("\n") for line in fp]

    @staticmethod
    def _split_csv_line(s):
        return [p.strip() for p in s.split(",") if p.strip()]

    @staticmethod
    def _find_kw(header, key):
        m = re.search(
            rf"\b{re.escape(key)}\s*=\s*([^,\s]+)",
            header,
            flags=re.I,
        )
        return m.group(1) if m else None

    def parse(self, filename):
        section = None

        elset = None
        source_type = None
        nset_name = None

        for line in self._read_file(filename):
            raw = line.strip()

            if not raw or raw.startswith("**"):
                continue

            # Keywords
            if raw.startswith("*"):
                header = raw.upper()

                if re.match(r"\*NODE\s*(,|$)", header):
                    section = "NODE"

                elif re.match(r"\*ELEMENT\s*(,|$)", header):
                    section = "ELEMENT"
                    source_type = self._find_kw(header, "TYPE")
                    elset = self._find_kw(header, "ELSET")

                    if source_type is None:
                        raise ValueError(f"No element TYPE specified in {raw!r}")

                    source_type = source_type.upper()

                    # Validate now
                    get_abaqus_element_type(source_type)

                    if elset is None:
                        elset = "__DEFAULT__"
                    else:
                        elset = elset.upper()

                    key = (elset, source_type)
                    self._elements.setdefault(key, {})

                elif header.startswith("*NSET"):
                    section = "NSET"

                    nset_name = self._find_kw(header, "NSET")

                    if nset_name is None:
                        raise ValueError(f"No NSET name specified in {raw!r}")

                    nset_name = nset_name.upper()
                    self.node_sets.setdefault(nset_name, [])

                else:
                    section = None

                continue

            # Section data
            parts = self._split_csv_line(raw)

            if section == "NODE":
                abaqus_nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3]) if len(parts) > 3 else 0.0

                self._nodes[abaqus_nid] = (x, y, z)

            elif section == "ELEMENT":
                abaqus_eid = int(parts[0])

                conn = tuple(int(p) for p in parts[1:])

                self._elements[(elset, source_type)][abaqus_eid] = conn

            elif section == "NSET":
                self.node_sets[nset_name].extend(int(p) for p in parts)

        # Convert ABAQUS IDs into internal contiguous IDs
        self._finalize()

    def _finalize(self):
        self._build_nodes()
        self._build_element_blocks()
        self._convert_node_sets()

    def _build_nodes(self):
        """
        Convert arbitrary ABAQUS node IDs into contiguous internal
        numbering 0 ... num_nodes-1.
        """

        node_ids = sorted(self._nodes)
        self.node_id_map = {nid: i for i, nid in enumerate(node_ids)}
        self.X = np.array([self._nodes[nid] for nid in node_ids], dtype=float)

    def _build_element_blocks(self):
        """
        Convert parsed ABAQUS element sections into generic ElementBlock objects.
        """

        next_element = 0

        for (domain, source_type), elements in self._elements.items():
            element_type = get_abaqus_element_type(source_type)
            abaqus_eids = sorted(elements)
            nelem = len(abaqus_eids)

            if nelem == 0:
                continue

            nnode = len(elements[abaqus_eids[0]])
            conn = np.empty((nelem, nnode), dtype=np.int64)
            element_ids = np.empty(nelem, dtype=np.int64)

            for i, abaqus_eid in enumerate(abaqus_eids):
                abaqus_conn = elements[abaqus_eid]

                if len(abaqus_conn) != nnode:
                    raise ValueError(
                        f"Inconsistent number of nodes in " f"{source_type} block"
                    )

                conn[i, :] = [self.node_id_map[nid] for nid in abaqus_conn]
                element_ids[i] = next_element
                self.element_id_map[abaqus_eid] = next_element

                next_element += 1

            block = ElementBlock(
                domain=domain,
                cell_type=element_type.cell_type,
                degree=element_type.degree,
                family=element_type.family,
                element_type=element_type,
                connectivity=conn,
                element_ids=element_ids,
                source_type=source_type,
            )

            self.element_blocks.append(block)

    def _convert_node_sets(self):
        """
        Convert ABAQUS node IDs in NSETs into internal node IDs.
        """

        for name, node_ids in self.node_sets.items():
            self.node_sets[name] = np.array(
                [self.node_id_map[nid] for nid in node_ids], dtype=np.int64
            )

    def get_nodes(self):
        return self.X

    def get_element_blocks(self):
        return self.element_blocks

    def get_node_sets(self):
        return self.node_sets


# class BdfParser:
#     def __init__(self, filename: str, debug: bool = False):
#         try:
#             import pyNastran.bdf.bdf as pn

#             self.pn = pn
#         except:
#             raise RuntimeError("Must install pyNastran to use .bdf files")

#         self.scan_bdf(filename)

#     def scan_bdf(self, filename: str, debug: bool = False):
#         info = self.pn.read_bdf(filename, validate=False, xref=False, debug=debug)

#         info.missing_properties = False
#         for element_id in info.elements:
#             element = info.elements[element_id]
#             if element.pid not in info.property_ids:
#                 # If no material properties were found,
#                 # add dummy properties and materials
#                 matID = 1
#                 E = 70.0
#                 G = 35.0
#                 nu = 0.3
#                 info.add_mat1(matID, E, G, nu)
#                 info.add_pbar(element.pid, matID)
#                 # Warn the user that the property card is missing
#                 # and should not be read in using pytacs elemCallBackFromBDF method
#                 info.missing_properties = True

#         self.property_to_elements = info.get_property_id_to_element_ids_map()

#         pid_list = list(self.property_to_elements.keys())
#         for pid in pid_list:
#             # If there are no elements referencing this property card, remove it
#             if len(self.property_to_elements[pid]) == 0:
#                 info.properties.pop(pid)
#                 self.property_to_elements.pop(pid)

#         # Map to contiguous ordering of nodes, components and elements
#         self.node_map = dict(zip(info.node_ids, range(info.nnodes)))
#         self.property_map = dict(zip(info.property_ids, range(info.nproperties)))
#         self.elem_map = dict(zip(info.element_ids, range(info.nelements)))

#         # Try to get the node x,y,z locations from bdf file
#         try:
#             self.Xpts = info.get_xyz_in_coord(fdtype=float, sort_ids=False)

#         # If this fails, the file may reference multiple coordinate systems
#         # and will have to be cross-referenced to work
#         except:
#             info.cross_reference()
#             info.is_xrefed = True
#             self.Xpts = info.get_xyz_in_coord(fdtype=float, sort_ids=False)

#         # Create the element connectivity
#         self.elem_conn = {}
#         for pid in self.property_to_elements:
#             element_ids = self.property_to_elements[pid]

#             elset = str(self.property_map[pid])
#             if not elset in self.elem_conn:
#                 self.elem_conn[elset] = {}

#             for id in element_ids:
#                 # Get the element
#                 element = info.elements[id]
#                 element_type = element.type.upper()

#                 # Map the id to the contiguous ordering
#                 elem_id = self.elem_map[id]

#                 # Check if this type of element been used
#                 if not element_type in self.elem_conn[elset]:
#                     self.elem_conn[elset][element_type] = {}

#                 nodes = [self.node_map[n] for n in element.nodes]
#                 self.elem_conn[elset][element_type][elem_id] = nodes

#         self.info = info

#     def get_nodes(self):
#         return self.Xpts

#     def get_domains(self):
#         names = {}
#         for elset in self.elem_conn:
#             names[elset] = []
#             for elem_type in self.elem_conn[elset]:
#                 names[elset].append(elem_type)

#         return names

#     def get_conn(self, elset, elem_type):
#         conn = self.elem_conn[elset.upper()][elem_type.upper()]
#         return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)


class BdfParser:
    def __init__(self):
        from pyNastran.bdf.bdf import BDF

        self.BDF = BDF

        self.X = {}
        self.elem_conn = {}

    def parse_bdf(self, filename):
        model = self.BDF()
        model.read_bdf(filename)

        # --------------------------------------------------------
        # Nodes
        # --------------------------------------------------------

        node_ids = sorted(model.nodes)

        node_map = {nid: i for i, nid in enumerate(node_ids)}

        for nid in node_ids:
            node = model.nodes[nid]

            # Position in the global coordinate system
            xyz = node.get_position()

            self.X[node_map[nid]] = np.asarray(xyz)

        # --------------------------------------------------------
        # Elements
        # --------------------------------------------------------

        for eid, elem in model.elements.items():

            card_type = elem.type
            node_ids = [nid for nid in elem.node_ids if nid is not None]

            elem_type = get_nastran_element_type(
                card_type,
                len(node_ids),
            )

            conn = [node_map[nid] for nid in node_ids]

            # Property ID is a reasonable default "domain"
            pid = elem.Pid()

            domain = f"PID_{pid}"

            key = (
                domain,
                elem_type.cell_type,
                elem_type.degree,
                elem_type.family,
            )

            self.elem_conn.setdefault(key, {})[eid] = conn
