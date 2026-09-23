from dataclasses import dataclass
from enum import Enum, auto
from reference_cells import CellType


class Space(Enum):
    H1 = auto()
    L2 = auto()
    HDIV = auto()
    HCURL = auto()
    CONST = auto()


class Conformity(Enum):
    GLOBAL = auto()
    COMPONENT = auto()
    DISCONTINUOUS = auto()


@dataclass(frozen=True)
class FunctionSpace:
    func_space: Space
    degree: int = 1
    conformity: Conformity = Conformity.GLOBAL


class SolutionSpace:
    def __init__(self, mapping):
        self.fields = []  # (name, FunctionSpace) in declaration order
        self.names = {}  # FunctionSpace -> [names]

        for names, space in mapping.items():
            if isinstance(names, str):
                names = (names,)
            if isinstance(space, str):
                try:
                    # Enum lookup is by name, not value: auto() values are ints.
                    space = FunctionSpace(Space[space.upper()])
                except KeyError:
                    raise ValueError(f"Unknown function space '{space}'")
            elif not isinstance(space, FunctionSpace):
                raise TypeError(f"Expected FunctionSpace or str, got {type(space)}")

            self.names.setdefault(space, []).extend(names)
            self.fields.extend((name, space) for name in names)

    def get_spaces(self):
        return list(self.names)

    def get_names(self, space):
        return self.names.get(space, [])

    def get_space(self, name):
        for field_name, space in self.fields:
            if field_name == name:
                return space
        raise KeyError(f"Unknown field '{name}'")


@dataclass(frozen=True)
class FiniteElementSpace:
    soln_space: SolutionSpace
    cell_type: CellType
