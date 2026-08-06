from dataclasses import dataclass
import numpy as np
import amigo as am
from amigo import ExpressionComponent as ExprComp
import argparse
from numpy.typing import ArrayLike, NDArray
from scipy.special import lambertw

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class ParallelSellarData:
    b: Vector
    c: Vector
    a: Vector
    r: Vector
    mu: Vector
    d: float
    omega: Vector
    gamma: Vector


@dataclass(frozen=True)
class ParallelSellarSolution:
    x: Vector
    z: Vector
    y: Vector
    q: Vector
    Y: float
    Q: float
    objective: float
    s_unconstrained: float
    active_set: str
    y_is_unique: bool


def _vector(name: str, values: ArrayLike) -> Vector:
    result = np.atleast_1d(np.asarray(values, dtype=float))
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def solve_parallel_sellar(
    data: ParallelSellarData,
    atol: float = 1.0e-12,
) -> ParallelSellarSolution:
    """Return an analytic solution of the parallel-group Sellar problem.

    The weights need not sum to one.  If the optimal aggregate Y is larger
    than C = dot(mu, c), the objective does not uniquely determine every y_i.
    In that case this routine selects the canonical allocation

        y_i = c_i + (Y - C)/sum(mu),

    which adds the same amount to every lower bound.
    """
    c = _vector("c", data.c)
    b = _vector("b", data.b)
    a = _vector("a", data.a)
    r = _vector("r", data.r)
    mu = _vector("mu", data.mu)
    omega = _vector("omega", data.omega)
    gamma = _vector("gamma", data.gamma)

    if not (len(c) == len(b) == len(mu)):
        raise ValueError("c, b, and mu must have the same length")
    if not (len(a) == len(r) == len(omega)):
        raise ValueError("a, r, and omega must have the same length")
    if np.any(c < 0.0):
        raise ValueError("c must be nonnegative so sqrt(Y) is well defined")
    if np.any(mu <= 0.0):
        raise ValueError("mu must be strictly positive")
    if np.any(omega < 0.0):
        raise ValueError("omega must be nonnegative")
    if len(gamma) != len(c) or np.any(gamma <= 0.0):
        raise ValueError("gamma must match c and be strictly positive")

    C = float(mu @ c)
    A = float(omega @ a)
    R = float(omega @ r)

    if A <= 0.0:
        raise ValueError("A = dot(omega, a) must be strictly positive")

    # With s = sqrt(Y), minimize F(s) = s^2 + exp(-R - A*s).
    s_lower = np.sqrt(C)
    s_upper = (float(data.d) - R) / A
    if s_upper < s_lower - atol:
        raise ValueError(
            "No feasible point exists on the nonnegative branch: "
            "d < R + A*sqrt(dot(mu, c))."
        )

    argument = 0.5 * A**2 * np.exp(-R)
    s0 = float(np.real(lambertw(argument))) / A
    s = float(np.clip(s0, s_lower, s_upper))

    if np.isclose(s, s_lower, atol=atol, rtol=0.0):
        active_set = "y lower bounds"
    elif np.isfinite(s_upper) and np.isclose(s, s_upper, atol=atol, rtol=0.0):
        active_set = "Q upper bound"
    else:
        active_set = "interior aggregate solution"

    Y = s**2
    extra = max(0.0, Y - C)
    y = c + extra / np.sum(mu)
    y_is_unique = extra <= atol

    q = a * s + r
    Q = float(omega @ q)
    x = np.zeros_like(c)

    radicand = y + b * Q
    if np.any(radicand < -atol):
        raise ValueError("The requested parameters make z_i**2 negative")
    z = np.sqrt(np.maximum(radicand, 0.0))

    objective = float(Y + np.exp(-Q) + gamma @ (x * x))

    return ParallelSellarSolution(
        x=x,
        z=z,
        y=y,
        q=q,
        Y=Y,
        Q=Q,
        objective=objective,
        s_unconstrained=s0,
        active_set=active_set,
        y_is_unique=y_is_unique,
    )


def generate_data(N: int = 5, M: int = 7, d: float = 10.0) -> ParallelSellarData:
    np.random.seed(0)
    b = np.random.uniform(size=N)
    c = np.random.uniform(size=N)
    a = np.random.uniform(size=M)
    r = np.random.uniform(size=M)
    mu = np.random.uniform(size=N)
    omega = np.random.uniform(size=M)
    gamma = np.random.uniform(size=N)

    return ParallelSellarData(c=c, b=b, a=a, r=r, mu=mu, omega=omega, d=d, gamma=gamma)


d = 10.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

comp1 = ExprComp(
    name="Comp1",
    inputs={
        "y": {},
        "z": {"value": 1.0, "lower": 0.0},
        "x": {"value": 1.0, "lower": 0.0},
        "Q": {},
    },
    data=["b", "c"],
    constraints={
        "con1": "y - z**2 - x + b * Q",
        "con2": {"expr": "y - c", "lower": 0.0, "upper": am.inf},
    },
)
comp2 = ExprComp(
    name="Comp2",
    inputs=["q", "Y"],
    data=["a", "r"],
    constraints={"con1": "q - a * sqrt(Y) - r"},
)

redy = ExprComp(
    name="ReduceY", inputs=["y"], data=["mu"], constraints={"con": "-mu * y"}
)
targety = ExprComp(
    name="TargetY", inputs={"Y": {"value": 1.0, "lower": 0.0}}, constraints={"con": "Y"}
)

redq = ExprComp(
    name="ReduceQ", inputs=["q"], data=["omega"], constraints={"con": "-omega * q"}
)
targetq = ExprComp(name="TargetQ", inputs={"Q": {"upper": d}}, constraints={"con": "Q"})

obj1 = ExprComp(name="Obj1", inputs=["Y", "Q"], objective={"obj": "Y + exp(-Q)"})
obj2 = ExprComp(
    name="Obj2", inputs=["x"], data=["gamma"], objective={"obj": "gamma * x**2"}
)

# Set up the problem data
N = 5
M = 4
sellar_data = generate_data(N, M, d=d)

# Create the model
model = am.Model("sellar")
model.add_component("comp1", N, comp1)
model.add_component("redy", N, redy)
model.add_component("targety", 1, targety)

model.add_component("comp2", M, comp2)
model.add_component("redq", M, redq)
model.add_component("targetq", 1, targetq)

# Add the objective terms
model.add_component("obj1", 1, obj1)
model.add_component("obj2", N, obj2)

# Link the variables
model.link("comp1.y", "redy.y")
model.link("targetq.Q", "comp1.Q")

model.link("comp2.q", "redq.q")
model.link("comp2.Y", "targety.Y")

model.link("targetq.Q", "obj1.Q")
model.link("targety.Y", "obj1.Y")
model.link("comp1.x", "obj2.x")

# Link the constraints
model.link("targety.con", "redy.con")
model.link("targetq.con", "redq.con")

values = model.get_meta_view("value")
data = model.get_meta_view("value", "data")
data["comp1.b"] = sellar_data.b
data["comp1.c"] = sellar_data.c
data["comp2.a"] = sellar_data.a
data["comp2.r"] = sellar_data.r
data["redy.mu"] = sellar_data.mu
data["redq.omega"] = sellar_data.omega
data["obj2.gamma"] = sellar_data.gamma

if args.build:
    model.build_module()

model.initialize()

x = model.create_vector()
opt = am.Optimizer(model, x)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_iterations": 500,
        "convergence_tolerance": 1e-12,
    }
)

solution = solve_parallel_sellar(sellar_data)

print(f"active set: {solution.active_set}")
print(f"objective:  {solution.objective:.12g}")
print(f"Y:          {solution.Y:.12g}   {x["obj1.Y"]}")
print(f"Q:          {solution.Q:.12g}   {x["obj1.Q"]}")
print(f"x:          {solution.x}        {x["obj2.x"]}")
print(f"z:          {solution.z}        {x["comp1.z"]}")
print(f"y:          {solution.y}        {x["comp1.y"]}")
print(f"q:          {solution.q}        {x["comp2.q"]}")


import matplotlib.pylab as plt
from scipy.sparse import csr_matrix

hessian = opt.solver.hessian
nrows, ncols, nnz, rowp, cols = hessian.get_nonzero_structure()
data = np.ones(nnz)
H = csr_matrix((data, cols, rowp), shape=(nrows, ncols))

plt.figure(figsize=(6, 6))
plt.spy(H, markersize=3.0)
plt.title("Sparsity pattern of matrix A")
plt.show()
