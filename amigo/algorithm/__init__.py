"""Amigo interior-point optimizer package.

Module map, grouped by concern:

  ipm_driver.py                  Optimizer class, main loop
  ipm_state.py                   Per-iteration state
  evaluator.py                   Gradient/residual/Hessian evaluation cache
  newton_direction.py            KKT solve and step assembly
  convergence_check.py           Convergence criteria, watchdogs
  iteration_logger.py            Iter-data assembly + progress table
  default_options.py             Default option values

  initialization/                Starting point construction
    iterate_initialization.py    Bounds push, slacks, bound duals
    multiplier_initialization.py Least-squares / affine / zero duals
    nlp_scaling.py               Gradient-based NLP scaling
    presolve.py                  Starting point via restoration

  barrier_strategy/              Barrier-parameter strategies
    base.py                      BarrierStrategy ABC, coupled tau
    monotone.py                  Monotone decrease with progress gate
    quality_function.py          Quality function + adaptive-mu globalization

  globalization/                 Step acceptance and rescue
    filter_line_search.py        Filter LS+ SOC + watchdog
    filter_acceptance.py         Two-dimensional filter
    funnel_line_search.py        Funnel LS
    merit_line_search.py         Backtracking Armijo merit LS
    feasibility_restoration.py   Damped Gauss-Newton restoration phase

  solvers/                       Linear solvers + inertia correction
    inertia_correction.py        Primal and dual regularization
"""

from .ipm_driver import Optimizer
from .default_options import get_default_options

from .solvers import (
    AmigoSolver,
    # DirectCudaSolver,
    # LNKSInexactSolver,
    MumpsSolver,
    # PardisoSolver,
    # DirectPetscSolver,
    # DirectScipySolver,
)
