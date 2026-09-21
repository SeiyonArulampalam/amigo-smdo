"""Default option values for the interior-point optimizer.

The options are split into one group per stage of the algorithm, in the
order a solve runs through them.
"""

import copy

# Iteration limit, convergence tolerances, and the divergence watchdogs
CONVERGENCE = {
    "max_iterations": 100,
    "convergence_tolerance": 1e-8,
    "dual_inf_tol": 1.0,
    "constr_viol_tol": 1e-4,
    "compl_inf_tol": 1e-4,
    # Acceptable convergence, a weaker solution held for several iterations
    "acceptable_tol": 1e-6,
    "acceptable_iter": 15,
    "acceptable_dual_inf_tol": 1e10,
    "acceptable_constr_viol_tol": 1e-2,
    "acceptable_compl_inf_tol": 1e-2,
    # Divergence watchdogs, both disabled by a zero tolerance
    "diverging_iterates_tol": 1e20,
    "diverging_step_tol": 1e8,
    "diverging_step_iters": 3,
}

# Starting point, multiplier estimate, warm start, and NLP scaling
STARTING_POINT = {
    "init_least_squares_multipliers": True,
    # Cap on the initial multiplier estimate magnitude
    "constr_mult_init_max": 1e3,
    # Reset the multipliers by least squares when a nearly feasible point has a large dual infeasibility
    "recompute_multipliers": True,
    "recompute_multiplier_tol": 1e-1,
    "recompute_dual_inf_trigger": 1e2,
    # Begin from a restored feasible point with centered duals
    "feasibility_presolve": False,
    # Relax bounds by factor * max(1, |b|), capped at constr_viol_tol
    "bound_relax_factor": 0.0,
    # Warm start keeping the supplied constraint multipliers
    "warm_start": False,
    "warm_start_mult_init_max": 1e3,
    "warm_start_mu_init": 1e-4,
    # Gradient based NLP scaling
    "nlp_scaling": True,
    "nlp_scaling_max_gradient": 100.0,
    "nlp_scaling_min_value": 1e-8,
}

# Barrier parameter and the monotone, heuristic, and quality function strategies
BARRIER = {
    "barrier_strategy": "monotone",
    "initial_barrier_param": 1.0,
    "fraction_to_boundary": 0.99,
    "adaptive_tau": True,
    "tau_min": 0.99,
    "verbose_barrier": False,
    "continuation_control": None,
    # Monotone strategy, initial mu is "fixed" or "complementarity"
    "mu_init_strategy": "fixed",
    "barrier_progress_tol": 10.0,
    "mu_linear_decrease_factor": 0.2,
    "mu_superlinear_decrease_power": 1.5,
    "mu_min": 1e-11,
    # Heuristic strategy
    "heuristic_barrier_gamma": 0.1,
    "heuristic_barrier_r": 0.95,
    # Quality function strategy and its adaptive mu globalization
    "mu_max_fact": 1e3,
    "barrier_tol_factor": 10.0,
    "adaptive_mu_globalization": "obj-constr-filter",
    "adaptive_mu_kkterror_red_iters": 4,
    "adaptive_mu_kkterror_red_fact": 0.9999,
    "adaptive_mu_monotone_init_factor": 0.8,
    # Lower safeguard, mu >= factor * scaled infeasibility
    "adaptive_mu_safeguard_factor": 0.0,
    "quality_function_sigma_max": 100.0,
    "quality_function_sigma_min": 1e-6,
    "quality_function_section_sigma_tol": 1e-2,
    "quality_function_section_qf_tol": 0.0,
    "quality_function_golden_iters": 8,
    "quality_function_norm_scaling": True,
    # "none", "log", "reciprocal", or "cubed-reciprocal"
    "quality_function_centrality": "none",
    # "none" or "cubic"
    "quality_function_balancing_term": "none",
    "quality_function_predictor_corrector": False,
    # Barrier increase when the line search keeps rejecting
    "max_consecutive_rejections": 5,
    "barrier_increase_factor": 5.0,
}

# Linear solver, regularization, and inertia correction
LINEAR_SOLVER = {
    "solver": "amigo",
    "amigo_pivot_tol": 1e-14,
    # GPU solver, pivots below the epsilon are statically perturbed
    "cuda_pivot_eps": 1e-8,
    "cuda_ir_steps": 2,
    "cuda_check_residual": True,
    "cuda_residual_rtol": 1e-4,
    # Constraint regularization delta_c = value * mu^exponent
    "jacobian_regularization_value": 1e-8,
    "jacobian_regularization_exponent": 0.25,
    # Apply delta_c on every factorization, for persistently singular KKTs
    "perturb_always_cd": False,
    # Resume from the decayed previous delta_x instead of restarting at 0
    "hessian_perturbation_warm_start": False,
    # Always-on dual regularization delta_c = coeff * max(||c||, mu) and 0 disables
    "dual_reg_infeas_coeff": 0.0,
    # Accept a non-singular factorization with wrong inertia once delta_w reaches this cap
    "max_regularization": 0.0,
    # Tolerated departure from the expected positive and negative counts
    "inertia_tolerance": 0,
}

# Line search with the filter, merit, and funnel acceptance rules
LINE_SEARCH = {
    "line_search": "filter",
    "max_line_search_iterations": 40,
    "backtracking_factor": 0.5,
    "armijo_constant": 1e-4,
    "second_order_correction": True,
    # Clip the bound duals into a band around mu / gap and 0 disables
    "kappa_sigma": 1e4,
    # Filter line search
    # Violation ceiling theta_max = fact * max(1, theta0)
    "filter_theta_max_fact": 1e4,
    "filter_gamma_theta": 1e-5,
    "filter_gamma_phi": 1e-8,
    "filter_delta": 1.0,
    "filter_s_theta": 1.1,
    "filter_s_phi": 2.3,
    "filter_eta_phi": 1e-8,
    "filter_max_soc": 4,
    "filter_kappa_soc": 0.99,
    "filter_reset_trigger": 5,
    "max_filter_resets": 5,
    # Merit line search
    "merit_penalty": 1.0,
    "merit_armijo_constant": 1e-4,
    # Funnel line search
    "funnel_ubd": 1.0,
    "funnel_fact": 1.5,
    "funnel_beta": 0.9999,
    "funnel_gamma": 0.001,
    "funnel_kappa": 0.5,
    "funnel_update_strategy": 1,
    "funnel_require_acceptance_wrt_current_iterate": False,
    "funnel_switching_delta": 0.999,
    "funnel_switching_infeasibility_exponent": 2,
    "funnel_armijo_fraction": 1e-4,
    "funnel_armijo_tolerance": 1e-9,
    "funnel_min_step_length": 1e-12,
}

# Feasibility restoration
RESTORATION = {
    "feasibility_restoration": True,
    "max_restorations": 5,
    "resto_max_iterations": 100,
    # Steady progress refills the restoration budget and 0 disables
    "restoration_budget_refresh_iters": 20,
    "resto_required_reduction": 0.5,
    # A small ||J^T c|| at a failed restoration signals local infeasibility
    "resto_infeas_tol": 1e-6,
    # Trigger after this many accepted steps below the alpha threshold
    "resto_trigger_alpha": 1e-3,
    "resto_trigger_iters": 5,
    # Interior push at entry, unpinning bound-stuck variables
    "resto_bound_push": 1e-2,
}

# Legacy options kept so old scripts still run with no effect
LEGACY = {
    "bound_push": 1e-2,
    "monotone_barrier_fraction": 0.1,
    "check_update_step": False,
    "record_components": [],
    "init_affine_step_multipliers": False,
    "use_armijo_line_search": True,
}

GROUPS = (
    CONVERGENCE,
    STARTING_POINT,
    BARRIER,
    LINEAR_SOLVER,
    LINE_SEARCH,
    RESTORATION,
    LEGACY,
)


def get_default_options(options={}):
    """Return the merged options dict with all defaults filled in."""
    default = {}
    for group in GROUPS:
        overlap = default.keys() & group.keys()
        if overlap:
            raise ValueError(f"Option defined in two groups: {sorted(overlap)}")
        default.update(copy.deepcopy(group))

    options = dict(options)

    # Legacy alias for the line-search selector
    if "filter_line_search" in options:
        flag = options.pop("filter_line_search")
        options.setdefault("line_search", "filter" if flag else "merit")

    # Legacy alias for the renamed warm start push
    if "warm_start_bound_push" in options:
        options.setdefault("bound_push", options.pop("warm_start_bound_push"))

    for name in options:
        if name in default:
            default[name] = options[name]
        else:
            raise ValueError(f"Unrecognized option {name}")

    return default
