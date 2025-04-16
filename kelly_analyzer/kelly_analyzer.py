# kelly_analyzer.py
# Library for optimizing Kelly betting for single or multiple simultaneous games,
# with optional fixed bonuses. Uses KKT analytical solver for single, no-bonus
# cases (optional) and CVXPY for general cases (multi-game, bonuses, fixed F).

import numpy as np
import cvxpy as cp
import itertools
from datetime import datetime
import time
import warnings

try:
    # Optional, for timezone-aware timestamps in output
    import pytz
except ImportError:
    pytz = None

# --- Configuration ---
# Suppress common numerical warnings during calculations, handle outcomes explicitly
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
# Suppress CVXPY/SCS specific warnings if desired (can be noisy)
# warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')

DEFAULT_EPS = 1e-9 # Tolerance for floating point comparisons and zero checks
DEFAULT_CVXPY_SOLVER = cp.SCS # Default CVXPY solver if user doesn't specify

# --- Helper: Timestamp ---
def get_current_time_string():
    """Gets a formatted timestamp string for reporting (Eastern Time)."""
    try:
        if pytz:
            target_tz = pytz.timezone('America/New_York')
            now = datetime.now(target_tz)
            return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        else:
            # Fallback if pytz is not installed
            now = datetime.now()
            return f"{now.strftime('%Y-%m-%d %H:%M:%S')} (Local Time, pytz not installed)"
    except Exception as e:
        # Fallback on timezone error
        now = datetime.now()
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')} (Local Time - TZ Error: {e})"

# --- Solver Implementations ---

# Region: KKT Analytical Solver (Single Game, No Bonuses ONLY)
# Based on the iterative approach of finding exit points F_k and checking
# for an interior maximum using the lambda=1 condition. Suitable only for
# the specific case of one game without any fixed bonuses.

def _calculate_optimal_fractions_kkt(F, active_set_indices, p, o, eps=DEFAULT_EPS):
    """ (Internal KKT Helper) Calculates optimal fractions f_i for KKT method."""
    n_outcomes = len(p)
    f_vec = np.zeros(n_outcomes)
    F = max(0.0, F) # Ensure F is non-negative

    if not active_set_indices or F < eps: return f_vec

    active_set_indices = [i for i in active_set_indices if 0 <= i < n_outcomes and p[i] > eps]
    if not active_set_indices: return f_vec

    active_p = p[active_set_indices]; active_o = o[active_set_indices]

    # Check odds within the active set
    if np.any(active_o <= eps):
        # print(f"KKT Warning: Non-positive odds {active_o[active_o <= eps]} found in active set {active_set_indices}. Cannot calculate fractions.")
        return f_vec * np.nan # Indicate error state

    S_p = np.sum(active_p)
    # Handle S_o calculation carefully if odds are extremely large
    safe_active_o = np.maximum(active_o, eps) # Prevent division by zero
    S_o = np.sum(1.0 / safe_active_o)

    if S_p < eps: return f_vec * np.nan

    # Calculate 1/lambda (no bonus formula)
    one_div_lambda = (F + (1.0 - F) * S_o) / S_p

    for i in active_set_indices:
        if o[i] > eps:
            # Calculate f_i (no bonus formula)
            f_i = p[i] * one_div_lambda - (1.0 - F) / o[i]
            f_vec[i] = max(0.0, f_i) # Clamp f_i >= 0
        # else: f_vec[i] remains 0

    # Renormalize f_vec slightly if sum is not exactly F due to clamping/precision
    current_sum = np.sum(f_vec)
    if F > eps and current_sum > eps and not np.isclose(current_sum, F, atol=eps*10): # Slightly looser tolerance after clamping
        scale_factor = F / current_sum
        f_vec = f_vec * scale_factor
        f_vec[f_vec < eps] = 0.0 # Ensure non-negativity and cleanup near-zeros
        # Ensure sum is close to F after renormalization
        final_sum = np.sum(f_vec)
        if final_sum > eps and not np.isclose(final_sum, F, atol=eps*10):
             # If still significantly off, might indicate deeper issues, but try one more scaling
             f_vec = f_vec * (F / final_sum)
             f_vec[f_vec < eps] = 0.0

    return f_vec

def _calculate_log_growth_kkt(F, f_vec, p, o, eps=DEFAULT_EPS):
    """ (Internal KKT Helper) Calculates log growth G for KKT method."""
    F = max(0.0, F)
    if F < eps and np.allclose(f_vec, 0.0, atol=eps): return 0.0

    f_vec = np.asarray(f_vec)
    # Calculate returns R_k = 1 - F + f_k*o_k (no bonus formula)
    returns = 1.0 - F + f_vec * o

    non_positive_mask = returns <= eps
    # Check if any outcome with positive probability has non-positive return
    if np.any(non_positive_mask & (p > eps)):
        # print(f"KKT Warning: Non-positive return for F={F} with p>0. G is -inf.")
        return -np.inf

    valid_p_mask = p > eps
    log_returns = np.zeros_like(p, dtype=float)
    relevant_returns = returns[valid_p_mask]

    # Calculate log only for relevant returns, ensure they are positive
    log_returns[valid_p_mask] = np.log(np.maximum(relevant_returns, eps))

    G = np.sum(p[valid_p_mask] * log_returns[valid_p_mask])
    if abs(G) < eps: G = 0.0 # Treat near-zero growth as zero
    return G

def _solve_kkt_no_bonus(game_data, verbose=False, eps=DEFAULT_EPS):
    """
    (Internal) Solves GLOBAL optimum for a SINGLE game with NO BONUSES
    using the iterative KKT boundary evaluation approach.

    Args:
        game_data (dict): Processed game data {'p':..., 'o':...}.
        verbose (bool): Enables verbose output.
        eps (float): Tolerance.

    Returns:
        dict: Results including status, F_opt, G_opt, optimal_fractions.
    """
    overall_start_time = time.time()
    func_name = "_solve_kkt_no_bonus"
    if verbose: print(f"--- Running {func_name} ---")

    results = { "status": "Error_KKT_Initialization", "F_opt": np.nan, "G_opt": np.nan,
                "optimal_fractions": None, "lambda_at_opt": np.nan}

    try:
        p = game_data['p']; o = game_data['o']; n_outcomes = len(p)
        # Assume basic validation (p>0, o>0, sum(p)=1) done by caller
        results["optimal_fractions"] = [np.zeros(n_outcomes)] # Default

        # Pre-check: If no p*o > 1, optimal F is 0
        po_products = p * o
        if not np.any(po_products > 1.0 + eps):
            if verbose: print(f"{func_name}: No p*o > 1. Optimal F=0.")
            results.update({"status": "Optimal_F_Zero", "F_opt": 0.0, "G_opt": 0.0})
            return results

        initial_active_indices = np.where(p > eps)[0]
        active_set = set(initial_active_indices)
        critical_points_data = [] # Store tuples: (F_k, k_exiting, active_set_before)
        max_G_found = -np.inf
        best_candidate_details = None
        max_growth_found_interior = False
        F_high = 1.0

        iteration = 1
        while len(active_set) >= 1: # Need >=1 for S_p, S_o calc
            if verbose: print(f"\n KKT Iter {iteration}: ActiveSet={sorted(list(active_set))}, F_high={F_high:.6f}")
            current_active_indices = sorted(list(active_set))

            active_p = p[current_active_indices]; active_o = o[current_active_indices]
            if np.any(active_o <= eps): raise ValueError("Non-positive odds in active set") # Should be caught earlier
            S_p = np.sum(active_p); S_o = np.sum(1.0 / active_o)

            # --- Check for potential interior maximum (lambda=1) ---
            F_potential_opt = np.nan
            denominator_opt_check = 1.0 - S_o
            if np.isclose(denominator_opt_check, 0):
                if S_p > 1.0 + eps: raise ValueError(f"Unbounded growth detected (1-So=0, Sp={S_p}>1)")
                # Else: max is likely at F=1, handled by boundary check
            else:
                F_potential_opt = (S_p - S_o) / denominator_opt_check

            # --- Find next exit point F_k < F_high ---
            k_to_exit = -1 ; F_k = np.nan
            eligible_exits = {}
            for k in current_active_indices:
                pk, ok = p[k], o[k]
                numerator_Fk = S_p - pk * ok * S_o
                denominator_Fk = S_p + pk * ok * (1.0 - S_o)
                if np.isclose(denominator_Fk, 0): continue # F_k is infinite/undefined
                Fk_cand = numerator_Fk / denominator_Fk
                if np.isfinite(Fk_cand) and Fk_cand < F_high - eps:
                    eligible_exits[k] = Fk_cand

            if eligible_exits:
                k_to_exit = max(eligible_exits, key=eligible_exits.get)
                F_k = eligible_exits[k_to_exit]
                if verbose: print(f"  Next exit k={k_to_exit} at F_k={F_k:.6f}")
                critical_points_data.append((F_k, k_to_exit, current_active_indices))
            else:
                if verbose: print(f"  No further valid exit points found below F_high={F_high:.6f}")
                # Break after checking this region's potential max

            # --- Check if F_potential_opt is the Global Max ---
            if not np.isnan(F_potential_opt) and not max_growth_found_interior:
                F_k_lower_bound = F_k if k_to_exit != -1 else 0.0
                # Check if valid: (0 < F_opt <= 1) AND (F_k < F_opt < F_high)
                if F_k_lower_bound + eps < F_potential_opt < F_high - eps and eps < F_potential_opt <= 1.0 + eps:
                    F_opt = max(0.0, min(1.0, F_potential_opt))
                    if verbose: print(f"  Found valid interior max candidate F={F_opt:.6f}")
                    active_set_at_max = current_active_indices
                    f_opt_vec = _calculate_optimal_fractions_kkt(F_opt, active_set_at_max, p, o, eps)
                    G_opt = _calculate_log_growth_kkt(F_opt, f_opt_vec, p, o, eps)

                    best_candidate_details = {"F": F_opt, "G": G_opt, "f": f_opt_vec, "active_set": active_set_at_max, "source": "Interior (lambda=1)"}
                    max_growth_found_interior = True
                    if verbose: print(f"    --> Confirmed Global Max G={G_opt:.6f}")
                    break # Found global max

            # --- Prepare for next iteration ---
            if k_to_exit != -1:
                F_high = F_k
                active_set.remove(k_to_exit)
            else:
                break # No more exits to process
            iteration += 1

        # --- If no interior max, check boundaries (F=0, F=1) and critical points ---
        if not max_growth_found_interior:
            if verbose: print("\n KKT: No interior max found. Evaluating boundaries & critical points.")
            candidate_details = []
            # Add F=0
            candidate_details.append({"F": 0.0, "G": 0.0, "f": np.zeros(n_outcomes), "active_set": [], "source": "Boundary F=0.0"})
            # Add F=1
            f_at_F1 = _calculate_optimal_fractions_kkt(1.0, list(range(n_outcomes)), p, o, eps)
            G_at_F1 = _calculate_log_growth_kkt(1.0, f_at_F1, p, o, eps)
            candidate_details.append({"F": 1.0, "G": G_at_F1, "f": f_at_F1, "active_set": list(range(n_outcomes)), "source": "Boundary F=1.0"}) # Active set approx
            # Add critical points F_k
            critical_points_data.sort(key=lambda x: x[0], reverse=True)
            for Fk_val, k_exit, active_set_b4 in critical_points_data:
                 Fk_eval = max(0.0, min(1.0, Fk_val)) # Clamp F_k to [0,1] for evaluation
                 f_at_Fk = _calculate_optimal_fractions_kkt(Fk_eval, active_set_b4, p, o, eps)
                 G_at_Fk = _calculate_log_growth_kkt(Fk_eval, f_at_Fk, p, o, eps)
                 if verbose: print(f"  Eval Critical F_k={Fk_val:.6f} (using F={Fk_eval:.6f}): G={G_at_Fk:.6f}")
                 candidate_details.append({"F": Fk_eval, "G": G_at_Fk, "f": f_at_Fk, "active_set": active_set_b4, "source": f"Critical F_k={Fk_val:.4f}"})

            # Find best among candidates
            max_G_found = -np.inf
            best_candidate_details = candidate_details[0] # Default to F=0
            for cand in candidate_details:
                 # Allow slightly negative G if it's the max found (e.g. -1e-10)
                 if np.isfinite(cand['G']) and cand['G'] > max_G_found + eps:
                     max_G_found = cand['G']
                     best_candidate_details = cand
            if verbose: print(f" KKT: Best G after boundary check: {max_G_found:.6f} at F={best_candidate_details['F']:.6f}")

        # Final result processing
        if best_candidate_details and best_candidate_details['G'] > -eps: # Use -eps threshold
            results.update({
                "status": "Optimal",
                "F_opt": best_candidate_details['F'],
                "G_opt": best_candidate_details['G'],
                "optimal_fractions": [best_candidate_details['f']], # Return list for consistency
                # Lambda calculation can be added here if needed, similar to older versions
            })
        elif best_candidate_details and best_candidate_details['G'] <= -eps: # Max G is negative or -inf
             if verbose: print(f"KKT: Max G found ({best_candidate_details['G']:.6f}) is non-positive. Overriding to F=0.")
             results.update({ "status": "Optimal_F_Zero", "F_opt": 0.0, "G_opt": 0.0,
                              "optimal_fractions": [np.zeros(n_outcomes)] })
        else:
             # This case should ideally not be reached if F=0 is always a candidate
             results["status"] = "Error_KKT_No_Optimum_Found"

    except Exception as e:
        print(f"!!! Error during KKT solver: {e} !!!")
        import traceback
        traceback.print_exc()
        results["status"] = f"Error_KKT_{type(e).__name__}"
        # Ensure fractions list exists even on error
        if results["optimal_fractions"] is None:
             results["optimal_fractions"] = [np.zeros(len(game_data['p']))] if 'p' in game_data else None

    duration = time.time() - overall_start_time
    if verbose: print(f"{func_name} duration: {duration:.4f} seconds")
    return results

# EndRegion: KKT Solver

# Region: CVXPY Convex Optimization Solver
# Handles single or multiple games, with or without bonuses.
# Solves max E[log(Wealth)] = max sum(p_joint * log(1 - F_total + Sum_gains))
# Can operate in global (F<=1), fixed F (F==F_target), or fractional modes.

def _solve_cvxpy(games_data_list, F_total_constraint, solver_name, verbose=False, eps=DEFAULT_EPS):
    """
    (Internal) Solves Kelly problem using CVXPY. Handles single/multi-game,
    bonuses, and F constraints.

    Args:
        games_data_list (list): List of processed game dicts [{'p', 'o', 'x'}, ...].
        F_total_constraint (float or None): Target F if fixed/fractional, None for global.
        solver_name (str or cp.Solver): CVXPY solver to use.
        verbose (bool): Enables CVXPY verbose output.
        eps (float): Tolerance.

    Returns:
        dict: Results including status, F_total, G, fractions_list.
    """
    overall_start_time = time.time()
    func_name = "_solve_cvxpy"
    if verbose: print(f"--- Running {func_name} ---")

    results = { 'status': 'Error_CVXPY_Initialization', 'F_total': np.nan, 'G': np.nan,
                'fractions_list': None, 'fractions_flat': None }

    try:
        num_games = len(games_data_list)
        game_outcomes = [len(g['p']) for g in games_data_list]
        total_vars = sum(game_outcomes)

        # Check scalability based on joint outcomes
        try:
            num_joint_outcomes = np.prod(game_outcomes, dtype=np.int64)
            if verbose or num_joint_outcomes > 10000: # Be more verbose for larger problems
                 print(f"CVXPY: {num_games} games, {total_vars} variables, {num_joint_outcomes} joint outcomes.")
            if num_joint_outcomes > 1_000_000: print(f"Warning: High number of joint outcomes ({num_joint_outcomes}). CVXPY might be slow or run out of memory.")
            if num_joint_outcomes > 10_000_000: # Stricter limit
                print("Error: CVXPY - Number of joint outcomes exceeds safety limit (~10M).")
                results['status'] = 'Error_CVXPY_Too_Large'
                return results
        except OverflowError:
            print("Error: CVXPY - Number of joint outcomes causes overflow. Problem too large.")
            results['status'] = 'Error_CVXPY_Overflow'
            return results

        # Prepare data structures for CVXPY
        var_indices = [] ; prob_vectors = [] ; odds_vectors = [] ; bonus_vectors = []
        current_idx = 0
        for g in games_data_list:
            n_out = len(g['p'])
            var_indices.append((current_idx, current_idx + n_out))
            prob_vectors.append(g['p']); odds_vectors.append(g['o']); bonus_vectors.append(g['x'])
            current_idx += n_out

        # Define CVXPY variable for all fractions flattened
        f = cp.Variable(total_vars, name="fractions", nonneg=True)

        # Define constraints based on mode
        constraints = []
        is_global_mode = False
        if F_total_constraint is None: # Global mode: sum(f) <= 1
            constraints.append(cp.sum(f) <= 1.0)
            is_global_mode = True
            current_F = cp.sum(f) # F is a variable in the objective
        else: # Fixed/Fractional mode: sum(f) == F_target
            F_target = max(0.0, F_total_constraint) # Ensure target F is non-negative
            constraints.append(cp.sum(f) == F_target)
            current_F = F_target # F is a constant in the objective

        # --- Objective Function: Maximize E[log(Wealth)] ---
        joint_outcome_indices = list(itertools.product(*[range(n) for n in game_outcomes]))
        objective_terms = []

        for joint_outcome in joint_outcome_indices:
            prob = np.prod([prob_vectors[g][joint_outcome[g]] for g in range(num_games)])
            if prob < eps: continue

            gain_sum_term = 0
            for g in range(num_games):
                k_g = joint_outcome[g]
                start_idx, end_idx = var_indices[g]
                o_gk = odds_vectors[g][k_g]
                x_gk = bonus_vectors[g][k_g]
                f_gk = f[start_idx + k_g]
                gain_sum_term += (f_gk * o_gk + x_gk)

            return_expression = 1.0 - current_F + gain_sum_term

            # *** CORRECTED DCP APPROACH ***
            # Add constraint to ensure log argument is positive
            constraints.append(return_expression >= eps)
            # Apply log directly to the affine return expression
            objective_terms.append(prob * cp.log(return_expression))
            # *** END CORRECTION ***

        if not objective_terms:
             print("Error: CVXPY - No valid objective terms generated. Check input probabilities.")
             results['status'] = 'Error_CVXPY_NoObjective'
             return results

        expected_log_growth = cp.sum(objective_terms)
        objective = cp.Maximize(expected_log_growth)

        # --- Define and Solve Problem ---
        problem = cp.Problem(objective, constraints)
        # Determine solver name string correctly if it's a Solver object
        if isinstance(solver_name, str):
            solver_used_name = solver_name
        elif hasattr(solver_name, 'name'):
             solver_used_name = solver_name.name() # For solver objects like cp.SCS
        else:
             solver_used_name = repr(solver_name) # Fallback

        print(f" Solving CVXPY problem using {solver_used_name}...")

        solve_kwargs = {'solver': solver_name, 'verbose': verbose}
        # Add specific solver params if needed, e.g., for SCS:
        if solver_name == cp.SCS or solver_used_name == 'SCS': # Check name too
             solve_kwargs.update({'eps': 1e-6, 'max_iters': 5000}) # Example adjustments

        try:
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as e:
            print(f" Solver {solver_used_name} failed ({e}).")
            # Try fallback to SCS if a different solver was initially tried
            if solver_name != cp.SCS and solver_used_name != 'SCS':
                print(" Trying fallback solver SCS...")
                try:
                    problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6, max_iters=5000)
                    solver_used_name = 'SCS (Fallback)'
                except cp.error.SolverError as e_scs:
                     print(f" Fallback solver SCS also failed ({e_scs}).")
                     raise e_scs # Raise the SCS error
            else:
                 raise e # Re-raise original error if SCS was already tried

    except (cp.error.DCPError, cp.error.SolverError, ValueError, Exception) as e:
        print(f"!!! Error during CVXPY setup or solve: {e} !!!")
        import traceback
        traceback.print_exc()
        results['status'] = f'Error_CVXPY_{type(e).__name__}'
        return results # Return error status

    # --- Process Results ---
    results['status'] = problem.status
    print(f" CVXPY Solver Status: {problem.status}")

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        optimal_f = f.value if f.value is not None else np.zeros(total_vars)
        optimal_f[optimal_f < eps] = 0.0 # Clean up small values
        optimal_f = np.maximum(optimal_f, 0.0) # Ensure non-negativity

        opt_F_calc = np.sum(optimal_f)
        if not is_global_mode and not np.isclose(opt_F_calc, F_target, atol=eps*100):
              print(f" Warning: CVXPY optimal sum {opt_F_calc:.6f} differs significantly from requested F={F_target:.6f}. Scaling...")
              if opt_F_calc > eps:
                   optimal_f *= (F_target / opt_F_calc)
                   optimal_f[optimal_f < eps] = 0.0
                   opt_F_calc = np.sum(optimal_f)
              else:
                   optimal_f.fill(0.0)
                   opt_F_calc = 0.0

        max_G = problem.value if problem.value is not None else -np.inf

        if is_global_mode and max_G < eps:
            print(" CVXPY Solver: Max Log Growth is non-positive or zero. Optimal F=0.")
            results['status'] = 'Optimal_F_Zero'
            results['F_total'] = 0.0
            results['G'] = 0.0
            results['fractions_flat'] = np.zeros(total_vars)
        else:
            results['F_total'] = opt_F_calc if is_global_mode else F_target
            results['G'] = max_G
            results['fractions_flat'] = optimal_f

        optimal_fractions_list = []
        flat_f = results['fractions_flat']
        if flat_f is not None:
            for g in range(num_games):
                start_idx, end_idx = var_indices[g]
                optimal_fractions_list.append(flat_f[start_idx:end_idx])
        results['fractions_list'] = optimal_fractions_list

    else:
        print(" CVXPY Optimization did not find an optimal solution.")
        results['F_total'] = np.nan
        results['G'] = np.nan
        results['fractions_list'] = [np.zeros(len(g['p'])) for g in games_data_list]
        results['fractions_flat'] = np.zeros(total_vars)

    duration = time.time() - overall_start_time
    if verbose: print(f"{func_name} duration: {duration:.4f} seconds")
    results['solver_actually_used'] = solver_used_name
    return results

# EndRegion: CVXPY Solver

# --- Main Library Function ---
def optimize_kelly(games_data, mode='global', F_total=None, kelly_fraction=None,
                   solver=None, use_kkt=False, verbose=False, eps=DEFAULT_EPS):
    """
    Optimizes Kelly betting fractions for single or multiple simultaneous games,
    potentially including fixed bonuses per outcome.

    Selects between an analytical KKT-based solver (only for a single game
    with no bonuses, if requested via use_kkt=True) and a general convex
    optimization solver (CVXPY) for all other cases (multiple games, bonuses,
    fixed total fraction, or if KKT is not requested).

    Args:
        games_data (dict or list):
            A single game dictionary or a list of game dictionaries.
            Each dict requires:
                'p' (list/array): Probabilities for each outcome (must sum approx to 1).
                'o' (list/array): Decimal odds for each outcome (must be > 0).
            Optional keys per dict:
                'x' (list/array): Fixed bonus fraction received if outcome wins
                                  (defaults to 0 for all outcomes if missing).
        mode (str): Optimization mode:
            'global' (default): Find the global optimal total fraction F_opt <= 1
                                 and the corresponding allocation maximizing
                                 expected log growth (G).
            'fixed_f': Find the allocation maximizing G for a specific, user-provided
                       total fraction F_total (uses CVXPY).
            'fractional': Find the allocation for a total fraction F = k * F_opt,
                          where F_opt is the global optimum total fraction and k is
                          the kelly_fraction (requires 2 optimizations).
        F_total (float, optional):
            Required total fraction (0 < F <= 1) for mode 'fixed_f'. Ignored otherwise.
        kelly_fraction (float, optional):
            Required fraction k (0 < k <= 1) of global F_opt for mode 'fractional'.
            Ignored otherwise.
        solver (str or cp.Solver, optional):
            Solver to use for CVXPY optimizations (e.g., 'SCS', 'ECOS', cp.MOSEK).
            Defaults to cp.SCS. Ignored if KKT solver is used.
        use_kkt (bool, optional):
            If True, attempt to use the KKT analytical solver *only if* exactly
            one game is provided *and* it has no bonuses *and* mode is 'global'
            or 'fractional' (for finding F_opt). Defaults to False.
        verbose (bool, optional):
            If True, enables verbose output from internal solvers. Defaults to False.
        eps (float, optional):
            Tolerance for floating point comparisons. Defaults to DEFAULT_EPS.

    Returns:
        dict: A dictionary containing the optimization results:
            'status' (str): Solver status ('Optimal', 'Optimal_F_Zero', 'Error_...').
            'mode' (str): The optimization mode used.
            'solver_used' (str): Name of the solver ('KKT' or CVXPY solver name).
            'optimal_F_total' (float): Optimal total fraction F (global) or target F.
            'max_log_growth' (float): Max expected log growth (G) at optimal_F_total.
            'optimal_fractions' (list): List of np.arrays, one per game, with fractions.
            'lambda_at_opt' (float): Lambda value from KKT (NaN otherwise).
            'global_optimum_details' (dict): Results of global opt step (for fractional mode).
            'timestamp' (str): Start time of the optimization.
            'duration' (float): Total execution time in seconds.
            Returns None only on critical initial input errors.
    """
    start_time = time.time()
    timestamp = get_current_time_string()
    print(f"\n>>> Running Kelly Optimization <<<")
    print(f"    Mode: {mode}, Use KKT: {use_kkt}, Timestamp: {timestamp}")

    # --- Input Validation and Preparation ---
    if isinstance(games_data, dict):
        games_data_list = [games_data]
    elif isinstance(games_data, list):
        games_data_list = games_data
    else:
        print("Error: optimize_kelly - games_data must be a dict or a list of dicts.")
        return None

    num_games = len(games_data_list)
    if num_games == 0:
        print("Error: optimize_kelly - No games provided.")
        return None

    processed_games = []
    has_bonuses = False
    total_outcomes = 0
    for i, g in enumerate(games_data_list):
        if not isinstance(g, dict) or 'p' not in g or 'o' not in g:
            print(f"Error: optimize_kelly - Game {i} is not a dict or missing 'p'/'o' keys.")
            return None
        try:
            p = np.array(g['p'], dtype=float)
            o = np.array(g['o'], dtype=float)
            n_out = len(p)

            if n_out < 2:
                 raise ValueError(f"Game {i} must have at least 2 outcomes (has {n_out}).")

            x_in = g.get('x', None) # Check if 'x' exists
            if x_in is None:
                 x = np.zeros(n_out, dtype=float) # Default bonus to 0 if 'x' key is missing
            else:
                 x = np.array(x_in, dtype=float)

            if not (len(o) == n_out and len(x) == n_out):
                 raise ValueError(f"Game {i} has length mismatch: p({len(p)}), o({len(o)}), x({len(x)}).")

            # Validate probabilities
            if np.any(p < 0): raise ValueError(f"Game {i} probabilities must be non-negative.")
            prob_sum = np.sum(p)
            if prob_sum <= eps: raise ValueError(f"Game {i} sum of probabilities is near zero.")
            if not np.isclose(prob_sum, 1.0):
                 print(f"Warning: Game {i} probabilities sum to {prob_sum:.4f}. Normalizing.")
                 p = p / prob_sum

            # Validate odds and bonuses
            if np.any(o <= 0): raise ValueError(f"Game {i} odds must be positive.")
            if np.any(x < 0): raise ValueError(f"Game {i} bonuses must be non-negative.")
            if np.any(x > eps): has_bonuses = True # Check if any bonus > tolerance

            processed_games.append({'p': p, 'o': o, 'x': x})
            total_outcomes += n_out
        except ValueError as e:
            print(f"Error: optimize_kelly - Input validation failed for game {i}: {e}")
            return None
        except Exception as e_gen:
             print(f"Error: optimize_kelly - Unexpected error processing game {i}: {e_gen}")
             return None

    # --- Solver Selection Logic ---
    can_use_kkt = (num_games == 1) and (not has_bonuses) and use_kkt
    must_use_cvxpy = (num_games > 1) or has_bonuses

    solver_type = 'CVXPY' # Default
    if can_use_kkt and mode == 'fixed_f':
        print("Warning: KKT solver cannot be used for mode 'fixed_f'. Forcing CVXPY.")
        solver_type = 'CVXPY'
    elif can_use_kkt and mode in ['global', 'fractional']:
        solver_type = 'KKT'
    elif must_use_cvxpy and use_kkt:
        print("Warning: KKT solver requested but ineligible (multiple games or bonuses). Using CVXPY instead.")
        solver_type = 'CVXPY'
    # else: stays CVXPY

    cvxpy_solver_to_use = DEFAULT_CVXPY_SOLVER if solver is None else solver
    # Determine final solver name for reporting (handle solver objects)
    if solver_type == 'KKT':
        final_solver_name_report = 'KKT'
    elif isinstance(cvxpy_solver_to_use, str):
        final_solver_name_report = cvxpy_solver_to_use
    elif hasattr(cvxpy_solver_to_use, 'name'):
        final_solver_name_report = cvxpy_solver_to_use.name()
    else:
        final_solver_name_report = repr(cvxpy_solver_to_use)


    # --- Initialize Results ---
    results = {
        'status': 'Not Run', 'mode': mode, 'solver_used': final_solver_name_report,
        'optimal_F_total': np.nan, 'max_log_growth': np.nan,
        'optimal_fractions': None, # Will be list of arrays
        'lambda_at_opt': np.nan,   # Specific to KKT for now
        'global_optimum_details': None, # For fractional mode
        'timestamp': timestamp, 'duration': 0.0
    }
    # Pre-populate zero fractions based on input structure
    zero_fractions = [np.zeros(len(g['p'])) for g in processed_games]
    results['optimal_fractions'] = zero_fractions.copy()


    # --- Execute Optimization ---
    try:
        if mode == 'global':
            if solver_type == 'KKT':
                print(" Using KKT solver for global optimum...")
                kkt_result = _solve_kkt_no_bonus(processed_games[0], verbose=verbose, eps=eps)
                results['status'] = kkt_result['status']
                results['optimal_F_total'] = kkt_result['F_opt']
                results['max_log_growth'] = kkt_result['G_opt']
                results['optimal_fractions'] = kkt_result.get('optimal_fractions', zero_fractions.copy())
                results['lambda_at_opt'] = kkt_result['lambda_at_opt']
                results['solver_used'] = 'KKT'
            else:
                print(" Using CVXPY solver for global optimum (F<=1)...")
                cvx_result = _solve_cvxpy(processed_games, None, cvxpy_solver_to_use, verbose=verbose, eps=eps)
                results['status'] = cvx_result['status']
                results['optimal_F_total'] = cvx_result['F_total']
                results['max_log_growth'] = cvx_result['G']
                results['optimal_fractions'] = cvx_result['fractions_list']
                results['solver_used'] = cvx_result.get('solver_actually_used', final_solver_name_report)


        elif mode == 'fixed_f':
            if F_total is None or not (0 < F_total <= 1.0 + eps):
                 raise ValueError("Valid F_total (0 < F <= 1) required for fixed_f mode.")
            if solver_type == 'KKT':
                 raise RuntimeError("KKT solver selected for fixed_f mode - internal logic error.")

            print(f" Using CVXPY solver for fixed F = {F_total:.6f}...")
            cvx_result = _solve_cvxpy(processed_games, F_total, cvxpy_solver_to_use, verbose=verbose, eps=eps)
            results['status'] = cvx_result['status']
            results['optimal_F_total'] = F_total
            results['max_log_growth'] = cvx_result['G']
            results['optimal_fractions'] = cvx_result['fractions_list']
            results['solver_used'] = cvx_result.get('solver_actually_used', final_solver_name_report)


        elif mode == 'fractional':
            if kelly_fraction is None or not (0 < kelly_fraction <= 1.0):
                 raise ValueError("Valid kelly_fraction (0 < k <= 1) required for fractional mode.")

            # 1. Find Global F_opt
            print(f" Fractional Mode: Step 1 - Finding global optimum using {solver_type}...")
            global_opt_result = None
            global_solver_name_step1 = 'KKT'
            if solver_type == 'KKT':
                global_opt_result = _solve_kkt_no_bonus(processed_games[0], verbose=verbose, eps=eps)
                global_solver_name_step1 = 'KKT'
            else: # Use CVXPY for global step
                global_opt_result = _solve_cvxpy(processed_games, None, cvxpy_solver_to_use, verbose=verbose, eps=eps)
                global_solver_name_step1 = global_opt_result.get('solver_actually_used', final_solver_name_report)


            # Store global results using the keys returned by the respective solvers
            if solver_type == 'KKT':
                global_F_opt = global_opt_result.get('F_opt', np.nan)
                global_G_opt = global_opt_result.get('G_opt', np.nan)
            else: # CVXPY was used for global step
                global_F_opt = global_opt_result.get('F_total', np.nan)
                global_G_opt = global_opt_result.get('G', np.nan)

            global_fractions_raw = global_opt_result.get('optimal_fractions', None)
            if global_fractions_raw is None: global_fractions = zero_fractions.copy()
            else: global_fractions = global_fractions_raw
            global_status = global_opt_result.get('status', 'Error_Unknown') # Get the status string

            results['global_optimum_details'] = {
                'F_opt': global_F_opt, 'G_opt': global_G_opt,
                'fractions': global_fractions, 'status': global_status,
                'solver_used': global_solver_name_step1
            }
            print(f" Fractional Mode: Global F_opt = {global_F_opt:.6f}, Global G_opt = {global_G_opt:.6f} (Status: {global_status}, Solver: {global_solver_name_step1})")

            # *** CORRECTED STATUS CHECK HERE ***
            # 2. Check if global opt was successful AND F_opt > 0
            # Include lowercase status strings and check status is not None
            accepted_statuses = [
                'Optimal', 'optimal',
                'Optimal_Inaccurate', 'optimal_inaccurate',
                'Optimal_F_Zero' # Custom status for F=0 case
            ]
            if global_status is not None and global_status in accepted_statuses and np.isfinite(global_F_opt):
            # *** END CORRECTION ***
                if global_F_opt > eps:
                    F_target = kelly_fraction * global_F_opt
                    print(f" Fractional Mode: Step 2 - Target F = {kelly_fraction:.3f} * {global_F_opt:.4f} = {F_target:.6f}")

                    if F_target < eps :
                        print(" Fractional Mode: Target F is near zero. Setting result to F=0.")
                        results['status'] = 'Optimal_F_Zero'
                        results['optimal_F_total'] = 0.0
                        results['max_log_growth'] = 0.0
                        results['optimal_fractions'] = zero_fractions.copy()
                        results['solver_used'] = global_solver_name_step1
                    else:
                        # 3. Re-solve using CVXPY for fixed F_target
                        print(f" Fractional Mode: Step 3 - Re-solving using CVXPY for fixed F = {F_target:.6f}...")
                        cvx_frac_result = _solve_cvxpy(processed_games, F_target, cvxpy_solver_to_use, verbose=verbose, eps=eps)
                        results['status'] = cvx_frac_result['status']
                        results['optimal_F_total'] = F_target
                        results['max_log_growth'] = cvx_frac_result['G']
                        results['optimal_fractions'] = cvx_frac_result['fractions_list']
                        results['solver_used'] = cvx_frac_result.get('solver_actually_used', final_solver_name_report)
                else: # Global F_opt was <= eps (or Optimal_F_Zero status)
                     print(" Fractional Mode: Global optimum F is zero. Resulting fractional bet is zero.")
                     results['status'] = 'Optimal_F_Zero'
                     results['optimal_F_total'] = 0.0
                     results['max_log_growth'] = 0.0
                     results['optimal_fractions'] = zero_fractions.copy()
                     results['solver_used'] = global_solver_name_step1
            else: # Global optimum failed (status bad OR F_opt is NaN/inf OR status is None)
                 print(f" Fractional Mode: Failed to find global optimum (Status: {global_status}, F_opt finite: {np.isfinite(global_F_opt)}). Cannot calculate fractional bet.")
                 results['status'] = f'Error_Fractional_Global_Failed ({global_status})'
                 results['optimal_F_total'] = np.nan
                 results['max_log_growth'] = np.nan
                 results['optimal_fractions'] = zero_fractions.copy()
                 results['solver_used'] = global_solver_name_step1
        else:
            raise ValueError(f"Invalid mode '{mode}' specified.")

    except Exception as e:
        print(f"!!! An error occurred during top-level optimization: {e} !!!")
        import traceback
        traceback.print_exc()
        results['status'] = f'Error_{type(e).__name__}'
        results['optimal_fractions'] = results.get('optimal_fractions', zero_fractions.copy())

    results['duration'] = time.time() - start_time
    final_status = results.get('status', 'Error_Unknown')

    if final_status in ['Optimal', 'optimal', 'Optimal_inaccurate', 'optimal_inaccurate'] \
       and results.get('optimal_F_total', 0) > eps \
       and results.get('max_log_growth', 0) < -eps:
         print(f"Final Check: Status {final_status} but G ({results['max_log_growth']:.6f}) < 0 and F ({results['optimal_F_total']:.6f}) > 0. Setting status to Optimal_F_Zero.")
         results['status'] = 'Optimal_F_Zero'
         results['optimal_F_total'] = 0.0
         results['max_log_growth'] = 0.0
         results['optimal_fractions'] = zero_fractions.copy()
    elif np.isclose(results.get('optimal_F_total', np.nan), 0.0):
         if final_status not in ['Optimal_F_Zero', 'Error_Input_No_Positive_Prob']:
            # Don't overwrite specific F=0 statuses from KKT pre-check etc.
            if results.get('status') not in ['Optimal_F_Zero']:
                 print(f"Final Check: F_total is {results.get('optimal_F_total', np.nan):.6f}. Setting status to Optimal_F_Zero.")
                 results['status'] = 'Optimal_F_Zero'
         results['max_log_growth'] = 0.0
         results['optimal_fractions'] = zero_fractions.copy()

    print(f">>> Kelly Optimization Finished: Status = {results.get('status', 'Unknown')} <<<")
    return results

# --- Example Usage ---
if __name__ == "__main__":
    print(f"\n{'='*25} Kelly Analyzer Library Example Usage {'='*25}")
    print(f"    Timestamp: {get_current_time_string()}")
    print("-" * 70)

    # --- Example Game Definitions ---
    # Game 1: Simple Binary (No Bonus) - KKT eligible
    game1 = {'p': [0.6, 0.4], 'o': [1.8, 2.3]} # p*o = [1.08, 0.92] -> Bet on outcome 0
    # Game 2: 3-Outcome (With Bonus) - CVXPY only
    game2 = {'p': [0.5, 0.25, 0.25], 'o': [1.5, 3.75, 6.0], 'x': [0.05, 0.0, 0.10]}
    # Game 3: Skewed Binary (No Bonus) - KKT eligible
    game3 = {'p': [0.9, 0.1], 'o': [1.15, 9.5]} # p*o = [1.035, 0.95] -> Bet on outcome 0
    # Game 4: Suboptimal bet (No Bonus) - KKT eligible, expect F=0
    game4 = {'p': [0.5, 0.5], 'o': [1.9, 1.9]} # p*o = [0.95, 0.95] -> Expect F=0
    # Game 5: Three outcomes, No Bonus - KKT eligible
    game5 = {'p': [0.4, 0.35, 0.25], 'o': [2.6, 3.0, 4.5]}
    # Game 6: Four outcomes, with Bonus - CVXPY only
    game6 = {'p': [0.4, 0.3, 0.2, 0.1], 'o': [2.6, 3.5, 5.1, 11.0], 'x':[0, 0.01, 0, 0.02]}
    # Game 2: 3-Outcome (No Bonus) - KKT eligible
    game7 = {'p': [0.5, 0.25, 0.25], 'o': [1.5, 3.75, 6.0]}
    # Games for Multi-Game Tests (up to 6)
    g_multi1 = {'p': [0.55, 0.45], 'o': [1.9, 2.1]} # Simple near coin flip
    g_multi2 = {'p': [0.7, 0.3], 'o': [1.5, 3.5]}   # Favored bet
    g_multi3 = {'p': [0.3, 0.7], 'o': [4.0, 1.4], 'x': [0.01, 0]} # Underdog + bonus
    g_multi4 = {'p': [0.25, 0.75], 'o': [5.0, 1.3]} # Strong favorite
    g_multi5 = {'p': [0.6, 0.4], 'o': [1.7, 2.4]}   # Similar to game1
    g_multi6 = {'p': [0.5, 0.5], 'o': [2.0, 2.0]}   # Fair coin flip

    # Helper to display results consistently
    def print_results(test_name, results_dict):
        print(f"\n--- {test_name} Results ---")
        if not results_dict: print(" Error: No results returned."); return
        status = results_dict.get('status', 'N/A')
        solver = results_dict.get('solver_used', 'N/A')
        mode = results_dict.get('mode', 'N/A')
        print(f" Status: {status}, Solver: {solver}, Mode: {mode}")
        if status not in ['Not Run'] and 'optimal_F_total' in results_dict:
            F_opt = results_dict['optimal_F_total']
            G_opt = results_dict['max_log_growth']
            print(f" Optimal Total F: {F_opt:.6f}" if np.isfinite(F_opt) else " Optimal Total F: N/A")
            print(f" Max Log Growth (G): {G_opt:.6f}" if np.isfinite(G_opt) else " Max Log Growth (G): N/A")

            if results_dict.get('optimal_fractions') is not None:
                 print(" Optimal Fractions per Game:")
                 for i, f_g in enumerate(results_dict['optimal_fractions']):
                     if f_g is not None:
                          f_str = ", ".join([f"{f:.4f}" for f in f_g])
                          print(f"   Game {i}: [{f_str}] (Sum: {np.sum(f_g):.4f})")
                     else:
                          print(f"   Game {i}: N/A")
            else:
                 print(" Optimal Fractions: N/A")

            if results_dict.get('lambda_at_opt') is not None and np.isfinite(results_dict['lambda_at_opt']):
                 print(f" Lambda at Opt (KKT): {results_dict['lambda_at_opt']:.6f}")

            if results_dict.get('global_optimum_details'):
                 glob_dets = results_dict['global_optimum_details']
                 print(" Fractional Mode Base Global Opt:")
                 print("   F_opt={:.4f}, G_opt={:.4f}, Status={}, Solver={}".format(
                     glob_dets.get('F_opt',np.nan), glob_dets.get('G_opt',np.nan),
                     glob_dets.get('status','N/A'), glob_dets.get('solver_used','N/A')))
        else:
             print(" Optimization did not run successfully or results format error.")
        print(f" Duration: {results_dict.get('duration', 0):.4f} seconds")
        print("-" * (len(test_name) + 16))

    # --- Test Cases ---
    print("\n" + "="*20 + " SINGLE GAME TESTS " + "="*20)

    # Test 1: Single Game 1 (No Bonus), Global Opt, KKT vs CVXPY
    print("\n--- Test 1: Single Game 1 (No Bonus), Global ---")
    res_1a = optimize_kelly(game1, mode='global', use_kkt=True, verbose=False)
    print_results("Test 1a: KKT", res_1a)
    res_1b = optimize_kelly(game1, mode='global', use_kkt=False, solver='SCS') # Default solver
    print_results("Test 1b: CVXPY(SCS)", res_1b)
    try: # Test with ECOS, might not be installed
         if 'ECOS' in cp.installed_solvers():
              res_1c = optimize_kelly(game1, mode='global', use_kkt=False, solver='ECOS')
              print_results("Test 1c: CVXPY(ECOS)", res_1c)
    except Exception as e: print(f"Could not run ECOS test: {e}")


    # Test 2: Single Game 1 (No Bonus), Fractional k=0.5
    print("\n--- Test 2: Single Game 1 (No Bonus), Fractional k=0.5 ---")
    res_2a = optimize_kelly(game1, mode='fractional', kelly_fraction=0.5, use_kkt=True)
    print_results("Test 2a: KKT (Global) + CVXPY (Fixed)", res_2a)
    res_2b = optimize_kelly(game1, mode='fractional', kelly_fraction=0.5, use_kkt=False)
    print_results("Test 2b: CVXPY Only", res_2b)

    # Test 3: Single Game 1 (No Bonus), Fixed F=0.05 (Must use CVXPY)
    print("\n--- Test 3: Single Game 1 (No Bonus), Fixed F=0.05 ---")
    res_3 = optimize_kelly(game1, mode='fixed_f', F_total=0.05)
    print_results("Test 3: CVXPY", res_3)

    # Test 4: Single Game 2 (With Bonus), Global (Must use CVXPY, KKT ignored)
    print("\n--- Test 4: Single Game 2 (With Bonus), Global ---")
    res_4a = optimize_kelly(game2, mode='global', use_kkt=True) # Should ignore use_kkt
    print_results("Test 4a: CVXPY (KKT Ignored)", res_4a)
    res_4b = optimize_kelly(game2, mode='global', use_kkt=False)
    print_results("Test 4b: CVXPY (KKT False)", res_4b)


    # Test 5: Single Suboptimal Game (Game 4), Global (KKT vs CVXPY, expect F=0)
    print("\n--- Test 5: Single Game 4 (Suboptimal), Global ---")
    res_5a = optimize_kelly(game4, mode='global', use_kkt=True)
    print_results("Test 5a: KKT", res_5a)
    res_5b = optimize_kelly(game4, mode='global', use_kkt=False)
    print_results("Test 5b: CVXPY", res_5b)

    # Test 6: Single Game 5 (No Bonus, 3 outcomes), KKT vs CVXPY
    print("\n--- Test 6: Single Game 5 (No Bonus, 3 outcomes), Global ---")
    res_6a = optimize_kelly(game5, mode='global', use_kkt=True, verbose=False)
    print_results("Test 6a: KKT", res_6a)
    res_6b = optimize_kelly(game5, mode='global', use_kkt=False)
    print_results("Test 6b: CVXPY", res_6b)

    # Test 7: Single Game 6 (No Bonus), Global Opt, KKT vs CVXPY
    print("\n--- Test 7': Single Game 5 (No Bonus), Global ---")
    res_7a = optimize_kelly(game7, mode='global', use_kkt=True, verbose=False)
    print_results("Test 1a: KKT", res_7a)
    res_7b = optimize_kelly(game7, mode='global', use_kkt=False, solver='SCS')
    print_results("Test 1b: CVXPY(SCS)", res_7b)
    try: # Test with CLARABEL, might not be installed
         if 'CLARABEL' in cp.installed_solvers():
              res_7c = optimize_kelly(game7, mode='global', use_kkt=False, solver='CLARABEL')
              print_results("Test 7c: CVXPY(CLARABEL)", res_7c)
    except Exception as e: print(f"Could not run CLARABEL test: {e}")

    print("\n" + "="*20 + " MULTI-GAME TESTS (CVXPY ONLY) " + "="*20)

    # Test 7: Multi-Game (g_multi1, g_multi2 - no bonus), Global
    print("\n--- Test 7: Multi-Game (2 Games, No Bonus), Global ---")
    res_7 = optimize_kelly([g_multi1, g_multi2], mode='global')
    print_results("Test 7: CVXPY", res_7)

    # Test 8: Multi-Game (g_multi1, g_multi3 - bonus), Global
    print("\n--- Test 8: Multi-Game (2 Games, With Bonus), Global ---")
    res_8 = optimize_kelly([g_multi1, g_multi3], mode='global')
    print_results("Test 8: CVXPY", res_8)

    # Test 9: Multi-Game (3 Games: g_multi1, g_multi2, g_multi4), Global
    print("\n--- Test 9: Multi-Game (3 Games, No Bonus), Global ---")
    res_9 = optimize_kelly([g_multi1, g_multi2, g_multi4], mode='global')
    print_results("Test 9: CVXPY", res_9)

    # Test 10: Multi-Game (3 Games: g_multi1, g_multi3, g_multi4 - bonus), Fractional k=0.7
    print("\n--- Test 10: Multi-Game (3 Games, With Bonus), Fractional k=0.7 ---")
    res_10 = optimize_kelly([g_multi1, g_multi3, g_multi4], mode='fractional', kelly_fraction=0.7)
    print_results("Test 10: CVXPY", res_10)

    # Test 11: Multi-Game (4 Games: g_multi1-4), Fixed F=0.4
    print("\n--- Test 11: Multi-Game (4 Games, With Bonus), Fixed F=0.4 ---")
    res_11 = optimize_kelly([g_multi1, g_multi2, g_multi3, g_multi4], mode='fixed_f', F_total=0.4)
    print_results("Test 11: CVXPY", res_11)

    # Test 12: Multi-Game (6 Games: g_multi1-6), Global
    print("\n--- Test 12: Multi-Game (6 Games, With Bonus), Global ---")
    res_12 = optimize_kelly([g_multi1, g_multi2, g_multi3, g_multi4, g_multi5, g_multi6], mode='global', verbose=False)
    print_results("Test 12: CVXPY", res_12)


    print(f"\n{'='*25} End Example Usage {'='*25}")

    # Restore warning settings if needed elsewhere in a larger application
    warnings.resetwarnings()
    # Re-apply the specific ignores if you run more code after this block
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
