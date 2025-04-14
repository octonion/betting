# --- Python Code Implementation ---
import cvxpy as cp
import numpy as np
import itertools
from datetime import datetime
import pytz # Optional, for timezone

# Function to get current time
def get_current_time_string():
    # ... (same as before) ...
    try:
        utc = pytz.utc
        now = datetime.now(utc)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except NameError:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


def optimize_simultaneous_bets(games, F_total=None, verbose=False, solver=cp.SCS):
    """
    Optimizes betting fractions across multiple simultaneous independent games
    using convex programming (CVXPY).

    Args:
        games (list): A list where each element is a dictionary representing a game:
                      {'p': [probabilities], 'o': [odds], 'x': [bonuses]}
        F_total (float, optional): The fixed total fraction to bet (0 < F_total <= 1).
                                   If None, optimizes globally to find the best F_total <= 1.
        verbose (bool): If True, prints solver output.
        solver (cp.Solver): The CVXPY solver to use (e.g., cp.SCS, cp.ECOS, cp.MOSEK).

    Returns:
        dict: A dictionary containing results:
              'status': Solver status.
              'optimal_F_total': The optimal total fraction bet (calculated or input).
              'max_log_growth': The maximum expected log growth (G_opt).
              'optimal_fractions': A list of numpy arrays, one per game, with optimal f_gi.
              'fractions_flat': The flat numpy array of all optimal fractions.
              'mode': 'Fixed F' or 'Global F <= 1'.
    """
    print(f"\n--- Running Optimization ({'Global F <= 1' if F_total is None else f'Fixed F = {F_total}'}) ---")
    start_time = datetime.now()

    num_games = len(games)
    if num_games == 0:
        print("Error: No games provided.")
        return None

    # --- Validate Inputs and Prepare Data ---
    game_outcomes = [] # Number of outcomes per game
    game_params = []   # List to store verified parameters
    total_vars = 0
    var_indices = []   # Store (start, end) index for each game in the flat variable vector
    prob_vectors = []
    odds_vectors = []
    bonus_vectors = []

    for i, g in enumerate(games):
        try:
            p = np.array(g['p'], dtype=float)
            o = np.array(g['o'], dtype=float)
            x = np.array(g['x'], dtype=float)
            n_out = len(p)

            if not (len(o) == n_out and len(x) == n_out):
                raise ValueError(f"Game {i}: Mismatched lengths of p, o, x.")
            if n_out == 0:
                raise ValueError(f"Game {i}: Must have at least one outcome.")
            if not np.isclose(np.sum(p), 1.0):
                print(f"Warning: Game {i} probabilities sum to {np.sum(p):.4f}. Normalizing.")
                p = p / np.sum(p)
            if np.any(p <= 0):
                raise ValueError(f"Game {i}: Probabilities must be positive.")
            if np.any(o <= 0):
                raise ValueError(f"Game {i}: Odds must be positive.")
            if np.any(x < 0):
                raise ValueError(f"Game {i}: Bonuses must be non-negative.")

            game_outcomes.append(n_out)
            game_params.append({'p': p, 'o': o, 'x': x})
            var_indices.append((total_vars, total_vars + n_out))
            total_vars += n_out
            prob_vectors.append(p)
            odds_vectors.append(o)
            bonus_vectors.append(x)

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing game {i}: {e}")
            return None

    # --- Check Scalability ---
    num_joint_outcomes = np.prod(game_outcomes)
    print(f"Number of games: {num_games}")
    print(f"Outcomes per game: {game_outcomes}")
    print(f"Total individual bet variables: {total_vars}")
    print(f"Total joint outcomes for objective: {num_joint_outcomes}")
    if num_joint_outcomes > 100000: # Threshold for warning
        print(f"Warning: High number of joint outcomes ({num_joint_outcomes}). Optimization might be slow or memory-intensive.")
    if num_joint_outcomes > 1000000:
        print("Error: Number of joint outcomes exceeds limit (1,000,000). Problem too large for this direct method.")
        return None


    # --- Define CVXPY Problem ---
    f = cp.Variable(total_vars, name="fractions", nonneg=True)

    # Constraints
    constraints = []
    mode = ""
    if F_total is None:
        # Global Optimization (Mode 1)
        constraints.append(cp.sum(f) <= 1.0)
        mode = "Global F <= 1"
    else:
        # Fixed F Optimization (Mode 2)
        if not (0 < F_total <= 1.0):
             print(f"Error: Fixed F_total={F_total} must be between 0 (exclusive) and 1 (inclusive).")
             return None
        constraints.append(cp.sum(f) == F_total)
        mode = f"Fixed F = {F_total}"

    # Objective Function Construction
    log_terms = []
    joint_outcome_indices = list(itertools.product(*[range(n) for n in game_outcomes]))

    # Pre-calculate probabilities for joint outcomes
    joint_probabilities = {}
    for joint_outcome in joint_outcome_indices:
        prob = np.prod([prob_vectors[g][joint_outcome[g]] for g in range(num_games)])
        joint_probabilities[joint_outcome] = prob

    # Build terms for the objective sum E[ln(Return)]
    # Return = 1 - sum(f) + sum_g (f_g,k_g * o_g,k_g + x_g,k_g)
    total_frac_sum = cp.sum(f)
    
    try:
        objective_terms = []
        for joint_outcome in joint_outcome_indices: # k_vec = (k_1, k_2, ...)
            prob = joint_probabilities[joint_outcome]
            
            # Calculate sum_g (f_g,k_g * o_g,k_g + x_g,k_g) for this joint outcome
            gain_sum_term = 0
            for g in range(num_games):
                k_g = joint_outcome[g] # Outcome index for game g
                start_idx, end_idx = var_indices[g]
                o_gk = odds_vectors[g][k_g]
                x_gk = bonus_vectors[g][k_g]
                # Select the correct variable f_g,k_g from the flat vector f
                f_gk = f[start_idx + k_g] 
                gain_sum_term += (f_gk * o_gk + x_gk)

            return_expression = 1.0 - total_frac_sum + gain_sum_term
            
            # Add term: p * log(return)
            # Need to handle log(non-positive) potentially, although constraints help
            # CVXPY's log is elementwise and handles domain.
            objective_terms.append(prob * cp.log(return_expression))
            
        # Define objective
        expected_log_growth = cp.sum(objective_terms)
        objective = cp.Maximize(expected_log_growth)

        # Define and Solve Problem
        problem = cp.Problem(objective, constraints)
        print(f"Solving optimization problem using {solver}...")
        # Note: MOSEK generally preferred for log/exp cones if available & licensed
        # SCS is a good open-source default. ECOS also works.
        problem.solve(solver=solver, verbose=verbose) 

    except cp.DCPError as e:
         print(f"CVXPY DCP Error: Problem is likely not formulated correctly as convex/concave. Error: {e}")
         return None
    except cp.SolverError as e:
         print(f"CVXPY Solver Error: Solver failed or encountered numerical issues. Error: {e}")
         return None
    except Exception as e:
         print(f"An unexpected error occurred during CVXPY problem setup or solve: {e}")
         return None


    # --- Process Results ---
    results = {'mode': mode}
    print(f"Solver Status: {problem.status}")
    results['status'] = problem.status

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        optimal_f = f.value
        # Clean near-zero values resulting from solver precision
        optimal_f[optimal_f < 1e-7] = 0.0

        # Calculate optimal F total from solution
        if F_total is None: # Global mode
            opt_F_calc = np.sum(optimal_f)
            results['optimal_F_total'] = opt_F_calc
        else: # Fixed F mode
            results['optimal_F_total'] = F_total
            # Check if sum matches requested F_total
            opt_F_calc = np.sum(optimal_f)
            if not np.isclose(opt_F_calc, F_total):
                 print(f"Warning: Solver optimal sum {opt_F_calc:.6f} differs from requested F_total={F_total}. May indicate inaccuracy.")
                 # Optionally rescale, but usually indicates solver issue if large difference
                 # If scaling:
                 # if opt_F_calc > 1e-9: optimal_f *= (F_total / opt_F_calc)

        results['max_log_growth'] = problem.value
        results['fractions_flat'] = optimal_f

        # Reshape fractions back into list per game
        optimal_fractions_list = []
        for g in range(num_games):
            start_idx, end_idx = var_indices[g]
            optimal_fractions_list.append(optimal_f[start_idx:end_idx])
        results['optimal_fractions'] = optimal_fractions_list

    else:
        print("Optimization did not converge to an optimal solution.")
        results['optimal_F_total'] = np.nan
        results['max_log_growth'] = np.nan
        results['optimal_fractions'] = None
        results['fractions_flat'] = None

    end_time = datetime.now()
    print(f"Optimization duration: {end_time - start_time}")
    return results

# --- Example Usage ---
if __name__ == "__main__":
    print(f"Script started at: {get_current_time_string()}")

    # Define 3 example games
    game1 = {'p': [0.6, 0.4],         'o': [1.8, 2.3],    'x': [0.0, 0.0]} # p*o = [1.08, 0.92]
    game2 = {'p': [0.5, 0.25, 0.25],  'o': [1.5, 3.75, 6.0], 'x': [0.05, 0.0, 0.10]} # p*o(raw) = [0.75, 0.9375, 1.5]
    game3 = {'p': [0.9, 0.1],         'o': [1.15, 9.5],   'x': [0.0, 0.02]} # p*o = [1.035, 0.95]

    all_games = [game1, game2, game3]

    # --- Run Mode 1: Global Optimization ---
    print("\n" + "="*60)
    print("Running Global Optimization (F_total <= 1)")
    print("="*60)
    global_results = optimize_simultaneous_bets(all_games, F_total=None, solver=cp.SCS) # Using SCS solver

    if global_results:
        print("\n--- Global Optimization Results ---")
        print(f"Status: {global_results['status']}")
        if global_results['optimal_fractions'] is not None:
            print(f"Optimal Total Fraction (F_opt): {global_results['optimal_F_total']:.6f}")
            print(f"Maximum Expected Log Growth (G_opt): {global_results['max_log_growth']:.6f}")
            print("Optimal Fractions per Game:")
            for i, f_g in enumerate(global_results['optimal_fractions']):
                f_str = ", ".join([f"{f:.4f}" for f in f_g])
                print(f"  Game {i}: [{f_str}]")
        else:
            print("  No optimal solution found.")

    # --- Run Mode 2: Fixed F Optimization ---
    fixed_F = 0.5 # Example fixed total fraction
    print("\n" + "="*60)
    print(f"Running Fixed F Optimization (F_total = {fixed_F})")
    print("="*60)
    fixed_results = optimize_simultaneous_bets(all_games, F_total=fixed_F, solver=cp.SCS)

    if fixed_results:
        print(f"\n--- Fixed F={fixed_F} Optimization Results ---")
        print(f"Status: {fixed_results['status']}")
        if fixed_results['optimal_fractions'] is not None:
            print(f"Specified Total Fraction (F_total): {fixed_results['optimal_F_total']:.6f}")
            print(f"Maximum Expected Log Growth (G): {fixed_results['max_log_growth']:.6f}")
            print("Optimal Fractions per Game:")
            for i, f_g in enumerate(fixed_results['optimal_fractions']):
                f_str = ", ".join([f"{f:.4f}" for f in f_g])
                print(f"  Game {i}: [{f_str}]")
        else:
            print("  No optimal solution found.")
