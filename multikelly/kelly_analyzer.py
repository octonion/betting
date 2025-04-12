import numpy as np
import warnings
from datetime import datetime
import pytz # Required for timezone handling

# Suppress RuntimeWarning for log(0) temporarily, handle it explicitly
warnings.filterwarnings('ignore', r'divide by zero encountered in log')
warnings.filterwarnings('ignore', r'invalid value encountered in log')
warnings.filterwarnings('ignore', r'divide by zero encountered in scalar divide') # For S_o=1 case
warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')

def calculate_optimal_fractions(F, active_set_indices, p, o):
    """ Calculates the optimal fractions f_i for a given F and active set """
    n_outcomes = len(p)
    f_vec = np.zeros(n_outcomes)

    if not active_set_indices or F <= 1e-12: # Handle empty set or F near zero
        return f_vec

    # Ensure indices are valid
    active_set_indices = [i for i in active_set_indices if i < n_outcomes]
    if not active_set_indices:
        return f_vec

    active_p = p[active_set_indices]
    active_o = o[active_set_indices]

    # Handle cases where active odds might be <= 0 if input validation allows
    if np.any(active_o <= 1e-12):
         print(f"Warning: Non-positive odds found in active set {active_set_indices} for F={F}. Cannot calculate fractions.")
         return f_vec # Or handle differently

    S_p = np.sum(active_p)
    S_o = np.sum(1.0 / active_o)

    if np.isclose(S_p, 0):
        # This shouldn't happen with p_i > 0 and non-empty active set
        return f_vec

    # Calculate 1/lambda
    # Avoid division by zero if S_p is somehow zero
    one_div_lambda = (F + (1.0 - F) * S_o) / S_p if S_p > 1e-12 else 0

    for idx, i in enumerate(active_set_indices):
        # Calculate f_i = p_i / lambda - (1 - F) / o_i
        # Ensure o[i] is not zero before division
        if o[i] > 1e-12:
            f_i = p[i] * one_div_lambda - (1.0 - F) / o[i]
            # Clamp f_i >= 0 due to potential floating point inaccuracies near exit point
            f_vec[i] = max(0.0, f_i)
        else:
             f_vec[i] = 0 # Cannot bet if odds are zero or less

    # Renormalize f_vec slightly if sum is not exactly F due to clamping/precision
    current_sum = np.sum(f_vec)
    # Only renormalize if F is positive and current_sum is positive and differs significantly from F
    if F > 1e-12 and current_sum > 1e-12 and not np.isclose(current_sum, F):
         f_vec = f_vec * (F / current_sum)
         # Ensure non-negativity after potential scaling
         f_vec[f_vec < 0] = 0
         # Check sum again after adjustment - minor deviations might remain
         final_sum = np.sum(f_vec)
         # if not np.isclose(final_sum, F):
         #     print(f"Debug: Post-normalization sum {final_sum} still differs from F={F}")


    return f_vec

def calculate_log_growth(F, f_vec, p, o):
    """ Calculates the expected log growth G for given F and fractions f_vec """
    n_outcomes = len(p)
    # Check if sum of calculated fractions exceeds F significantly
    if np.sum(f_vec) > F + 1e-9:
         # This warning might indicate issues in fraction calculation/normalization
         # print(f"Warning: Sum of fractions {np.sum(f_vec)} exceeds F={F} in G calculation.")
         pass # Proceed cautiously

    # Calculate return factor R_k for each outcome k
    # R_k = 1 - sum(f) + f_k*o_k = 1 - F + f_k*o_k (since sum f_i = F)
    returns = 1.0 - F + f_vec * o

    # Handle cases where return <= 0 (shouldn't happen with optimal fractions in valid range)
    if np.any(returns <= 1e-12):
        # Find where it's non-positive
        bad_indices = np.where(returns <= 1e-12)[0]
        # print(f"Warning: Non-positive return calculated for F={F:.4f}. Fractions: {[f'{x:.4f}' for x in f_vec]}. Returns: {[f'{x:.4f}' for x in returns]}. Indices: {bad_indices}")
        return -np.inf # Indicate invalid growth

    # Calculate log safely
    # Use np.logaddexp for stability? No, just log directly.
    # Add a small epsilon to avoid log(0) strictly, although handled above
    log_returns = np.log(np.maximum(returns, 1e-12)) # Ensure argument is positive

    G = np.sum(p * log_returns)
    return G

def analyze_kelly_regions(probabilities, odds):
    """
    Calculates Kelly exit fractions, optimal fractions at exits, log growth,
    and identifies the overall global maximum growth point using the lambda=1 condition.

    Args:
        probabilities (list or np.array): List of probabilities for each outcome.
        odds (list or np.array): List of decimal odds for each outcome.

    Returns:
        dict: A dictionary containing analysis results.
    """
    n_outcomes = len(probabilities)
    analysis = {
        "critical_points": [],
        "global_max_point": None # Will store the single global max details
    }
    if n_outcomes != len(odds) or n_outcomes == 0:
        print("Error: Probabilities and odds must be non-empty lists of the same length.")
        return analysis # Return empty analysis

    # Use numpy arrays
    p = np.array(probabilities, dtype=float)
    o = np.array(odds, dtype=float)

    # --- Basic input validation ---
    if not np.isclose(np.sum(p), 1.0):
         print(f"Warning: Probabilities do not sum to 1 (sum={np.sum(p):.4f}).")
    if np.any(p <= 0):
         print("Error: Probabilities must be positive.")
         return analysis
    if np.any(o <= 0): # Check for non-positive odds more strictly
         print("Error: Odds must be positive.")
         return analysis
    if np.any(o <= 1):
         print("Warning: Some odds are <= 1. Corresponding outcomes might exit immediately.")


    # --- Algorithm Implementation ---
    active_set = set(range(n_outcomes))
    po_products = p * o

    critical_points_data = [] # Store tuples: (F_k, k_exiting, active_set_before)

    # Variables to track global maximum found so far
    global_max_details = {"G_opt": -np.inf, "F_opt": np.nan, "source": "Initialization"}
    max_growth_found_interior = False # Flag if we found the max via lambda=1 condition

    F_high = 1.0 # Upper bound for F in the current region being checked

    print("\n--- Iteratively Finding Exit Points and Checking Regions for Max Growth ---")
    iteration = 1
    while len(active_set) > 1:
        print(f"\nIteration {iteration}: Analyzing region F > F_k (where F_k is exit point of outcome below)")
        current_active_indices = sorted(list(active_set))
        print(f"  Current Active Set I = {current_active_indices}")

        current_po = {i: po_products[i] for i in active_set}

        # Find outcome k with the minimum p*o product
        min_po = float('inf')
        k_to_exit = -1
        for i in current_active_indices: # Use sorted list for determinism
            if current_po[i] < min_po:
                min_po = current_po[i]
                k_to_exit = i

        if k_to_exit == -1: break # Should not happen

        print(f"  Min p*o = {min_po:.4f} found for outcome k = {k_to_exit} (Next to exit)")
        active_indices_before_exit = current_active_indices # Store before modification

        # Calculate S_p and S_o for the current active_set I
        if not active_indices_before_exit: continue # Should not happen here
        active_p = p[active_indices_before_exit]
        active_o = o[active_indices_before_exit]
        if np.any(active_o <= 1e-12):
            print(f"Error: Non-positive odds in active set {active_indices_before_exit}. Aborting iteration.")
            break
        S_p = np.sum(active_p)
        S_o = np.sum(1.0 / active_o)
        print(f"  S_p = {S_p:.6f}, S_o = {S_o:.6f}")

        # --- Check for potential interior maximum using lambda=1 condition ---
        # This condition F = (S_p - S_o) / (1 - S_o) corresponds to lambda=1
        F_potential_opt = np.nan
        denominator_opt_check = 1.0 - S_o
        if np.isclose(denominator_opt_check, 0):
             print("  Region Check: Cannot calculate potential interior max F (1 - S_o is zero).")
        else:
             F_potential_opt = (S_p - S_o) / denominator_opt_check
             print(f"  Region Check: Potential Max F from lambda=1 condition = {F_potential_opt:.6f}")


        # Calculate F_k (exit point for k_to_exit) which defines the LOWER bound for the region where this set I is active
        pk = p[k_to_exit]
        ok = o[k_to_exit]
        numerator_Fk = S_p - pk * ok * S_o
        denominator_Fk = S_p + pk * ok * (1.0 - S_o)
        F_k = np.nan # Default
        if np.isclose(denominator_Fk, 0):
            print(f"  Warning: Denominator zero for F_{k_to_exit} calculation.")
            # Interpretation depends: If Num > 0, F_k -> +inf; If Num < 0, F_k -> -inf.
            # If Num = 0, indeterminate. Assume it doesn't provide a finite boundary.
        elif denominator_Fk < 0:
             # Usually implies F_k > 1 or F_k < 0
             F_k = numerator_Fk / denominator_Fk
             print(f"  Warning: Denominator negative for F_{k_to_exit} calc (F_k = {F_k:.4f}).")
        else:
            F_k = numerator_Fk / denominator_Fk
            
        print(f"  Calculated exit threshold F_{k_to_exit} = {F_k:.6f} (Lower bound for current set's validity)")
        # Use max(0, F_k) for range checks as negative F is not relevant
        F_k_check_bound = max(0.0, F_k) if not np.isnan(F_k) else -np.inf # Use -inf if calc failed

        # Now, check if F_potential_opt falls strictly within (F_k_check_bound, F_high)
        if not np.isnan(F_potential_opt) and not np.isnan(F_k_check_bound) and not max_growth_found_interior:
            eps = 1e-9 # Tolerance for strict inequality
            # Check if F_potential_opt is within the meaningful range (0, 1] and within the region boundaries
            if F_k_check_bound + eps < F_potential_opt < F_high - eps and F_potential_opt > eps:
                 print(f"  !!! Found Valid Interior Max Candidate at F_opt = {F_potential_opt:.6f} in region ({F_k_check_bound:.4f}, {F_high:.4f}) !!!")
                 F_opt = F_potential_opt
                 active_set_at_max = active_indices_before_exit
                 f_opt_vec = calculate_optimal_fractions(F_opt, active_set_at_max, p, o)
                 G_opt = calculate_log_growth(F_opt, f_opt_vec, p, o)

                 # Since G is concave, this must be the global maximum if found interiorly
                 global_max_details = {
                     "F_opt": F_opt, "G_opt": G_opt,
                     "active_set": active_set_at_max, "fractions": f_opt_vec,
                     "source": f"Interior optimum based on lambda=1 in region F > {F_k:.4f}"
                 }
                 max_growth_found_interior = True # Stop checking in subsequent (lower F) regions
                 print(f"      --> Confirmed Global Max G = {G_opt:.6f}")
            else:
                 print(f"  Region Check: Potential Max F is outside the valid range ({F_k_check_bound:.4f}, {F_high:.4f}).")

        # Store critical point info AFTER checking the region
        if not np.isnan(F_k):
             critical_points_data.append((F_k, k_to_exit, active_indices_before_exit))
        else:
             print(f"  Skipping storing critical point for k={k_to_exit} due to failed F_k calculation.")

        # Update F_high (upper bound for the *next* region check)
        F_high = F_k_check_bound
        if np.isnan(F_high): F_high = 0 # Stop range checks if F_k calculation failed

        # Remove k from the active set
        active_set.remove(k_to_exit)
        iteration += 1

    # --- After Loop ---
    print("\n--- Finished Iterations ---")

    # --- Process critical points (calculate f_i and G at F_k) ---
    analysis["critical_points"] = []
    # Sort points by F_k descending (order of exit)
    critical_points_data.sort(key=lambda x: x[0], reverse=True)

    for F_k_val, k_exiting_val, active_set_before_val in critical_points_data:
         if np.isnan(F_k_val): continue # Skip points where F_k calculation failed

         # Ensure F_k is not negative for fraction calculation
         F_k_calc = max(0.0, F_k_val)

         f_at_Fk = calculate_optimal_fractions(F_k_calc, active_set_before_val, p, o)
         # Ensure f_k is zero at its exit point, handle potential small negative from precision
         f_at_Fk[k_exiting_val] = 0.0

         # Readjust sum if needed after forcing f_k=0, only if F_k_calc > 0
         current_sum = np.sum(f_at_Fk)
         if F_k_calc > 1e-12 and not np.isclose(current_sum, F_k_calc):
             if current_sum > 1e-12: # Avoid division by zero
                 f_at_Fk *= (F_k_calc / current_sum)
                 f_at_Fk[f_at_Fk < 0] = 0 # Ensure non-negativity
             else: # If sum became zero unexpectedly, set all to zero
                  f_at_Fk.fill(0.0)


         G_at_Fk = calculate_log_growth(F_k_calc, f_at_Fk, p, o)

         analysis["critical_points"].append({
             "F_k": F_k_val, # Store original calculated F_k
             "k_exiting": k_exiting_val,
             "active_set_before": active_set_before_val,
             "fractions_at_Fk": f_at_Fk, # Fractions corresponding to F_k_calc
             "G_at_Fk": G_at_Fk
         })


    # --- Determine and finalize Global Maximum ---
    if not max_growth_found_interior:
        print("\n--- No interior max found via lambda=1 condition. Evaluating boundary F=1.0 ---")
        F_opt = 1.0
        active_set_at_max = list(range(n_outcomes))
        # Calculate fractions at F=1 using the general function for consistency
        # Although theoretically f_i(1) = p_i
        f_opt_vec = calculate_optimal_fractions(F_opt, active_set_at_max, p, o)
        # Verify f_i(1) is close to p_i
        if not np.allclose(f_opt_vec, p, atol=1e-4):
             print(f"Warning: Calculated f(1) {f_opt_vec} differs from theoretical p {p}. Using calculated.")
             
        G_opt = calculate_log_growth(F_opt, f_opt_vec, p, o)

        # Check if F=1 is indeed better than G at the highest finite F_k found (if any)
        highest_finite_Fk_G = -np.inf
        if analysis["critical_points"]:
             # Find the first point with finite G
             for cp in analysis["critical_points"]:
                 if np.isfinite(cp["G_at_Fk"]):
                     highest_finite_Fk_G = cp["G_at_Fk"]
                     break

        if G_opt >= highest_finite_Fk_G:
             global_max_details = {
                 "F_opt": F_opt, "G_opt": G_opt,
                 "active_set": active_set_at_max, "fractions": f_opt_vec,
                 "source": "Boundary F=1.0"
             }
             print(f"  Global Max confirmed at F=1.0 with G={G_opt:.6f}")
        else:
             # This case implies G peaked *exactly* at the highest F_k, which is less common
             print(f"  Warning: G(1.0)={G_opt:.6f} is lower than G={highest_finite_Fk_G:.6f} at the highest exit point. Max might be at that F_k.")
             # Find the details for that point
             max_cp = analysis["critical_points"][0] # Assumes sorted desc
             global_max_details = {
                 "F_opt": max(0.0, max_cp["F_k"]), # Use non-negative F_k
                 "G_opt": max_cp["G_at_Fk"],
                 "active_set": max_cp["active_set_before"],
                 "fractions": max_cp["fractions_at_Fk"],
                 "source": f"Boundary at highest exit point F_k={max_cp['F_k']:.6f}"
             }

    analysis["global_max_point"] = global_max_details

    return analysis

# --- Get Current Time for Reporting ---
def get_current_time_string():
    try:
        # Use a known timezone, e.g., EST/EDT
        eastern = pytz.timezone('America/New_York')
        now = datetime.now(eastern)
        # Format with timezone offset
        return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except Exception as e:
        print(f"Timezone calculation error: {e}")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Fallback to local time

# --- Example Usage ---
# Example 1 (from previous interaction)
# probabilities_ex = [0.4, 0.3, 0.2, 0.1]
# odds_ex = [2.6, 3.5, 5.1, 11.0] # Max at F=1

# Example 2 (from previous interaction)
#probabilities_ex = [0.5, 0.25, 0.25]
#odds_ex = [1.5, 3.75, 6.0] # Interior Max Expected

# Example 3
probabilities_ex = [0.25, 0.1, 0.1, 0.4, 0.15]
odds_ex = [1/0.2, 1/0.06667059, 1/0.04, 1/0.4, 1/0.46976471]

print(f"Running Kelly Analysis at: {get_current_time_string()}")
print("\nAnalyzing Kelly Regions for the example:")
print(f"Probabilities: {probabilities_ex}")
print(f"Odds: {odds_ex}")
print("-" * 50)

analysis_results = analyze_kelly_regions(probabilities_ex, odds_ex)

print("\n\n--- FINAL ANALYSIS RESULTS ---")

print("\n1. Critical Points (Order of Exit):")
if analysis_results["critical_points"]:
     # Sort by F_k descending for clarity of exit order
     sorted_cps = sorted(analysis_results["critical_points"], key=lambda x: x['F_k'], reverse=True)
     print("-" * 80)
     print(f"{'Exiting k':<10} | {'Exit F_k':<12} | {'G at F_k':<12} | {'Active Set Before':<20} | {'Fractions f_i at F_k'}")
     print("-" * 80)
     for cp in sorted_cps:
         f_str = ", ".join([f"{f:.4f}" for f in cp['fractions_at_Fk']])
         # Use max(0, F_k) for display consistency if F_k was negative
         display_Fk = max(0.0, cp['F_k']) if not np.isnan(cp['F_k']) else cp['F_k']
         print(f"{cp['k_exiting']:<10} | {cp['F_k']:<12.6f} | {cp['G_at_Fk']:<12.6f} | {str(cp['active_set_before']):<20} | {f_str}")
     print("-" * 80)

else:
     print("  No critical exit points found (only 1 outcome?).")


print("\n2. Global Maximum Growth Point:")
max_gp = analysis_results["global_max_point"]
if max_gp and np.isfinite(max_gp['G_opt']):
    print(f"  Source: {max_gp['source']}")
    print(f"  Max Growth occurs at F_opt = {max_gp['F_opt']:.6f}")
    print(f"    Active Set at F_opt: {max_gp['active_set']}")
    print(f"    Optimal Fractions f_i at F_opt:")
    f_str = ", ".join([f"{f:.4f}" for f in max_gp['fractions']])
    print(f"      [{f_str}]")
    print(f"    Maximum Expected Log Growth G_opt = {max_gp['G_opt']:.6f}")
else:
    print("  Could not determine maximum growth point (G might be -inf or calculation failed).")

# Restore warning settings if needed elsewhere
warnings.resetwarnings()
