import numpy as np
import warnings
from datetime import datetime
import pytz # Required for timezone handling

# Suppress RuntimeWarning for log(0) etc., handle it explicitly
warnings.filterwarnings('ignore', r'divide by zero encountered in log')
warnings.filterwarnings('ignore', r'invalid value encountered in log')
warnings.filterwarnings('ignore', r'divide by zero encountered in scalar divide')
warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', r'invalid value encountered in true_divide') # For S_o etc.
warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')

def calculate_optimal_fractions(F, active_set_indices, p, o, x):
    """
    Calculates the optimal fractions f_i for a given F and active set,
    including fixed bonuses x_i.
    """
    n_outcomes = len(p)
    f_vec = np.zeros(n_outcomes)

    # Ensure indices are valid and F is positive for calculations
    active_set_indices = [i for i in active_set_indices if i < n_outcomes]
    if not active_set_indices or F < 1e-12:
        return f_vec

    active_p = p[active_set_indices]
    active_o = o[active_set_indices]
    active_x = x[active_set_indices]

    # Check for non-positive odds in the active set
    if np.any(active_o <= 1e-12):
        print(f"Error: Non-positive odds found in active set {active_set_indices} for F={F}. Cannot calculate fractions.")
        return f_vec * np.nan # Indicate failure

    S_p = np.sum(active_p)
    # Calculate S_o and S_xo carefully, avoiding division by zero
    valid_o_indices = np.where(active_o > 1e-12)[0] # Indices within active_o
    if len(valid_o_indices) != len(active_o):
        print(f"Warning: Zero or near-zero odds encountered in active set for F={F}")
    
    # Compute sums only over elements with valid odds
    S_o = np.sum(1.0 / active_o[valid_o_indices]) if len(valid_o_indices) > 0 else 0.0
    S_xo = np.sum(active_x[valid_o_indices] / active_o[valid_o_indices]) if len(valid_o_indices) > 0 else 0.0


    if np.isclose(S_p, 0):
        return f_vec # Should not happen if p_i > 0

    # Calculate 1/lambda (Generalized Eq D)
    one_div_lambda = (F + (1.0 - F) * S_o + S_xo) / S_p

    for idx_in_active, original_idx in enumerate(active_set_indices):
        # Check if odds for this specific index are valid
        if o[original_idx] > 1e-12:
            # Generalized Eq C/E
            f_i = p[original_idx] * one_div_lambda - (1.0 - F + x[original_idx]) / o[original_idx]
            f_vec[original_idx] = max(0.0, f_i)
        else:
             f_vec[original_idx] = 0 # Cannot bet if odds invalid

    # Renormalize f_vec if sum is not exactly F
    current_sum = np.sum(f_vec)
    if F > 1e-12 and current_sum > 1e-12 and not np.isclose(current_sum, F):
        f_vec = f_vec * (F / current_sum)
        f_vec[f_vec < 0] = 0
        # final_sum = np.sum(f_vec)
        # if not np.isclose(final_sum, F):
        #      print(f"Debug: Post-normalization sum {final_sum:.6f} still differs from F={F:.6f}")

    return f_vec

def calculate_log_growth(F, f_vec, p, o, x):
    """
    Calculates the expected log growth G for given F, fractions f_vec,
    and bonuses x_i.
    """
    n_outcomes = len(p)
    if np.sum(f_vec) > F + 1e-7: # Allow slightly larger tolerance
         # print(f"Warning: Sum of fractions {np.sum(f_vec):.6f} exceeds F={F:.6f} in G calculation.")
         pass

    # Calculate return factor R_k (Generalized)
    returns = 1.0 - F + f_vec * o + x

    # Handle cases where return <= 0
    if np.any(returns <= 1e-12):
        bad_indices = np.where(returns <= 1e-12)[0]
        # print(f"Warning: Non-positive return calculated for F={F:.6f}. Indices: {bad_indices}. Returns: {[f'{r:.4f}' for r in returns]}")
        return -np.inf

    # Calculate log safely
    log_returns = np.log(np.maximum(returns, 1e-12))
    G = np.sum(p * log_returns)
    return G

def analyze_kelly_regions(probabilities, odds, bonuses):
    """
    Analyzes Kelly strategy with fixed bonuses x_i, subject to sum f_i = F.
    Finds exit points, optimal fractions, log growth, and the global maximum.
    """
    n_outcomes = len(probabilities)
    analysis = {
        "critical_points": [],
        "global_max_point": None,
        "boundary_F1_details": None
    }
    if n_outcomes != len(odds) or n_outcomes != len(bonuses):
        print("Error: Probabilities, odds, and bonuses must be lists of the same length.")
        return analysis

    # Use numpy arrays
    p = np.array(probabilities, dtype=float)
    o = np.array(odds, dtype=float)
    x = np.array(bonuses, dtype=float) # Bonuses array

    # --- Input validation ---
    if not np.isclose(np.sum(p), 1.0): print(f"Warning: Probabilities sum to {np.sum(p):.4f}.")
    if np.any(p <= 0): print("Error: Probabilities must be positive."); return analysis
    if np.any(o <= 0): print("Error: Odds must be positive."); return analysis
    if np.any(x < 0): print("Error: Bonuses must be non-negative."); return analysis
    if np.any(o <= 1): print("Warning: Some odds <= 1.")


    # --- Algorithm Implementation ---
    active_set = set(range(n_outcomes))
    po_products = p * o # Still useful for context, though not sole driver of exit

    critical_points_data = [] # Store tuples: (F_k_exit_value, k_exiting, active_set_before)

    global_max_details = {"G_opt": -np.inf, "F_opt": np.nan, "source": "Initialization"}
    max_growth_found_interior = False

    F_high = 1.0 # Upper bound for F in the current region being checked

    print("\n--- Iteratively Finding Exit Points and Checking Regions for Max Growth (with Bonuses) ---")
    iteration = 1
    while len(active_set) > 1:
        print(f"\nIteration {iteration}: Analyzing region F < {F_high:.6f}")
        current_active_indices = sorted(list(active_set))
        print(f"  Current Active Set I = {current_active_indices}")

        # Calculate S_p, S_o, S_xo for the current active_set I
        active_p = p[current_active_indices]
        active_o = o[current_active_indices]
        active_x = x[current_active_indices]
        
        # Check for invalid odds before calculating sums involving division by odds
        valid_o_mask = active_o > 1e-12
        if not np.all(valid_o_mask):
            print(f"Error: Zero or negative odds found in active set {current_active_indices}. Cannot proceed.")
            break # Exit loop if odds are invalid
        
        S_p = np.sum(active_p)
        S_o = np.sum(1.0 / active_o) # Already checked active_o > 0 essentially
        S_xo = np.sum(active_x / active_o)
        print(f"  S_p = {S_p:.6f}, S_o = {S_o:.6f}, S_xo = {S_xo:.6f}")

        # --- Calculate F_k for all k in active_set (Generalized Eq F) ---
        Fk_values = {}
        for k in current_active_indices:
            pk, ok, xk = p[k], o[k], x[k]
            numerator_Fk = S_p * (1.0 + xk) - pk * ok * (S_o + S_xo)
            denominator_Fk = S_p + pk * ok * (1.0 - S_o)
            
            F_k = np.nan # Default
            if np.isclose(denominator_Fk, 0):
                # If denominator is 0 and numerator is non-zero, F_k is infinite (+ or -)
                # If both are zero, it's indeterminate. Treat as non-finite boundary.
                print(f"  F_k calc for k={k}: Denominator is zero.")
            else:
                F_k = numerator_Fk / denominator_Fk
            Fk_values[k] = F_k
            # print(f"  Debug: Calculated F_{k} = {F_k:.6f}") # Verbose debug

        # --- Find next outcome to exit ---
        # Find the highest F_k that is finite and strictly less than F_high
        eligible_Fks = {k: fk for k, fk in Fk_values.items() if np.isfinite(fk) and fk < F_high - 1e-9}

        if not eligible_Fks:
            print(f"  No valid exit threshold found below F_high={F_high:.6f}. Ending iterations.")
            break # Stop if no more outcomes exit below current F_high

        k_to_exit = max(eligible_Fks, key=eligible_Fks.get)
        F_exit_threshold = eligible_Fks[k_to_exit]
        print(f"  Highest valid F_k below {F_high:.6f} is F_{k_to_exit} = {F_exit_threshold:.6f}. Outcome k={k_to_exit} exits next.")

        active_indices_before_exit = current_active_indices # Copy before modification

        # Use max(0, F_exit_threshold) as the lower bound for the region check
        F_k_check_bound = max(0.0, F_exit_threshold)

        # --- Check for potential interior maximum using lambda=1 condition (Generalized Eq G) ---
        if not max_growth_found_interior:
            F_potential_opt = np.nan
            denominator_opt_check = 1.0 - S_o
            can_check_interior = True
            if np.isclose(denominator_opt_check, 0):
                 print("  Region Check: Cannot calculate potential interior max F (1 - S_o is zero).")
                 can_check_interior = False
            else:
                 F_potential_opt = (S_p - S_o - S_xo) / denominator_opt_check
                 print(f"  Region Check: Potential Max F from lambda=1 = {F_potential_opt:.6f}")
                 # Check if F_potential_opt is plausible (e.g., within (0, 1])
                 if F_potential_opt <= 1e-9 or F_potential_opt > 1.0 + 1e-9:
                     print("  Region Check: Potential Max F is outside plausible (0, 1] range.")
                     can_check_interior = False

            # Check if F_potential_opt falls strictly within (F_k_check_bound, F_high)
            if can_check_interior and not np.isnan(F_potential_opt):
                eps = 1e-9
                if F_k_check_bound + eps < F_potential_opt < F_high - eps:
                     print(f"  !!! Found Valid Interior Max Candidate at F_opt = {F_potential_opt:.6f} in region ({F_k_check_bound:.4f}, {F_high:.4f}) !!!")
                     F_opt = F_potential_opt
                     active_set_at_max = active_indices_before_exit
                     # Use generalized fraction/growth calculations
                     f_opt_vec = calculate_optimal_fractions(F_opt, active_set_at_max, p, o, x)
                     G_opt = calculate_log_growth(F_opt, f_opt_vec, p, o, x)

                     # Store as global max (since G is concave)
                     global_max_details = {
                         "F_opt": F_opt, "G_opt": G_opt,
                         "active_set": active_set_at_max, "fractions": f_opt_vec,
                         "source": f"Interior optimum (lambda=1) in region F > {F_exit_threshold:.4f}"
                     }
                     max_growth_found_interior = True
                     print(f"      --> Confirmed Global Max G = {G_opt:.6f}")
                else:
                     print(f"  Region Check: Potential Max F is outside the valid region ({F_k_check_bound:.4f}, {F_high:.4f}).")

        # Store critical point info AFTER checking the region
        critical_points_data.append((F_exit_threshold, k_to_exit, active_indices_before_exit))

        # Update F_high for the next iteration's region check
        F_high = F_k_check_bound

        # Remove k from the active set
        active_set.remove(k_to_exit)
        iteration += 1

    # --- After Loop ---
    print("\n--- Finished Iterations ---")

    # --- Process critical points (calculate f_i and G at F_k) ---
    analysis["critical_points"] = []
    critical_points_data.sort(key=lambda x: x[0], reverse=True) # Sort by F_k descending

    for F_k_val, k_exiting_val, active_set_before_val in critical_points_data:
         if np.isnan(F_k_val): continue

         F_k_calc = max(0.0, F_k_val) # Use non-negative F for calculations
         # Use generalized fraction calculation
         f_at_Fk = calculate_optimal_fractions(F_k_calc, active_set_before_val, p, o, x)
         f_at_Fk[k_exiting_val] = 0.0 # Ensure exiting fraction is zero

         # Readjust sum if needed after forcing f_k=0
         current_sum = np.sum(f_at_Fk)
         if F_k_calc > 1e-12 and not np.isclose(current_sum, F_k_calc):
             if current_sum > 1e-12:
                 f_at_Fk *= (F_k_calc / current_sum)
                 f_at_Fk[f_at_Fk < 0] = 0
             else: f_at_Fk.fill(0.0)
         # Use generalized growth calculation
         G_at_Fk = calculate_log_growth(F_k_calc, f_at_Fk, p, o, x)

         analysis["critical_points"].append({
             "F_k": F_k_val, # Store original calculated F_k
             "k_exiting": k_exiting_val,
             "active_set_before": active_set_before_val,
             "fractions_at_Fk": f_at_Fk,
             "G_at_Fk": G_at_Fk
         })

    # --- Determine and finalize Global Maximum ---
    if not max_growth_found_interior:
        print("\n--- No interior max found. Evaluating boundary F=1.0 ---")
        F_opt = 1.0
        active_set_at_1 = list(range(n_outcomes))
        # Use generalized fraction calculation for F=1
        f_vec_at_1 = calculate_optimal_fractions(F_opt, active_set_at_1, p, o, x)
        actual_active_set_at_1 = [i for i, f in enumerate(f_vec_at_1) if f > 1e-9]
        # Use generalized growth calculation for F=1
        G_at_1 = calculate_log_growth(F_opt, f_vec_at_1, p, o, x)
        print(f"  Calculated G(1.0) = {G_at_1:.6f}")
        print(f"  Calculated f_i(1.0) = {[f'{f:.4f}' for f in f_vec_at_1]}")


        # Compare G(1) with G at highest finite F_k (if any)
        highest_finite_Fk_G = -np.inf
        highest_finite_Fk_details = None
        for cp in analysis["critical_points"]:
             if np.isfinite(cp["G_at_Fk"]):
                  highest_finite_Fk_G = cp["G_at_Fk"]
                  highest_finite_Fk_details = cp
                  break # Found the highest valid one

        if np.isfinite(G_at_1) and G_at_1 >= highest_finite_Fk_G:
             global_max_details = {
                 "F_opt": F_opt, "G_opt": G_at_1,
                 "active_set": actual_active_set_at_1,
                 "fractions": f_vec_at_1,
                 "source": "Boundary F=1.0"
             }
             print(f"  Global Max occurs at F=1.0")
        elif highest_finite_Fk_details is not None:
             print(f"  Warning: G(1.0)={G_at_1:.6f} is lower than G={highest_finite_Fk_G:.6f} at F_k={highest_finite_Fk_details['F_k']:.6f}. Using F_k point as max.")
             global_max_details = {
                 "F_opt": max(0.0, highest_finite_Fk_details['F_k']),
                 "G_opt": highest_finite_Fk_details['G_at_Fk'],
                 "active_set": highest_finite_Fk_details['active_set_before'],
                 "fractions": highest_finite_Fk_details['fractions_at_Fk'],
                 "source": f"Boundary at highest exit point F_k={highest_finite_Fk_details['F_k']:.6f}"
             }
        else: # No interior max, G(1) is invalid, and no valid critical points
             print("Error: Could not determine Global Maximum point.")
             global_max_details = None # Indicate failure


    analysis["global_max_point"] = global_max_details

    # --- Calculate and Store F=1 Details Separately ---
    print("\n--- Calculating details specifically for F=1.0 boundary ---")
    f_vec_at_1_final = calculate_optimal_fractions(1.0, list(range(n_outcomes)), p, o, x)
    G_at_1_final = calculate_log_growth(1.0, f_vec_at_1_final, p, o, x)
    actual_active_set_at_1_final = [i for i, f in enumerate(f_vec_at_1_final) if f > 1e-9]
    analysis["boundary_F1_details"] = {
        "F": 1.0,
        "G": G_at_1_final,
        "active_set": actual_active_set_at_1_final,
        "fractions": f_vec_at_1_final
    }
    print(f"  G(1.0) = {G_at_1_final:.6f}")
    print(f"  f_i(1.0) = {[f'{f:.4f}' for f in f_vec_at_1_final]}")
    print(f"  Active Set at F=1.0: {actual_active_set_at_1_final}")

    return analysis

# --- Get Current Time for Reporting ---
def get_current_time_string():
    # Use UTC for consistency or a specific timezone if pytz is available
    try:
        utc = pytz.utc
        now = datetime.now(utc)
        # Format with timezone offset
        return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except NameError: # pytz not imported or available
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") # Assume UTC if pytz fails


# --- Example Usage ---
probabilities_ex = [0.5, 0.25, 0.25]
odds_ex = [1.5, 3.75, 6.0]
# Add bonuses (as fractions of bankroll)
bonuses_ex = [0.05, 0.0, 0.10] # Bonus on outcome 0 and 2

print(f"Running Kelly Analysis (with Bonuses) at: {get_current_time_string()}")
print("\nAnalyzing Kelly Regions for the example:")
print(f"Probabilities: {probabilities_ex}")
print(f"Odds: {odds_ex}")
print(f"Bonuses (x_i): {bonuses_ex}")
print("-" * 50)

analysis_results = analyze_kelly_regions(probabilities_ex, odds_ex, bonuses_ex)

print("\n\n--- FINAL ANALYSIS RESULTS (with Bonuses) ---")

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
if max_gp and np.isfinite(max_gp.get('G_opt', -np.inf)): # Check if G_opt exists and is finite
    print(f"  Source: {max_gp['source']}")
    print(f"  Max Growth occurs at F_opt = {max_gp['F_opt']:.6f}")
    print(f"    Active Set at F_opt: {max_gp['active_set']}")
    print(f"    Optimal Fractions f_i at F_opt:")
    f_str = ", ".join([f"{f:.4f}" for f in max_gp['fractions']])
    print(f"      [{f_str}]")
    print(f"    Maximum Expected Log Growth G_opt = {max_gp['G_opt']:.6f}")
else:
    print("  Could not determine maximum growth point (G might be -inf or calculation failed).")

print("\n3. Details at F=1.0 Boundary:")
f1_details = analysis_results["boundary_F1_details"]
if f1_details:
    print(f"  Expected Log Growth G(1.0) = {f1_details['G']:.6f}")
    print(f"    Active Set at F=1.0: {f1_details['active_set']}")
    print(f"    Optimal Fractions f_i at F=1.0:")
    f_str = ", ".join([f"{f:.4f}" for f in f1_details['fractions']])
    print(f"      [{f_str}]")
else:
    print("  Details for F=1.0 could not be calculated.")


# Restore warning settings if needed elsewhere
warnings.resetwarnings()
