import numpy as np

def calculate_kelly_exit_details(probabilities, odds):
    """
    Calculates critical exit fractions F_k, optimal betting fractions f_i
    at those points, and expected log growth G.

    Args:
        probabilities (list or np.array): List of probabilities for each outcome.
        odds (list or np.array): List of decimal odds for each outcome.

    Returns:
        list: A list of dictionaries, each containing details for an exit event:
              {'exiting_k': k, 'F_k': F_k, 'fractions': f_vector, 'log_growth': G}
              Sorted by F_k descending (earliest exit first).
              Returns an empty list if input is invalid or calculation fails.
    """
    n_outcomes = len(probabilities)
    if n_outcomes != len(odds) or n_outcomes == 0:
        print("Error: Probabilities and odds must be non-empty lists of the same length.")
        return []

    p = np.array(probabilities, dtype=float)
    o = np.array(odds, dtype=float)

    if not np.isclose(np.sum(p), 1.0):
         print(f"Warning: Probabilities do not sum to 1 (sum={np.sum(p):.4f}).")
    if np.any(p <= 0):
         print("Error: Probabilities must be positive.")
         return []
    if np.any(o <= 1):
         print("Warning: Some odds are <= 1.")

    active_set = set(range(n_outcomes))
    exit_results = [] # Store results as a list of dicts
    po_products = p * o

    print("\n--- Calculation Steps ---")
    iteration = 1
    while len(active_set) > 1:
        print(f"\nIteration {iteration}: Active Set = {sorted(list(active_set))}")
        current_po = {i: po_products[i] for i in active_set}
        min_po = float('inf')
        k_to_exit = -1
        for i in sorted(list(active_set)):
            if current_po[i] < min_po:
                min_po = current_po[i]
                k_to_exit = i

        if k_to_exit == -1: print("Error: Could not find minimum p*o outcome."); return []
        print(f"  Min p*o = {min_po:.4f} triggers exit for outcome k = {k_to_exit}")

        # Calculate F_k using the current active_set (includes k_to_exit)
        active_indices = list(active_set)
        S_p = np.sum(p[active_indices])
        S_o = np.sum(1.0 / o[active_indices])
        pk = p[k_to_exit]; ok = o[k_to_exit]
        numerator = S_p - pk * ok * S_o
        denominator = S_p + pk * ok * (1.0 - S_o)

        if np.isclose(denominator, 0):
            print(f"  Warning: Denominator near zero for F_{k_to_exit}. Skipping point.")
            F_k = np.nan # Indicate failure for this point
        elif denominator < 0:
             print(f"  Warning: Negative denominator for F_{k_to_exit}.")
             F_k = numerator / denominator
        else:
            F_k = numerator / denominator
        print(f"  Calculated Exit Threshold F_{k_to_exit} = {F_k:.6f}")

        # --- Calculate optimal fractions and log growth AT F = F_k ---
        f_vector_at_Fk = np.zeros_like(p)
        G_at_Fk = np.nan # Default to NaN

        # Active set excluding the one that just exited
        I_prime = active_set - {k_to_exit}

        if I_prime and not np.isnan(F_k) and F_k < 1.0 : # Ensure F_k is valid and sensible
            try:
                # Calculate S_p', S_o' for the remaining set I'
                active_indices_prime = list(I_prime)
                S_p_prime = np.sum(p[active_indices_prime])
                S_o_prime = np.sum(1.0 / o[active_indices_prime])

                if np.isclose(S_p_prime, 0): # Avoid division by zero if I_prime is empty or has 0 prob
                     print(f"  Warning: S_p_prime is zero for F_{k_to_exit}. Cannot calculate fractions.")
                else:
                    term1_factor = (F_k + (1.0 - F_k) * S_o_prime) / S_p_prime
                    term2_factor = (1.0 - F_k)

                    for i in I_prime:
                        # Formula requires i to be in I_prime
                        f_i = p[i] * term1_factor - term2_factor / o[i]
                        f_vector_at_Fk[i] = max(0.0, f_i) # Ensure non-negative

                    # Verify sum f_i approx F_k (optional check)
                    if not np.isclose(np.sum(f_vector_at_Fk), F_k, atol=1e-6):
                        print(f"  Warning: Sum of fractions {np.sum(f_vector_at_Fk):.6f} != F_k {F_k:.6f}")
                        # This might happen due to precision or if F_k>=1 forced fractions to 0

                    # Calculate Log Growth G
                    R_vector = 1.0 - F_k + f_vector_at_Fk * o
                    if np.any(R_vector <= 1e-9): # Check for non-positive values before log
                        print(f"  Warning: Non-positive wealth factor detected for F_{k_to_exit}. Cannot calculate G.")
                    else:
                        G_at_Fk = np.sum(p * np.log(R_vector))
                        print(f"  Optimal Fractions at F_{k_to_exit}: {[f'{x:.4f}' for x in f_vector_at_Fk]}")
                        print(f"  Expected Log Growth G at F_{k_to_exit}: {G_at_Fk:.6f}")

            except Exception as e:
                 print(f"  Error during fraction/growth calculation for F_{k_to_exit}: {e}")

        elif F_k >= 1.0:
             print(f"  Skipping fraction/growth calculation as F_k >= 1.")
             # Log growth is likely -inf or undefined if forced F=F_k>=1 with bad bets
        elif not I_prime:
             print(f"  Last outcome exiting. Portfolio is empty below F_{k_to_exit}.")
             # Growth would be log(1-F_k) if F_k applied to no bets.
             if F_k < 1.0: G_at_Fk = np.log(1.0 - F_k) # Growth from cash only

        # Store result
        exit_results.append({
            'exiting_k': k_to_exit,
            'F_k': F_k,
            'fractions': f_vector_at_Fk,
            'log_growth': G_at_Fk
        })

        # Remove k from the active set for the next iteration
        active_set.remove(k_to_exit)
        iteration += 1

    print("\n--- Calculation Complete ---")
    # Sort results by F_k descending for presentation
    exit_results.sort(key=lambda x: x['F_k'], reverse=True)
    return exit_results

# --- Example Usage ---
#probabilities_ex = [0.4, 0.3, 0.2, 0.1]
#odds_ex = [2.6, 3.5, 5.1, 11.0]
probabilities_ex = [0.5, 0.25, 0.25]
odds_ex = [1.5, 3.75, 6]

print("Calculating Kelly exit details for the example:")
print(f"Probabilities: {probabilities_ex}")
print(f"Odds: {odds_ex}")
print("-" * 50)

results = calculate_kelly_exit_details(probabilities_ex, odds_ex)

print("\n--- Results: Exit Points, Fractions, and Log Growth ---")
print("(Sorted by F_k descending: Earlier exits appear first)")
print("(Fractions & Growth are calculated AT the moment outcome 'k' exits)")

if results:
    print("-" * 80)
    print(f"{'Exit k':<8} | {'Exit F_k':<12} | {'Log Growth G':<15} | {'Optimal Fractions f_i at F_k':<40}")
    print("-" * 80)
    for res in results:
        k = res['exiting_k']
        Fk = res['F_k']
        G = res['log_growth']
        f_vec = res['fractions']
        frac_str = "[" + ", ".join([f"{f:.4f}" for f in f_vec]) + "]"
        print(f"{k:<8} | {Fk:<12.6f} | {G:<15.6f} | {frac_str:<40}")
    print("-" * 80)
    # Find remaining outcome
    all_outcomes = set(range(len(probabilities_ex)))
    exited_k = {res['exiting_k'] for res in results}
    remaining = list(all_outcomes - exited_k)
    if remaining:
        print(f"Outcome {remaining[0]} remains active until F approaches 0.")
else:
    print("Calculation failed or no results found.")
