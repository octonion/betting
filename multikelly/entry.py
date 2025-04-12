import numpy as np

def calculate_kelly_exit_fractions(probabilities, odds):
    """
    Calculates the critical total fraction F at which each outcome exits
    the optimal Kelly portfolio (constrained sum f_i = F).

    Args:
        probabilities (list or np.array): List of probabilities for each outcome.
        odds (list or np.array): List of decimal odds for each outcome.

    Returns:
        dict: A dictionary mapping outcome index to its exit fraction F_k.
              The outcome exits the portfolio when F drops *below* this value.
              Returns an empty dict if input is invalid or calculation fails.
    """
    n_outcomes = len(probabilities)
    if n_outcomes != len(odds) or n_outcomes == 0:
        print("Error: Probabilities and odds must be non-empty lists of the same length.")
        return {}

    # Use numpy arrays for easier calculations
    p = np.array(probabilities, dtype=float)
    o = np.array(odds, dtype=float)

    # --- Basic input validation ---
    if not np.isclose(np.sum(p), 1.0):
         print(f"Warning: Probabilities do not sum to 1 (sum={np.sum(p):.4f}). Proceeding cautiously.")
         # Consider normalizing or raising an error depending on requirements
    if np.any(p <= 0):
         print("Error: Probabilities must be positive.")
         return {}
    if np.any(o <= 1):
         # Allow odds <= 1, but p*o will be small, likely exiting first.
         print("Warning: Some odds are <= 1. Corresponding outcomes might exit immediately (F_k near 1 or >1).")

    # --- Algorithm Implementation ---
    active_set = set(range(n_outcomes))
    exit_points = {}
    po_products = p * o # Pre-calculate p*o products

    print("\n--- Calculation Steps ---")

    iteration = 1
    while len(active_set) > 1:
        print(f"\nIteration {iteration}: Active Set = {sorted(list(active_set))}")

        current_po = {i: po_products[i] for i in active_set}

        # Find outcome k with the minimum p*o product in the active set
        min_po = float('inf')
        k_to_exit = -1
        # Iterate deterministically through sorted indices for consistent tie-breaking
        for i in sorted(list(active_set)):
            if current_po[i] < min_po:
                min_po = current_po[i]
                k_to_exit = i
            # If po products are equal, the one with the lower index (checked first) is chosen

        if k_to_exit == -1:
             print("Error: Could not find minimum p*o outcome. Aborting.")
             return {} # Should not happen with valid inputs > 1 outcome

        print(f"  Min p*o = {min_po:.4f} found for outcome k = {k_to_exit}")

        # Calculate S_p and S_o for the current active_set I
        active_indices = list(active_set)
        S_p = np.sum(p[active_indices])
        S_o = np.sum(1.0 / o[active_indices])
        print(f"  S_p (sum p_i for active) = {S_p:.6f}")
        print(f"  S_o (sum 1/o_i for active) = {S_o:.6f}")

        # Calculate F_k for the exiting outcome k
        pk = p[k_to_exit]
        ok = o[k_to_exit]

        numerator = S_p - pk * ok * S_o
        denominator = S_p + pk * ok * (1.0 - S_o)
        print(f"  Calculating F_{k_to_exit}: Numerator = {numerator:.6f}, Denominator = {denominator:.6f}")


        # Check for potential division by zero or unusual cases
        if np.isclose(denominator, 0):
            print(f"  Warning: Denominator near zero for outcome {k_to_exit}. Setting F_k to NaN.")
            F_k = float('nan')
        elif denominator < 0:
             print(f"  Warning: Negative denominator for outcome {k_to_exit}. May indicate unusual odds/probs. F_k = {numerator/denominator:.6f}")
             F_k = numerator / denominator # Report raw value but flag it
        else:
            F_k = numerator / denominator
            # F_k should ideally be between 0 and 1. Values outside this range can occur
            # if p*o is very low (F_k > 1) or in edge cases.

        print(f"  => F_{k_to_exit} = {F_k:.6f}")
        exit_points[k_to_exit] = F_k

        # Remove k from the active set for the next iteration
        active_set.remove(k_to_exit)
        iteration += 1

    print("\n--- Calculation Complete ---")
    if active_set:
        print(f"Final remaining outcome (active down to F=0): {list(active_set)[0]}")
    else:
        print("No outcomes remaining.")

    return exit_points

# --- Example Usage ---
# Probabilities (summing to 1)
#probabilities_ex = [0.4, 0.3, 0.2, 0.1]
probabilities_ex = [0.5, 0.25, 0.25]
# Decimal Odds (>1)
#odds_ex = [2.6, 3.5, 5.1, 11.0] # p*o: 1.04, 1.05, 1.02, 1.10

odds_ex = [1.5, 3.75, 6]

print("Calculating Kelly exit fractions for the example:")
print(f"Probabilities: {probabilities_ex}")
print(f"Odds: {odds_ex}")
print("-" * 50)

exit_fractions = calculate_kelly_exit_fractions(probabilities_ex, odds_ex)

print("\n--- Results: Exit Points (F_k) ---")
if exit_fractions:
    # Sort by exit fraction (value of F below which outcome is excluded)
    # Higher F_k means it exits earlier (at higher total fraction F)
    sorted_exits = sorted(exit_fractions.items(), key=lambda item: item[1], reverse=True)

    print("(Outcome exits when F drops BELOW the stated F_k value)")
    print("-" * 65)
    print(f"{'Outcome':<10} | {'Exit F_k':<15} | {'p':<8} | {'o':<8} | {'p*o':<10}")
    print("-" * 65)
    for outcome_index, exit_f in sorted_exits:
         p_val = probabilities_ex[outcome_index]
         o_val = odds_ex[outcome_index]
         po_val = p_val * o_val
         print(f"{outcome_index:<10} | {exit_f:<15.6f} | {p_val:<8.3f} | {o_val:<8.1f} | {po_val:<10.4f}")
    print("-" * 65)

    # Find the outcome(s) not in exit_points (the one remaining)
    all_outcomes = set(range(len(probabilities_ex)))
    remaining = list(all_outcomes - set(exit_fractions.keys()))
    if remaining:
        print(f"Outcome {remaining[0]} remains active until F approaches 0.")

else:
    print("Calculation failed or no exit points found.")
