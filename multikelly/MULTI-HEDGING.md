# Constrained Kelly Criterion Analysis with Bonuses (multi-hedging)

*Generated: Sunday, April 13, 2025 at 12:40:27 AM EDT (Lexington, Kentucky, United States)*

## Overview

This repository contains the Python script `multi-hedging.py` that analyzes the optimal Kelly betting strategy for multiple mutually exclusive outcomes. This version generalizes the problem by including **fixed, outcome-dependent bonuses `x_i`** (expressed as fractions of the initial bankroll) that are awarded if outcome `i` occurs, regardless of any bets placed.

The analysis is performed under the constraint that the **total fraction `F`** of the bankroll wagered across all outcomes is fixed (`sum(f_i) = F`, where `0 < F <= 1`).

The script implements an algorithm to:

1.  **Calculate Critical Exit Points (`F_k`):** Determine the specific values of the total bet fraction `F` at which individual outcomes are dropped from the optimal betting portfolio as `F` decreases, considering the impact of bonuses.
2.  **Determine Optimal Fractions (`f_i`):** Calculate the optimal fraction of the bankroll to allocate to each active bet at these critical exit points and at the point of maximum growth, incorporating bonuses.
3.  **Calculate Expected Log-Growth (`G`):** Compute the expected logarithmic growth rate of the bankroll (`E[ln(W_1/W_0)]`) at the critical exit points and at the maximum growth point, including bonus effects.
4.  **Identify Global Maximum Growth Point (`F_opt`, `G_opt`):** Find the total fraction `F` (`F_opt`) that yields the highest possible expected log-growth (`G_opt`) within the allowed range (0 < F <= 1), using generalized theoretical criteria to check for interior optima.

This analysis helps understand how the optimal hedging strategy adapts as the total allowed wager `F` changes, revealing which bets are prioritized and when less favorable opportunities (considering odds, probabilities, and bonuses) are excluded, and identifying the overall optimal risk level `F_{opt}`.

## Background: Constrained Kelly Criterion with Bonuses

The core objective remains maximizing the expected logarithm of wealth (`G = E[ln(W_1/W_0)]`). With bonuses `x_i`, the wealth `W_1^{(k)}` if outcome `k` occurs becomes `W_0 (1 - F + f_k o_k + x_k)`. The optimization problem is:

`max G = sum_{k=1 to N} p_k * ln(1 - F + f_k o_k + x_k)`
subject to `sum f_i = F` and `f_i >= 0`.

This constrained optimization problem can be solved using the Karush-Kuhn-Tucker (KKT) conditions. The **Lagrange multiplier** (`lambda`) associated with the constraint `sum(f_i) = F` represents the marginal increase in log-growth `G` for a marginal increase in `F` (`lambda ≈ dG/dF`). The theory suggests the maximum unconstrained growth occurs when this marginal gain equals a benchmark (typically `lambda = 1`). This insight is used to locate potential interior optima.

A key finding is that even with bonuses, the optimal fractions `f_i(F)` remain **linear functions** of `F` within regions where the set of active bets (`I = {i | f_i > 0}`) is constant.

## The Algorithm (Generalized with Bonuses)

The script uses an iterative algorithm that conceptually decreases `F` from 1:

1.  **Initialization:** Start with the full active set `I = {1, \dots, N}`. Set region upper bound `F_high = 1.0`. Initialize list for critical points `CP = []`.
2.  **Identify Exit Order:** In each iteration:
    * Calculate the potential exit threshold `F_k` for **every** outcome `k` currently in the active set `I` using the generalized formula (see below).
    * The next outcome to exit, `k_to_exit`, is the one with the **highest** calculated `F_k` that is still strictly less than `F_high`. Let this threshold be `F_exit_threshold`.
    * *(Note: With bonuses, the simple `min(p*o)` rule no longer solely determines the exit order.)*
3.  **Calculate Exit Threshold (`F_k`):** The generalized formula for the exit threshold `F_k` for outcome `k`, derived from setting the generalized `f_k(F)=0`, is:
    `F_k = [ S_p (1 + x_k) - p_k o_k (S_o + S_{xo}) ] / [ S_p + p_k o_k (1 - S_o) ]`
    where `S_p = sum_{j in I} p_j`, `S_o = sum_{j in I} (1/o_j)`, and `S_{xo} = sum_{j in I} (x_j / o_j)` are calculated over the *current* active set `I`. Handle potential division by zero or edge cases.
4.  **Check for Interior Maximum (`F_opt`):** Before removing `k_to_exit`, check if the global maximum growth occurs *within* the current region `(F_exit_threshold, F_high]`.
    * **Motivation:** The max occurs where `lambda = 1`.
    * **Calculation:** Use the parameters (`S_p`, `S_o`, `S_{xo}`) of the *current* active set `I` to calculate the candidate fraction `F_potential_opt` corresponding to `lambda = 1`:
        `F_potential_opt = (S_p - S_o - S_{xo}) / (1 - S_o)` (Handle `S_o = 1`).
    * **Validation:** Check if this candidate falls strictly within the valid range for this region: `F_exit_threshold < F_potential_opt < F_high`.
    * **Store if Valid:** If it falls within the range, this `F_{opt} = F_{potential_opt}` is the unique interior global maximum (due to concavity). Store its details and stop checking in later regions.
5.  **Store Critical Point:** Record the actual exit threshold `F_exit_threshold`, the exiting outcome `k_to_exit`, and the active set `I` *before* `k_to_exit` was removed.
6.  **Update:** Set `F_high = F_exit_threshold` for the next region check and remove `k_to_exit` from the active set `I`.
7.  **Repeat:** Continue steps 2-6 until only one outcome remains or no further valid exit points are found below `F_high`.
8.  **Determine Global Maximum & Post-Processing:**
    * If a valid interior maximum `F_opt` was found, report it as the global maximum.
    * If not, the maximum must occur at the boundary `F=1.0`. Calculate the optimal fractions and growth at `F=1` using the generalized formulas (note: `f_i(1)` generally differs from `p_i` when `x_i > 0`). Report `F=1` details as the global maximum, potentially comparing `G(1)` with `G` at the highest `F_k` as a sanity check.
    * Calculate the optimal fractions `f_i` and log-growth `G` at each recorded critical exit point `F_k` using the generalized helper functions.
    * Calculate and store details (fractions, growth, active set) specifically for the `F=1.0` boundary.

## Code Explanation (`multi-hedging.py`)

* **`analyze_kelly_regions(probabilities, odds, bonuses)`:** The main function implementing the generalized algorithm. Takes probabilities, odds, and the new `bonuses` array as input. Returns a dictionary with analysis results.
* **`calculate_optimal_fractions(F, active_set_indices, p, o, x)`:** Generalized helper function to calculate optimal fractions `f_i`, including bonus effects.
* **`calculate_log_growth(F, f_vec, p, o, x)`:** Generalized helper function to calculate expected log-growth `G`, including bonus effects on returns.

## Usage

1.  **Dependencies:** Requires Python, NumPy, and optionally Pytz for timezone-aware timestamps.
    ```bash
    pip install numpy pytz
    ```
2.  **Save the Code:** Save the generalized Python script as `multi-hedging.py`.
3.  **Define Inputs:** Modify the example usage section at the bottom of the script or call `analyze_kelly_regions` with your specific inputs:
    * `probabilities`: List/array where elements sum to 1.0, `p_i > 0`.
    * `odds`: List/array of decimal odds, `o_i > 1` recommended.
    * `bonuses`: List/array of non-negative fixed bonuses `x_i` (as fractions of bankroll).
4.  **Run the Script:**
    ```bash
    python multi-hedging.py
    ```

## Output Interpretation

The script output includes:

1.  **Iterative Steps:** Details for each iteration, including the active set, sums (`S_p`, `S_o`, `S_{xo}`), the potential `F` from the `lambda=1` condition, the calculated exit thresholds `F_k` for all active outcomes, and the identified `k_to_exit` with its `F_exit_threshold`. Explicitly states if a valid interior maximum is found.
2.  **Critical Points Summary:** A table summarizing the exit points (where `F_k < 1`):
    * `Exiting k`: Index of the outcome exiting.
    * `Exit F_k`: Critical value of `F`. Outcome `k` is excluded when `F` drops **below** this value. Listed in order of exit (highest `F_k` first).
    * `G at F_k`: Expected log-growth at this threshold.
    * `Active Set Before`: Indices active just before `k` exits.
    * `Fractions f_i at F_k`: Optimal allocation at the threshold (with `f_k=0`).
3.  **Global Maximum Growth Point:** Details about the overall optimum:
    * `Source`: Indicates if the maximum was found as an "Interior optimum (lambda=1 condition)" or at the "Boundary F=1.0" (or potentially at the highest F_k if G(1) is lower).
    * `F_opt`: The total fraction `F` yielding the maximum growth.
    * `Active Set at F_opt`: Outcomes included at the optimum.
    * `Optimal Fractions f_i at F_opt`: The specific fractions allocated.
    * `Maximum Expected Log Growth G_opt`: The highest achievable growth rate.
4.  **Details at F=1.0 Boundary:** Information specifically for `F=1`:
    * `G(1.0)`: Expected log-growth when betting the entire bankroll.
    * `Active Set at F=1.0`: Outcomes receiving non-zero allocation.
    * `Optimal Fractions f_i at F=1.0`: The specific allocation fractions (note: `f_i(1) != p_i` typically).

## Example (with Bonuses)

Using the following inputs:

* `probabilities = [0.5, 0.25, 0.25]`
* `odds = [1.5, 3.75, 6.0]`
* `bonuses = [0.05, 0.0, 0.10]`

The script (`multi-hedging.py`) produces the following summarized output:

```
Running Kelly Analysis (with Bonuses) at: 2025-04-13 04:26:01 UTC+0000

Analyzing Kelly Regions for the example:
Probabilities: [0.5, 0.25, 0.25]
Odds: [1.5, 3.75, 6.0]
Bonuses (x_i): [0.05, 0.0, 0.1]
--------------------------------------------------

--- Iteratively Finding Exit Points and Checking Regions for Max Growth (with Bonuses) ---

Iteration 1: Analyzing region F < 1.000000
  Current Active Set I = [0, 1, 2]
  S_p = 1.000000, S_o = 1.100000, S_xo = 0.050000
  Highest valid F_k below 1.000000 is F_0 = 0.202703. Outcome k=0 exits next.
  Region Check: Potential Max F from lambda=1 = 1.500000
  Region Check: Potential Max F is outside plausible (0, 1] range.

Iteration 2: Analyzing region F < 0.202703
  Current Active Set I = [1, 2]
  S_p = 0.500000, S_o = 0.433333, S_xo = 0.016667
  Highest valid F_k below 0.202703 is F_1 = 0.075758. Outcome k=1 exits next.
  Region Check: Potential Max F from lambda=1 = 0.088235
  !!! Found Valid Interior Max Candidate at F_opt = 0.088235 in region (0.0758, 0.2027) !!!
      --> Confirmed Global Max G = 0.065739

--- Finished Iterations ---

--- Calculating details specifically for F=1.0 boundary ---
  G(1.0) = -0.009819
  f_i(1.0) = ['0.4917', '0.2625', '0.2458']
  Active Set at F=1.0: [0, 1, 2]


--- FINAL ANALYSIS RESULTS (with Bonuses) ---

1. Critical Points (Order of Exit):
--------------------------------------------------------------------------------
Exiting k  | Exit F_k     | G at F_k     | Active Set Before    | Fractions f_i at F_k
--------------------------------------------------------------------------------
0          | 0.202703     | 0.063369     | [0, 1, 2]            | 0.0000, 0.0698, 0.1329
1          | 0.075758     | 0.065063     | [1, 2]               | 0.0000, 0.0000, 0.0758
--------------------------------------------------------------------------------

2. Global Maximum Growth Point:
  Source: Interior optimum (lambda=1) in region F > 0.0758
  Max Growth occurs at F_opt = 0.088235
    Active Set at F_opt: [1, 2]
    Optimal Fractions f_i at F_opt:
      [0.0000, 0.0069, 0.0814]
    Maximum Expected Log Growth G_opt = 0.065739

3. Details at F=1.0 Boundary:
  Expected Log Growth G(1.0) = -0.009819
    Active Set at F=1.0: [0, 1, 2]
    Optimal Fractions f_i at F=1.0:
      [0.4917, 0.2625, 0.2458]
```

## Limitations and Assumptions

* Assumes accurate probabilities (`p_i > 0`, sum=1), odds (`o_i > 0`), and bonus fractions (`x_i >= 0`).
* Assumes bonuses `x_i` are fixed fractions of the initial bankroll.
* Floating-point precision may affect results slightly.
* Edge cases like `S_o = 1` or denominators becoming zero in formulas are handled with warnings but might warrant closer inspection.
* Assumes simultaneous bets based on `F`, ignores transaction costs or parameter uncertainty.
* The underlying theory assumes logarithmic utility of wealth.
