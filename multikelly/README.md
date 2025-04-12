# Kelly Criterion Analysis for Constrained Total Bet Fraction

*Generated: Saturday, April 12, 2025 at 7:30:30 AM EDT*

## Overview

This repository contains a Python script that analyzes the behavior of the optimal Kelly betting strategy when applied to multiple mutually exclusive outcomes, specifically under the constraint that the *total fraction* of the bankroll wagered (`F`) is fixed (i.e., `sum(f_i) = F`).

The script implements an algorithm to:

1.  **Calculate Critical Exit Points (`F_k`):** Determine the specific values of the total bet fraction `F` at which individual outcomes are dropped from the optimal betting portfolio as `F` decreases.
2.  **Determine Optimal Fractions (`f_i`):** Calculate the optimal fraction of the bankroll to allocate to each active bet at these critical exit points and at the point of maximum growth.
3.  **Calculate Expected Log-Growth (`G`):** Compute the expected logarithmic growth rate of the bankroll at the critical exit points and at the maximum growth point.
4.  **Identify Global Maximum Growth Point (`F_opt`, `G_opt`):** Find the total fraction `F` (`F_opt`) that yields the highest possible expected log-growth (`G_opt`) within the allowed range (0 < F <= 1), using theoretical insights to check for interior optima, along with the corresponding optimal allocation strategy.

This analysis helps understand how the optimal betting strategy adapts as the total risk (represented by `F`) changes, revealing which bets are prioritized and when less favorable bets are excluded, and where the peak long-term growth rate lies.

## Background: Constrained Kelly Criterion & Optimization

The standard Kelly Criterion aims to maximize the expected logarithm of wealth, `E[ln(W_1/W_0)]`, which corresponds to maximizing long-term capital growth. When applied to `N` mutually exclusive outcomes with probabilities `p_i` and decimal odds `o_i`, we seek fractions `f_i` to bet on each outcome.

This script focuses on a *constrained* version where the sum of these fractions must equal a specific total fraction `F`:
`sum(f_i) = F` (where `0 < F <= 1`)

The objective is to maximize:
`G(F) = E[ln(Wealth_after / Wealth_before)] = sum_{i=1 to N} p_i * ln(1 - F + f_i * o_i)`
subject to `sum(f_i) = F` and `f_i >= 0`.

Key properties observed in this scenario include:

* As `F` decreases from 1, outcomes are progressively removed from the optimal betting set `I`.
* The optimal fractions `f_i` scale linearly with `F` within regions where the set `I` is constant.

The analysis leverages concepts from constrained optimization. The **Lagrange multiplier** (`lambda`) associated with the constraint `sum(f_i) = F` represents the marginal increase in the objective function (log-growth `G`) for a marginal increase in the total allowed fraction `F` (i.e., `lambda ≈ dG/dF`). The point of maximum unconstrained growth often corresponds to where this marginal gain equals a benchmark value, which is typically `lambda = 1` for log-growth maximization relative to holding cash.

## The Algorithm

The script uses an iterative algorithm that conceptually decreases `F` and analyzes the portfolio changes:

1.  **Initialization:** Start with the full set `I` of all possible outcomes. Set the upper bound of the first F-region `F_high = 1.0`.
2.  **Identify Exit Order:** In each iteration, identify the outcome `k` currently in the active set `I` that has the **minimum `p_k * o_k` product**. This outcome is the next one to exit the portfolio as `F` decreases.
3.  **Calculate Exit Threshold (`F_k`):** For the identified outcome `k`, calculate the critical fraction `F_k` at which its optimal allocation `f_k` would become zero, defining the lower bound of the current region. This uses the formula derived from KKT conditions:
    `F_k = (S_p - p_k * o_k * S_o) / (S_p + p_k * o_k * (1 - S_o))`
    where `S_p = sum(p_j)` and `S_o = sum(1/o_j)` are calculated over the *current* active set `I`.
4.  **Check for Interior Maximum (`F_opt`):** Before removing outcome `k`, check if the global maximum growth occurs *within* the current region `(F_k, F_high]`.
    * **Motivation:** The maximum growth occurs where the marginal gain `lambda = 1`.
    * **Calculation:** Calculate the candidate fraction `F_potential_opt` that corresponds to `lambda = 1` using the parameters (`S_p`, `S_o`) of the *current* active set `I`:
        `F_potential_opt = (S_p - S_o) / (1 - S_o)` (Handle the case where `S_o = 1`).
    * **Validation:** Check if this candidate falls strictly within the valid range for this region: `F_k < F_potential_opt < F_high`.
    * **Store if Valid:** If it falls within the range, this `F_potential_opt` identifies the location of the interior **global maximum** (due to the concavity of the log-growth function). Store its details (`F_opt`, `G_opt`, `f_i`, active set) and stop checking for interior points in subsequent regions.
5.  **Store Critical Point:** Record the calculated `F_k`, the exiting outcome `k`, and the active set `I` *before* `k` was removed.
6.  **Update:** Set `F_high = F_k` for the next region check and remove outcome `k` from the active set `I`.
7.  **Repeat:** Continue steps 2-6 until only one outcome remains in `I`.
8.  **Determine Global Maximum & Post-Processing:**
    * If a valid interior maximum `F_opt` was found (via the `lambda=1` check), report it as the global maximum.
    * If no interior maximum was found after checking all regions, the maximum must occur at the boundary `F=1.0`. Report the details for `F=1` (`f_i=p_i`).
    * Calculate the optimal fractions `f_i` and expected log-growth `G` at each recorded critical exit point `F_k`.

## Code Explanation

* **`analyze_kelly_regions(probabilities, odds)`:** The main function orchestrating the analysis. It implements the iterative algorithm described above, including the check for the interior maximum based on the `lambda=1` condition (`F = (S_p - S_o) / (1 - S_o)`).
* **`calculate_optimal_fractions(F, active_set_indices, p, o)`:** Helper function to calculate the optimal fractions `f_i` for a given `F` and active set `I`.
* **`calculate_log_growth(F, f_vec, p, o)`:** Helper function to calculate the expected log-growth `G` for a given `F` and fraction vector `f_vec`.

## Usage

1.  **Dependencies:** Requires Python and NumPy.
    ```bash
    pip install numpy pytz # pytz used for timestamp timezone
    ```
2.  **Save the Code:** Save the Python script as a file (e.g., `kelly_analyzer.py`).
3.  **Define Inputs:** Modify the example usage section at the bottom of the script or call the `analyze_kelly_regions` function with your specific `probabilities` and `odds`.
    * `probabilities`: List/array where elements sum to 1.0, `p_i > 0`.
    * `odds`: List/array of decimal odds, `o_i > 1` recommended.
4.  **Run the Script:**
    ```bash
    python kelly_analyzer.py
    ```

## Output Interpretation

The script output includes:

1.  **Iterative Steps:** Details showing the active set, the exiting outcome (`k`), calculated `S_p`, `S_o`, the potential `F` from the `lambda=1` condition, and the calculated exit threshold `F_k` for each iteration. It explicitly states if a valid interior maximum candidate is found.
2.  **Critical Points Summary:** A table summarizing the exit points:
    * `Exiting k`: Index of the outcome exiting.
    * `Exit F_k`: Critical value of `F`. Outcome `k` is excluded when `F` drops **below** this value. Listed in order of exit (highest `F_k` first).
    * `G at F_k`: Expected log-growth at this threshold.
    * `Active Set Before`: Indices active just before `k` exits.
    * `Fractions f_i at F_k`: Optimal allocation at the threshold (with `f_k=0`).
3.  **Global Maximum Growth Point:** Details about the overall optimum:
    * `Source`: Indicates if the maximum was found as an "Interior optimum (lambda=1 condition)" or at the "Boundary F=1.0".
    * `F_opt`: The total fraction `F` yielding the maximum growth.
    * `Active Set at F_opt`: Outcomes included at the optimum.
    * `Optimal Fractions f_i at F_opt`: The optimal allocation strategy.
    * `Maximum Expected Log Growth G_opt`: The highest achievable growth rate.

## Example

Using the following inputs:

* `probabilities = [0.5, 0.25, 0.25]`
* `odds = [1.5, 3.75, 6.0]`

The script produces the following summarized output:

```
Running Kelly Analysis at: 2025-04-12 07:28:39 EDT-0400

Analyzing Kelly Regions for the example:
Probabilities: [0.5, 0.25, 0.25]
Odds: [1.5, 3.75, 6.0]
--------------------------------------------------

--- Iteratively Finding Exit Points and Checking Regions for Max Growth ---

Iteration 1: Analyzing region F > F_k (where F_k is exit point of outcome below)
  Current Active Set I = [0, 1, 2]
  Min p*o = 0.7500 found for outcome k = 0 (Next to exit)
  S_p = 1.000000, S_o = 1.100000
  Region Check: Potential Max F from lambda=1 condition = 1.000000
  Calculated exit threshold F_0 = 0.189189 (Lower bound for current set's validity)
  Region Check: Potential Max F is outside the valid range (0.1892, 1.0000).

Iteration 2: Analyzing region F > F_k (where F_k is exit point of outcome below)
  Current Active Set I = [1, 2]
  Min p*o = 0.9375 found for outcome k = 1 (Next to exit)
  S_p = 0.500000, S_o = 0.433333
  Region Check: Potential Max F from lambda=1 condition = 0.117647
  Calculated exit threshold F_1 = 0.090909 (Lower bound for current set's validity)
  !!! Found Valid Interior Max Candidate at F_opt = 0.117647 in region (0.0909, 0.1892) !!!
      --> Confirmed Global Max G = 0.022650

--- Finished Iterations ---


--- FINAL ANALYSIS RESULTS ---

1. Critical Points (Order of Exit):
--------------------------------------------------------------------------------
Exiting k  | Exit F_k     | G at F_k     | Active Set Before    | Fractions f_i at F_k
--------------------------------------------------------------------------------
0          | 0.189189     | 0.019352     | [0, 1, 2]            | 0.0000, 0.0541, 0.1351
1          | 0.090909     | 0.022191     | [1, 2]               | 0.0000, 0.0000, 0.0909
--------------------------------------------------------------------------------

2. Global Maximum Growth Point:
  Source: Interior optimum based on lambda=1 in region F > 0.0909
  Max Growth occurs at F_opt = 0.117647
    Active Set at F_opt: [1, 2]
    Optimal Fractions f_i at F_opt:
      [0.0000, 0.0147, 0.1029]
    Maximum Expected Log Growth G_opt = 0.022650
```

## Limitations and Assumptions

* Requires accurate probabilities (`p_i > 0`, sum=1) and odds (`o_i > 0`). Meaningful Kelly betting typically requires `o_i > 1`.
* Floating-point precision can lead to minor variations or require tolerances in calculations.
* The calculation `F_potential_opt = (S_p - S_o) / (1 - S_o)` assumes `S_o != 1`. The code handles the `S_o ≈ 1` case by skipping the interior check for that region.
* Assumes simultaneous bets placed according to the calculated fractions for a given `F`.
* The underlying theory assumes logarithmic utility of wealth.
