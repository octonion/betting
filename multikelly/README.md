# Kelly Criterion Analysis for Constrained Total Bet Fraction

## Overview

This repository contains a Python script that analyzes the behavior of the optimal Kelly betting strategy when applied to multiple mutually exclusive outcomes, specifically under the constraint that the *total fraction* of the bankroll wagered (`F`) is fixed (i.e., `sum(f_i) = F`).

The script implements an algorithm to:

1.  **Calculate Critical Exit Points (`F_k`):** Determine the specific values of the total bet fraction `F` at which individual outcomes are dropped from the optimal betting portfolio as `F` decreases.
2.  **Determine Optimal Fractions (`f_i`):** Calculate the optimal fraction of the bankroll to allocate to each active bet at these critical exit points and at the point of maximum growth.
3.  **Calculate Expected Log-Growth (`G`):** Compute the expected logarithmic growth rate of the bankroll at the critical exit points and at the maximum growth point.
4.  **Identify Maximum Growth Point (`F_opt`, `G_opt`):** Find the total fraction `F` (`F_opt`) that yields the highest possible expected log-growth (`G_opt`) within the allowed range (0 < F <= 1), along with the corresponding optimal fractions.

This analysis helps understand how the optimal betting strategy adapts as the total risk (represented by `F`) changes, revealing which bets are prioritized and when less favorable bets are excluded.

## Background: Constrained Kelly Criterion

The standard Kelly Criterion aims to maximize the expected logarithm of wealth, which corresponds to maximizing long-term capital growth. When applied to `N` mutually exclusive outcomes with probabilities `p_i` and decimal odds `o_i`, we seek fractions `f_i` to bet on each outcome.

This script focuses on a *constrained* version where the sum of these fractions must equal a specific total fraction `F`:
`sum(f_i) = F` (where `0 < F <= 1`)

The objective is to maximize:
`G(F) = E[ln(Wealth_after / Wealth_before)] = sum_{i=1 to N} p_i * ln(1 - F + f_i * o_i)`
subject to `sum(f_i) = F` and `f_i >= 0`.

Key properties observed in this scenario include:

* As `F` decreases from 1, outcomes are progressively removed from the optimal betting set.
* The optimal fractions `f_i` scale linearly with `F` within regions where the set of active bets is constant.

## The Algorithm

The script uses an iterative algorithm to analyze the behavior as `F` decreases:

1.  **Initialization:** Start with the full set `I` of all possible outcomes. Assume `F=1` initially.
2.  **Identify Exit Order:** In each iteration, identify the outcome `k` currently in the active set `I` that has the **minimum `p_k * o_k` product**. This outcome is the next one to exit the portfolio as `F` decreases.
3.  **Calculate Exit Threshold (`F_k`):** For the identified outcome `k`, calculate the critical fraction `F_k` at which its optimal allocation `f_k` becomes zero. This is done using the formula derived from the Karush-Kuhn-Tucker (KKT) conditions of the constrained optimization problem:
    `F_k = (S_p - p_k * o_k * S_o) / (S_p + p_k * o_k * (1 - S_o))`
    where `S_p = sum(p_j)` and `S_o = sum(1/o_j)` are calculated over the *current* active set `I`.
4.  **Store Critical Point:** Record `F_k`, the exiting outcome `k`, and the active set `I` *before* `k` was removed.
5.  **Check for Interior Maximum (`F_opt`):** Within the F-range corresponding to the current active set `I` (between the previously calculated exit threshold `F_high` and the current `F_k`), check if the potential unconstrained optimum lies within this range. This optimum occurs where the Lagrange multiplier `lambda` equals 1, which corresponds to `F_potential_opt = (S_p - S_o) / (1 - S_o)`. If `F_k < F_potential_opt < F_high`, then `F_opt = F_potential_opt` is identified as the point of maximum growth.
6.  **Update Set:** Remove outcome `k` from the active set `I`.
7.  **Repeat:** Continue steps 2-6 until only one outcome remains in `I`.
8.  **Post-Processing:**
    * If no interior maximum `F_opt` was found, assume the maximum occurs at the boundary `F=1.0`.
    * Calculate the optimal fractions `f_i` and expected log-growth `G` at each critical exit point `F_k` (using the active set *before* exit) and at the identified maximum growth point `F_opt`.

## Code Explanation

* **`analyze_kelly_regions(probabilities, odds)`:** The main function that takes lists/arrays of probabilities and decimal odds as input and returns a dictionary containing the analysis results.
* **`calculate_optimal_fractions(F, active_set_indices, p, o)`:** A helper function that calculates the vector of optimal fractions `f_i` for a given total fraction `F` and the list of indices in the currently active set.
* **`calculate_log_growth(F, f_vec, p, o)`:** A helper function that calculates the expected log-growth `G` based on the total fraction `F`, the vector of allocated fractions `f_vec`, probabilities `p`, and odds `o`.

## Usage

1.  **Dependencies:** Make sure you have Python and NumPy installed:
    ```bash
    pip install numpy
    ```
2.  **Save the Code:** Save the Python script provided previously as a file (e.g., `kelly_analyzer.py`).
3.  **Define Inputs:** Modify the example usage section at the bottom of the script to define your specific `probabilities` and `odds` lists.
    * `probabilities`: A list or NumPy array where elements sum to 1.0, and each `p_i > 0`.
    * `odds`: A list or NumPy array of corresponding decimal odds (must be > 1 for meaningful betting).
4.  **Run the Script:**
    ```bash
    python kelly_analyzer.py
    ```

## Output Interpretation

The script will print:

1.  **Calculation Steps (Optional):** Detailed steps showing which outcome exits in each iteration and intermediate calculations (can be commented out if desired).
2.  **Critical Points (Order of Exit):** A summary table showing:
    * `Outcome k`: The index of the outcome exiting.
    * `Exit F_k`: The critical value of `F`. The outcome `k` is excluded from the portfolio when the total fraction `F` drops **below** this value. Outcomes are listed starting with the one that exits first (highest `F_k`).
    * `Active Set Before Exit`: The indices of outcomes included just before `k` exits.
    * `Optimal Fractions f_i at F_k`: The calculated optimal fractions for all outcomes at the moment `k` exits (note `f_k` will be 0).
    * `Expected Log Growth G at F_k`: The growth rate achievable at this specific threshold `F_k`.
3.  **Maximum Growth Point:** Information about the point yielding the highest growth rate:
    * `F_opt`: The total fraction `F` at which maximum growth occurs.
    * `Note`: Indicates if the maximum occurs at the boundary `F=1.0` or is an interior point (`F_opt < 1.0`).
    * `Active Set at F_opt`: The outcomes included in the portfolio at `F_opt`.
    * `Optimal Fractions f_i at F_opt`: The specific fractions allocated at the maximum growth point.
    * `Maximum Expected Log Growth G_opt`: The highest achievable growth rate.

## Example

Using the following inputs:

* `probabilities = [0.5, 0.25, 0.25]`
* `odds = [1.5, 3.75, 6.0]`

The script produces the following summarized output:

Calculating Kelly exit details for the example:
Probabilities: [0.5, 0.25, 0.25]
Odds: [1.5, 3.75, 6]
--------------------------------------------------

--- Calculation Steps ---

Iteration 1: Active Set = [0, 1, 2]
  Min p*o = 0.7500 triggers exit for outcome k = 0
  Calculated Exit Threshold F_0 = 0.189189
  Optimal Fractions at F_0: ['0.0000', '0.0541', '0.1351']
  Expected Log Growth G at F_0: 0.019352

Iteration 2: Active Set = [1, 2]
  Min p*o = 0.9375 triggers exit for outcome k = 1
  Calculated Exit Threshold F_1 = 0.090909
  Optimal Fractions at F_1: ['0.0000', '0.0000', '0.0909']
  Expected Log Growth G at F_1: 0.022191

--- Calculation Complete ---

--- Results: Exit Points, Fractions, and Log Growth ---
(Sorted by F_k descending: Earlier exits appear first)
(Fractions & Growth are calculated AT the moment outcome 'k' exits)
--------------------------------------------------------------------------------
Exit k   | Exit F_k     | Log Growth G    | Optimal Fractions f_i at F_k            
--------------------------------------------------------------------------------
0        | 0.189189     | 0.019352        | [0.0000, 0.0541, 0.1351]                
1        | 0.090909     | 0.022191        | [0.0000, 0.0000, 0.0909]                
--------------------------------------------------------------------------------
Outcome 2 remains active until F approaches 0.

