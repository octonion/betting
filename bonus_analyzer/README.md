# Log Utility Betting Analyzer (`bonus_analyzer.py`)

This script analyzes betting scenarios under the log-utility framework, considering potential fixed bonuses for specific outcomes. It performs several key tasks:

1.  **Traces Optimal Strategy:** Determines the optimal set of outcomes to bet on (the "active set") as the total fraction of the bankroll wagered ($F$) varies from 1 down to 0.
2.  **Predicts Global Optimum:** Uses an analytical test based on marginal utility to predict the region where the globally optimal total fraction $F^*$ (when $F$ itself is optimized between 0 and 1) should lie.
3.  **Numerical Verification:** Compares the results of the analytical algorithm against the `cvxpy` convex optimization library for fixed $F$ values.
4.  **Calculates Global Optimum:** Computes the globally optimal strategy ($F^*$ and the corresponding fractions $f_i^*$) using `cvxpy`.
5.  **Final Verification:** Compares the analytically predicted region for $F^*$ with the value found by `cvxpy`.

## Mathematical Model

We consider a betting scenario with $n$ mutually exclusive outcomes.
* **$p_i$**: Probability of outcome $i$ occurring ($\sum p_i = 1$).
* **$d_i$**: Decimal odds for outcome $i$ ($d_i > 0$).
* **$x_i$**: Fixed bonus received *only* if outcome $i$ occurs ($x_i \ge 0$).
* **$f_i$**: Fraction of the total bankroll (assumed to be 1) bet on outcome $i$ ($f_i \ge 0$).
* **$F$**: Total fraction of the bankroll wagered, $F = \sum_{i=1}^n f_i$.

**Objective Function:**
The goal is to maximize the expected logarithm of final wealth, $E[\log W]$. We use a standard wealth model where the unbet portion $(1-F)$ is kept safe:
If outcome $i$ occurs, the final wealth is $W_i = (1-F) + f_i d_i + x_i$.
The objective is:
$$ \max_{f_1,...,f_n} \sum_{i=1}^n p_i \log((1-F) + f_i d_i + x_i) $$

**Constraints:**
The script analyzes two scenarios:
1.  **Fixed F:** The total fraction $F$ is fixed. The constraints are $\sum_{i=1}^n f_i = F$ and $f_i \ge 0$. The trace algorithm maps the optimal strategy for $F \in (0, 1]$.
2.  **Variable F:** The total fraction $F$ is also optimized. The constraints are $f_i \ge 0$ and $F = \sum_{i=1}^n f_i \le 1$.

## Convex Optimization

This problem is an instance of **convex optimization**.
* The objective function $E[\log W]$ is **concave** because the logarithm is concave and the wealth $W_i$ is an affine function of the variables $f_i$.
* The constraints ($f_i \ge 0$, $\sum f_i = F$ or $\sum f_i \le 1$) define a **convex set**.
* Maximizing a concave function over a convex set has the crucial property that **any locally optimal solution is also globally optimal**. Furthermore, the Karush-Kuhn-Tucker (KKT) conditions are both necessary and **sufficient** for global optimality.

This means we can confidently find the single best strategy using methods based on the KKT conditions.

## Algorithm Description

The script uses an analytical approach based on solving the KKT conditions.

1.  **Finding the Active Set $S^*$ for Fixed F (`find_active_set`)**:
    * For a given $F$, the optimal strategy involves betting only on a subset of outcomes (the active set $S$, where $f_i > 0$).
    * This function iteratively adjusts a candidate set $S$ based on the KKT conditions until the correct set $S^*$ is found. It checks if currently active bets should become inactive or if inactive bets should become active based on the Lagrange multiplier $\lambda(F)$.
    * $\lambda(F) = \frac{\sum_{i \in S} p_i}{F + \sum_{i \in S} (1-F+x_i)/d_i}$
    * $f_i(F) = p_i/\lambda(F) - (1-F+x_i)/d_i$ for $i \in S$.

2.  **Tracing Deactivations (`trace_deactivations`)**:
    * Starts with $F=1$ and finds the initial active set $S^*(1)$.
    * It calculates, for each outcome $k$ currently in the active set $S$, the critical fraction $F_k$ at which it would become inactive ($f_k=0$):
      $ F_k = \frac{P_S(1+x_k) - p_k d_k(D_S + X_S)}{P_S + p_k d_k (1-D_S)} $
      (where $P_S, D_S, X_S$ are sums of $p_i, 1/d_i, x_i/d_i$ over the current set $S$).
    * It identifies the largest valid $F_k$ below the current $F$ value. This $F_{crit}$ is the next point where the active set changes.
    * It removes the corresponding outcome(s) from $S$ and repeats the process until $F=0$ or $S$ is empty.
    * This maps out intervals $(F_{low}, F_{high}]$ and the corresponding constant active set $S$ for each interval.
    * It also identifies potential "flat regions" where the optimal utility might be constant over a range of $F$ (occurs if $P_S \approx 1$ and $D_S \approx 1$).

3.  **Global Optimum $F^*$ Prediction (Test Function $T(F, S)$):**
    * Calculates $T(F, S) = dG/dF = \lambda(F)(1-D_S) - \sum_{j \notin S} p_j/(1-F+x_j)$, which represents the marginal gain from increasing $F$ given active set $S$.
    * **Predicts $F^*=1$** if $T(1, S^*(1)) \ge 0$.
    * **Predicts interior $F^*$** lies in $(F_{low}, F_{high}]$ if $T(F, S)$ brackets zero within that interval (i.e., $T(F_{high}) \le 0$ and $T(F_{low}+\epsilon) \ge 0$).
    * **Predicts $F^*=0$** if $T(F, S) < 0$ for all $F > 0$.

## Verification using CVXPY

To ensure the correctness of the analytical algorithm and formulas:
* **Fixed-F Comparison:** For various test values of $F$, the script compares the active set $S_{analytic}$ and fractions $f_{analytic}$ derived from the trace algorithm against the results ($S_{cvxpy}, f_{cvxpy}$) obtained by solving the fixed-F optimization problem directly using the `cvxpy` library with appropriate numerical solvers (ECOS, SCS). It also compares the resulting $E[\log W]$ values.
* **Mismatch Handling:** Discrepancies are flagged. However, if a mismatch in $S$ or $f$ occurs very close to a transition point $F_{crit}$ AND the $E[\log W]$ values match, it's noted as a likely numerical boundary issue and not counted as a fundamental mismatch in the final summary.
* **Variable-F Optimum Calculation:** The script uses `cvxpy` to solve the variable-F problem (optimizing $f_i$ subject to $\sum f_i \le 1$) to find the numerical global optimum $F^*$ and $f^*$.
* **Final Verification:** It compares the numerically found $F^*$ with the region predicted by the analytical $T(F,S)$ test, checking for consistency, especially in cases of potential non-unique optima (flat regions).

## How to Run

1.  **Dependencies:** Install Python 3 and the required libraries:
    ```bash
    pip install numpy cvxpy ecos scs
    ```
    (Depending on your system, `ecos` and `scs` might need specific build tools or can sometimes be installed via conda).

2.  **Input File (`betting_examples.txt`):**
    * Create a text file named `betting_examples.txt` (or use a different name and pass it as a command-line argument).
    * Each line represents one outcome for a given example.
    * Each line should contain three space-separated numbers: `probability odds bonus` (i.e., $p_i$ $d_i$ $x_i$).
    * Use a **blank line** to separate different examples.
    * Lines starting with `#` are treated as comments and ignored.
    * Probabilities for each example *must* sum to 1.0 (the script includes a check and normalization). $d_i$ must be $>0$, $x_i$ must be $\ge 0$.

    *Example `betting_examples.txt` content:*
    ```text
    # Example 1: Simple 2-outcome
    0.6 2.0 0.0
    0.4 2.2 0.0

    # Example 2: 3 outcomes with bonus
    0.5 2.0 0.1
    0.3 3.0 0.0
    0.2 5.0 0.05

    ```

3.  **Running the Script:**
    * Save the Python code as `bonus_analyzer.py`.
    * Make sure `betting_examples.txt` is in the same directory.
    * Run from the terminal:
        ```bash
        python bonus_analyzer.py
        ```
    * *Optional:* To use a different input filename:
        ```bash
        python bonus_analyzer.py your_custom_examples.txt
        ```

4.  **Interpreting Output:**
    * For each example, the script will print:
        * The input parameters.
        * The results of the trace algorithm: intervals of $F$ and the corresponding optimal active set $S$.
        * The prediction for the global optimum $F^*$ based on the $T(F,S)$ test.
        * A summary of any mismatches found during the fixed-F comparison against CVXPY (distinguishing boundary discrepancies).
        * The globally optimal $F^*$, $f_i^*$, $E[\log W]^*$, and $S^*$ found by CVXPY for the variable-F problem.
        * The final verification comparing the analytical $F^*$ prediction region with the CVXPY $F^*$ result.
    * A final summary across all examples is printed at the end. Check this for any significant mismatch counts or solver failures.
