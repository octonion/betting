# Kelly Criterion Optimization Library (`kelly_analyzer.py`)

## Introduction

This library provides tools to calculate optimal betting strategies according to the **Kelly Criterion**. The core principle of the Kelly Criterion is to maximize the expected value of the logarithm of wealth, which, under certain assumptions, maximizes the long-term growth rate of capital.

The library supports:
* Analysis of single or multiple simultaneous betting opportunities (games).
* Inclusion of fixed fractional bonuses awarded upon winning specific outcomes.
* Calculation of the global optimum allocation (maximizing long-term growth).
* Calculation of allocations for fixed total betting fractions.
* Calculation of allocations for fractional Kelly strategies (betting a fraction of the optimal total).
* Choice between a fast analytical KKT solver (for single games without bonuses) and a general-purpose convex optimization solver (using CVXPY).

## Mathematical Formulation

Let's define the key variables:

* $W_0$: Initial bankroll (typically assumed to be 1 for calculation).
* $g$: Index for a game (if multiple games).
* $i$: Index for an outcome within a game.
* $f_{g,i}$: Fraction of the *initial* bankroll bet on outcome $i$ of game $g$.
* $o_{g,i}$: Decimal odds received for winning outcome $i$ of game $g$ (total return is $f_{g,i} \times o_{g,i}$).
* $p_{g,i}$: Probability of outcome $i$ occurring in game $g$.
* $x_{g,i}$: Fixed bonus fraction (relative to $W_0$) received if outcome $i$ of game $g$ occurs.
* $F$: Total fraction of the bankroll bet across all outcomes and all games, $F = \sum_{g,i} f_{g,i}$.

**Wealth After One Period:**

Consider a scenario where multiple independent games are bet simultaneously. The outcome is a *joint outcome* $J$, which specifies the winning outcome $k_g$ for each game $g$. The probability of this joint outcome is $p_J = \prod_g p_{g, k_g}$.

If the joint outcome $J$ occurs, the wealth $W_{1,J}$ after this period, starting from $W_0=1$, is:

$W_{1,J} = 1 - \sum_{g',i'} f_{g',i'} + \sum_{\text{game } g} (f_{g, k_g} \times o_{g, k_g} + x_{g, k_g})$

Or, more compactly:

$W_{1,J} = 1 - F + \text{Gains}_J$

where $\text{Gains}_J$ represents the sum of payouts from winning bets ($f_{g, k_g} o_{g, k_g}$) and bonuses ($x_{g, k_g}$) for the specific outcomes $k_g$ that occurred in joint outcome $J$.

**Objective Function (Kelly Criterion):**

The goal is to choose the fractions $f_{g,i}$ to maximize the expected value of the logarithm of the wealth after one period:

Maximize $G = E[\log(W_1)] = \sum_{\text{all joint outcomes } J} p_J \log(W_{1,J})$

**Constraints:**

The optimization is performed subject to the following constraints:

1.  **Non-negativity:** $f_{g,i} \ge 0$ for all $g, i$. (You cannot bet a negative fraction).
2.  **Budget Constraint:**
    * For *global optimization*: $\sum_{g,i} f_{g,i} \le 1$. (You cannot bet more than your total bankroll).
    * For *fixed total fraction F*: $\sum_{g,i} f_{g,i} = F_{\text{target}}$.

## Optimization Approaches

The library implements two methods to solve this maximization problem.

### 1. Convex Optimization (CVXPY Approach - General Case)

**Theory:**

The Kelly Criterion objective function, $G = E[\log(W_1)]$, is a **concave function** of the betting fractions $f_{g,i}$. The logarithm function is concave, and the expectation (a sum weighted by positive probabilities) preserves concavity. The constraints ($f_{g,i} \ge 0$ and the budget constraint $\sum f_{g,i} \le 1$ or $\sum f_{g,i} = F_{\text{target}}$) are linear.

Maximizing a concave function subject to linear constraints is a standard **convex optimization problem**. These problems have the desirable property that any locally optimal solution is also globally optimal.

**Implementation (`_solve_cvxpy`):**

This library uses the `cvxpy` package to solve the general Kelly problem.
* **CVXPY:** A Python-embedded modeling language for convex optimization problems. It allows you to express the objective function and constraints in a natural mathematical syntax.
* **DCP:** CVXPY uses **Disciplined Convex Programming (DCP)** rules to analyze the problem structure and verify its convexity (or concavity for maximization). If the problem follows DCP rules, CVXPY can convert it into a standard form that numerical solvers can understand.
* **Solvers:** CVXPY interfaces with backend numerical solvers (like SCS, ECOS, MOSEK) that perform the actual computation to find the optimal $f_{g,i}$ values. SCS is the default in this library.

**Advantages:**
* Handles the most general form of the problem: multiple games, bonuses, different budget constraints.
* Relatively easy to formulate complex scenarios.

**Limitations:**
* Requires `cvxpy` and a compatible solver installation.
* Can become computationally intensive for problems with many games, as the number of joint outcomes ($N = \prod_g |\text{outcomes}_g|$) grows exponentially. The complexity is roughly related to $N$.

### 2. KKT Analytical Approach (Single Game, No Bonuses ONLY)

**Restriction:**
* The implementation (`_solve_kkt_no_bonus`) provided in this library is specifically derived for and **only applicable to a single game ($g=1$) with zero bonuses ($x_i = 0$ for all $i$)**. It cannot be directly used if bonuses are present or if multiple games are involved.

**Theory:**

For constrained optimization problems like Kelly, the **Karush-Kuhn-Tucker (KKT) conditions** provide necessary conditions for optimality. For convex problems, they are also sufficient. The KKT conditions essentially state that at an optimal point, the gradient of the objective function must be expressible as a linear combination of the gradients of the *active* constraints (constraints that hold with equality), with non-negative multipliers for inequality constraints.

**Derivation (Single Game, No Bonus):**

Consider a single game with $n$ outcomes, probabilities $p_i$, odds $o_i$, and fractions $f_i$. We want to find the optimal *total* fraction $F = \sum f_i$ and the allocation $f_i$ that maximize $G$. We can think of optimizing $G$ with respect to $f_i$ for a *fixed* $F$, and then optimizing over $F$.

Objective: $G(f_1,...,f_n; F) = \sum_{i=1}^n p_i \log(1 - F + f_i o_i)$
Constraints:
* $\sum_{i=1}^n f_i = F$ (Budget for the fixed F)
* $f_i \ge 0$ (Non-negativity)

We introduce **Lagrange Multipliers**:
* $\lambda$: Associated with the budget constraint $\sum f_i = F$.
* $\mu_i$: Associated with the non-negativity constraints $f_i \ge 0$ (or $-f_i \le 0$).

The Lagrangian is:
$\mathcal{L}(f, \lambda, \mu) = \sum p_i \log(1 - F + f_i o_i) - \lambda (\sum f_i - F) - \sum \mu_i (-f_i)$

The relevant KKT conditions are:

1.  **Stationarity:** $\frac{\partial \mathcal{L}}{\partial f_i} = \frac{p_i o_i}{1 - F + f_i o_i} - \lambda - \mu_i = 0$ for all $i$.
2.  **Primal Feasibility:** $\sum f_i = F$ and $f_i \ge 0$.
3.  **Dual Feasibility:** $\mu_i \ge 0$.
4.  **Complementary Slackness:** $\mu_i f_i = 0$ for all $i$.

*Interpretation:*
* If $f_i > 0$ (an outcome is actively bet on), complementary slackness implies $\mu_i = 0$. The stationarity condition becomes $\frac{p_i o_i}{1 - F + f_i o_i} = \lambda$.
* If $f_i = 0$ (an outcome is not bet on), complementary slackness allows $\mu_i \ge 0$. The stationarity condition implies $\frac{p_i o_i}{1 - F} \le \lambda$.

Solving for $f_i$ when $f_i > 0$ gives: $f_i = \frac{p_i}{\lambda} - \frac{1 - F}{o_i}$.

**Algorithm Description (`_solve_kkt_no_bonus`):**

The library's KKT implementation leverages these conditions to find the global optimum $F$ without needing to solve the full system for every possible $F$.

1.  **Find Critical Points ($F_k$):** It calculates the theoretical total fraction $F_k$ at which each fraction $f_k$ would become zero ($f_k \to 0$). This occurs when the conditions shift from $f_k > 0$ to $f_k = 0$. Using the formulas, this leads to a calculation for $F_k$ based on $p_i$, $o_i$ for the currently active set of bets.
2.  **Iterative Refinement:** It starts assuming all outcomes are active ($F=1$) and iteratively finds the outcome $k$ with the highest $F_k < F_{high}$ (where $F_{high}$ is the upper bound of the current F-region being considered, initially 1.0). This $F_k$ becomes the *lower* bound for the region where the current set was active. The outcome $k$ is removed from the active set, and $F_{high}$ is updated to $F_k$.
3.  **Check for Interior Maximum:** Within each F-region defined by consecutive exit points $(F_k, F_{high})$, the algorithm checks if the condition for maximum growth ($dG/dF = 0$) occurs. For the no-bonus case, this simplifies to checking if $\lambda(F) = 1$. The value of $F$ where $\lambda=1$ can be calculated as $F_{\lambda=1} = (S_p - S_o) / (1 - S_o)$, where $S_p = \sum p_i$ and $S_o = \sum (1/o_i)$ for the *current active set*. If a valid $F_{\lambda=1}$ (i.e., $0 < F_{\lambda=1} \le 1$ and $F_k < F_{\lambda=1} < F_{high}$) is found, this is the global maximum $F_{opt}$ because $G(F)$ is concave.
4.  **Boundary Evaluation:** If no valid interior maximum is found during the iterations, the algorithm evaluates the growth rate $G$ at the boundaries ($F=0$, $F=1$) and at all the calculated critical points $F_k$ (clamped to the [0, 1] range). The $F$ value yielding the highest valid $G$ is chosen as $F_{opt}$.
5.  **Zero Growth:** If the maximum calculated $G$ is non-positive, the optimal strategy is to bet nothing ($F=0, G=0$).

**Advantages:**
* Generally faster and potentially more numerically precise than iterative convex solvers *for the specific case it handles*.
* Provides insights into the structure of the solution (exit points).

**Limitations:**
* **Current implementation only handles single games with no bonuses.** Extending it analytically for bonuses or multiple games is significantly more complex.

## Library Usage

### Running the Code

1.  Save the library code as `kelly_analyzer.py`.
2.  Ensure dependencies (`numpy`, `cvxpy`, optionally `pytz`) are installed.
3.  Import the main function: `from kelly_analyzer import optimize_kelly`
4.  Prepare your game data as a dictionary (for a single game) or a list of dictionaries (for multiple games).
5.  Call `optimize_kelly`, passing the game data and any desired options (`mode`, `kelly_fraction`, `F_total`, `use_kkt`, `solver`).
6.  Interpret the returned dictionary.

### Main Function Signature

```python
optimize_kelly(games_data, mode='global', F_total=None, kelly_fraction=None,
               solver=None, use_kkt=False, verbose=False, eps=DEFAULT_EPS)
```
