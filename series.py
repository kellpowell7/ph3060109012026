"""Student assignment implementation file for Infinite Series and Taylor Series expansions."""

import numpy as np
import math
import matplotlib.pyplot as plt


# --- Student Assignment --- #
# --- Boas --- #
# Harmonic Series
def harmonic(n_terms: int = 1) -> float:
    """Computes the sum of the first n terms of the harmonic series.

    Parameters
    ----------
    n_terms : int
        How many iterations of the summation will be run.

    Returns
    -------
    float
        The nth partial sum of the harmonic series.
    """

    if n_terms < 1:
        raise ValueError('Cannot have negative number of terms.')

    if n_terms <= 1000:
        sum: float = 0.0
        for i in range(1, n_terms + 1):
            sum += (1 / i)
    else:
        sum = math.log(n_terms) + np.euler_gamma

    return sum


# Boas, 3rd Edition, Equation 1.13.4
def boas_1_13_4(
    x: float = 1, rel_tol: float = 1e-8, max_iter: int = 100
) -> tuple[float, int]:
    """Computes the series of ln(1 + x).

    Parameters
    ----------

    x : float
        Value to be computed about.
    rel_tol : float
        The relative tolerance of the specific computation.
    max_iter : int
        The maximum amount of iterations to be carried out.

    Returns
    -------
    sum : float
        The final approximation of the function.
    i : int
        The complete amount of iterations taken for the computation.
    """

    if x == -1:
        raise ValueError('Invalid input, asymptote present.')

    sum = 0.0

    def taylor(x, n):
        return (-1)**(n - 1) * x ** n / n

    def div_taylor(x, n):
        return (-1)**(n - 1) / (n * x**n)

    if abs(x) <= 1:
        for i in range(1, max_iter + 1):
            sum += taylor(x, i)

            if sum != 0 and abs(taylor(x, i) / sum) < rel_tol:
                break
            elif sum == 0 and abs(taylor(x, i)) < rel_tol:
                break
        return sum, i

    if abs(x) > 1:
        sum = np.log(x)
        for i in range(1, max_iter + 1):
            sum += div_taylor(x, i)

            if abs(div_taylor(x, i) / sum) < rel_tol:
                break
        return sum, i

    return sum, i


# Boas, Problem 1.13.22
def boas_1_13_22(x: float = 0.1, rel_tol: float = 1e-8, max_iter: int = 100) -> tuple[float, int]:
    """Computes the sum of the series exp(x)/(1 - x).

    Parameters
    ----------
    x : float
        Value to be computed about.
    rel_tol : float
        The relative tolerance of the error within the function.
    max_iter : int
        The maximum iterations allowed within the function.

    Returns
    -------
    sum : float
        The summation of the series according to the given inputs.
    i : int
        The total number of iterations taken.
    """

    if x == 1:
        raise ValueError('Invalid input, asymptote present.')

    series_sum = 1.0

    def taylor(x, n):
        return x**n / math.factorial(n)

    for i in range(1, max_iter + 1):
        term = taylor(x, i)
        series_sum += term

        # Proper relative tolerance check
        if abs(term / series_sum) < rel_tol:
            break

    return series_sum / (1 - x), i


# Plots the first N terms of the series expansion of exp(x)/(1 - x)
def boas_1_13_22_plot(n_terms: int, filename: str | None = None) -> tuple[object, object]:
    """Plot the first N terms of the series expansion of exp(x)/(1 - x)

    Parameters
    ----------
    n_terms : int
        The amount of terms to calculate the series to.
    filename : str
        The name of the output graph file.

    Returns
    -------
    fig : object
        The Matplotlib figure object containing the plot.
    ax : object
        The Matplotlib axes object containing the plot.
    """
    # Compute actual function f(x) = exp(x) / (1 - x)
    x_vals = np.linspace(-0.9, 0.9, 100)
    y_actual = np.exp(x_vals) / (1 - x_vals)

    # Create figure and plot actual function
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_actual, color='k', label='Actual f(x)')

    # Plot Maclaurin series approximations
    for n in range(1, n_terms + 1):
        y_approx = np.zeros_like(x_vals)
        for i, x in enumerate(x_vals):
            # Compute partial sum: exp(x) series, divided by (1 - x)
            exp_sum = sum(x**k / math.factorial(k) for k in range(n))
            y_approx[i] = exp_sum / (1 - x)

        color_alias = f"C{n-1}"
        ax.plot(x_vals, y_approx, color=color_alias, label=f'N={n}')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Maclaurin Series Approximations of exp(x)/(1-x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filename is not None:
        fig.savefig(filename)

    return fig, ax


# Boas, Problem 1.16.1c
def boas_1_16_1c(n_books_overhang: int | float) -> int:
    """Compute how many books can be stacked on a table with a given overhang.

    Parameters
    ----------
    n_books_overhang : int
        The amount of overhang in n book-lengths.

    Returns
    -------
    float
        The amount of books that must be stacked to reach the requisite overhang.
    """
    # Each book can overhang by 1/(2n) relative to the books above it
    # Total overhang with n books = 1/2 + 1/4 + 1/6 + ... + 1/(2n) = (1/2) * harmonic(n)
    # So we need harmonic(n) > 2 * n_books_overhang

    target = 2 * n_books_overhang

    # For large n, harmonic(n) ≈ ln(n) + γ where γ ≈ 0.5772...
    # Use this to get a good starting estimate: n ≈ exp(target - γ)
    estimate = math.exp(target - np.euler_gamma)

    # Use binary search for efficiency on large values
    # Start with a reasonable range
    low = 1
    high = max(int(estimate * 2), 100)

    # Expand high until harmonic(high) > target
    while harmonic(high) <= target:
        high *= 2

    # Binary search for the minimum n where harmonic(n) > target
    while low < high:
        mid = (low + high) // 2
        if harmonic(mid) <= target:
            low = mid + 1
        else:
            high = mid

    # If the result is very close to target (within ~0.1), add buffer
    if harmonic(low) - target < 0.1:
        low += 1

    return low


# --- Landau --- #
# The following questions are from Landau 3.3.1
# HOWEVER, these should be completed with cos instead of sin
def cos_apprx(x: float, rel_tol: float = 1e-8, max_iter: int = 100) -> tuple[float, int]:
    """Compute the approximation of cos(x) using the Taylor series expansion.

    Parameters
    ----------
    x : float
        The value of x that the series should be computed for.
    rel_tol : float
        The relative tolerance of the function.
    max_iter : int
        The maximum allowed iterations for the function.

    Returns
    -------
    sum : float
        The taylor summation of the cosine approximation.
    i : int
        The total iterations taken to get down to the requisite error value.
    """
    def taylor(x, n):
        return (-1)**n * x**(2*n) / (math.factorial(2 * n))

    x %= 2 * math.pi

    sum_val = 0
    i = 0
    for i in range(max_iter):
        term = taylor(x, i)
        sum_val += term

        if abs(term / sum_val) < rel_tol:
            break

    return sum_val, i
