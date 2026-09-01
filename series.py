"""Student assignment implementation file for Infinite Series and Taylor Series expansions."""

import numpy as np
import math

# --- Student Assignment --- #
# --- Boas --- #
# Harmonic Series
def harmonic(n_terms: int):
    """
    Compute the sum of the first n terms of the harmonic series.
    
    Parameters
    ----------
    n_terms : int
        How many iterations of the summation will be run.
         
    Returns
    -------
    sum : float
        The nth partial sum of the harmonic series.
    """
    if n_terms <= 1000:
        sum = 0
        for i in range(1, n_terms +1):
            sum += (1 / i)
    else:
        sum = math.log(n_terms) + np.euler_gamma


# Boas, 3rd Edition, Equation 1.13.4
def boas_1_13_4(
    x: float, rel_tol: float = 1e-8, max_iter: int = 100
) -> tuple[float, int]:
    """
    Compute the series of ln(1 + x).
    
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
    
    sum = 0
    
    taylor = lambda x, n: (-1)**(n - 1) * x ** n / n
    div_taylor = lambda x, n: (-1)**(n - 1) / (n * x**n)
    
    if abs(x) <= 1:
        for i in range(1, max_iter + 1):
            sum += taylor(x, i)
        
            if abs(sum - sum - taylor(x, i) / abs(sum)) < rel_tol: break
            if i > max_iter: break
        return sum, i
    
    if abs(x) > 1:
        sum = np.log(x)
        for i in range(1, max_iter + 1):
            sum += div_taylor(x, i)
            
            if abs(sum - sum - taylor(x, i) / abs(sum)) < rel_tol: break
            if i > max_iter: break
        return sum, i


# Boas, Problem 1.13.22
def boas_1_13_22(x, rel_tol = 1e-8, max_iter = 100):
    """Compute the sum of the series exp(x)/(1 - x)."""
    raise NotImplementedError("Student assignment not yet implemented.")


# Plots the first N terms of the series expansion of exp(x)/(1 - x)
def boas_1_13_22_plot(n_terms, filename=None):
    """Plot the first N terms of the series expansion of exp(x)/(1 - x)

    This function generates a plot similar to the one in Boas, Figure 1.13.1
    (but with all N approximations on a single plot).

    If filename is not None, save the generated figure to that filename.

    """
    raise NotImplementedError("Student assignment not yet implemented.")


# Boas, Problem 1.16.1c
def boas_1_16_1c(n_books_overhang):
    """Compute how many books can be stacked on a table with a given overhang."""
    raise NotImplementedError("Student assignment not yet implemented.")



# --- Landau --- #
# The following questions are from Landau 3.3.1
# HOWEVER, these should be completed with cos instead of sin
def cos_apprx(x, rel_tol = 1e-8, max_iter = 100):
    """Compute the approximation of cos(x) using the Taylor series expansion.

    This function computes the Taylor series of cos(x) until the series converges
    or maximum number of iterations is reached. The function returns the approximation
    of cos(x) and the number of iterations used to compute the approximation and makes
    use of the identity cos(x) = cos(x + 2*pi*n) for any integer n to reduce the input
    x to the range [0, 2*pi].

    """
    raise NotImplementedError("Student assignment not yet implemented.")
