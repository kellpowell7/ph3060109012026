"""Student assignment implementation file for complex analysis."""
import numpy as np
import matplotlib.pyplot as plt


# --- Student Assignment --- #
# --- General --- #
def complex_polar(z):
    """Convert a complex number to polar form.

    Parameters
    ----------
    z: np.complex128
        Complex number for the function to evaluate.
    
    Returns
    -------
    r: float
        The length of the vector specified.
    theta: float
        The angle of the vector, counterclockwise from x axis in radians.
    """
    
    r = np.abs(z)
    theta = np.angle(z)
    
    return r, theta


def nth_root(z, n):
    """Compute the n-th roots of a complex number.

    Parameters
    ----------
    z: np.complex128
        Complex number for the function to evaluate.
    n: int
        The amount of roots to be found.
    
    Returns
    -------
    roots: tuple
        A list of all nth roots of the complex number. 
    """
    
    r = np.abs(z)
    theta = np.angle(z)

    k = np.arange(n)
    angles = (theta + 2 * np.pi * k) / n
    roots = (r**(1 / n)) * (np.cos(angles) + 1j * np.sin(angles))
        
    return roots


# --- Boas --- #
def complex_impedance(resistance, inductance, capacitance, omega):
    """Compute the complex impedance of a series RLC circuit.

    Parameters
    ----------
    resistance: float
        The real portion of resistance within the circuit.
    inductance: float
        The inductance of the circuit.
    capacitance: float
        The capacitance of the circuit.
    omega: float
        The frequency of the current within the circuit.
    
    Returns
    -------
    z: np.complex128
        The complex impedence of the circuit itself.
    """
    
    z = resistance + 1j * (omega*inductance - 1 / (omega * capacitance))
    return z


# See Boas Example 2.16 - Electricity
def plot_rlc(resistance, inductance, capacitance, omega, time, max_current, filename=None):
    """Plot the current and voltage time series of a series RLC circuit.

    Parameters
    ----------
    resistance: float
        The resistance of the circuit.
    inductance: float
        The inductance of the circuit.
    capacitance: float
        The capacitance of the circuit.
    omega: float
        The angular frequency of the circuit's signal.
    time: float
        How long the time series should run.
    max_current: float
        The maximum amperage allowed within the function.
    filename: string
        The intended name for the output plot. 
    """
    xl = omega * inductance
    xc = 1 / (omega * capacitance)
    impedance = np.sqrt(resistance**2 + (xl - xc)**2)

    max_voltage = max_current * impedance

    phi = np.arctan2((xl - xc), resistance)

    current = max_current * np.cos(omega * time)
    voltage = max_voltage * np.cos(omega * time + phi)

    fig, ax = plt.subplots()
    ax.plot(time, current)
    ax.plot(time, voltage)

    fig.savefig(filename)
    plt.show()

    return fig
