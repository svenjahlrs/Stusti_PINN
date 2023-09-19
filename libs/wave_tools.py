from scipy.optimize import root
import numpy as np
import cmath

def convert_to_wavevector(H, x, t_inc, omega, kp, bool=True, shift=0):
    A = np.zeros(len(H), dtype=complex)
    print(len(A))
    for t in range(0, len(H)):
        A[t] = H[t] * cmath.exp(+1j * (kp * x - omega * t * t_inc+shift))
        if bool == False:
           A[t] = H[t] * cmath.exp(-1j * (kp * x - omega * t * t_inc+shift))
    return A


def wavevector_to_envelope(A, x_grid, t_grid, omega, kp):
    H = np.zeros(A.shape, dtype=complex)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            H[i, j] = A[i, j] * cmath.exp(-1j * (kp * x_grid[j] - omega * t_grid[i]))
            #H[i, j] = A[i, j] * cmath.exp(+1j * (kp * x_grid[j] - omega * t_grid[i]))
    return H

def dispersion(k, *args):
    """
    definition of the rearranged dispersion relation f(k) = omega^2 - k * g * tanh(k * d) (=!0)
    :param k: wavenumber
    :param args: omega = angular frequency [rad/s], d = water depth [m], g = gravitational acceleration constant [m/s^2]
    :return: f(k_p)
    """
    omega = args[0]['omega']
    g = args[0]['g']
    d = args[0]['d']

    return np.square(omega) - k * g * np.tanh(k * d)


def wavenumber(omega, d, g=9.81):
    """
    function for calculating the wavenumber k associated to a given angular frequency omega and water depth d using the dispersion relation for finite water depth
    :param omega: angular frequency [rad/s]
    :param d: water depth [m]
    :param g: gravitational acceleration constant [m/s^2]
    :return: wavenumber k [1/m]
    """
    k_guess = np.divide(np.square(omega), g)  # initial guess for k from deepwater dispersion relation omega^2 = k * g
    k_return = root(dispersion, x0=k_guess, args={'omega': omega, 'd': d, 'g': g})  # find roots of rearraged dispersion relation
    if k_return.success:
        k = k_return.x[0]

    return k


def NLSE_coefficients_marco(omega_p, d, g=9.81):
    k_p = wavenumber(omega=omega_p, d=d)  # calculates the wavenumber to angular frequency using the dispersion relation

    nu = 1 + np.divide(2 * k_p * d, np.sinh(2 * k_p * d))

    C_g = np.divide(omega_p, 2 * k_p) * nu
    alpha = - np.square(nu) + 2 + 8 * np.square(k_p * d) * np.divide(np.cosh(2 * k_p * d),
                                                                     np.square(np.sinh(2 * k_p * d)))
    alpha_ = np.divide(omega_p * alpha, 8 * np.square(k_p) * np.power(C_g, 3))

    beta = np.divide(np.cosh(4 * k_p * d) + 8 - 2 * np.square(np.tanh(k_p * d)), 8 * np.power(np.sinh(k_p * d), 4)) - np.divide(np.square(2 * np.square(np.cosh(k_p * d)) + 0.5 * nu),np.square(np.sinh(2 * k_p * d)) * (np.divide(k_p * d, np.tanh(k_p * d)) - np.divide(np.square(nu), 4)))

    beta_ = np.divide(omega_p * np.square(k_p) * beta, 2 * C_g)

    return k_p, C_g, alpha_, beta_


def NLSE_coefficients_chabchoub(omega_p, d, g=9.81):
    """calculates the NLSE coefficients for time-like und space-like form of NLSE O(eps^3) according to Chabchoub2016 https://www.mdpi.com/2311-5521/1/3/23
    :param omega: angular frequency [rad/s]
    :param d: water depth [m]
    :param g: gravitational acceleration constant [m/s^2]
    :return peak wavenumber k_p, group velocity C_g, coefficients for space-like form lamb und mu, coefficients for time-like-form delta und nu"""
    k_p = wavenumber(omega=omega_p, d=d)  # calculates the wavenumber to angular frequency using the dispersion relation

    C_g = np.divide(omega_p, 2 * k_p)
    lamb = - np.divide(omega_p, 8*np.square(k_p))
    mu = - np.divide(omega_p*np.square(k_p), 2)
    delta = -1/g
    nu = -k_p**3

    return k_p, C_g, lamb, mu, delta, nu

print(wavenumber(omega=3,d=1))