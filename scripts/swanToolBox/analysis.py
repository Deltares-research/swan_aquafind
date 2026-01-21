from netCDF4 import Dataset
import datetime
import os
import numpy as np
import pandas as pd


def read_pnt(res_path, colnames, fname="POINTS.TAB"):

    dateparse = lambda x: datetime.datetime.strptime(x, "%Y%m%d.%H%M%S")
    pnt = pd.read_csv(
        os.path.join(res_path, fname),
        skiprows=7,
        header=None,
        delim_whitespace=True,
        names=colnames,
        index_col=False,
        parse_dates=["Time"],
        date_parser=dateparse,
    )
    return pnt


def compute_moment_tail(f, S, n):
    """compute contribution of the tail

    Args:
        f (array): 1d array with frequencies
        S (array): 1d array with f^n x E
        n (int): power in moment

    Returns:
        float: contribution of the tail
    """
    ## last value
    S_end = S[-1]
    f1 = f[-1]
    f2 = 100

    t = S_end / (n - 3) * f1**4 * (f2 ** (n - 3) - f1 ** (n - 3))
    return t


def compute_Hm0(E, f, delta_D, tail=True):
    """Compute significant wave height (Hm0)

    Args:
        E (array): 2d array with energy density. First dimension is the direction and second dimension is the frequency
        f (array): 1d array with frequency axis
        delta_D (float): directional resolution.
        tail (logical): logical indicating whether a tail with power 4 law should be included

    Returns:
        float: significant wave height
    """
    assert E.ndim == 2, "E does not have two dimensions"
    assert E.shape[1] == f.shape[0], "second dimension of E does not correspond to f"

    ## compute 1d energy density
    S = np.sum(E, axis=0) * delta_D
    if tail:
        t0 = compute_moment_tail(f, S, 0)
    else:
        t0 = 0
    m0 = np.trapz(S, f)
    Hm0 = 4 * np.sqrt(m0 + t0)
    return Hm0


def compute_Hswell(self, E, f, delta_D, f_cutoff=0.1, add_last_contribution=True):
    """Compute swell significant wave height (HSWELL)

    Args:
        E (array): 2d array with energy density. First dimension is the direction and second dimension is the frequency
        f (array): 1d array with frequency axis
        delta_D (float): directional resolution.
        add_last_contribution(logical): logical indicating whether the energy between fcutoff and the last frequency smaller than fcutoff is added

    Returns:
        float: significant swell wave height
    """
    assert E.ndim == 2, "E does not have two dimensions"
    assert f_cutoff < f[-1], "f_cutoff is larger than the last frequency value"
    assert E.shape[1] == f.shape[0], "second dimension of E does not correspond to f"
    index = f <= f_cutoff
    E_dummy = np.sum(E, axis=0) * delta_D

    last_index = np.where(index)[0][-1]
    df_last = f_cutoff - f[last_index]
    if add_last_contribution and len(f) > last_index + 1:
        E_estimate = np.interp(
            f_cutoff,
            [f[last_index], f[last_index + 1]],
            [E_dummy[last_index], E_dummy[last_index + 1]],
        )
        m0_last = df_last * 0.5 * (E_estimate + E_dummy[last_index])
    else:
        m0_last = 0

    hswell = 4 * np.sqrt(np.trapz(E_dummy[index], f[index]) + m0_last)
    return hswell


def compute_Tmm10(self, E, f, delta_D, tail=True):
    """Compute spectral period (tmm10)

    Args:
        E (array): 2d array with energy density. First dimension is the direction and second dimension is the frequency
        f (array): 1d array with frequency axis
        delta_D (float): directional resolution.
        tail(logical): logical indicating whether a tail with power 4 law should be included

    Returns:
        float: Mean absolute wave period
    """
    assert E.ndim == 2, "E does not have two dimensions"
    assert E.shape[1] == f.shape[0], "second dimension of E does not correspond to f"

    ## compute 1d energy density
    S0 = np.sum(E, axis=0) * delta_D
    m0_part1 = np.trapz(S0, f)
    Sm1 = np.sum(E * f**-1, axis=0) * delta_D
    mm1_part1 = np.trapz(Sm1, f)
    ## first and last frequency for tail
    if tail:
        t0 = compute_moment_tail(f, S0, 0)
        tm1 = compute_moment_tail(f, Sm1, -1)
    else:
        t0 = 0
        tm1 = 0

    tmm10 = (mm1_part1 + tm1) / (m0_part1 + t0)
    return tmm10


def compute_Tm02(E, f, delta_D, tail=True):
    """Compute wave period (tm02)

    Args:
        E (array): 2d array with energy density. First dimension is the direction and second dimension is the frequency
        f (array): 1d array with frequency axis
        delta_D (float): directional resolution.
        tail(logical): logical indicating whether a tail with power 4 law should be included

    Returns:
        float: Wave period
    """
    assert E.ndim == 2, "E does not have two dimensions"
    assert E.shape[1] == f.shape[0], "second dimension of E does not correspond to f"

    ## compute 1d energy density
    S0 = np.sum(E, axis=0) * delta_D
    m0_part1 = np.trapz(S0, f)
    S2 = np.sum(E * f**2, axis=0) * delta_D
    m2_part1 = np.trapz(S2, f)
    if tail:
        t0 = compute_moment_tail(f, S0, 0)
        t2 = compute_moment_tail(f, S2, 2)
    else:
        t0 = 0
        t2 = 0

    tm02 = ((m2_part1 + t2) / (m0_part1 + t0)) ** -0.5

    return tm02


def compute_Tm01(E, f, delta_D, tail=True):
    """Compute wave period (tm01)

    Args:
        E (array): 2d array with energy density. First dimension is the direction and second dimension is the frequency
        f (array): 1d array with frequency axis
        delta_D (float): directional resolution.
        tail(logical): logical indicating whether a tail with power 4 law should be included

    Returns:
        float: Wave period
    """
    assert E.ndim == 2, "E does not have two dimensions"
    assert E.shape[1] == f.shape[0], "second dimension of E does not correspond to f"

    ## compute 1d energy density
    S0 = np.sum(E, axis=0) * delta_D
    m0_part1 = np.trapz(S0, f)
    S1 = np.sum(E * f, axis=0) * delta_D
    m1_part1 = np.trapz(S1, f)
    ## first and last frequency for tail
    f1 = f[-1]
    f2 = 100
    if tail:
        t0 = compute_moment_tail(f, S0, 0)
        t1 = compute_moment_tail(f, S1, 1)
    else:
        t0 = 0
        t1 = 0

    tm01 = ((m1_part1 + t1) / (m0_part1 + t0)) ** -1

    return tm01


def compute_Dir(E, f, D, delta_D):
    """Compute Wave direction (Dir)

    Args:
        E (array): 2d array with energy density. First dimension is the direction and second dimension is the frequency
        f (array): 1d array with frequency axis
        D (array): 1d array with direction axis
        delta_D (float): directional resolution.

    Returns:
        float: Wave direction
    """
    assert E.ndim == 2, "E does not have two dimensions"
    assert E.shape[1] == f.shape[0], "second dimension of E does not correspond to f"
    assert E.shape[0] == D.shape[0], "first dimension of E does not correspond to D"

    dir = np.rad2deg(
        np.arctan2(
            np.trapz(np.sum(np.sin(np.deg2rad(D)) * E.T, axis=1) * delta_D, f),
            np.trapz(np.sum(np.cos(np.deg2rad(D)) * E.T, axis=1) * delta_D, f),
        )
    )

    return dir
