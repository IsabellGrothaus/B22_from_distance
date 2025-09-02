"""b22_from_distances.py — Tools for computing and visualizing the second osmotic virial coefficient (B22)

Copyright (c) 2025

Authors
-------
- Jürgen Köfinger <Juergen.Koefinger@biophys.mpg.de>
- Isabell Louise Grothaus <grothaus@uni-bremen.de>
- Contributors welcome via Pull Requests

Project
-------
This module provides utilities to compute B22 from distance distributions (e.g. from conventional MD simulations, umbrella or replica-exchange sampling) 
by block-averaging and uncertainty estimation, plus plotting helpers.

Expected `data` layout
----------------------
`data` is a mapping with (at least) these keys:
- data['distance']: np.ndarray shaped (N, M>=2). Column 0 is the time and column 1 is the distance r (nm) for system `key`.
- data['label']: label of data inferred from directory name
- data['path']: directory path + label
- data['temperature']: simulation temperature in Kelvin
- data['is_restrained']: if molecules are restrained in x, y - True or False
- data['k_in_kJ_per_M_per_nm2']: spring constant in kJ/M/nm^2 used for biasing
- data['k_in_kbT'][key]: float or array-like; spring constant in kBT/nm^2 used for biasing.

Public API
----------
- block_averaging
- get_blocked_std_fixed_M
- calc_plot_FEP
- plot_B22_for_R1_range
- calculate_B22_for_R1_range
- calculate_B22_for_R1_range_convergence
- calculate_and_plot_B22
- get_k
- gamma
- get_B22_from_pv
- get_indicator
- get_weights
- apply_pbcs
- get_corrected_pv

License
-------
GNU GENERAL PUBLIC LICENSE
"""

from __future__ import annotations

from itertools import groupby
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.special import dawsn

__all__ = [
    "block_averaging",
    "get_blocked_std_fixed_M",
    "calc_plot_FEP",
    "plot_B22_for_R1_range",
    "calculate_B22_for_R1_range",
    "calculate_B22_for_R1_range_convergence",
    "calculate_and_plot_B22",
    "get_k",
    "gamma",
    "get_B22_from_pv",
    "get_indicator",
    "get_weights",
    "apply_pbcs",
    "get_corrected_pv",
    "kb",
    "T",
    "unit_conv_fac",
]

__version__ = "0.1.0"
__author__ = "Jürgen Köfinger, Isabell Grothaus and contributors"

# Physical constants and default settings
kb: float = 0.008314462618  # kJ/(mol*K)
T: float = 300.0  # K
unit_conv_fac: float = 1e-24 * (6.022e23)  # nm^3 -> L/mol

# ---------------------------
# Core computations & helpers
# ---------------------------

def block_averaging(
    data: np.ndarray,
    weights: np.ndarray,
    R1: float,
    R2: float,
    n_blocks: int,
) -> np.ndarray:
    """Block-averaging of B22 for a fixed number of blocks.

    Parameters
    ----------
    data : np.ndarray
        1D array of distances (nm). Only the distances column should be passed in here.
    weights : np.ndarray
        Statistical weights associated with each distance sample (sum to 1 over ``data``).
    R1 : float
        Inner radius (nm) for defining the sub-volume.
    R2 : float
        Outer cutoff radius (nm). Used to compute dv = V(R2) - V(R1).
    n_blocks : int
        Number of contiguous blocks used for averaging.

    Returns
    -------
    np.ndarray
        Array of per-block B22 values (nm^3).
    """
    block_size = len(data) // n_blocks
    if block_size == 0:
        raise ValueError("n_blocks is too large relative to data length")

    B22_blocks: List[float] = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_distances = data[start_idx:end_idx]
        block_weights = weights[start_idx:end_idx]

        pv_block = get_corrected_pv(block_distances, block_weights, [R1])
        if pv_block == 1:
            pv_block = 0.9999  # numerical guard against division by zero

        v = 4.0 * np.pi * R1 ** 3 / 3.0
        V = 4.0 * np.pi * R2 ** 3 / 3.0
        dv = V - v

        B22_block = get_B22_from_pv(pv_block, v, dv)
        B22_blocks.append(B22_block)

    return np.asarray(B22_blocks)


def get_blocked_std_fixed_M(B22_blocks: np.ndarray, M: int) -> Tuple[float, float, float]:
    """Compute mean, standard deviation, and SEM of block-averaged B22.

    Parameters
    ----------
    B22_blocks : np.ndarray
        Per-block B22 values (nm^3).
    M : int
        Number of blocks.

    Returns
    -------
    (mean, std, sem) : tuple of floats
        Mean B22, population std over blocks, and standard error of the mean (all in nm^3).
    """
    mean = float(B22_blocks.mean())
    var = float(((B22_blocks - mean) ** 2).sum() / float(M))  # population variance
    std = float(np.sqrt(var))
    bse = std / np.sqrt(M)
    return mean, std, float(bse)


# ---------------------------
# Plotting utilities
# ---------------------------

def calc_plot_FEP(
    data: Dict,
    key: str,
    fit_range_lower: float,
    fit_range_upper: float,
    threshold: float,
    max_gap: float,
    color: str,
    fontsize: int,
) -> List[Tuple[float, float]]:
    """Calculate and plot the free energy profile from distance data.

    Identifies regions where the free energy is within ``threshold`` of the
    average free energy in the interval [fit_range_lower, fit_range_upper].
    Neighboring regions separated by gaps <= ``max_gap`` are merged.

    Returns the list of merged (start_x, end_x) regions.
    """
    P, bin_edges = np.histogram(
        data["distance"][key][:, 1],
        bins=260,
        range=(0, float(data["distance"][key][:, 1].max())),
        density=True,
    )
    F = -kb * T * np.log(P)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = (bin_centers >= fit_range_lower) & (bin_centers <= fit_range_upper)
    F_avg = np.mean(F[mask])
    F_shifted = F - F_avg

    within_band = np.abs(F_shifted) < threshold

    # Step 1: contiguous subregions within threshold
    raw_ranges: List[Tuple[float, float]] = []
    for kflag, g in groupby(enumerate(within_band), key=lambda x: x[1]):
        if kflag:
            group = list(g)
            start_idx = group[0][0]
            end_idx = group[-1][0]
            raw_ranges.append((bin_centers[start_idx], bin_centers[end_idx]))

    # Step 2: merge close subregions
    merged_ranges: List[Tuple[float, float]] = []
    if raw_ranges:
        current_start, current_end = raw_ranges[0]
        for i in range(1, len(raw_ranges)):
            next_start, next_end = raw_ranges[i]
            if next_start - current_end <= max_gap:
                current_end = next_end
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_ranges.append((current_start, current_end))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(bin_centers, F_shifted, lw=3, color=color)
    for (start_x, end_x) in merged_ranges:
        ax.axvspan(start_x, end_x, color=color, alpha=0.3)
    ax.set_ylabel("Free energy [kJ/mol]", fontsize=fontsize)
    ax.set_xlabel("r [nm]", fontsize=fontsize)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.tick_params(labelsize=fontsize)

    # Inset: 1/3 width of main plot
    axins = inset_axes(ax, width="33%", height="33%", loc="upper right", borderpad=1)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    axins.plot(bin_centers, F_shifted, lw=2, color=color)
    for (start_x, end_x) in merged_ranges:
        axins.axvspan(start_x, end_x, color=color, alpha=0.3)

    if merged_ranges:
        region_start = min(r[0] for r in merged_ranges)
        region_end = max(r[1] for r in merged_ranges)
        padding = 0.05 * (region_end - region_start)
        axins.set_xlim(region_start - padding, region_end + padding)

    axins.set_ylim(-1, 1)
    axins.axhline(0, color="black", linewidth=1, linestyle="-")
    axins.tick_params(labelsize=fontsize - 2)

    if merged_ranges:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.2)

    plt.show()
    return merged_ranges


def calculate_B22_for_R1_range(
    data: Dict,
    R1_range: Sequence[float],
    R2: Dict[str, float],
    n_blocks: int,
    key: str,
) -> pd.DataFrame:
    """Compute B22 vs. R1 for a fixed R2 for the given system key.

    Returns
    -------
    pandas.DataFrame with columns:
    ['Key', 'R1 (nm)', 'R2 (nm)', 'B22 mean (l/mol)', 'B22 mean (nm³)',
     'B22_st_error (nm³)', 'B22_st_error (l/mol)']
    """
    results: List[Dict] = []
    distances = apply_pbcs(data["distance"][key], R2[key])
    w = get_weights(distances, data["k_in_kbT"][key])

    for R1 in R1_range:
        B22_blocks = block_averaging(distances, w, float(R1), R2[key], n_blocks)
        B22_average, B22_std, B22_st_error = get_blocked_std_fixed_M(B22_blocks, n_blocks)
        B22_in_l_per_mol = B22_average * unit_conv_fac
        B22_st_error_in_l_per_mol = B22_st_error * unit_conv_fac
        results.append(
            {
                "Key": key,
                "R1 (nm)": float(R1),
                "R2 (nm)": float(R2[key]),
                "B22 mean (l/mol)": B22_in_l_per_mol,
                "B22 mean (nm³)": B22_average,
                "B22_st_error (nm³)": B22_st_error,
                "B22_st_error (l/mol)": B22_st_error_in_l_per_mol,
            }
        )

    return pd.DataFrame(results)


def calculate_B22_for_R1_range_convergence(
    data: Dict,
    R1_range: Sequence[float],
    R2: float,
    n_blocks: int,
    key: str,
    window: int,
) -> pd.DataFrame:
    """Compute B22 vs. R1 using only the first ``window`` samples (convergence over time).

    Parameters
    ----------
    window : int
        Number of frames/samples from the beginning to include.
    """
    results: List[Dict] = []

    distances = apply_pbcs(data["distance"][key][:window], R2)
    w = get_weights(distances, data["k_in_kbT"][key])

    for R1 in R1_range:
        B22_blocks = block_averaging(distances, w, float(R1), R2, n_blocks)
        B22_average, B22_std, B22_st_error = get_blocked_std_fixed_M(B22_blocks, n_blocks)
        B22_in_l_per_mol = B22_average * unit_conv_fac
        B22_st_error_in_l_per_mol = B22_st_error * unit_conv_fac
        results.append(
            {
                "Key": key,
                "R1 (nm)": float(R1),
                "R2 (nm)": float(R2),
                "B22 mean (l/mol)": B22_in_l_per_mol,
                "B22 mean (nm³)": B22_average,
                "B22_st_error (nm³)": B22_st_error,
                "B22_st_error (l/mol)": B22_st_error_in_l_per_mol,
            }
        )

    return pd.DataFrame(results)


def plot_B22_for_R1_range(
    data: Dict,
    R1_range: Sequence[float],
    R2: Dict[str, float],
    key: str,
    n_blocks: int = 5,
    color: str = "black",
    fontsize: int = 16,
    ylim_min: float | None = None,
    ylim_max: float | None = None,
) -> None:
    """Plot B22 values for a range of R1 values at fixed R2.
    Y-axis is shown both in nm^3 and L/mol via a twin axis.
    """
    df = calculate_B22_for_R1_range(data, R1_range, R2, n_blocks, key)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300, sharex=True)
    ax1.bar(
        R1_range,
        height=df["B22 mean (l/mol)"],
        yerr=df["B22_st_error (l/mol)"],
        capsize=10,
        linewidth=2,
        width=0.08,
        color=color,
        alpha=0.80,
    )
    ax1.set_xlabel(r"$\mathrm{R1}$ [nm]", fontsize=fontsize)
    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize=fontsize)

    if ylim_min is None:
        ylim_min = float(min(0, df["B22 mean (l/mol)"].min() - 1.5 * df["B22_st_error (l/mol)"].max()))
    if ylim_max is None:
        ylim_max = float(df["B22 mean (l/mol)"].max() + 1.5 * df["B22_st_error (l/mol)"].max())
    ax1.set_ylim(ylim_min, ylim_max)

    ax1a = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize=fontsize)

    plt.axhline(y=0, color="black", linestyle="--")
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", which="major", labelsize=fontsize)
    ax1a.tick_params(axis="both", which="major", labelsize=fontsize)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    plt.show()


def calculate_and_plot_B22(
    data: Dict,
    keys: Sequence[str],
    R1_ranges: Dict[str, Sequence[float]],
    colors: Sequence[str],
    labels: Sequence[str],
    ylim: Tuple[float, float],
    fontsize: int = 16,
) -> Tuple[List[float], List[float]]:
    """Compute weighted B22 across provided R1 ranges for each key and plot as bars.

    Returns
    -------
    (means, stderrs) in L/mol.
    """
    B22_weighted_list: List[float] = []
    B22_weighted_stderr_list: List[float] = []
    R2: Dict[str, float] = {}

    for key, c in zip(keys, colors):
        R1_range = R1_ranges[key]
        R2[key] = float(data["distance"][key][:, 1].max() * 0.9)

        df = calculate_B22_for_R1_range(data, R1_range, R2, n_blocks=5, key=key)
        weights = 1.0 / (df["B22_st_error (l/mol)"] ** 2)
        weighted_mean = float(np.sum(weights * df["B22 mean (l/mol)"]) / np.sum(weights))
        weighted_stderr = float(np.sqrt(1.0 / np.sum(weights)))
        B22_weighted_list.append(weighted_mean)
        B22_weighted_stderr_list.append(weighted_stderr)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300, sharex=True)
    ax1.bar(
        range(len(keys)),
        height=B22_weighted_list,
        yerr=B22_weighted_stderr_list,
        capsize=10,
        linewidth=2,
        width=0.7,
        color=colors,
    )
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor", fontsize=fontsize)
    ax1.set_xlabel("Water model", fontsize=fontsize)
    ax1.set_ylabel(r"$\bar B_\mathrm{22}$ [nm$^3$]", fontsize=fontsize)
    ax1.set_ylim(*ylim)

    ax1a = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
    ax1a.set_ylabel(r"$\bar B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize=fontsize)

    plt.axhline(y=0, color="black", linestyle="--")
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", which="major", labelsize=fontsize)
    ax1a.tick_params(axis="both", which="major", labelsize=fontsize)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    plt.show()
    return B22_weighted_list, B22_weighted_stderr_list

# ---------------------------
# Physics helpers
# ---------------------------

def get_k(k_in_kj_per_mol_per_nm2: float, T: float = 298.0) -> float:
    """Convert k from kJ/mol/nm^2 to kBT/nm^2 at temperature ``T``.
    Uses 1 kBT = 2.479 kJ/mol at 298 K.
    """
    k_in_kbT_per_nm2 = k_in_kj_per_mol_per_nm2 / 2.479 * (298.0 / T)
    return float(k_in_kbT_per_nm2)


def gamma(r: np.ndarray, k: float) -> np.ndarray:
    """Distance distribution factor for a harmonic constraint in (x, y).

    Parameters
    ----------
    r : array-like
        Distances (nm).
    k : float
        Spring constant in kBT/nm^2.
    """
    f = 1.0
    return r * np.sqrt(2.0) * np.sqrt(k) * dawsn(f * np.sqrt(k) * r / np.sqrt(2.0))


def get_B22_from_pv(pv: float, v: float, dv: float) -> float:
    """Compute B22 (nm^3) from the sub-volume probability ``pv``.

    Parameters
    ----------
    pv : float
        Probability of finding a particle within sub-volume V(R1).
    v : float
        Volume of inner sphere V(R1).
    dv : float
        Shell volume V(R2) - V(R1).
    """
    return float(-v / 2.0 * (pv / (1.0 - pv) * dv / v - 1.0))


def get_indicator(energies: np.ndarray, energy_cutoff: float) -> np.ndarray:
    """Indicator function 1[ energy < cutoff ]."""
    return np.where(energies < energy_cutoff, 1.0, 0.0)


def get_weights(distances: np.ndarray, spring_constant: float) -> np.ndarray:
    """Unbiasing weights for distances under harmonic bias.

    Parameters
    ----------
    distances : np.ndarray
        Distances r (nm).
    spring_constant : float
        Bias strength in kBT/nm^2.

    Returns
    -------
    np.ndarray
        Normalized weights (sum to 1).
    """
    g = gamma(distances, float(spring_constant))
    w = distances ** 2 / g
    w = w / w.sum()
    return w


def apply_pbcs(data: np.ndarray, r_max: float, ind_dist: int = 1, ind_ener: int = 2) -> np.ndarray:
    """Apply spherical cutoff (PBC-like) and return distances < r_max.

    Parameters
    ----------
    data : np.ndarray
        Array where column ``ind_dist`` is the distance r.
    r_max : float
        Maximum distance to retain (nm).
    ind_dist : int
        Column index containing r.
    ind_ener : int
        Unused (kept for back-compatibility).
    """
    distances = data[:, ind_dist]
    ix = np.where(distances < r_max)[0]
    return distances[ix]


def get_corrected_pv(distances: np.ndarray, w: np.ndarray, R1_list: Sequence[float]) -> float | np.ndarray:
    """Weighted probability of being inside spheres of radii in ``R1_list``.

    If a single radius is given, returns the scalar probability for that R1.
    Otherwise returns an array with columns [R1, pv].
    """
    pv_corr = []
    for R1 in R1_list:
        iv = get_indicator(distances, float(R1))
        pv = float(np.average(iv, weights=w))
        pv_corr.append([float(R1), pv])

    pv_corr = np.asarray(pv_corr)
    if len(R1_list) > 1:
        return pv_corr
    else:
        return float(pv_corr[0, 1])


















#####################################################################

#import numpy as np
#import pandas as pd
#import scipy.special
#import matplotlib.pyplot as plt
#import scipy
#import pandas as pd
#import pickle as pkl
#from itertools import groupby
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#import matplotlib.ticker as ticker
#
#
##Constants
#kb=0.008314462618
#T=300
#unit_conv_fac=1e-24*(6.022e23)  #conversion factor from nm^3 to liter
#
#def block_averaging(data, weights, R1, R2, n_blocks):
#    """
#    Block averaging of B22 values for fixed number of blocks.
#    """
#    block_size = len(data) // n_blocks
#    B22_blocks = []
#
#    for i in range(n_blocks):
#        start_idx = i * block_size
#        end_idx = start_idx + block_size
#        block_distances = data[start_idx:end_idx]
#        block_weights = weights[start_idx:end_idx]
#
#        pv_block = get_corrected_pv(block_distances, block_weights, [R1])
#        if pv_block == 1:
#            pv_block = 0.9999
#        else:
#            pv_block = pv_block
#        v = 4 * np.pi * R1**3 / 3.
#        V = 4 * np.pi * R2**3 / 3.
#        dv = V - v
#
#        B22_block = get_B22_from_pv(pv_block, v, dv)
#        B22_blocks.append(B22_block)
#
#    B22_blocks = np.array(B22_blocks)
#
#    return B22_blocks                   #returns B22 for every block
#
#def get_blocked_std_fixed_M(B22_blocks,M):                      
#    """
#    B22_blocks: Observable (in this case: block averages of B22 for every block)
#    M: Numbers of blocks
#    Returns: Estimates B22 mean, standard deviation and standard error of the mean for a given block length/block number
#    """
#    
#    mean=B22_blocks.mean()                        #this line calculates the mean value over all block mean values
#    var=((B22_blocks-mean)**2).sum()/float(M)       #calculation of variance for overall population (in this case: calculates the variability between blocks) (Calculation differs from the calculation in the function above, in the function "get_bootstrap_error" the sample variance is used)
#    std=np.sqrt(var)                                 # Calculation of standard deviation as square root of variance
#    bse = std/np.sqrt(M)                  # Calculation of standard error of the mean (= standard deviation of block means / squareroot(number of blocks))
#    return mean, std, bse
#
#def calc_plot_FEP(data, key, fit_range_lower, fit_range_upper, threshold, max_gap, color, fontsize):
#    '''
#    Calculate and plot free energy profile from distance data.
#    Identify regions where free energy is within a certain threshold of the average free energy in a specified fit range.
#    Merge regions that are separated by gaps smaller than max_gap.  
#    Plot the free energy profile and highlight the identified regions.
#    Include an inset zoomed plot in the upper-right corner with connecting lines.
#    '''
#
#    # Free energy
#    P, bin_edges = np.histogram(
#        data['distance'][key][:,1],
#        bins=130*2,
#        range=(0, data['distance'][key][:,1].max()),
#        density=True
#    )
#    F = -kb * T * np.log(P)
#    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#    mask = (bin_centers >= fit_range_lower) & (bin_centers <= fit_range_upper)
#    F_avg = np.mean(F[mask])
#    F_shifted = F - F_avg
#
#    within_band = np.abs(F_shifted) < threshold
#
#    # Step 1: find all subregions
#    raw_ranges = []
#    for k, g in groupby(enumerate(within_band), key=lambda x: x[1]):
#        if k:
#            group = list(g)
#            start_idx = group[0][0]
#            end_idx = group[-1][0]
#            start_x = bin_centers[start_idx]
#            end_x = bin_centers[end_idx]
#            raw_ranges.append((start_x, end_x))
#
#    # Step 2: merge subregions with gaps ≤ max_gap
#    merged_ranges = []
#    if raw_ranges:
#        current_start, current_end = raw_ranges[0]
#        for i in range(1, len(raw_ranges)):
#            next_start, next_end = raw_ranges[i]
#            if next_start - current_end <= max_gap:
#                current_end = next_end  # extend current region
#            else:
#                merged_ranges.append((current_start, current_end))
#                current_start, current_end = next_start, next_end
#        merged_ranges.append((current_start, current_end))  # add the last one
#
#    # Main plot   
#    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
#    ax.plot(bin_centers, F_shifted, lw=3, color=color)
#    for (start_x, end_x) in merged_ranges:
#        ax.axvspan(start_x, end_x, color=color, alpha=0.3)
#    ax.set_ylabel("Free energy [kJ/mol]", fontsize=fontsize)
#    ax.set_xlabel("r [nm]", fontsize=fontsize)
#    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
#    ax.tick_params(labelsize=fontsize)
#
#    # Inset: 1/3 width of main plot
#    axins = inset_axes(ax, width="33%", height="33%", loc='upper right',
#                       borderpad=1)  # small padding
#    axins.set_facecolor("white")     # solid white background
#    axins.spines[:].set_visible(True) # make sure all spines (box) are visible
#    for spine in axins.spines.values():
#        spine.set_edgecolor("black")
#        spine.set_linewidth(1.5)
#
#    # Plot inside inset
#    axins.plot(bin_centers, F_shifted, lw=2, color=color)
#    for (start_x, end_x) in merged_ranges:
#        axins.axvspan(start_x, end_x, color=color, alpha=0.3)
#
#    # Zoom to cover all merged regions (min start → max end, with padding)
#    if merged_ranges:
#        region_start = min(r[0] for r in merged_ranges)
#        region_end   = max(r[1] for r in merged_ranges)
#        padding = 0.05 * (region_end - region_start)
#        axins.set_xlim(region_start - padding, region_end + padding)
#
#    axins.set_ylim(-1, 1)
#    axins.axhline(0, color='black', linewidth=1, linestyle='-')
#    axins.tick_params(labelsize=fontsize-2)
#
#    # Draw connectors between inset and main plot
#    if merged_ranges:
#        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.2)
#
#    plt.show()
#
#    return merged_ranges
#
#def plot_B22_for_R1_range(data, R1_range, R2, key, n_blocks = 5, color = "black", fontsize = 16, ylim_min = None, ylim_max = None):
#    """
#    Plot B22 values for a range of R1 values at fixed R2.
#    """
#    df = calculate_B22_for_R1_range(data, R1_range, R2, n_blocks, key)
#    fig, ax1 = plt.subplots(figsize=(10,6), dpi=300 ,sharex=True) 
#    bars = ax1.bar(R1_range, height=df['B22 mean (l/mol)'], bottom=0, yerr=df['B22_st_error (l/mol)'], capsize = 10, linewidth=2, width=0.08,
#        color=color,  # visible fill
#        alpha=0.80)
#    ax1.set_xlabel(r"$\mathrm{R1}$ [nm]", fontsize = fontsize)
#    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize = fontsize)
#    if ylim_min is None:
#        ylim_min = min(0, df['B22 mean (l/mol)'].min() - 1.5 * df['B22_st_error (l/mol)'].max())
#    if ylim_max is None:
#        ylim_max = df['B22 mean (l/mol)'].max() + 1.5 * df['B22_st_error (l/mol)'].max()
#    ax1.set_ylim(ylim_min, ylim_max)
#    ax1a = ax1.twinx()
#    mn, mx = ax1.get_ylim()
#    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
#    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize = fontsize)
#    plt.axhline(y=0, color='black', linestyle='--')
#    ax1.set_axisbelow(True)
#    ax1.tick_params(axis='both', which='major', labelsize = fontsize)
#    ax1a.tick_params(axis='both', which='major', labelsize = fontsize)
#    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#    plt.show()
#
#def calculate_B22_for_R1_range(data, R1_range, R2, n_blocks, key):
#    """
#    Calculate B22 values for a range of R1 values at fixed R2.
#    """
#    results = []
#    for R1 in R1_range:
#        distances = apply_pbcs(data['distance'][key], R2[key])  # return distances and corresponding energies when distances are smaller than R2
#        w = get_weights(distances, data['k_in_kbT'][key])  # calculate weights from gamma distribution
#        B22_blocks = block_averaging(distances, w, R1, R2[key], n_blocks) #calculate B22 values for every block via block averaging for the distances
#        B22_average, B22_std, B22_st_error = get_blocked_std_fixed_M(B22_blocks, n_blocks) 
#        B22_in_l_per_mol = B22_average * (1e-24 * (6.022e23))
#        B22_st_error_in_l_per_mol = B22_st_error * (1e-24 * (6.022e23))
#        results.append({
#            'Key': key, 
#            'R1 (nm)': R1,
#            'R2 (nm)': R2[key],
#            'B22 mean (l/mol)': B22_in_l_per_mol,
#            'B22 mean (nm³)': B22_average,
#            'B22_st_error (nm³)': B22_st_error,
#            'B22_st_error (l/mol)': B22_st_error_in_l_per_mol 
#        }) 
#                
#    B22_over_R1_df = pd.DataFrame(results)
#    return B22_over_R1_df
#
#def calculate_B22_for_R1_range_convergence(data, R1_range, R2, n_blocks, key, window):
#    """
#    Calculate B22 values for a range of R1 values at fixed R2 with convergence over time.
#    """
#    results = []
#    for R1 in R1_range:
#        distances = apply_pbcs(data['distance'][key][0:window], R2)  # return distances and corresponding energies when distances are smaller than R2
#        w = get_weights(distances, data['k_in_kbT'][key])  # calculate weights from gamma distribution
#        B22_blocks = block_averaging(distances, w, R1, R2, n_blocks) #calculate B22 values for every block via block averaging for the distances
#        B22_average, B22_std, B22_st_error = get_blocked_std_fixed_M(B22_blocks, n_blocks) 
#        B22_in_l_per_mol = B22_average * (1e-24 * (6.022e23))
#        B22_st_error_in_l_per_mol = B22_st_error * (1e-24 * (6.022e23))
#        results.append({
#            'Key': key, 
#            'R1 (nm)': R1,
#            'R2 (nm)': R2,
#            'B22 mean (l/mol)': B22_in_l_per_mol,
#            'B22 mean (nm³)': B22_average,
#            'B22_st_error (nm³)': B22_st_error,
#            'B22_st_error (l/mol)': B22_st_error_in_l_per_mol 
#        }) 
#
#def calculate_and_plot_B22(data, keys, R1_ranges, colors, labels, ylim, fontsize=16):
#    """
#    Compute weighted B22 values for given keys and plot them as a bar chart.
#    
#    Parameters
#    ----------
#    data : dict
#        Data dictionary containing 'distance' and whatever b22 needs.
#    keys : list
#        List of system keys (e.g. ["Man9/TIP3P", "Man9/TIP4PD", ...]).
#    R1_ranges : dict
#        Dictionary mapping each key to its chosen R1_range (numpy array).
#        Example: {"Man9/TIP3P": np.arange(3,4,0.1), ...}
#    colors : list
#        Colors for each bar.
#    labels : list
#        X-axis labels corresponding to keys.
#    fontsize : int
#        Font size for labels and ticks.
#    ylim : tuple
#        y-axis limits in nm^3.
#    unit_conv_fac : float
#        Conversion factor from nm^3 to L/mol.
#    """
#    B22_weighted_list = []
#    B22_weighted_stderr_list = []
#    R2 = {}
#
#    for key, c in zip(keys, colors):
#        # let the USER supply R1_range
#        R1_range = R1_ranges[key]
#        R2[key] = data['distance'][key][:,1].max() * 0.9
#
#        # Calculate B22
#        df = calculate_B22_for_R1_range(data, R1_range, R2, n_blocks=5, key=key)
#        weights = 1.0 / (df['B22_st_error (l/mol)'] ** 2)
#        weighted_mean = np.sum(weights * df['B22 mean (l/mol)']) / np.sum(weights)
#        weighted_stderr = np.sqrt(1.0 / np.sum(weights))
#        B22_weighted_list.append(weighted_mean)
#        B22_weighted_stderr_list.append(weighted_stderr)
#
#    fig, ax1 = plt.subplots(figsize=(10,6), dpi=300, sharex=True)
#    ax1.bar(
#        range(len(keys)),
#        height=B22_weighted_list,
#        yerr=B22_weighted_stderr_list,
#        capsize=10,
#        linewidth=2,
#        width=0.7,
#        color=colors,
#    )
#    ax1.set_xticks(np.arange(len(labels)))
#    ax1.set_xticklabels(labels, rotation=20, ha='right', rotation_mode="anchor", fontsize=fontsize)
#    ax1.set_xlabel("Water model", fontsize=fontsize)
#    ax1.set_ylabel(r"$\bar B_\mathrm{22}$ [nm$^3$]", fontsize=fontsize)
#    ax1.set_ylim(*ylim)
#
#    # Twin y-axis
#    ax1a = ax1.twinx()
#    mn, mx = ax1.get_ylim()
#    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
#    ax1a.set_ylabel(r"$\bar B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize=fontsize)
#
#    plt.axhline(y=0, color='black', linestyle='--')
#    ax1.set_axisbelow(True)
#    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
#    ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
#    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#    plt.show()
#
#    return B22_weighted_list, B22_weighted_stderr_list
#
#def plot_B22_over_time(data, R1_range, key, window, n_blocks=5, color='black', n_ticks=10, ticks=None, ticklabels=None, fontsize=16): 
#    """
#    Plot B22 values over time for a given R1 range and fixed R2.
#    """
#    B22_weighted_list = []
#    B22_weighted_stderr_list = []
#    R2 = data['distance'][key][:,1].max() * 0.9
#
#    for w in window:   
#        df = calculate_B22_for_R1_range_convergence(data, R1_range, R2, n_blocks, key, int(w))
#        weights = 1.0 / (df['B22_st_error (l/mol)'] ** 2)
#        B22_weighted = np.sum(weights * df['B22 mean (l/mol)']) / np.sum(weights)
#        B22_weighted_stderr = np.sqrt(1.0 / np.sum(weights))
#        B22_weighted_list.append(B22_weighted)
#        B22_weighted_stderr_list.append(B22_weighted_stderr)
#
#    fig, ax1 = plt.subplots(figsize=(10,6), dpi=300 ,sharex=True) 
#    fig.subplots_adjust(hspace=0.08)
#    bars = ax1.bar(window, height=B22_weighted_list, bottom=0, yerr=B22_weighted_stderr_list, capsize = 5,
#        width=len(data['log'][key])/n_ticks*0.8,
#        color=color,
#        alpha=0.8)
#
#    alphas = np.linspace(0.1, 0.8, n_ticks).tolist()
#    for i, bar in enumerate(bars):
#        bar.set_alpha(alphas[i])  # Set individual alpha values
#    ax1.set_xticks(ticks=ticks)
#    ax1.set_xticklabels(ticklabels, rotation=0, fontsize=fontsize)
#    ax1.set_xlabel(r"$\mathrm{Time}$ [µs]", fontsize = fontsize)
#    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize =fontsize)
#    #ax1.set_ylim(ymin, ymax)
#    ax1a = ax1.twinx()
#    mn, mx = ax1.get_ylim()
#    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
#    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize = fontsize)
#    ax1.set_axisbelow(True)
#    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
#    ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
#    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#
#def get_k(k_in_kj_per_mol_per_nm2, T=298.):
#    """
#    Unit conversion kJ/mol/nm2 to kbT/nm2
#    """
#    k_in_kbT_per_nm2=k_in_kj_per_mol_per_nm2/2.479*(298./T)
#    return k_in_kbT_per_nm2
#
#def gamma(r, k):
#    """
#    Distance distribution function using harmonic constraint in in (x,y)
#    """
#    f=1
#    return f*r*np.sqrt(2)*np.sqrt(k)*scipy.special.dawsn(f*np.sqrt(k)*r/np.sqrt(2))
#
#def get_B22_from_pv(pv, v, dv):
#    """
#    Calculate 2nd osmotic virial coeff. from sub-vol probability p_v. 
#    """
#    return -v/2.*(pv/(1.-pv)*dv/v-1.)
#
#def get_indicator(energies, energy_cutoff):
#    # ind=np.where(energies<energy_cutoff)[0]
#    # h_bound=np.zeros(len(energies))
#    # h_bound[ind]=1.
#    # return h_bound
#    """
#    Returns indicator function.
#    """
#    return np.where(energies < energy_cutoff, 1., 0)
#
#def get_weights(distances, spring_constant):
#    """
#    Get statistical weights for unbiased average calculation.
#    """
#    g=gamma(distances, spring_constant)
#    w=distances**2/g
#    w/=w.sum()
#    return w
#
#def apply_pbcs(data, r_max, ind_dist=1, ind_ener=2):
#    distances=data[:,ind_dist]
#    ix=np.where(distances<r_max)[0]
#    distances=distances[ix]
#    
#    #energies=data[ix, ind_ener]
#    return distances#, energies
#
#def get_corrected_pv(distances, w, R1_list):
#    pv_corr=[]
#    for R1 in R1_list:
#        iv=get_indicator(distances, R1)
#        pv=np.average(iv, weights=w)
#        pv_corr.append([R1, pv])
#    pv_corr=np.asarray(pv_corr)
#    if len(R1_list)>1:
#        return pv_corr
#    else:
#        return pv_corr[0,1] 
