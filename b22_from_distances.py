"""b22_from_distances.py — Tools for computing and visualizing the second osmotic virial coefficient (B22)

Copyright (c) 2026

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

License
-------
GNU GENERAL PUBLIC LICENSE
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.special import dawsn
from itertools import groupby
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
from typing import Sequence

__version__ = "1.0.0"
__author__ = "Jürgen Köfinger, Isabell Grothaus and contributors"

# Physical constants and default settings
kb: float = 0.008314462618  # kJ/(mol*K)
T: float = 300.0  # K
unit_conv_fac: float = 1e-24 * (6.022e23)  # nm^3 -> L/mol
fontsize = 20 
threshold = 0.02
max_gap = 0.2

class ManuscriptFigureGenerator:
    """
    Automates generation of manuscript figures (free energy, transitions,
    BSE plateau, B22 variation, B22 bar chart) for a set of keys.
    """
    def __init__(self, data, keys, color, path, molecule,
                 R1_value, n_blocks, ylim = None, n=100, ylim_padding = 0.2, show=False):
        self.data = data
        self.keys = keys["all"]
        self.color = color
        self.path = path
        self.molecule = molecule
        self.n_blocks = n_blocks
        self.n = n
        self.ylim = ylim
        self.ylim_padding = ylim_padding
        self.show = show

        self.R1 = self._resolve_per_key(R1_value, "R1_value")
        self.R2 = {k: data["boxsize_z/2"][k] * 0.95 for k in self.keys}

        self.bse_plateau = []
        self.B22_df = None

    def _resolve_per_key(self, value, name):
        """
        Accepts either a single scalar (broadcast to all keys) or a dict
        already keyed per dataset. Returns a dict covering all self.keys.
        """
        if isinstance(value, dict):
            missing = [k for k in self.keys if k not in value]
            if missing:
                raise KeyError(f"{name} is missing entries for keys: {missing}")
            return {k: value[k] for k in self.keys}
        else:
            return {k: value for k in self.keys}
    
    def _fname(self, suffix, key=None):
        key_part = self.data["key"][key] if key is not None else ""
        return "{}{}{}{}".format(self.path, self.molecule, key_part, suffix)

    def plot_free_energy(self):
        for key, c in zip(self.keys, self.color):
            calc_plot_FEP(self.data, key, c, show=self.show,
                           save_path=self._fname("_free_energy.pdf", key))

    def plot_transitions(self):
        for key, c in zip(self.keys, self.color):
            plot_transitions(self.data, key, self.n, c, show=self.show,
                              save_path=self._fname("_transitions.pdf", key))

    def plot_bse_plateau(self):
        self.bse_plateau = []
        for key in self.keys:
            results = determine_bse_plateau(self.data, key, self.R1[key],
                                             R2=self.R2[key],
                                             n_blocks_range=range(2, 200))
            self.bse_plateau.append(results)
        plot_bse_plateau(self.bse_plateau, self.color, show=self.show,
                          save_path="{}{}bse_plateau.pdf".format(self.path, self.molecule))

    def plot_B22_validation(self):
        for key, c in zip(self.keys, self.color):
            R2_single = {key: self.R2[key]}

            plot_B22_variation(self.data, key, self.n_blocks, c, mode='R1',
                                R1_range=self.data["boxsize_z/2"][key] * np.arange(0.6, 0.8 + 1e-9, 0.025),
                                R2=R2_single, show=self.show,
                                save_path=self._fname("_B22_over_R1.pdf", key), ylim_padding=self.ylim_padding)

            plot_B22_variation(self.data, key, self.n_blocks, c, mode='R2',
                                R1=self.R1,
                                R2_range=self.data["boxsize_z/2"][key] * np.arange(0.85, 1.0 + 1e-9, 0.025),
                                show=self.show,
                                save_path=self._fname("_B22_over_R2.pdf", key), ylim_padding=self.ylim_padding)

            plot_B22_variation(self.data, key, self.n_blocks, c, mode='time',
                                R1=self.R1, R2=R2_single, show=self.show,
                                save_path=self._fname("_B22_over_time.pdf", key), ylim_padding=self.ylim_padding)

    def plot_B22_bar_chart(self):
        B22_list = []
        for key in self.keys:
            R2_single = {key: self.R2[key]}
            df = calculate_B22(self.data, self.R1, R2_single, self.n_blocks, key)
            B22_list.append(df)

        self.B22_df = pd.concat(B22_list, ignore_index=True)
        height = self.B22_df['B22 mean (nm³)'].values
        yerr = self.B22_df['B22_st_error (nm³)'].values
        labels_list = [self.data["labels"][key] for key in self.keys]

        exp_B22 = next(
            (self.data["experimental_B22"][key] for key in self.keys if key in self.data["experimental_B22"]),
            None
        )

        fig, ax1, ax1a = plot_B22_bar_chart(
            height, yerr, labels_list, color=self.color, show=self.show, ylim=self.ylim, ylim_padding=self.ylim_padding,
            save_path="{}{}B22.pdf".format(self.path, self.molecule),
            experimental_B22=exp_B22
        )
        return fig, ax1, ax1a

    def run_all(self):
        self.plot_free_energy()
        self.plot_transitions()
        self.plot_bse_plateau()
        self.plot_B22_validation()
        return self.plot_B22_bar_chart()

# ---------------------------
# Core computations
# ---------------------------
def determine_bse_plateau(
    data,
    key,
    R1,
    R2,
    n_blocks_range,
):
    """
    Determine the plateau of the blocked standard error of B22.

    Parameters
    ----------
    data: dict
        Dictionary containing numpy arrays with columns [time, distance].
    R1 : float
        Inner radius used for B22 calculation.
    R2 : float or dict
        Outer radius. Can either be a single value or one value per dictionary key.
    n_blocks_range : iterable
        Numbers of blocks to test.

    Returns
    -------
    results : dict
        - n_blocks
        - block_size
        - bse
        - plateau_bse
        - plateau_n_blocks
        - plateau_block_size
    """

    distances = apply_pbcs(data['log'][key], R2)  # return distances and corresponding energies when distances are smaller than R2
    weights = get_weights(distances, data['k_in_kbT'][key])  # calculate weights from gamma distribution
    values = []
    for n_blocks in n_blocks_range:
        if n_blocks >= len(distances):
            continue
        B22, std, bse, pv_blocks, B22_blocks = block_averaging(
            distances,
            weights,
            R1,
            R2,
            n_blocks
        )

        pv = pv_blocks.mean()
        
        values.append([
            n_blocks,
            len(distances) // n_blocks,
            bse,
            B22,
            std,
            pv
        ])
    values = np.asarray(values)

    results = {
        "n_blocks": values[:, 0].astype(int),
        "block_size": values[:, 1].astype(int),
        "bse": values[:, 2],
        "B22": values[:, 3],
        "B22_std" : values[:, 4],
        "pv_mean" : values[:, 5]
    }

    return results

def block_averaging(data, weights, R1, R2, n_blocks):
    block_size = len(data) // n_blocks
    pv_blocks = []
    B22_blocks = []
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_distances = data[start_idx:end_idx]
        block_weights = weights[start_idx:end_idx]

        pv_block = get_corrected_pv(block_distances, block_weights, [R1])
        if pv_block == 1:
            pv_block = 0.9999
        else:
            pv_block = pv_block
        v = 4 * np.pi * R1**3 / 3.
        V = 4 * np.pi * R2**3 / 3.
        dv = V - v

        B22_block = get_B22_from_pv(pv_block, v, dv)
        B22_blocks.append(B22_block)
        pv_blocks.append(pv_block)
        
    B22_blocks = np.array(B22_blocks)
    pv_blocks = np.array(pv_blocks)

    mean = pv_blocks.mean()
    B22 = get_B22_from_pv(mean, v, dv)
    std = np.std(B22_blocks, ddof=1)
    bse = std / np.sqrt(n_blocks)

    return B22, std, bse, pv_blocks, B22_blocks                     

def calculate_B22(data, R1, R2, n_blocks, key, window = None):

    if window == None:
        distances = apply_pbcs(data['log'][key], R2[key])  # return distances and corresponding energies when distances are smaller than R2
    else:
        distances = apply_pbcs(data['log'][key][0:window], R2[key])
    w = get_weights(distances, data['k_in_kbT'][key])  # calculate weights from gamma distribution
    B22, _, bse, _, _ = block_averaging(distances, w, R1[key], R2[key], n_blocks)
    B22_in_l_per_mol = B22 * (1e-24 * (6.022e23))
    B22_st_error_in_l_per_mol = bse * (1e-24 * (6.022e23))
    results = {
        'Key': key, 
        'R1 (nm)': R1[key],
        'R2 (nm)': R2[key],
        'B22 mean (l/mol)': B22_in_l_per_mol,
        'B22 mean (nm³)': B22,
        'B22_st_error (nm³)': bse,
        'B22_st_error (l/mol)': B22_st_error_in_l_per_mol 
    } 
                
    B22 = pd.DataFrame([results])
    return B22

def find_flat_region(x, y, window_frac=0.10, min_window=10, slope_weight=1.0, min_start_frac=0.75):
    """
    Find the sub-region of (x, y) that is flattest (near-zero slope, low scatter).
    Returns (x_lower, x_upper) for the best window.

    min_start_frac: fraction of the domain to skip before starting the search
                     (e.g. 0.5 restricts the search to the outer half of the data).
    """
    valid = np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    window = max(int(n * window_frac), min_window)

    start_min = int(n * min_start_frac) 

    best_score = np.inf
    best_start = start_min
    for start in range(start_min, n - window):  
        xs = x[start:start+window]
        ys = y[start:start+window]
        slope, intercept = np.polyfit(xs, ys, 1)
        residuals = ys - (slope * xs + intercept)
        score = np.var(residuals) + slope_weight * slope**2
        if score < best_score:
            best_score = score
            best_start = start

    return x[best_start], x[best_start + window - 1]

# ---------------------------
# Plotting utilities
# ---------------------------
def calc_plot_FEP(data, key, color, show=True, save_path=None):
    # Free energy
    P, bin_edges = np.histogram(data['log'][key][:,1], bins=130*2, range=(0, data["boxsize_z/2"][key]), density=True)
    F = -kb * data['temperature'][key] * np.log(P)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fit_range_lower, fit_range_upper = find_flat_region(bin_centers, F, window_frac=0.10)
    mask = (bin_centers >= fit_range_lower) & (bin_centers <= fit_range_upper)
    F_avg = np.mean(F[mask])
    F_shifted = F - F_avg

    within_band = np.abs(F_shifted) < threshold

    # Step 1: find all subregions
    raw_ranges = []
    for k, g in groupby(enumerate(within_band), key=lambda x: x[1]):
        if k:
            group = list(g)
            start_idx = group[0][0]
            end_idx = group[-1][0]
            start_x = bin_centers[start_idx]
            end_x = bin_centers[end_idx]
            raw_ranges.append((start_x, end_x))

    # Step 2: merge subregions with gaps ≤ max_gap
    merged_ranges = []
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

    axins.set_ylim(-0.5, 0.5)
    axins.axhline(0, color="black", linewidth=1, linestyle="-")
    axins.tick_params(labelsize=fontsize - 2)

    if merged_ranges:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_transitions(data, key, n, color, show=True, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(data['log'][key][::n,0], data['log'][key][::n,1], color=color, s=0.1)
    ax.set_xlabel(r"Time [$\mu$s]", fontsize=fontsize)
    ax.set_ylabel("COM distance [nm]", fontsize=fontsize)
    ax.tick_params(axis="x",labelsize=fontsize)
    ax.tick_params(axis="y",labelsize=fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_bse_plateau(bse_plateau, color, ylim = None, show=True, save_path=None):

    fig = plt.figure(dpi = 300) 
    for i,c in zip(range(0,len(bse_plateau)), color):
        plt.plot(bse_plateau[i]["n_blocks"],bse_plateau[i]["bse"], color = c)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel("No. of blocks")
    plt.ylabel("Standard error of the mean")
    if ylim is not None:
        plt.ylim(0,ylim)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def build_axis_info(log_length, base_unit=100001, target_ticks=10, n_labels=8):
    """
    Adaptive axis info for time-based plots.

    Picks a step size (unit) so that roughly `target_ticks` windows result,
    working for both very short (<1 µs) and very long simulations.
    base_unit=100001 raw steps corresponds to 1 µs.

    Returns
    -------
    window     : array of raw step positions (for slicing data)
    n_ticks    : number of windows
    ticks      : raw step positions to place tick marks at (subset of window)
    ticklabels : corresponding time values in µs (rounded, for display)
    """
    unit = base_unit
    n_ticks = log_length // unit
    while n_ticks < target_ticks and unit > 1001:
        unit = max(unit // 2, 1001)
        n_ticks = log_length // unit

    n_ticks = max(n_ticks, 1)
    window = unit * np.arange(1, n_ticks + 1)

    time_values = window / base_unit 

    n_show = min(n_labels, n_ticks)
    label_idx = np.unique(np.linspace(0, n_ticks - 1, n_show).astype(int))

    ticks = window[label_idx]
    ticklabels = [round(t, 2) for t in time_values[label_idx]]

    return window, n_ticks, ticks, ticklabels

def plot_B22_variation(data, key, n_blocks, color, mode,
                        R1=None, R2=None, R1_range=None, R2_range=None,
                        window=None, n_ticks=None, ticks=None, ticklabels=None,
                        ylim=None, ylim_padding=0.35, show=True, save_path=None,
                        marker_size=120):
    """
    General B22 plotting function. Plots B22 (with error bars) as a function of
    R1, R2, or simulation time, depending on `mode`.
    ... (docstring unverändert) ...
    """
    
    B22_list = []
    bse_list = []

    if mode == 'R1':
        assert R1_range is not None and R2 is not None, "mode='R1' requires R1_range and R2"
        x_values = R1_range
        xlabel = r"$\mathrm{R1}$ [nm]"
        for r1_val in R1_range:
            R1_i = {key: r1_val}
            df = calculate_B22(data, R1_i, R2, n_blocks, key)
            B22_list.append(df["B22 mean (nm³)"].iloc[0])
            bse_list.append(df["B22_st_error (nm³)"].iloc[0])

    elif mode == 'R2':
        assert R1 is not None and R2_range is not None, "mode='R2' requires R1 and R2_range"
        x_values = R2_range
        xlabel = r"$\mathrm{R2}$ [nm]"
        for r2_val in R2_range:
            R2_i = {key: r2_val}
            df = calculate_B22(data, R1, R2_i, n_blocks, key)
            B22_list.append(df["B22 mean (nm³)"].iloc[0])
            bse_list.append(df["B22_st_error (nm³)"].iloc[0])

    elif mode == 'time':
        assert R1 is not None and R2 is not None, \
            "mode='time' requires R1, R2"
        window, n_ticks, ticks, ticklabels = build_axis_info(len(data['log'][key][:, 0]))
        x_values = window
        xlabel = r"$\mathrm{Time}$ [µs]"
        for w in window:
            df = calculate_B22(data, R1, R2, n_blocks, key, int(w))
            B22_list.append(df["B22 mean (nm³)"].iloc[0])
            bse_list.append(df["B22_st_error (nm³)"].iloc[0])

    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'R1', 'R2', or 'time'")

    if ylim is None:
        lows = np.array(B22_list)
        highs = np.array(B22_list)
        data_min = np.min(lows)
        data_max = np.max(highs)
        if data["experimental_B22"][key] != "None":
            data_min = min(data_min, data["experimental_B22"][key])
            data_max = max(data_max, data["experimental_B22"][key])
        span = data_max - data_min if data_max != data_min else (abs(data_max) or 1.0)
        pad = span * ylim_padding
        ylim_min, ylim_max = data_min - pad, data_max + pad
    else:
        ylim_min, ylim_max = ylim

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    x_values_arr = np.asarray(x_values, dtype=float)
    x_min, x_max = x_values_arr.min(), x_values_arr.max()
    x_pad = (x_max - x_min) * 0.08 if x_max != x_min else 1.0
    xlim_min, xlim_max = x_min - x_pad, x_max + x_pad

    gradient = np.linspace(ylim_max, ylim_min, 500).reshape(-1, 1)

    if ylim_min < 0 < ylim_max:
        norm = matplotlib.colors.TwoSlopeNorm(
            vmin=ylim_min,
            vcenter=0,
            vmax=ylim_max
        )
    else:
        norm = matplotlib.colors.Normalize(
            vmin=ylim_min,
            vmax=ylim_max
        )
    ax1.imshow(
        gradient, aspect='auto', cmap='RdBu_r', norm=norm,
        extent=[xlim_min, xlim_max, ylim_min, ylim_max],
        zorder=0, alpha=0.3
    )

    # --- Datenpunkte als Disks mit Errorbars ---
    ax1.errorbar(
        x_values, B22_list, yerr=bse_list,
        fmt='o', markersize=np.sqrt(marker_size), color=color,
        ecolor='black', elinewidth=1.5, capsize=6,
        markeredgecolor='black', markeredgewidth=1.2,
        zorder=3
    )

    if mode == 'time':
        # Fade von hell nach dunkel zur zeitlichen Progression -> als Alpha pro Punkt
        alphas = np.linspace(0.3, 1.0, n_ticks).tolist()
        for i, a in enumerate(alphas):
            ax1.errorbar(
                x_values[i], B22_list[i], yerr=bse_list[i],
                fmt='o', markersize=np.sqrt(marker_size), color=color,
                ecolor='black', elinewidth=1.5, capsize=5,
                markeredgecolor='black', markeredgewidth=1.0,
                alpha=a, zorder=3
            )
        ax1.set_xticks(ticks=ticks)
        ax1.set_xticklabels(ticklabels, rotation=0, fontsize=fontsize)

    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize=fontsize)
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    ax1a = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize=fontsize)
    ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    if data["experimental_B22"][key] != "None":
        ax1.axhline(y=data["experimental_B22"][key], color='black', linestyle='--', zorder=2)
    else:
        pass

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax1, ax1a

def plot_B22_bar_chart(height, yerr, labels,
                        color=None,
                        ylim=None, ylim_padding=0.25,
                        figsize=(10, 6), dpi=300,
                        show=True, save_path=None,
                        experimental_B22=None,
                        marker_size=120):
    """
    Plot B22 values (as disks with error bars) across systems, with an optional
    secondary axis (L/mol), a diverging background gradient centered at 0,
    and an optional experimental reference line.

    Parameters
    ----------
    height : array-like
        B22 mean values (e.g. in nm^3).
    yerr : array-like
        Error bar values, same length as height.
    labels : list of str
        X-axis tick labels, same length as height.
    color : list, optional
        Marker colors. If None, uses matplotlib's default color cycle.
    fontsize : int
        Base fontsize for labels/ticks.
    ylim : tuple (min, max), optional
        Explicit y-limits. If None, computed automatically from the data.
    ylim_padding : float
        Fractional padding added above/below the data range when auto-computing ylim.
    figsize : tuple
        Figure size.
    dpi : int
        Figure resolution.
    save_path : str, optional
        If given, saves the figure to this path.
    marker_size : float
        Marker area (passed through to markersize via sqrt scaling).

    Returns
    -------
    fig, ax1, ax1a
    """
    height = np.asarray(height)
    yerr = np.asarray(yerr)
    x_values = np.arange(len(height))

    assert len(height) == len(yerr) == len(labels), \
        "height, yerr, and labels must all be the same length"

    if color is not None:
        assert len(color) == len(height), "color must be the same length as height"

    if ylim is None:
        lows = height - yerr
        highs = height + yerr
        data_min = np.min(lows)
        data_max = np.max(highs)
        if experimental_B22 is not None:
            data_min = min(data_min, experimental_B22)
            data_max = max(data_max, experimental_B22)
        span = data_max - data_min
        if span == 0:
            span = abs(data_max) if data_max != 0 else 1.0
        pad = span * ylim_padding
        ylim_min = data_min - pad
        ylim_max = data_max + pad
    else:
        ylim_min, ylim_max = ylim

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

    x_pad = 0.6
    xlim_min, xlim_max = x_values.min() - x_pad, x_values.max() + x_pad

    gradient = np.linspace(ylim_max, ylim_min, 500).reshape(-1, 1)

    if ylim_min < 0 < ylim_max:
        norm = matplotlib.colors.TwoSlopeNorm(
            vmin=ylim_min,
            vcenter=0,
            vmax=ylim_max
        )
    else:
        norm = matplotlib.colors.Normalize(
            vmin=ylim_min,
            vmax=ylim_max
        )
    ax1.imshow(
        gradient, aspect='auto', cmap='RdBu_r', norm=norm,
        extent=[xlim_min, xlim_max, ylim_min, ylim_max],
        zorder=0, alpha = 0.3
    )

    if color is None:
        color = [f"C{i}" for i in range(len(height))]

    for xi, hi, ei, ci in zip(x_values, height, yerr, color):
        ax1.errorbar(
            xi, hi, yerr=ei,
            fmt='o', markersize=np.sqrt(marker_size), color=ci,
            ecolor='black', elinewidth=1.5, capsize=6,
            markeredgecolor='black', markeredgewidth=1.2,
            zorder=3
        )

    for xi, hi in zip(x_values, height):
        ax1.annotate(f"{hi:.1f}", (xi, hi),
                     textcoords="offset points", xytext=(15, 0),
                     ha='left', va='center', fontsize=fontsize - 4, zorder=4)

    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(labels, rotation=20, ha='right', rotation_mode="anchor", fontsize=fontsize)
    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize=fontsize)
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    ax1a = ax1.twinx()
    ax1a.set_ylim(ylim_min * unit_conv_fac, ylim_max * unit_conv_fac)
    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize=fontsize)
    ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    if experimental_B22 != "None":
        ax1.axhline(y=experimental_B22, color='black', linestyle='--', zorder=2)
    else:
        pass
        
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax1, ax1a

# ---------------------------
# Physics helpers
# ---------------------------

def get_k(k_in_kj_per_mol_per_nm2: float, T: float = 300, scaled: bool = False, Lx: float = 1) -> float:
    """Convert k from kJ/mol/nm^2 to kBT/nm^2 at temperature ``T``.
    Uses 1 kBT = 2.479 kJ/mol at 298 K.
    """
    if scaled is True:
        k_phys = k_in_kj_per_mol_per_nm2 / Lx**2
        k_in_kbT_per_nm2 = k_phys/ 2.479 * (298 / T)
    else:
        k_in_kbT_per_nm2 = k_in_kj_per_mol_per_nm2 / 2.479 * (298 / T)
    
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
    #return f*r*np.sqrt(2)*np.sqrt(k)*scipy.special.dawsn(f*np.sqrt(k)*r/np.sqrt(2)) # Dawson function for restraining only 1 of the 2 molecules
    return f*r*np.sqrt(k)*dawsn(r*np.sqrt(k*f)/2) # Dawson function for restraining both molecules


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


def apply_pbcs(data: np.ndarray, r_max: float, ind_dist: int = 1) -> np.ndarray:
    """Apply spherical cutoff (PBC-like) and return distances < r_max.

    Parameters
    ----------
    data : np.ndarray
        Array where column ``ind_dist`` is the distance r.
    r_max : float
        Maximum distance to retain (nm).
    ind_dist : int
        Column index containing r.
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