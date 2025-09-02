import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import pickle as pkl
from itertools import groupby
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker


#Constants
kb=0.008314462618
T=300
unit_conv_fac=1e-24*(6.022e23)  #conversion factor from nm^3 to liter

def block_averaging(data, weights, R1, R2, n_blocks):
    """
    Block averaging of B22 values for fixed number of blocks.
    """
    block_size = len(data) // n_blocks
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

    B22_blocks = np.array(B22_blocks)

    return B22_blocks                   #returns B22 for every block

def get_blocked_std_fixed_M(B22_blocks,M):                      
    """
    B22_blocks: Observable (in this case: block averages of B22 for every block)
    M: Numbers of blocks
    Returns: Estimates B22 mean, standard deviation and standard error of the mean for a given block length/block number
    """
    
    mean=B22_blocks.mean()                        #this line calculates the mean value over all block mean values
    var=((B22_blocks-mean)**2).sum()/float(M)       #calculation of variance for overall population (in this case: calculates the variability between blocks) (Calculation differs from the calculation in the function above, in the function "get_bootstrap_error" the sample variance is used)
    std=np.sqrt(var)                                 # Calculation of standard deviation as square root of variance
    bse = std/np.sqrt(M)                  # Calculation of standard error of the mean (= standard deviation of block means / squareroot(number of blocks))
    return mean, std, bse

def calc_plot_FEP(data, key, fit_range_lower, fit_range_upper, threshold, max_gap, color, fontsize):
    '''
    Calculate and plot free energy profile from distance data.
    Identify regions where free energy is within a certain threshold of the average free energy in a specified fit range.
    Merge regions that are separated by gaps smaller than max_gap.  
    Plot the free energy profile and highlight the identified regions.
    Include an inset zoomed plot in the upper-right corner with connecting lines.
    '''

    # Free energy
    P, bin_edges = np.histogram(
        data['distance'][key][:,1],
        bins=130*2,
        range=(0, data['distance'][key][:,1].max()),
        density=True
    )
    F = -kb * T * np.log(P)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
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
                current_end = next_end  # extend current region
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_ranges.append((current_start, current_end))  # add the last one

    # Main plot   
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(bin_centers, F_shifted, lw=3, color=color)
    for (start_x, end_x) in merged_ranges:
        ax.axvspan(start_x, end_x, color=color, alpha=0.3)
    ax.set_ylabel("Free energy [kJ/mol]", fontsize=fontsize)
    ax.set_xlabel("r [nm]", fontsize=fontsize)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.tick_params(labelsize=fontsize)

    # Inset: 1/3 width of main plot
    axins = inset_axes(ax, width="33%", height="33%", loc='upper right',
                       borderpad=1)  # small padding
    axins.set_facecolor("white")     # solid white background
    axins.spines[:].set_visible(True) # make sure all spines (box) are visible
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    # Plot inside inset
    axins.plot(bin_centers, F_shifted, lw=2, color=color)
    for (start_x, end_x) in merged_ranges:
        axins.axvspan(start_x, end_x, color=color, alpha=0.3)

    # Zoom to cover all merged regions (min start → max end, with padding)
    if merged_ranges:
        region_start = min(r[0] for r in merged_ranges)
        region_end   = max(r[1] for r in merged_ranges)
        padding = 0.05 * (region_end - region_start)
        axins.set_xlim(region_start - padding, region_end + padding)

    axins.set_ylim(-1, 1)
    axins.axhline(0, color='black', linewidth=1, linestyle='-')
    axins.tick_params(labelsize=fontsize-2)

    # Draw connectors between inset and main plot
    if merged_ranges:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.2)

    plt.show()

    return merged_ranges

def plot_B22_for_R1_range(data, R1_range, R2, key, n_blocks = 5, color = "black", fontsize = 16, ylim_min = None, ylim_max = None):
    """
    Plot B22 values for a range of R1 values at fixed R2.
    """
    df = calculate_B22_for_R1_range(data, R1_range, R2, n_blocks, key)
    fig, ax1 = plt.subplots(figsize=(10,6), dpi=300 ,sharex=True) 
    bars = ax1.bar(R1_range, height=df['B22 mean (l/mol)'], bottom=0, yerr=df['B22_st_error (l/mol)'], capsize = 10, linewidth=2, width=0.08,
        color=color,  # visible fill
        alpha=0.80)
    ax1.set_xlabel(r"$\mathrm{R1}$ [nm]", fontsize = fontsize)
    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize = fontsize)
    if ylim_min is None:
        ylim_min = min(0, df['B22 mean (l/mol)'].min() - 1.5 * df['B22_st_error (l/mol)'].max())
    if ylim_max is None:
        ylim_max = df['B22 mean (l/mol)'].max() + 1.5 * df['B22_st_error (l/mol)'].max()
    ax1.set_ylim(ylim_min, ylim_max)
    ax1a = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize = fontsize)
    plt.axhline(y=0, color='black', linestyle='--')
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='major', labelsize = fontsize)
    ax1a.tick_params(axis='both', which='major', labelsize = fontsize)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.show()

def calculate_B22_for_R1_range(data, R1_range, R2, n_blocks, key):
    """
    Calculate B22 values for a range of R1 values at fixed R2.
    """
    results = []
    for R1 in R1_range:
        distances = apply_pbcs(data['distance'][key], R2[key])  # return distances and corresponding energies when distances are smaller than R2
        w = get_weights(distances, data['k_in_kbT'][key])  # calculate weights from gamma distribution
        B22_blocks = block_averaging(distances, w, R1, R2[key], n_blocks) #calculate B22 values for every block via block averaging for the distances
        B22_average, B22_std, B22_st_error = get_blocked_std_fixed_M(B22_blocks, n_blocks) 
        B22_in_l_per_mol = B22_average * (1e-24 * (6.022e23))
        B22_st_error_in_l_per_mol = B22_st_error * (1e-24 * (6.022e23))
        results.append({
            'Key': key, 
            'R1 (nm)': R1,
            'R2 (nm)': R2[key],
            'B22 mean (l/mol)': B22_in_l_per_mol,
            'B22 mean (nm³)': B22_average,
            'B22_st_error (nm³)': B22_st_error,
            'B22_st_error (l/mol)': B22_st_error_in_l_per_mol 
        }) 
                
    B22_over_R1_df = pd.DataFrame(results)
    return B22_over_R1_df

def calculate_B22_for_R1_range_convergence(data, R1_range, R2, n_blocks, key, window):
    """
    Calculate B22 values for a range of R1 values at fixed R2 with convergence over time.
    """
    results = []
    for R1 in R1_range:
        distances = apply_pbcs(data['distance'][key][0:window], R2)  # return distances and corresponding energies when distances are smaller than R2
        w = get_weights(distances, data['k_in_kbT'][key])  # calculate weights from gamma distribution
        B22_blocks = block_averaging(distances, w, R1, R2, n_blocks) #calculate B22 values for every block via block averaging for the distances
        B22_average, B22_std, B22_st_error = get_blocked_std_fixed_M(B22_blocks, n_blocks) 
        B22_in_l_per_mol = B22_average * (1e-24 * (6.022e23))
        B22_st_error_in_l_per_mol = B22_st_error * (1e-24 * (6.022e23))
        results.append({
            'Key': key, 
            'R1 (nm)': R1,
            'R2 (nm)': R2,
            'B22 mean (l/mol)': B22_in_l_per_mol,
            'B22 mean (nm³)': B22_average,
            'B22_st_error (nm³)': B22_st_error,
            'B22_st_error (l/mol)': B22_st_error_in_l_per_mol 
        }) 

def calculate_and_plot_B22(data, keys, R1_ranges, colors, labels, ylim, fontsize=16):
    """
    Compute weighted B22 values for given keys and plot them as a bar chart.
    
    Parameters
    ----------
    data : dict
        Data dictionary containing 'distance' and whatever b22 needs.
    keys : list
        List of system keys (e.g. ["Man9/TIP3P", "Man9/TIP4PD", ...]).
    R1_ranges : dict
        Dictionary mapping each key to its chosen R1_range (numpy array).
        Example: {"Man9/TIP3P": np.arange(3,4,0.1), ...}
    colors : list
        Colors for each bar.
    labels : list
        X-axis labels corresponding to keys.
    fontsize : int
        Font size for labels and ticks.
    ylim : tuple
        y-axis limits in nm^3.
    unit_conv_fac : float
        Conversion factor from nm^3 to L/mol.
    """
    B22_weighted_list = []
    B22_weighted_stderr_list = []
    R2 = {}

    for key, c in zip(keys, colors):
        # let the USER supply R1_range
        R1_range = R1_ranges[key]
        R2[key] = data['distance'][key][:,1].max() * 0.9

        # Calculate B22
        df = calculate_B22_for_R1_range(data, R1_range, R2, n_blocks=5, key=key)
        weights = 1.0 / (df['B22_st_error (l/mol)'] ** 2)
        weighted_mean = np.sum(weights * df['B22 mean (l/mol)']) / np.sum(weights)
        weighted_stderr = np.sqrt(1.0 / np.sum(weights))
        B22_weighted_list.append(weighted_mean)
        B22_weighted_stderr_list.append(weighted_stderr)

    fig, ax1 = plt.subplots(figsize=(10,6), dpi=300, sharex=True)
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
    ax1.set_xticklabels(labels, rotation=20, ha='right', rotation_mode="anchor", fontsize=fontsize)
    ax1.set_xlabel("Water model", fontsize=fontsize)
    ax1.set_ylabel(r"$\bar B_\mathrm{22}$ [nm$^3$]", fontsize=fontsize)
    ax1.set_ylim(*ylim)

    # Twin y-axis
    ax1a = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
    ax1a.set_ylabel(r"$\bar B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize=fontsize)

    plt.axhline(y=0, color='black', linestyle='--')
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.show()

    return B22_weighted_list, B22_weighted_stderr_list

def plot_B22_over_time(data, R1_range, key, window, n_blocks=5, color='black', n_ticks=10, ticks=None, ticklabels=None, fontsize=16): 
    """
    Plot B22 values over time for a given R1 range and fixed R2.
    """
    B22_weighted_list = []
    B22_weighted_stderr_list = []
    R2 = data['distance'][key][:,1].max() * 0.9

    for w in window:   
        df = calculate_B22_for_R1_range_convergence(data, R1_range, R2, n_blocks, key, int(w))
        weights = 1.0 / (df['B22_st_error (l/mol)'] ** 2)
        B22_weighted = np.sum(weights * df['B22 mean (l/mol)']) / np.sum(weights)
        B22_weighted_stderr = np.sqrt(1.0 / np.sum(weights))
        B22_weighted_list.append(B22_weighted)
        B22_weighted_stderr_list.append(B22_weighted_stderr)

    fig, ax1 = plt.subplots(figsize=(10,6), dpi=300 ,sharex=True) 
    fig.subplots_adjust(hspace=0.08)
    bars = ax1.bar(window, height=B22_weighted_list, bottom=0, yerr=B22_weighted_stderr_list, capsize = 5,
        width=len(data['log'][key])/n_ticks*0.8,
        color=color,
        alpha=0.8)

    alphas = np.linspace(0.1, 0.8, n_ticks).tolist()
    for i, bar in enumerate(bars):
        bar.set_alpha(alphas[i])  # Set individual alpha values
    ax1.set_xticks(ticks=ticks)
    ax1.set_xticklabels(ticklabels, rotation=0, fontsize=fontsize)
    ax1.set_xlabel(r"$\mathrm{Time}$ [µs]", fontsize = fontsize)
    ax1.set_ylabel(r"$B_\mathrm{22}$ [nm$^3$]", fontsize =fontsize)
    #ax1.set_ylim(ymin, ymax)
    ax1a = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax1a.set_ylim(mn * unit_conv_fac, mx * unit_conv_fac)
    ax1a.set_ylabel(r"$B_\mathrm{22}$ [L/mol]", rotation=-90, labelpad=30, fontsize = fontsize)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1a.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

def get_k(k_in_kj_per_mol_per_nm2, T=298.):
    """
    Unit conversion kJ/mol/nm2 to kbT/nm2
    """
    k_in_kbT_per_nm2=k_in_kj_per_mol_per_nm2/2.479*(298./T)
    return k_in_kbT_per_nm2

def gamma(r, k):
    """
    Distance distribution function using harmonic constraint in in (x,y)
    """
    f=1
    return f*r*np.sqrt(2)*np.sqrt(k)*scipy.special.dawsn(f*np.sqrt(k)*r/np.sqrt(2))

def get_B22_from_pv(pv, v, dv):
    """
    Calculate 2nd osmotic virial coeff. from sub-vol probability p_v. 
    """
    return -v/2.*(pv/(1.-pv)*dv/v-1.)

def get_indicator(energies, energy_cutoff):
    # ind=np.where(energies<energy_cutoff)[0]
    # h_bound=np.zeros(len(energies))
    # h_bound[ind]=1.
    # return h_bound
    """
    Returns indicator function.
    """
    return np.where(energies < energy_cutoff, 1., 0)

def get_weights(distances, spring_constant):
    """
    Get statistical weights for unbiased average calculation.
    """
    g=gamma(distances, spring_constant)
    w=distances**2/g
    w/=w.sum()
    return w

def apply_pbcs(data, r_max, ind_dist=1, ind_ener=2):
    distances=data[:,ind_dist]
    ix=np.where(distances<r_max)[0]
    distances=distances[ix]
    
    #energies=data[ix, ind_ener]
    return distances#, energies

def get_corrected_pv(distances, w, R1_list):
    pv_corr=[]
    for R1 in R1_list:
        iv=get_indicator(distances, R1)
        pv=np.average(iv, weights=w)
        pv_corr.append([R1, pv])
    pv_corr=np.asarray(pv_corr)
    if len(R1_list)>1:
        return pv_corr
    else:
        return pv_corr[0,1] 