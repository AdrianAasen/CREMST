import numpy as np
from scipy.stats import unitary_group
import scipy as sp
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import os
import glob
import uuid
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
import scipy.cluster.hierarchy as sch
from EMQST_lib.qrem import QREM

def plot_POVM_folder(path_to_folder = None):
    if path_to_folder is None:
        path_to_folder = "data/raw_povms/"        
    filenames = glob.glob(f'{path_to_folder}*.npy')
    print(f'Plotting POVM lists from {path_to_folder}.')
    X = np.array([[0,1],[1,0]],dtype = complex)
    Y = np.array([[0,-1j],[1j,0]],dtype = complex)
    Z = np.array([[1,0],[0,-1]],dtype = complex)
    sigma = np.array([X,Y,Z],dtype=complex)
    
    def rot(axis,angle):
        return sp.linalg.expm(-1/2j * angle * np.einsum('j,jkl->kl',axis,sigma))
    
    rot_x_to_z = rot([0,-1,0],np.pi/2)
    rot_y_to_z = rot([1,0,0],np.pi/2)
    rot_dict = {
            "X": rot_x_to_z,
            "Y": rot_y_to_z,
            "Z": np.eye(2)
        }
    outcome_list = ["up up","up down", "down up", "down down"]
    label_list = ["XX","XY","XZ","YX","YY","YZ","ZX","ZY","ZZ"]
    
    for name in filenames:
        exp_povm = np.load(name)
        base_name = os.path.basename(name)
        base_name = os.path.splitext(base_name)[0]

        print(base_name)

        for k in range(len(exp_povm)):

            matrices = np.einsum("ij,kjl,lm->kim",np.kron(rot_dict[label_list[k][0]],rot_dict[label_list[k][1]]).conj().T,exp_povm[k],
                                np.kron(rot_dict[label_list[k][0]],rot_dict[label_list[k][1]]))
            fig, axes = plt.subplots(len(matrices), 2, figsize=(8, 15))
            # Compute the maximum absolute value across all real and imaginary parts of all matrices
            max_abs = np.max([np.max(np.abs(np.real(matrix))) for matrix in matrices])
            max_abs1 = np.max([np.max(np.abs(np.imag(matrix))) for matrix in matrices])
            max_abs = np.max([max_abs,max_abs1])
            
            # Iterate through each matrix and its corresponding subplot
            for j, matrix in enumerate(matrices):
                
                for i, part in enumerate(['Real', 'Imaginary']):
                    if part == 'Real':
                        data = np.real(matrix)
                    else:
                        data = np.imag(matrix)

                    # Plotting real or imaginary part
                    ax = axes[j, i]
                    im = ax.imshow(data, cmap='RdYlBu', vmin=-max_abs, vmax=max_abs)
                    ax.set_title(f'Outcome {outcome_list[j]} ({part} Part)')
                    plt.colorbar(im, ax=ax)

            # Adjust layout
            fig.suptitle(f"{base_name} rotation to comp basis elements: {label_list[k]}")
            plt.tight_layout()
            save_path = f"{path_to_folder}/POVM_plots/{base_name}"
            path_exists=os.path.exists(save_path)
            if not path_exists:
                print(f'Making directory {save_path}.')
                os.makedirs(save_path)
            plt.savefig(f"{save_path}/{label_list[k]}.png")
            plt.close()
    return 1


def visualize_state(rho):
    max_abs = np.max(np.abs(np.imag(rho)))
    max_abs = np.max([max_abs,np.max(np.abs(np.real(rho)))])
                     
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, part in enumerate(['Real', 'Imaginary']):
        if part == 'Real':
            data = np.real(rho)
        else:
            data = np.imag(rho)

        # Plotting real or imaginary part
        ax = axes[i]
        im = ax.imshow(data, cmap='RdBu',vmin=-max_abs, vmax=max_abs)
        ax.set_title(f'{part} Part')
        #ax.set_title(f'Outcome {outcome_list[j]} ({part} Part)')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    return 1


def plot_dendrogram(data, plot_shape = (7,4), cutoff=None, save_path = None, x_lim = None, y_lim = None):
    

    if isinstance(data, QREM):
        print()
        fig, ax = plt.subplots(1, 1, figsize=plot_shape)
        if cutoff is None:
            cutoff = data.cluster_cutoff
        dn1 = sch.dendrogram(data.Z, ax=ax, above_threshold_color='C0',
                                orientation='top', color_threshold=cutoff)
        ax.plot([0, 1000], [cutoff, cutoff], 'r--',  label = 'Threshold' )

        
    else: # Data is a linkage map
        if cutoff is None:
            cutoff = 0.9


        fig, ax = plt.subplots(1, 1, figsize=plot_shape)
        dn1 = sch.dendrogram(data, ax=ax, above_threshold_color='C0',
                                orientation='top', color_threshold=cutoff)

        ax.plot([0, 1000], [cutoff, cutoff], 'r--',  label = 'Threshold' )

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim) 
    ax.set_ylabel('Distance')
    ax.set_xlabel('Qubit index')
    plt.xticks(fontsize=10)
    ax.legend()
    sch.set_link_color_palette(None)  # reset to default after use
    if save_path is not None:
        plt.savefig(f'dendrogram.pdf', dpi=300, bbox_inches='tight')


def power_law(x,a,b):
    return a*x**(b)

def plot_infidelity_curves(qst, adaptive_burnin = None):
    """
    Plots infidelity curves from a qst object.
    """
    plt.rcParams.update({'font.size': 20})
    cutoff = 100
    fitcutoff = 1000
    plt.figure(figsize=(15, 9))
    x = np.arange(len(qst.infidelity[0]))
    for i in range(qst.n_averages):

        plt.plot(x[cutoff:], qst.infidelity[i][cutoff:], alpha=0.3)
    mean_infidelity = np.mean(qst.infidelity, axis=0)
    plt.plot(x[cutoff:], mean_infidelity[cutoff:], label='Mean Infidelity', color='black', linewidth=2, linestyle='--')
    popt,pcov=curve_fit(power_law,x[fitcutoff:],mean_infidelity[fitcutoff:],p0=np.array([1,-0.5]))
    infiFit=power_law(x[fitcutoff:],popt[0],popt[1])

    #plt.plot(x[cutoff:],infiFit,label='Power Law Fit',color='red',linewidth=2)
    plt.plot(x[fitcutoff:],infiFit,'r--',label=rf'Fit, $N^a, a={"%.2f" % popt[1]}$')
    if adaptive_burnin is not None:
        plt.axvline(x=adaptive_burnin, color='red', linestyle='--', label='Adaptive starts')
    plt.xlabel('Number of measurements')
    plt.ylabel('Infidelity')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='both')
    plt.minorticks_on()
    plt.show()
    print(f'Mean final infidelity {mean_infidelity[-1]}')
    return 1


def plot_infidelity_from_folders(base_path):
    folders = [f for f in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(f)]
    print(folders)
    adaptive_infidelity_container = []
    nonadaptive_infidelity_container = []
    
    plt.rcParams.update({'font.size': 20})
    plot_cutoff = 10
    fitcutoff = 1000
    plt.figure(figsize=(15, 9))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    for i, folder in enumerate(folders):
        with open(f'{folder}/infidelity_container.npy', 'rb') as f:
            infidelity_dict = np.load(f, allow_pickle=True).item()
        with open(f'{folder}/settings.npy', 'rb') as f:    
            settings = np.load(f, allow_pickle=True).item()
        adaptive_infidelity_container.append(infidelity_dict['adaptive_infidelity_container'])
        nonadaptive_infidelity_container.append(infidelity_dict['nonadaptive_infidelity_container'])
    adaptive_infidelity_container = np.array(adaptive_infidelity_container)
    container_shape = adaptive_infidelity_container.shape
    print(adaptive_infidelity_container.shape)
    adaptive_infidelity_container = np.reshape(adaptive_infidelity_container, (container_shape[0]*container_shape[2], -1))

    mean_infidelities = np.mean(adaptive_infidelity_container, axis=0)
    print(mean_infidelities)
    x = np.arange(len(mean_infidelities))
    plt.plot(x[plot_cutoff:], mean_infidelities[plot_cutoff:], color=colors[i])
    popt,pcov=curve_fit(power_law,x[fitcutoff:],mean_infidelities[fitcutoff:],p0=np.array([1,-0.5]))
    infiFit=power_law(x[fitcutoff:],popt[0],popt[1])
    plt.plot(x[fitcutoff:],infiFit,'--',label=rf'Fit, $N^a, a={"%.2f" % popt[1]}$',color=colors[i])
        
    plt.xlabel('Number of measurements')
    plt.ylabel('Infidelity')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()


def plot_from_infidelity_container(infidelity_container_array, adaptive_burnin = None, labels = None):
    """
    Plots the average infidelity from a container of infidelity data.
    """
    plt.rcParams.update({'font.size': 20})
    cutoff = 10
    fitcutoff = 100
    plt.figure(figsize=(15, 9))

    for k, infidelity_container in enumerate(infidelity_container_array):
        
        
        x = np.arange(len(infidelity_container[0][0]))
        mean_infidelities = np.mean(infidelity_container, axis=1)
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        for i in range(len(mean_infidelities)):
            plt.plot(x[cutoff:], mean_infidelities[i][cutoff:], label=labels[i] if labels else None, color=colors[k])
            popt,pcov=curve_fit(power_law,x[fitcutoff:],mean_infidelities[i][fitcutoff:],p0=np.array([1,-0.5]))
            infiFit=power_law(x[fitcutoff:],popt[0],popt[1])
            plt.plot(x[fitcutoff:],infiFit,'--',label=rf'Fit, $N^a, a={"%.2f" % popt[1]}$',color=colors[k])
        if adaptive_burnin is not None:
            plt.axvline(x=adaptive_burnin, color='red', linestyle='--', label='Adaptive starts')
    # for i in range(len(infidelity_container)):
    #     plt.plot(x, infidelity_container[i], alpha=0.3)
    # mean_infidelity = np.mean(infidelity_container, axis=0)
    # plt.plot(x, mean_infidelity, label='Mean Infidelity', color='black', linewidth=2, linestyle='--')
    plt.xlabel('Number of measurements')
    plt.ylabel('Infidelity')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()
    return 1




from scipy.optimize import curve_fit
import numpy as np

def exponent_series_curvefit_logspan(
    y, x, span=1.5, step_factor=1.25, start_idx=0,
    p0=(1.0, -0.5), maxfev=10000, min_points=20,
    include_final_window=True
):
    """
    Slide full log windows: for left edge Xl, window is [Xl, Xl*span].
    Only evaluate windows that are COMPLETELY inside the valid data range
    (x>0, y>0, finite). No truncated windows at either end.

    Args:
        y, x: arrays (x must be non-decreasing)
        span (float): multiplicative width (>1), e.g., 3 means 1.0 decade ~0.477
        step_factor (float): multiplicative step between windows (>1)
        start_idx (int): first index allowed (e.g., fitcutoff)
        p0, maxfev: curve_fit params
        min_points (int): minimum points required in a window
        include_final_window (bool): also try a last full-span window whose
            right edge touches the max valid x (i.e., Xl = x_max_valid/span)

    Returns:
        x_centers (geom. means), alphas (exponents)
    """
    x = np.asarray(x); y = np.asarray(y)
    assert np.all(np.diff(x) >= 0), "x must be non-decreasing"

    # Keep only fully valid (positive, finite) samples
    m_full = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if not np.any(m_full):
        return np.array([]), np.array([])

    xv = x[m_full]; yv = y[m_full]

    # Respect start_idx by x value (align against the filtered xv)
    x_start_val = x[min(max(start_idx, 0), len(x)-1)]
    # Effective left boundary is the first valid x >= x_start_val
    x_left_min = xv[np.searchsorted(xv, x_start_val, side='left')]
    x_right_max = xv[-1]

    # For full coverage, left edge must satisfy: Xl * span <= x_right_max
    # Also must be >= x_left_min
    Xl = max(x_left_min, xv[0])
    # If the first candidate violates the right bound, push it up
    if Xl * span > x_right_max:
        # No full-span window fits at all
        return np.array([]), np.array([])

    xs, alphas = [], []

    def fit_window(Xl_candidate):
        """Fit a single full-span window [Xl, Xl*span], return (center, alpha) or None."""
        Xr_candidate = Xl_candidate * span
        if Xr_candidate > x_right_max:
            return None

        left = np.searchsorted(xv, Xl_candidate, side='left')
        right = np.searchsorted(xv, Xr_candidate, side='right')

        xx = xv[left:right]; yy = yv[left:right]
        # Need enough points
        if len(xx) < min_points:
            return None

        try:
            popt, _ = curve_fit(power_law, xx, yy, p0=np.array(p0), maxfev=maxfev)
            x_center = np.sqrt(Xl_candidate * Xr_candidate)  # geometric mean
            return x_center, popt[1]
        except Exception:
            return None

    # Main sweep with multiplicative steps
    while True:
        if Xl * span > x_right_max:
            break
        out = fit_window(Xl)
        if out is not None:
            xc, alpha = out
            xs.append(xc); alphas.append(alpha)

        # advance
        Xl *= step_factor
        if not np.isfinite(Xl) or Xl <= 0:
            break

    # Optional: ensure we also include a last full-span window that ends at x_right_max
    # (i.e., left edge at x_right_max/span), without duplicating an already-close window.
    if include_final_window:
        Xl_last = x_right_max / span
        if Xl_last >= x_left_min:
            out = fit_window(Xl_last)
            if out is not None:
                xc_last, a_last = out
                # Avoid near-duplicate centers
                if len(xs) == 0 or np.abs(np.log(xc_last) - np.log(xs[-1])) > np.log(step_factor)/2:
                    xs.append(xc_last); alphas.append(a_last)

    # Sort by x (in case final window appended out of order)
    if xs:
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        alphas = np.asarray(alphas)[order]
    else:
        xs = np.array([]); alphas = np.array([])

    return xs, alphas



