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
        plt.savefig(f'dendrogram.png', dpi=300, bbox_inches='tight')


def power_law(x,a,b):
    return a*x**(b)

def plot_infidelity_curves(qst):
    """
    Plots infidelity curves from a qst object.
    """
    cutoff = 100
    fitcutoff = 3000
    plt.figure(figsize=(10, 6))
    x = np.arange(len(qst.infidelity[0]))
    for i in range(qst.n_averages):

        plt.plot(x[cutoff:], qst.infidelity[i][cutoff:], alpha=0.3)
    mean_infidelity = np.mean(qst.infidelity, axis=0)
    plt.plot(x[cutoff:], mean_infidelity[cutoff:], label='Mean Infidelity', color='black', linewidth=2, linestyle='--')
    popt,pcov=curve_fit(power_law,x[fitcutoff:],mean_infidelity[fitcutoff:],p0=np.array([1,-0.5]))
    infiFit=power_law(x[fitcutoff:],popt[0],popt[1])

    #plt.plot(x[cutoff:],infiFit,label='Power Law Fit',color='red',linewidth=2)
    plt.plot(x[fitcutoff:],infiFit,'r--',label=rf'Fit, $N^a, a={"%.2f" % popt[1]}$')
    plt.xlabel('Shot Index')
    plt.ylabel('Infidelity')
    plt.title('Infidelity Curves')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()
    return 1