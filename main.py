import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
import cProfile, pstats
import re

import scipy
import sys
import os
import uuid
from EMQST_lib import unit_test as ut
from EMQST_lib import support_functions as sf
from EMQST_lib import emqst
import time



def main():
    # Checks if the code is being run on the cluster
    # On cluster the main script is run as 'python main.py True' in the sbatch script
    if (len(sys.argv)>1 and sys.argv[-1]=='True'):
        print("We are on the cluster!")
        boolOnCluster=True
    else:
        print("We are NOT on the cluster!")
        boolOnCluster=False

    np.random.seed(0)

    n_qubits=2
    n_QST_shots=int(10**4)+2
    n_calibration_shots=10**4
    n_cores=8
    if boolOnCluster:
        n_cores=48
    noise_mode=2
    # 0: No noise
    # 1: Depolarizing noise
    # 2: Stronger depolarized noise
    # 3: Amplitude damping
    # 4: Constant rotation around x-axis

    n_averages=2
    exp_dictionary={}
    list_of_true_angles=np.array([[np.pi/2,0],[np.pi/2,np.pi],
                        [np.pi/2,np.pi/2],[np.pi/2,3*np.pi/2],
                        [0,0],[np.pi,0]])
    #list_of_true_angles=np.array([[np.pi/2,0]])
    start=time.time()
    list_of_true_states=np.array([sf.get_projector_from_angles(np.array([angles])) for angles in list_of_true_angles])
    #print(list_of_true_states)
    #list_of_true_states=np.array([[[1/2,1/2],[1/2,1/2]]],dtype=complex)
    list_of_true_states=np.array([sf.generate_random_pure_state(n_qubits) for _ in range(n_averages)])
    #n_averages=len(list_of_true_states)
    #print(list_of_true_states)
    
    uncorrected_infidelity, corrected_infidelity,rho_estm=emqst.emqst(n_qubits,n_QST_shots,n_calibration_shots,list_of_true_states, n_cores=n_cores,noise_mode=noise_mode )

    

    end = time.time()
    print(f'Total dt runtime: {end - start}')
    #dicta={"Function_name": sp_file_test}
    #ut.main(dicta)
    
    return 1


def main2():
        #dicta={"Function_name": sp_file_test}
    ut.main()
    
    return 1

def profiler():
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()
    #stats.dump_stats('profiler/export-data.txt')
    now=datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    string_name= now_string+str(uuid.uuid4())
    with open(f'profiler\profiling_{string_name}.txt', "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats()
    #cProfile.run('main()') 

if __name__=="__main__":
    #main2()
    main()
    #profiler()
    
