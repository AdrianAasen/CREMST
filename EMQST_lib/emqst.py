import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy.optimize import curve_fit


from EMQST_lib import support_functions as sf
#from EMQST_lib import measurement_functions as mf
from EMQST_lib import dt
from EMQST_lib.qst import QST
from EMQST_lib.povm import POVM


def emqst(n_qubits,n_QST_shots_each,n_calibration_shots_each,true_state_list, 
          calibration_mode=None,bool_exp_measurements=False,exp_dictionary={},
          n_cores=1,noise_mode=0, true_state_angles_list=None,method="MLE", perform_unmitigated_QST=True):
    """
    Performs a complete cycle of noise corrected POVM with random sampling of states.
    Takes in experimental parameters. To be passed to both POVM calibration and QST. 
    Returns estimator dictionary.

    n_QST_shots_each        # shots in QST reconstruction for each POVM used. 
    n_calibraion_shots_each # shots for each calibration measurement.
    trueAngleList           List of random true states to average over.
    calibration_mode        Defines list of calibration states, Pauli (default) or SIC.
    

    returns results dictionary
    """

    

    # Check if restuls exist:
    check_path='results'
    path_exists=os.path.exists(check_path)
    if not path_exists:
        print("Created results dictionary.")
        os.makedirs('results')



    # Generate new dictionary for current run
    now=datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    dir_name= now_string+str(uuid.uuid4())


    data_path=f'results/{dir_name}'
    os.mkdir(data_path)

    with open(f'{data_path}/experimental_settings.npy','wb') as f:
        np.save(f,exp_dictionary)


    
    
    POVM_list=POVM.generate_Pauli_POVM(n_qubits)
    
    print('Starting EMQST.')
    print(f'----------------------------')
    print(f'Error corrected {method}.')
    if calibration_mode is None or calibration_mode == 'Pauli':
        print(f'POVM calibration states are Pauli eigenstates.')
        calibration_states,calibration_angles=sf.get_calibration_states(n_qubits)    
    elif calibration_mode == 'SIC':
        print(f'POVM calibration states are SIC.')
        calibration_states,calibration_angles=sf.get_calibration_states(n_qubits,"SIC")
    print(f'{n_qubits} qubit(s).')
    print(f'Total shot count using spin measurements: ')
    print(f'{n_calibration_shots_each*len(calibration_states)*len(POVM_list):,} POVM calibration shots.')
    #print(f'(Using spin meaurements, : {n_calibration_shots_each*len(calibration_states)*len(POVM_list):,})')
    if n_QST_shots_each == 0:
        print(f'QST not performed.')
    else: 
        print(f'{n_QST_shots_each*len(POVM_list):,} QST shots.')
        print(f'{len(true_state_list):,} QST averages.')
    print(f'----------------------------')


    # If experimental measurements are set, do not apply noise methods.
    if bool_exp_measurements:
        noise_mode=0
        print("Noise mode is disabled as experimental measurements are performed.")

    if noise_mode:
        print(f'Synthetic noise mode {noise_mode}.')
        if  n_qubits<=2:
            noisy_POVM_list=np.array([POVM.generate_noisy_POVM(povm,noise_mode) for povm in POVM_list])
            
        else: # Only depolarizing noise is implemented for more than 2 qubits
            noisy_POVM_list=np.array([POVM.depolarized_POVM(povm) for povm in POVM_list])
    else:
        noisy_POVM_list=POVM_list
        print("No synthetic noise.")
    dt_start=time.time()
    
    reconstructed_POVM_list = dt.device_tomography(n_qubits,n_calibration_shots_each,noisy_POVM_list,calibration_states,n_cores=n_cores, bool_exp_meaurements=bool_exp_measurements,exp_dictionary=exp_dictionary,initial_guess_POVM=POVM_list,calibration_angles=calibration_angles)

    dt_end = time.time()
    print(f'Runtime of DT reconstruction {dt_end - dt_start}')
    DT_settings={
        "n_qubits": n_qubits,
        "calibration_states": calibration_states,
        "n_calibration_shots": n_calibration_shots_each,
        "initial_POVM": POVM_list,
        "reconstructed_POVM_list": reconstructed_POVM_list,
        "bool_exp_meaurements": bool_exp_measurements,
        "noise_mode": noise_mode,
        "noisy_POVM_list" : noisy_POVM_list,
        "reconstructed_POVM_matrix":  np.array([povm.get_POVM() for povm in reconstructed_POVM_list])
    }

    with open(f'{data_path}/DT_settings.npy','wb') as f:
        np.save(f,DT_settings)

    

    for i in range (len(reconstructed_POVM_list)):
        print(f'Distance between reconstructed and noisy POVM: {sf.POVM_distance(reconstructed_POVM_list[i].get_POVM(),noisy_POVM_list[i].get_POVM())}')

    print("POVM calibration complete.\n----------------------------")
    
    # Define placeholder variables in case QST is not run. 
    uncorrected_infidelity=np.empty(0)
    corrected_infidelity=np.empty(0)
    corrected_rho_estm=np.empty(0)
    uncorrected_rho_estm=np.empty(0)
    
    if n_QST_shots_each == 0:
        print("QST not performed.")
    else:       
        qst=QST(POVM_list,true_state_list,n_QST_shots_each,n_qubits,bool_exp_measurements,exp_dictionary,n_cores=n_cores,noise_corrected_POVM_list=reconstructed_POVM_list,true_state_angles_list=true_state_angles_list)
        qst.generate_data(override_POVM_list=noisy_POVM_list)
        
        # Save data settings
        qst.save_QST_settings(data_path,noise_mode)
        print("Generated data.")

        print("Start corrected QST.")
        if method=="MLE":
            qst.perform_MLE(override_POVM_list=reconstructed_POVM_list)
        elif method=="BME":
            qst.perform_BME(override_POVM_list=reconstructed_POVM_list)
        corrected_infidelity=qst.get_infidelity()
        corrected_rho_estm=qst.get_rho_estm()

        print("Corrected QST complete.\n----------------------------")
        
        # Run comparative BME with uncorrected POVMs
        
        if perform_unmitigated_QST:
            print("Start uncorrected QST.")
            if method=="MLE":
                qst.perform_MLE()
            elif method=="BME":
                qst.perform_BME()
            uncorrected_infidelity=qst.get_infidelity()
            uncorrected_rho_estm=qst.get_rho_estm()
            
            
            print("Uncorrected QST complete.\n----------------------------") 

        n_averages=len(true_state_list)
        sample_step=np.arange(len(corrected_infidelity[0]))
        corrected_average=np.sum(corrected_infidelity,axis=0)/n_averages
        uncorrected_average=np.sum(uncorrected_infidelity,axis=0)/n_averages

        with open(f'{data_path}/QST_results.npy','wb') as f:
            np.save(f,corrected_infidelity )
            np.save(f,uncorrected_infidelity)
            np.save(f,corrected_rho_estm)
            np.save(f,uncorrected_rho_estm)

        
        # Generate plots if not run on a cluster.
        if n_cores < 10 and method == "BME" and perform_unmitigated_QST and n_QST_shots_each > 999:
            cutoff = 10
            popt_corr,pcov_corr = curve_fit(sf.power_law,sample_step[1000:],corrected_average[1000:], p0 = np.array([1,-0.5]))
            corr_fit = sf.power_law(sample_step[cutoff:],popt_corr[0],popt_corr[1])
            popt_uncorr,pcov_uncorr = curve_fit(sf.power_law,sample_step[1000:],uncorrected_average[1000:], p0 = np.array([1,-0.5]))
            uncorr_fit = sf.power_law(sample_step[cutoff:],popt_uncorr[0],popt_uncorr[1])

            plt.figure(figsize=(8,6))
            plt.plot(sample_step[cutoff:],corrected_average[cutoff:],'r', label="Corrected")
            plt.plot(sample_step[cutoff:],uncorrected_average[cutoff:],'b',label="Uncorrected")
            plt.plot(sample_step[cutoff:],corr_fit,'r--',label=rf'Fit, $N^a, a={"%.2f" % popt_corr[1]}$')
            plt.plot(sample_step[cutoff:],uncorr_fit,'b--',label=rf'Fit, $N^a, a={"%.2f" % popt_uncorr[1]}$')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim(100,len(sample_step))
            #plt.ylim(10**(-5),10**(-0))
            plt.ylabel('Mean Infidelity')
            plt.xlabel('Number of shots')
            plt.legend(loc="lower left",prop={'size': 16})
            plt.tight_layout

            plt.savefig(f'{data_path}/Averaged_infidelities.png')
            plt.savefig('latest_run.png')

    # Pack everything you want to return into the results dictionary    
    result = {
        "corrected_infidelity" : corrected_infidelity,
        "uncorrected_infidelity": uncorrected_infidelity,
        "corrected_rho_estm" : corrected_rho_estm,
        "uncorrected_rho_estm" : uncorrected_rho_estm,
        "reconstructed_POVM" : np.array([povm.get_POVM() for povm in reconstructed_POVM_list]),
        "synthetic_POVM": np.array([povm.get_POVM() for povm in noisy_POVM_list])
    }
    print("EMQST complete.")
    return result



      

