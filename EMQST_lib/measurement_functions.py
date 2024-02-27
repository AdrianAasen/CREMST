import numpy as np
from EMQST_lib import support_functions as sf



def measurement(n_shots,povm,rho,bool_exp_measurements,exp_dictionary,state_angle_representation=None, custom_measurement_function = None):
    """
    Measurment settings and selects either experimental or simulated measurements. 
    For experimental measurements some settings are converted to angle arrays. 
    """

    if bool_exp_measurements:
        if custom_measurement_function is None:
            if state_angle_representation is None:
                print("Experimental measurement: No angle representation has been given! Returning None.")
                return np.array([None]*n_shots) 
            outcome_index = exp_dictionary["standard_measurement_function"](n_shots,povm.get_angles(),state_angle_representation,exp_dictionary)
        else:
            outcome_index = custom_measurement_function(n_shots,povm.get_angles(),exp_dictionary)
    else:
        outcome_index = simulated_measurement(n_shots,povm,rho)
        
    return outcome_index


def simulated_measurement(n_shots,povm,rho):

    """
    Takes in number of shots required from a single POVM on a single quantum states.
    Returns and outcome_index vector where the index corresponds the the POVM that occured.
    """

    # Find probabilites for different outcomes
    histogram=povm.get_histogram(rho)
    #print(histogram)
    cumulative_sum=np.cumsum(histogram)

    # Sample outcomes 
    r=np.random.random(n_shots)

    # Return index list of outcomes 
    return np.searchsorted(cumulative_sum,r)
