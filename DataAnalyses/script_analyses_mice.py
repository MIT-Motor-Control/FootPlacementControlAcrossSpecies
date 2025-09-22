import os, sys
import pickle
import numpy as np 
import warnings 
import scipy
import scipy.stats
from tqdm import tqdm
# from utils.test_plots_humans import *
from utils.utils_mice import *
from utils.plot_functions_mice import *
warnings.filterwarnings('ignore')

# Loading the data
body_io, self_io, timing_io = load_all_data()
body_i_abs, body_o_abs = load_data_absolute()



tot_input_list, tot_input_self, tot_output_list, tot_output_self, tot_animal = concatenate_data(body_io[0], body_io[1], self_io[0], self_io[1])
tot_input_list_od, tot_input_self_od, tot_output_list_od, tot_output_self_od, tot_animal_od = concatenate_data(body_io[2], body_io[3], self_io[2], self_io[3])

tot_input_absolute, tot_output_absolute = body_i_abs[0], body_o_abs[0]
tot_animal_absolute = 0 * np.ones((body_i_abs[0].shape[0],1))
for animal in range(1, 80):
    tot_input_absolute = np.vstack((tot_input_absolute, body_i_abs[animal]))
    tot_output_absolute = np.vstack((tot_output_absolute, body_o_abs[animal]))
    tot_animal_absolute = np.vstack((tot_animal_absolute, animal*np.ones((body_i_abs[animal].shape[0],1))))


difference_matrix = np.load(os.path.join(os.getcwd(),'Datasets','Mouse','Klibaite','Control','difference_matrix_mice.npy'))


vel_tier = add_velocity_tier(tot_input_list, tot_input_self)
vel_tier_od = add_velocity_tier(tot_input_list_od, tot_input_self_od)
velocity_vector = get_velocity_vector(tot_input_list, tot_input_self)
velocity_vector_od = get_velocity_vector(tot_input_list_od, tot_input_self_od)


###################################################
### PART 1 - Velocity-dependent ffwd controller ###
###################################################
plot_feedforward_controller_final(velocity_vector, tot_output_list, tot_animal, tot_input_absolute, tot_output_absolute, tot_animal_absolute, bool_plot=True, bool_save=False, figname='feedforward_figures_mouse')


list_input_horizon, list_output_horizon = [], []
list_input_horizon.append(tot_input_self), list_output_horizon.append(tot_output_self)
for horizon in range(3,6):
    input_list_horizon, output_list_horizon = load_data_horizon(horizon)
    tot_input_self_horizon, tot_output_self_horizon = concatenate_data_horizon(input_list_horizon, output_list_horizon)
    list_input_horizon.append(tot_input_self_horizon), list_output_horizon.append(tot_output_self_horizon)

plot_correlation_horizon_final(list_input_horizon, tot_animal, bool_plot=True, bool_save=True, figname='horizon_correlation')


###############################################################
### PART 2 - Controller inference on the remaining variance ###
###############################################################
# Get the rsquares matrix for the location 
rsquare_front_foreaft_body,_ = get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind=0, bool_lat=0)
rsquare_front_lateral_body,gains_front_lateral = get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind=0, bool_lat=1)
rsquare_hind_foreaft_body,_ = get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind=1, bool_lat=0)
rsquare_hind_lateral_body,_ = get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind=1, bool_lat=1)
rsquare_front_foreaft_self = get_rsquare_matrix_self(tot_animal, tot_animal_od, tot_input_list, tot_input_self, tot_input_list_od, tot_input_self_od, tot_output_list, tot_output_self_od, bool_hind=0, bool_lat=0)
rsquare_front_lateral_self = get_rsquare_matrix_self(tot_animal, tot_animal_od, tot_input_list, tot_input_self, tot_input_list_od, tot_input_self_od, tot_output_list, tot_output_self_od, bool_hind=0, bool_lat=1)
rsquare_hind_foreaft_self = get_rsquare_matrix_self(tot_animal, tot_animal_od, tot_input_list, tot_input_self, tot_input_list_od, tot_input_self_od, tot_output_list, tot_output_self_od, bool_hind=1, bool_lat=0)
rsquare_hind_lateral_self = get_rsquare_matrix_self(tot_animal, tot_animal_od, tot_input_list, tot_input_self, tot_input_list_od, tot_input_self_od, tot_output_list, tot_output_self_od, bool_hind=1, bool_lat=1)
# Get the rsquares matrix for the timing 
rsquare_front_time_body = get_rsquare_matrix_feedback_time(tot_animal, tot_input_list, tot_output_absolute, bool_hind=0)
rsquare_hind_time_body = get_rsquare_matrix_feedback_time(tot_animal, tot_input_list, tot_output_absolute, bool_hind=1)
rsquare_front_time_self = get_rsquare_matrix_self_time(tot_animal, tot_input_list, tot_input_self, tot_output_absolute, bool_hind=0)
rsquare_hind_time_self = get_rsquare_matrix_self_time(tot_animal, tot_input_list, tot_input_self, tot_output_absolute, bool_hind=1)

plot_rsquares_time(rsquare_front_time_body, rsquare_hind_time_body, rsquare_front_time_self, rsquare_hind_time_self, bool_plot=False, bool_save=False, figname='rsquares_contact_timing')




ind_difference = np.zeros((rsquare_front_foreaft_body.shape[0],4))
for animal in range(ind_difference.shape[0]):
    ind_difference[animal,0] = np.max(rsquare_front_foreaft_body[animal,:] - rsquare_front_foreaft_self[animal,:])
    ind_difference[animal,1] = np.max(rsquare_front_lateral_body[animal,:] - rsquare_front_lateral_self[animal,:])
    ind_difference[animal,2] = np.max(rsquare_hind_foreaft_body[animal,:] - rsquare_hind_foreaft_self[animal,:])
    ind_difference[animal,3] = np.max(rsquare_hind_lateral_body[animal,:] - rsquare_hind_lateral_self[animal,:])


plot_rsquares_final(rsquare_front_foreaft_body, rsquare_front_lateral_body, rsquare_hind_foreaft_body, rsquare_hind_lateral_body,
                    rsquare_front_foreaft_self, rsquare_front_lateral_self, rsquare_hind_foreaft_self, rsquare_hind_lateral_self,
                    bool_plot=False, bool_save=False, figname='rsquares_final_mice')


############################################################
### PART 3 - Characterization of the feedback controller ###
############################################################

# Correlation between successive contacts 
list_input_horizon, list_output_horizon = [], []
list_input_horizon.append(tot_input_self), list_output_horizon.append(tot_output_self)
for horizon in range(3,6):
    input_list_horizon, output_list_horizon = load_data_horizon(horizon)
    tot_input_self_horizon, tot_output_self_horizon = concatenate_data_horizon(input_list_horizon, output_list_horizon)
    list_input_horizon.append(tot_input_self_horizon), list_output_horizon.append(tot_output_self_horizon)

plot_correlation_horizon_final(list_input_horizon, tot_animal, bool_plot=False, bool_save=False, figname='horizon_correlation')

# 3.3 Normalized feedback gains for the lateral deviation
regression_matrix = compute_regression_fr(tot_animal, tot_input_list, tot_output_list, vel_tier,0)
plot_normalized_feedback_gains(regression_matrix, bool_plot=False, bool_save=False, figname='fbgains_lat_lat')

###############################################################################
### PART 4 - Investigation of the laterality for the foot placement control ###
###############################################################################
regression_small, regression_large = regression_fr_laterality_velidp(tot_animal, tot_input_list, tot_output_list, 0)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality')
