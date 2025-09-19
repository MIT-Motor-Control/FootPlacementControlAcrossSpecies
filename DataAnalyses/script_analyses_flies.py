import os, sys
import pickle
import numpy as np 
import warnings 
import scipy
from tqdm import tqdm 
from utils.test_plots_humans import *
from utils.utils_flies import *
from utils.plot_functions_flies import *
warnings.filterwarnings('ignore')

# Loading and pre-processing the data 
body_io, self_io = load_all_data_mm()
body_io_, self_io_ = load_all_data_mm_()
video_list, video_list_od = load_video_data_mm()
video_list_ = load_video_data_mm_()

tot_input_list, tot_input_self, tot_output_list, tot_output_self = concatenate_data(body_io[0], body_io[1], self_io[0], self_io[1])
tot_input_list_, tot_input_self_, tot_output_list_, tot_output_self_ = concatenate_data(body_io_[0], body_io_[1], self_io_[0], self_io_[1])
print(tot_input_list.shape, tot_output_list.shape)

# Concatenate the two parts of the data here 
tot_input_list = np.concatenate((tot_input_list, tot_input_list_),0)
tot_output_list = np.concatenate((tot_output_list, tot_output_list_),0)
tot_input_self = np.concatenate((tot_input_self, tot_input_self_),0)
tot_output_self = np.concatenate((tot_output_self, tot_output_self_),0)
tot_input_list_od = np.concatenate((tot_input_list_od, tot_input_list_od_),0)
tot_output_list_od = np.concatenate((tot_output_list_od, tot_output_list_od_),0)
tot_input_self_od = np.concatenate((tot_input_self_od, tot_input_self_od_),0)
tot_output_self_od = np.concatenate((tot_output_self_od, tot_output_self_od_),0)
video_list = np.concatenate((video_list, video_list_),0)



plot_feedforward_controller_final_time(tot_input_time, tot_output_time, video_list_time, bool_plot=False, bool_save=False, figname='time_ffwd_output_fly')
plot_feedforward_controller_final(tot_input_list, tot_output_list, video_list, bool_plot=False, bool_save=False, figname='feedforward_outputs_fly')
plot_feedforward_body_flies_final_outliers(tot_input_list, tot_output_list, video_list, bool_plot=False, bool_save=False, figname='tmp')
plot_feedforward_body_flies_final(tot_input_list, video_list, bool_plot=False, bool_save=False, figname='ffwd_body_flies')   




rsquare_time1_body = get_rsquare_time_body(tot_input_list, tot_output_time, video_list, 0)
rsquare_time2_body = get_rsquare_time_body(tot_input_list, tot_output_time, video_list, 1)
rsquare_time3_body = get_rsquare_time_body(tot_input_list, tot_output_time, video_list, 2)

rsquare_time1_self = get_rsquare_time_self(tot_input_self, tot_output_time, video_list, 0)
rsquare_time2_self = get_rsquare_time_self(tot_input_self, tot_output_time, video_list, 1)
rsquare_time3_self = get_rsquare_time_self(tot_input_self, tot_output_time, video_list, 2)
list_body_time = [rsquare_time1_body, rsquare_time2_body, rsquare_time3_body]
list_self_time = [rsquare_time1_self, rsquare_time2_self, rsquare_time3_self]

# Body prediction
rsquare_leg1_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 0, 0)
rsquare_leg1_lat,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 0, 1)

rsquare_leg2_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 1, 0)
rsquare_leg2_lat,fb_gains = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 1, 1)

rsquare_leg3_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 2, 0)
rsquare_leg3_lat,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 2, 1)

# Self prediction
rsquare_leg1_front_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 0, 0)
rsquare_leg1_lat_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 0, 1)

rsquare_leg2_front_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 1, 0)
rsquare_leg2_lat_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 1, 1)

rsquare_leg3_front_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 2, 0)
rsquare_leg3_lat_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 2, 1)

list_body = [rsquare_leg1_front, rsquare_leg1_lat, rsquare_leg2_front, rsquare_leg2_lat, rsquare_leg3_front, rsquare_leg3_lat]
list_self = [rsquare_leg1_front_self, rsquare_leg1_lat_self, rsquare_leg2_front_self, rsquare_leg2_lat_self, rsquare_leg3_front_self, rsquare_leg3_lat_self]


ind_difference = np.zeros((rsquare_leg1_front.shape[0],6))
for ii in range(ind_difference.shape[0]):
    ind_difference[ii,0] = np.max(rsquare_leg2_front[ii,:]-rsquare_leg2_front_self[ii,:])
    ind_difference[ii,1] = np.max(rsquare_leg2_lat[ii,:]-rsquare_leg2_lat_self[ii,:])
    ind_difference[ii,2] = np.max(rsquare_leg1_front[ii,:]-rsquare_leg1_front_self[ii,:])
    ind_difference[ii,3] = np.max(rsquare_leg1_lat[ii,:]-rsquare_leg1_lat_self[ii,:])
    ind_difference[ii,4] = np.max(rsquare_leg3_front[ii,:]-rsquare_leg3_front_self[ii,:])
    ind_difference[ii,5] = np.max(rsquare_leg3_lat[ii,:]-rsquare_leg3_lat_self[ii,:])



plot_rsquares_timing(list_body_time, list_self_time, bool_plot=False, bool_save=False, figname='final_rsquares_time')
plot_rsquares_final(list_body, list_self, bool_plot=False, bool_save=False, figname='final_rsquares_flies_video')


# Temporary location for the laterality plots
vel_tier = add_velocity_tier(tot_input_list)
regression_small, regression_large = regression_fr_laterality_velidp(tot_input_list, tot_output_list, video_list, 1)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_final')


list_self_with_horizons = [] 
list_input_horizon, list_output_horizon = [], []
list_self_with_horizons.append(tot_input_self)
for horizon in tqdm(range(2,6)):
    body_io_h, self_io_h = load_all_data_horizon(horizon)
    tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_output_self_h = concatenate_data(body_io_h[0], body_io_h[1], self_io_h[0], self_io_h[1])
    tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_ouptut_self_h = flip_data(tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_output_self_h)
    list_self_with_horizons.append(tot_input_self_h)
    list_input_horizon.append(tot_input_list_h)
    list_output_horizon.append(tot_output_list_h)


plot_successive_contacts_fly(list_self_with_horizons, video_list, bool_plot=False, bool_save=False, figname='successive_contacts')



vel_tier = add_velocity_tier(tot_input_list)
regression_matrix_diag_leg2 = regression_fr_body(vel_tier, tot_input_list, tot_output_list, video_list, 1)

plot_normalized_feedback_gains(regression_matrix_diag_leg2, bool_plot=False, bool_save=False, figname='fb_gains_lat_lat')



regression_small, regression_large = regression_fr_laterality_velidp(tot_input_list, tot_output_list, video_list, 1)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_final')
