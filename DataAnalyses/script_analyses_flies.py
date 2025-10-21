# This script investigates the feedforward and feedback components of the fly locomotion datasets presented in the paper.
import os, sys
import pickle
import numpy as np 
import warnings 
import scipy
from tqdm import tqdm 
from utils.utils_flies import *
from utils.plot_functions_flies import *
warnings.filterwarnings('ignore')



# Loading and pre-processing the data 
body_io, self_io = load_all_data_mm()
video_list, video_list_od = load_video_data_mm()
# Extracting the input and output data for the body-state and baseline foot placement control models
tot_input_list, tot_input_self, tot_output_list, tot_output_self = concatenate_data(body_io[0], body_io[1], self_io[0], self_io[1])
tot_input_list, tot_input_self, tot_output_list, tot_output_self = flip_data(tot_input_list, tot_input_self, tot_output_list, tot_output_self)


###################################################
### PART 1 - Velocity dependent ffwd controller ###
###################################################

plot_feedforward_controller_final(tot_input_list, tot_output_list, video_list, bool_plot=False, bool_save=False, figname='feedforward_outputs_fly')


########################################################
### PART 2 - Body-state depedent feedback controller ###
########################################################

# Computing the rsquares matrices for the body prediction
rsquare_leg1_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 0, 0)
rsquare_leg1_lat,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 0, 1)

rsquare_leg2_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 1, 0)
rsquare_leg2_lat,fb_gains = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 1, 1)

rsquare_leg3_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 2, 0)
rsquare_leg3_lat,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 2, 1)

# Computing the rsquares matrices for the self prediction
rsquare_leg1_front_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 0, 0)
rsquare_leg1_lat_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 0, 1)

rsquare_leg2_front_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 1, 0)
rsquare_leg2_lat_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 1, 1)

rsquare_leg3_front_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 2, 0)
rsquare_leg3_lat_self = get_rsquare_matrix(tot_input_self, tot_output_list, video_list, 2, 1)

list_body = [rsquare_leg1_front, rsquare_leg1_lat, rsquare_leg2_front, rsquare_leg2_lat, rsquare_leg3_front, rsquare_leg3_lat]
list_self = [rsquare_leg1_front_self, rsquare_leg1_lat_self, rsquare_leg2_front_self, rsquare_leg2_lat_self, rsquare_leg3_front_self, rsquare_leg3_lat_self]

plot_rsquares_final(list_body, list_self, bool_plot=False, bool_save=False, figname='final_rsquares_flies_video')


############################################################
### PART 3 - Characterization of the feedback controller ###
############################################################

# Computing the difference between the body state and baseline models.

ind_difference = np.zeros((rsquare_leg1_front.shape[0],6))
for ii in range(ind_difference.shape[0]):
    ind_difference[ii,0] = np.max(rsquare_leg2_front[ii,:]-rsquare_leg2_front_self[ii,:])
    ind_difference[ii,1] = np.max(rsquare_leg2_lat[ii,:]-rsquare_leg2_lat_self[ii,:])
    ind_difference[ii,2] = np.max(rsquare_leg1_front[ii,:]-rsquare_leg1_front_self[ii,:])
    ind_difference[ii,3] = np.max(rsquare_leg1_lat[ii,:]-rsquare_leg1_lat_self[ii,:])
    ind_difference[ii,4] = np.max(rsquare_leg3_front[ii,:]-rsquare_leg3_front_self[ii,:])
    ind_difference[ii,5] = np.max(rsquare_leg3_lat[ii,:]-rsquare_leg3_lat_self[ii,:])

# Representation of the control magnitude
fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Control magnitude', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.linspace(2.8,3.2,8),ind_difference[:,1],color='r',s=5, alpha=0.5)
axs.scatter(3, np.nanmedian(ind_difference[:,1]), color='r',s=20)
axs.plot([3,3], [np.nanmedian(ind_difference[:,1])+scipy.stats.iqr(ind_difference[:,1], nan_policy='omit'),np.nanmedian(ind_difference[:,1])-scipy.stats.iqr(ind_difference[:,1], nan_policy='omit')],'r',lw=2)
axs.set_xlim([0.5,3.5]), axs.set_ylabel('Control Magnitude'), axs.set_ylim([0,1])
plt.tight_layout()

# Control timescale timescale analyses 
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


###########################################################################
### PART 4 - Investigation of the laterality for foot placement control ###
###########################################################################


regression_small, regression_large = regression_fr_laterality_velidp(tot_input_list, tot_output_list, video_list, 1)
plot_laterality_regression(regression_small, regression_large, bool_plot=True, bool_save=False, figname='laterality_final')
