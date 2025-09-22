import os, sys
import numpy as np
import pickle
import warnings
import scipy
from tqdm import tqdm 
from scipy.stats import linregress
import matplotlib.pyplot as plt 

from utils.utils_humans import *
from utils.plot_functions_humans import *
# from utils.test_plots_humans import *
warnings.filterwarnings('ignore')
PATH_CAMARGO_DATASET = os.path.join(os.getcwd(), 'Datasets','Humans','Camargo','processed')
PATH_CAMARGO_DATASET_TIME = os.path.join(os.getcwd(), 'Datasets','Humans','Camargo','processed_time_bis_250')
# difference_matrix = np.load(os.path.join(PATH_CAMARGO_DATASET_TIME,'difference_matrix_humans.npy'))

tmp_output_path = os.path.join(os.getcwd(),'round_2_analysis')
os.makedirs(tmp_output_path, exist_ok=True)

# Loading the datasets 
list_io1, list_io2, list_sub = load_data_camargo()
tot_input_leg1, tot_self_input_leg1, tot_output_leg1, tot_self_output_leg1 = list_io1[0], list_io1[1], list_io1[2], list_io1[3]
tot_input_leg2, tot_self_input_leg2, tot_output_leg2, tot_self_output_leg2 = list_io2[0], list_io2[1], list_io2[2], list_io2[3]
tot_sub1, tot_sub2 = list_sub[0], list_sub[1]

# Loading the datasets containing contact timing
list_io1_time, list_io2_time, list_sub_time = load_data_camargo_timing()
tot_input_leg1_time, tot_self_input_leg1_time, tot_output_leg1_time, tot_self_output_leg1_time = list_io1_time[0], list_io1_time[1], list_io1_time[2], list_io1_time[3]
tot_input_leg2_time, tot_self_input_leg2_time, tot_output_leg2_time, tot_self_output_leg2_time = list_io2_time[0], list_io2_time[1], list_io2_time[2], list_io2_time[3]
tot_sub1_time, tot_sub2_time = list_sub_time[0], list_sub_time[1]


###################################################
### PART 1 - Velocity dependent ffwd controller ###
###################################################
plot_feedforward_output_humans_time_paper(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time, bool_plot=False, bool_save=False, bool_fb=False, figname='ffwd_output_h_final_time')
plot_feedforward_output_humans_paper(tot_input_leg1, tot_output_leg1, tot_sub1, bool_plot=True, bool_save=False, bool_fb=False, figname='ffwd_output_h_final')
plot_feedforward_body_humans_final(tot_input_leg1, tot_sub1, bool_plot=False, bool_save=False, figname='ffwd_body_h_final')


###############################################################
### PART 2 - Controller inference on the remaining variance ###
###############################################################
rsquare_time_body = get_rsquare_time_matrix_camargo(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time)
rsquare_time_self = get_rsquare_self_time_matrix_baseline(tot_self_input_leg1_time, tot_output_leg1_time, tot_sub1_time)

rsquare_foreaft_body,_ = get_rsquare_matrix_camargo(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time, 0)
rsquare_lateral_body, gains_matrix = get_rsquare_matrix_camargo(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time, 1)

rsquare_foreaft_self = get_rsquare_self_matrix_baseline(tot_self_input_leg1, tot_output_leg1, tot_sub1, 0)
rsquare_lateral_self = get_rsquare_self_matrix_baseline(tot_self_input_leg1, tot_output_leg1, tot_sub1, 1)

plot_rsquares_final(rsquare_foreaft_body, rsquare_lateral_body, rsquare_foreaft_self, rsquare_lateral_self, bool_plot=True, bool_save=True, figname='rsquares_final_velocity_250')


############################################################
### PART 3 - Characterization of the feedback controller ###
############################################################

# Correlation between successive contacts 
list_input_time, list_subjects = [],[]
list_input_fr, list_output_fr = [], []
list_input_time.append(list_io1[1]), list_subjects.append(list_sub[0])
for horizon in range(2,5):
    list_horizon_input, list_horizon_sub = load_camargo_horizon(horizon)
    local_input_fr, local_output_fr, _ = load_camargo_horizon_io(horizon)
    list_input_time.append(list_horizon_input)
    list_subjects.append(list_horizon_sub)
    list_input_fr.append(local_input_fr)
    list_output_fr.append(local_output_fr)


plot_correlation_horizon_final(list_input_time, list_subjects, bool_plot=False, bool_save=False, figname='horizon_correlation')


###########################################################################
### PART 4 - Investigation of the laterality for foot placement control ###
###########################################################################

regression_small, regression_large = regression_laterality(tot_input_leg1, tot_output_leg1, tot_sub1)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_human')
