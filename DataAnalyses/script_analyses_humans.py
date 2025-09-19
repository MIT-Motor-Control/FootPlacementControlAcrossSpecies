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
from utils.test_plots_humans import *
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

# Loading the datasets with time
list_io1_time, list_io2_time, list_sub_time = load_data_camargo_timing()
tot_input_leg1_time, tot_self_input_leg1_time, tot_output_leg1_time, tot_self_output_leg1_time = list_io1_time[0], list_io1_time[1], list_io1_time[2], list_io1_time[3]
tot_input_leg2_time, tot_self_input_leg2_time, tot_output_leg2_time, tot_self_output_leg2_time = list_io2_time[0], list_io2_time[1], list_io2_time[2], list_io2_time[3]
tot_sub1_time, tot_sub2_time = list_sub_time[0], list_sub_time[1]

# bootstrap_kl_laterality(tot_input_leg1, tot_output_leg1)
# laterality_bootstrap_humans(tot_input_leg1,tot_output_leg1)
# laterality_error_bar(tot_input_leg1, tot_output_leg1,bool_save=True, figname='seaborn_regressions')
# plot_scatter_laterality(tot_input_leg1, tot_output_leg1, bool_plot=True)
# sys.exit()

# np.save('AAATOTRANSFER.npy', tot_output_leg1_time)
# idx_nans = np.where(~np.isnan(tot_output_leg1_time[:,0]))[0]
# reg_test = scipy.stats.linregress(tot_output_leg1_time[idx_nans, 1], tot_output_leg1_time[idx_nans, 3])
# print(reg_test)
# sys.exit()


###################################################
### PART 1 - Velocity dependent ffwd controller ###
###################################################
# plot_feedforward_output_humans_time_paper(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time, bool_plot=False, bool_save=False, bool_fb=False, figname='ffwd_output_h_final_time')
# plot_feedforward_output_humans_paper(tot_input_leg1, tot_output_leg1, tot_sub1, bool_plot=False, bool_save=False, bool_fb=False, figname='ffwd_output_h_final')
# plot_feedforward_body_humans_final(tot_input_leg1, tot_sub1, bool_plot=False, bool_save=False, figname='ffwd_body_h_final')
# # regression_small, regression_large = regression_laterality(tot_input_leg1, tot_output_leg1, tot_sub1)


# ###############################################################
# ### PART 2 - Controller inference on the remaining variance ###
# ###############################################################
# rsquare_time_body = get_rsquare_time_matrix_camargo(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time)
# rsquare_time_self = get_rsquare_self_time_matrix_baseline(tot_self_input_leg1_time, tot_output_leg1_time, tot_sub1_time)

# print(tot_input_leg1.shape, tot_input_leg1_time.shape)

# rsquare_foreaft_body,_ = get_rsquare_matrix_camargo(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time, 0)
rsquare_lateral_body, gains_matrix = get_rsquare_matrix_camargo(tot_input_leg1_time, tot_output_leg1_time, tot_sub1_time, 1)
# print(gains_matrix.shape)
np.save(os.path.join(tmp_output_path,'matrix_gains_humans.npy'), gains_matrix)
print('Data saved')
sys.exit()
# rsquare_foreaft_self = get_rsquare_self_matrix_baseline(tot_self_input_leg1, tot_output_leg1, tot_sub1, 0)
# rsquare_lateral_self = get_rsquare_self_matrix_baseline(tot_self_input_leg1, tot_output_leg1, tot_sub1, 1)

# # Compute the stats for the rsquares plots ... 
# pvalues_matrix_rsquares = get_stats_feedback_controller(rsquare_foreaft_body, rsquare_foreaft_self,
#                                                         rsquare_lateral_body, rsquare_lateral_self)
# # boolean_matrix_rsquars = pvalues_matrix_rsquares<0.05
# # np.savetxt(os.path.join(os.getcwd(),'humans_results','stats','rsquares_matrix_pvalue.txt'), boolean_matrix_rsquars)

# plt.close('all')
# # plot_rsquares_time(rsquare_time_body, rsquare_time_self, bool_plot=False,bool_save=False, figname='rsquares_timing')
# plt.close('all')
# # plot_rsquares_final(rsquare_foreaft_body, rsquare_lateral_body, rsquare_foreaft_self, rsquare_lateral_self, bool_plot=True, bool_save=True, figname='rsquares_final_velocity_250')
# print(rsquare_foreaft_body.shape)
# print(rsquare_foreaft_self.shape)
# ind_max = np.zeros((rsquare_foreaft_body.shape[0],2))
# for subject in range(ind_max.shape[0]):
#     ind_max[subject,0] = np.max(rsquare_foreaft_body[subject,:] - rsquare_foreaft_self[subject,:])
#     ind_max[subject,1] = np.max(rsquare_lateral_body[subject,:] - rsquare_lateral_self[subject,:])


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# vec_subject_x = np.linspace(0.8,1.2,21)
# for subject in range(21):
#     axs.scatter(vec_subject_x[subject], ind_max[subject,1],color='k',s=5, alpha=0.2)
# axs.scatter(1,np.nanmean(ind_max[:,1]),color='k',s=20)
# print(np.nanmean(ind_max[:,1]))
# print(np.nanstd(ind_max[:,1]))
# axs.plot([1,1], [np.nanmean(ind_max[:,1])+np.nanstd(ind_max[:,1]), np.nanmean(ind_max[:,1])-np.nanstd(ind_max[:,1])],color='k',lw=2)
# axs.set_ylim([-0.05,1.05])
# axs.set_xlim([0,4])
# plt.tight_layout()
# # fig.savefig(os.path.join(os.getcwd(),'humans_results','figures','humans_improvement.png'),bbox_inches='tight')
# # fig.savefig(os.path.join(os.getcwd(),'humans_results','figures','humans_improvement.svg'),bbox_inches='tight')

# ############################################################
# ### PART 3 - Characterization of the feedback controller ###
# ############################################################

# # 3.0 Histograms of the total variance captured on average for each leg

# # 3.1. Total variance captured for the front limbs
# matrix_rsquares_ffwd = compute_feedforward_rsquare(tot_input_leg1, tot_output_leg1, tot_sub1)
# matrix_contribution_variances = compute_variance_contributions(matrix_rsquares_ffwd, rsquare_foreaft_body, rsquare_lateral_body)
# matrix_contribution_variances_time = compute_variance_contributions_time(matrix_rsquares_ffwd,rsquare_foreaft_body,rsquare_lateral_body)

# plot_histograms_total_variance(matrix_contribution_variances, bool_plot=False, bool_save=False, figname='total_variance')
# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.plot(np.nanmean(matrix_contribution_variances_time[:,0,0,:],0),'k',label='foreaft ffwd')
# axs.plot(np.nanmean(matrix_contribution_variances_time[:,0,1,:],0),'k:',label='foreaft fb')
# axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,0,:],0),'r',label='lateral ffwd')
# axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,1,:],0),'r:',label='lateral fb')
# axs.set_ylim([-0.05,1.05])
# axs.set_xlim([-2.5,102.5])
# plt.tight_layout()
# # fig.savefig(os.path.join(os.getcwd(),'humans_results','figures','gains_time.png'),bbox_inches='tight')
# # fig.savefig(os.path.join(os.getcwd(),'humans_results','figures','gains_time.svg'),bbox_inches='tight')

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.set_xlim([0,4])
# axs.set_ylim([-0.05,1.05])
# axs.scatter(1, np.nanmean(matrix_contribution_variances_time[:,1,1,75]),color='k',s=20)
# axs.plot([1,1], [np.nanmean(matrix_contribution_variances_time[:,1,1,75],0)+np.nanstd(matrix_contribution_variances_time[:,1,1,75],0),np.nanmean(matrix_contribution_variances_time[:,1,1,75],0)-np.nanstd(matrix_contribution_variances_time[:,1,1,75],0)],color='k',lw=2)
# fig.savefig(os.path.join(os.getcwd(),'humans_results','figures','gains_midstance.png'),bbox_inches='tight')
# fig.savefig(os.path.join(os.getcwd(),'humans_results','figures','gains_midstance.svg'),bbox_inches='tight')
# plt.tight_layout()


# plot_variance_contributions_final(matrix_contribution_variances, bool_plot=False, bool_save=False, figname='variance_contribution')
# plt.close('all')

# 3.2 Correlation between successive contacts 
print(tot_input_leg1.shape, tot_output_leg1.shape)
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
# print(list_input_fr[0].shape, list_input_fr[1].shape, list_input_fr[2].shape)
# print(list_output_fr[0].shape)


# diff_input_1 = tot_input_leg1[:,75,1] - np.nanmean(tot_input_leg1[:,75,1])
# diff_output_1 = tot_output_leg1[:,1] - np.nanmean(tot_output_leg1[:,1])

# diff_input_2 = list_input_fr[0][:,75,1] - np.nanmean(list_input_fr[0][:,75,1],0)
# diff_output_2 = list_output_fr[0][:,1]-np.nanmean(list_output_fr[0][:,1])

# diff_input_3 = list_input_fr[1][:,75,1] - np.nanmean(list_input_fr[1][:,75,1],0)
# diff_output_3 = list_output_fr[1][:,1]-np.nanmean(list_output_fr[1][:,1])

# diff_input_4 = list_input_fr[2][:,75,1] - np.nanmean(list_input_fr[2][:,75,1],0)
# diff_output_4 = list_output_fr[2][:,1]-np.nanmean(list_output_fr[2][:,1])

# idx_nans_1 = np.where(~np.isnan(diff_input_1))
# idx_nans_2 = np.where(~np.isnan(diff_input_2))
# idx_nans_3 = np.where(~np.isnan(diff_input_3))
# idx_nans_4 = np.where(~np.isnan(diff_input_4))

# reg_1 = scipy.stats.linregress(diff_input_1, diff_output_1)
# reg_2 = scipy.stats.linregress(diff_input_2, diff_output_2)
# reg_3 = scipy.stats.linregress(diff_input_3, diff_output_3)
# reg_4 = scipy.stats.linregress(diff_input_4, diff_output_4)

# print(reg_1.slope, reg_2.slope, reg_3.slope, reg_4.slope) 

# sys.exit()
plot_correlation_horizon_final(list_input_time, list_subjects, bool_plot=False, bool_save=False, figname='horizon_correlation')

# 3.3 Normalized feedback gains for the lateral deviation


regression_matrix = compute_regression_fr(tot_input_leg1, tot_output_leg1, tot_sub1)
plot_normalized_feedback_gains(regression_matrix, bool_plot=False, bool_save=False, figname='fbgains_lat_lat')



###########################################################################
### PART 4 - Investigation of the laterality for foot placement control ###
###########################################################################

regression_small, regression_large = regression_laterality(tot_input_leg1, tot_output_leg1, tot_sub1)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_human')
print(regression_large.shape, regression_small.shape)



idx_nan_l = np.where(~np.isnan(regression_large[:,75,1,1,0].flatten()))[0]
idx_nan_s = np.where(~np.isnan(regression_small[:,75,1,1,0].flatten()))[0]
shapiro_large = scipy.stats.shapiro(regression_large[idx_nan_l,75,1,1,0].flatten())
shapiro_small = scipy.stats.shapiro(regression_small[idx_nan_s,75,1,1,0].flatten())
print(shapiro_large)
print(shapiro_small)
wilco_2 = scipy.stats.wilcoxon(regression_large[idx_nan_l,75,1,1,0].flatten(),regression_small[idx_nan_s,75,1,1,0].flatten())
wilco_3 = scipy.stats.wilcoxon(regression_large[idx_nan_l,75,1,1,0].flatten(),regression_small[idx_nan_s,75,1,1,0].flatten(),alternative='greater')
wilco_4 = scipy.stats.wilcoxon(regression_large[idx_nan_l,75,1,1,0].flatten(),regression_small[idx_nan_s,75,1,1,0].flatten(),alternative='less')
ttest_stats = scipy.stats.ttest_rel(regression_large[:,75,1,1,0],regression_small[:,75,1,1,0],nan_policy='omit')
tmp_s = regression_large[:,75,1,1,0].flatten()
tmp_l = regression_small[:,75,1,1,0].flatten()
cohend = (np.nanmean(tmp_s) - np.nanmean(tmp_l)) / np.nanstd(np.concatenate((tmp_l,tmp_s)))
print(cohend)
print(ttest_stats)
print(wilco_2)
print(wilco_3)
print(wilco_4)