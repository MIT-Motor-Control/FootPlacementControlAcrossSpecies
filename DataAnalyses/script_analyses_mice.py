import os, sys
import pickle
import numpy as np 
import warnings 
import scipy
import scipy.stats
from tqdm import tqdm
from utils.test_plots_humans import *
from utils.utils_mice import *
from utils.plot_functions_mice import *
warnings.filterwarnings('ignore')

# Loading the data
body_io, self_io, timing_io = load_all_data()
body_i_abs, body_o_abs = load_data_absolute()



tmp = timing_io[0][0]
idx_nnans = np.where(~np.isnan(tmp[:,0,0]))[0]
tot_input_list, tot_input_self, tot_output_list, tot_output_self, tot_animal = concatenate_data(body_io[0], body_io[1], self_io[0], self_io[1])
tot_input_list_od, tot_input_self_od, tot_output_list_od, tot_output_self_od, tot_animal_od = concatenate_data(body_io[2], body_io[3], self_io[2], self_io[3])

tot_input_absolute, tot_output_absolute = body_i_abs[0], body_o_abs[0]
tot_animal_absolute = 0 * np.ones((body_i_abs[0].shape[0],1))
for animal in range(1, 80):
    tot_input_absolute = np.vstack((tot_input_absolute, body_i_abs[animal]))
    tot_output_absolute = np.vstack((tot_output_absolute, body_o_abs[animal]))
    tot_animal_absolute = np.vstack((tot_animal_absolute, animal*np.ones((body_i_abs[animal].shape[0],1))))

print(tot_input_list.shape, tot_input_self.shape, tot_input_absolute.shape, tot_output_absolute.shape)


diff_input_1 = tot_input_list[:,7,5]-np.nanmean(tot_input_list[:,7,5])
diff_output_1 = tot_input_self[:,10,1]-np.nanmean(tot_input_self[:,10,1])
diff_output_2 = tot_input_self[:,-1,1]-np.nanmean(tot_input_self[:,-1,1])
idx_nans_1 = np.where(~np.isnan(diff_output_1))[0]
idx_nans_2 = np.where(~np.isnan(diff_output_2))[0]

reg_1 = scipy.stats.linregress(diff_input_1[idx_nans_1], diff_output_1[idx_nans_1])
reg_2 = scipy.stats.linregress(diff_input_1[idx_nans_2], diff_output_2[idx_nans_2])
print(reg_1)
print(reg_2)

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(tot_input_list[:,7,5]-np.nanmean(tot_input_list[:,7,5]), tot_input_self[:,10,1]-np.nanmean(tot_input_self[:,10,1]),color='k',s=5)
# axs.scatter(tot_input_list[:,7,5]-np.nanmean(tot_input_list[:,7,5]), tot_input_self[:,-1,1]-np.nanmean(tot_input_self[:,-1,1]),color='r',s=5)
# plt.tight_layout()
# plt.show()

# sys.exit()
difference_matrix = np.load(os.path.join(os.getcwd(),'Datasets','Mouse','Klibaite','Control','difference_matrix_mice.npy'))


vel_tier = add_velocity_tier(tot_input_list, tot_input_self)
vel_tier_od = add_velocity_tier(tot_input_list_od, tot_input_self_od)
velocity_vector = get_velocity_vector(tot_input_list, tot_input_self)
velocity_vector_od = get_velocity_vector(tot_input_list_od, tot_input_self_od)


# laterality_bootstrap_mice(tot_input_list,tot_output_leg1)
# laterality_error_bar_mice(tot_input_list, tot_output_list,0,bool_save=True,figname='seaborn_regressions')
# plot_scatter_laterality(tot_animal, tot_input_list, tot_output_list, idx_leg=0)
# regression_small, regression_large = regression_fr_laterality_velidp(tot_animal, tot_input_list, tot_output_list, 0)
# plot_laterality_regression(regression_small, regression_large, bool_plot=True, bool_save=False, figname='laterality')

# sys.exit()

###################################################
### PART 1 - Velocity-dependent ffwd controller ###
###################################################
# plot_feedforward_controller_final(velocity_vector, tot_output_list, tot_animal, tot_input_absolute, tot_output_absolute, tot_animal_absolute, bool_plot=False, bool_save=False, figname='feedforward_figures_mouse')
# plot_feedforward_body_mice_final(tot_input_list, tot_animal, bool_plot=False, bool_save=False, figname='ffwd_body_mouse')
# sys.exit()


list_input_horizon, list_output_horizon = [], []
list_input_horizon.append(tot_input_self), list_output_horizon.append(tot_output_self)
for horizon in range(3,6):
    input_list_horizon, output_list_horizon = load_data_horizon(horizon)
    tot_input_self_horizon, tot_output_self_horizon = concatenate_data_horizon(input_list_horizon, output_list_horizon)
    list_input_horizon.append(tot_input_self_horizon), list_output_horizon.append(tot_output_self_horizon)

# plot_correlation_horizon_final(list_input_horizon, tot_animal, bool_plot=True, bool_save=True, figname='horizon_correlation')

print(len(list_input_horizon), list_input_horizon[0].shape, list_input_horizon[1].shape, list_output_horizon[2].shape)
print(len(list_output_horizon), list_output_horizon[0].shape, list_output_horizon[1].shape, list_output_horizon[2].shape)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmean(list_input_horizon[0][:,:,4],0),'k',lw=2)
axs.plot(np.nanmean(list_input_horizon[1][:,:,4],0),'r',lw=2)
axs.plot(np.nanmean(list_input_horizon[2][:,:,4],0),'b',lw=2)
plt.tight_layout()
# plt.show()

# sys.exit()
###############################################################
### PART 2 - Controller inference on the remaining variance ###
###############################################################

rsquare_front_foreaft_body,_ = get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind=0, bool_lat=0)
rsquare_front_lateral_body,gains_front_lateral = get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind=0, bool_lat=1)
tmp_output_path = os.path.join(os.getcwd(),'round_2_analysis')
os.makedirs(tmp_output_path, exist_ok=True)
np.save(os.path.join(tmp_output_path, 'gain_matrix_mice.npy'), gains_front_lateral)
print('Data saved')
# sys.exit()
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
plt.close('all')
plot_rsquares_time(rsquare_front_time_body, rsquare_hind_time_body, rsquare_front_time_self, rsquare_hind_time_self, bool_plot=False, bool_save=False, figname='rsquares_contact_timing')




ind_difference = np.zeros((rsquare_front_foreaft_body.shape[0],4))
for animal in range(ind_difference.shape[0]):
    ind_difference[animal,0] = np.max(rsquare_front_foreaft_body[animal,:] - rsquare_front_foreaft_self[animal,:])
    ind_difference[animal,1] = np.max(rsquare_front_lateral_body[animal,:] - rsquare_front_lateral_self[animal,:])
    ind_difference[animal,2] = np.max(rsquare_hind_foreaft_body[animal,:] - rsquare_hind_foreaft_self[animal,:])
    ind_difference[animal,3] = np.max(rsquare_hind_lateral_body[animal,:] - rsquare_hind_lateral_self[animal,:])

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
vec_animal_x = np.linspace(1.8,2.2,80)
for animal in range(80):
    axs.scatter(vec_animal_x[animal], ind_difference[animal,1], color='r',s=5, alpha=0.2)
axs.scatter(2, np.nanmean(ind_difference[:,1]),color='r', s=20)
axs.plot([2,2], [np.nanmean(ind_difference[:,1])+np.nanstd(ind_difference[:,1]),np.nanmean(ind_difference[:,1])-np.nanstd(ind_difference[:,1])],color='r',lw=2)
axs.set_ylim([-0.05,1.05])
axs.set_xlim([0,4])
print(np.nanmean(ind_difference[:,1]))
print(np.nanstd(ind_difference[:,1]))
plt.tight_layout()
# fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','mice_improvement.png'),bbox_inches='tight')
# fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','mice_improvement.svg'),bbox_inches='tight')


plot_rsquares_final(rsquare_front_foreaft_body, rsquare_front_lateral_body, rsquare_hind_foreaft_body, rsquare_hind_lateral_body,
                    rsquare_front_foreaft_self, rsquare_front_lateral_self, rsquare_hind_foreaft_self, rsquare_hind_lateral_self,
                    bool_plot=False, bool_save=False, figname='rsquares_final_mice')


############################################################
### PART 3 - Characterization of the feedback controller ###
############################################################
# 3.0 Total variance captured by the model 

# 3.1 Total variance captured for the front limbs 
matrix_rsquares_ffwd = compute_feedforward_rsquare(tot_input_list, tot_output_list, tot_animal)
matrix_contribution_variance = compute_variance_contributions(matrix_rsquares_ffwd, rsquare_front_foreaft_body, rsquare_front_lateral_body, rsquare_hind_foreaft_body, rsquare_hind_lateral_body)
matrix_contribution_variance_time = compute_variance_contributions_time(matrix_rsquares_ffwd, rsquare_front_foreaft_body, rsquare_front_lateral_body, rsquare_hind_foreaft_body, rsquare_hind_lateral_body)

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmean(matrix_contribution_variance_time[:,1,0,10:],0),'k',label='fb front foreaft')
axs.plot(np.nanmean(matrix_contribution_variance_time[:,1,1,10:],0),'k:',label='fb front lateral')
axs.plot(np.nanmean(matrix_contribution_variance_time[:,1,2,10:],0),'r',label='fb hind foreaft')
axs.plot(np.nanmean(matrix_contribution_variance_time[:,1,3,10:],0),'r:',label='fb hind lateral')
axs.legend(frameon=False)
axs.set_ylim([-0.05,1.05])
axs.set_xlim([-0.25,10.25])
plt.tight_layout()
# fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','gains_time.png'),bbox_inches='tight')
# fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','gains_time.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_ylim([-0.05,1.05])
axs.set_xlim([0,4])
axs.scatter(2, np.nanmean(matrix_contribution_variance_time[:,1,1,17],0),color='k',s=20)
axs.plot([2,2], [np.nanmean(matrix_contribution_variance_time[:,1,1,17],0)+np.nanstd(matrix_contribution_variance_time[:,1,1,17],0),np.nanmean(matrix_contribution_variance_time[:,1,1,17],0)-np.nanstd(matrix_contribution_variance_time[:,1,1,17],0)],color='k',lw=2)
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','gains_midstance.png'),bbox_inches='tight')
fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','gains_midstance.svg'),bbox_inches='tight')
plot_histograms_total_variance(matrix_contribution_variance, bool_plot=False, bool_save=False, figname='total_variance')
plot_variance_contributions_final(matrix_contribution_variance, bool_plot=False, bool_save=False, figname='variance_contribution')

# 3.2 Correlation between successive contacts 
list_input_horizon, list_output_horizon = [], []
list_input_horizon.append(tot_input_self), list_output_horizon.append(tot_output_self)
for horizon in range(3,6):
    input_list_horizon, output_list_horizon = load_data_horizon(horizon)
    tot_input_self_horizon, tot_output_self_horizon = concatenate_data_horizon(input_list_horizon, output_list_horizon)
    list_input_horizon.append(tot_input_self_horizon), list_output_horizon.append(tot_output_self_horizon)

plot_correlation_horizon_final(list_input_horizon, tot_animal, bool_plot=False, bool_save=False, figname='horizon_correlation')

print('ho')
# 3.3 Normalized feedback gains for the lateral deviation
regression_matrix = compute_regression_fr(tot_animal, tot_input_list, tot_output_list, vel_tier,0)
plot_normalized_feedback_gains(regression_matrix, bool_plot=False, bool_save=False, figname='fbgains_lat_lat')

print('ho')
###############################################################################
### PART 4 - Investigation of the laterality for the foot placement control ###
###############################################################################
plt.close('all')
print('ho')
regression_small, regression_large = regression_fr_laterality_velidp(tot_animal, tot_input_list, tot_output_list, 0)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality')
print(scipy.stats.shapiro(regression_small[:,17,1,5,0].flatten()))
print(scipy.stats.shapiro(regression_large[:,17,1,5,0].flatten()))
print(scipy.stats.ttest_rel(regression_small[:,17,1,5,0],regression_large[:,17,1,5,0],nan_policy='omit'))
tmp_s = regression_small[:,17,1,5,0].flatten()
tmp_l = regression_large[:,17,1,5,0].flatten()
cohend = (np.nanmean(tmp_s) - np.nanmean(tmp_l)) / (np.nanstd(np.concatenate((tmp_s,tmp_l))))
print(cohend)