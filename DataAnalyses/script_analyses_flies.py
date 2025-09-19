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
print(len(video_list))
print('Data loaded')
sys.exit()
#fly_list, fly_list_od = load_fly_data()

tot_input_list, tot_input_self, tot_output_list, tot_output_self = concatenate_data(body_io[0], body_io[1], self_io[0], self_io[1])
tot_input_list_, tot_input_self_, tot_output_list_, tot_output_self_ = concatenate_data(body_io_[0], body_io_[1], self_io_[0], self_io_[1])
print(tot_input_list.shape, tot_output_list.shape)
# sys.exit()
# difference_matrix = np.load(os.path.join(os.getcwd(),'Datasets','Fly','controllers_io_abs','difference_matrix_fliesnpy.npy'))
# print(difference_matrix.shape)


# tot_input_list_od, tot_input_self_od, tot_output_list_od, tot_output_self_od = concatenate_data(body_io[2], body_io[3], self_io[2], self_io[3])
# tot_input_list_od_, tot_input_self_od_, tot_output_list_od_, tot_output_self_od_ = concatenate_data(body_io_[2], body_io_[3], self_io_[2], self_io_[3])
# print(tot_input_list_.shape)
# tot_input_list, tot_input_self, tot_output_list, tot_output_self = flip_data(tot_input_list, tot_input_self, tot_output_list, tot_output_self)
# tot_input_list_, tot_input_self_, tot_output_list_, tot_output_self_ = flip_data(tot_input_list_, tot_input_self_, tot_output_list_, tot_output_self_)

# tot_input_list_od, tot_input_self_od, tot_output_list_od, tot_output_self_od = flip_data(tot_input_list_od, tot_input_self_od, tot_output_list_od, tot_output_self_od)
# tot_input_list_od_, tot_input_self_od_, tot_output_list_od_, tot_output_self_od_ = flip_data(tot_input_list_od_, tot_input_self_od_, tot_output_list_od_, tot_output_self_od_)
# print('Data concatenated')

# print(tot_input_list.shape, tot_output_list.shape)
# print(video_list.shape)
# idx_local = np.where((video_list[:,0]==1))[0]
# local_input = tot_input_list[idx_local,:,7] - np.nanmean(tot_input_list[idx_local,:,7],0)
# local_output = tot_output_list[idx_local,4] - np.nanmean(tot_output_list[idx_local,4])

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(local_input[:,17], local_output, color='k', s=5)
# plt.tight_layout()
# plt.show()

# sys.exit()


# Loading the time input data containing the absolute contact timing 
# body_io_time, self_io_time = load_all_data_mm_abs()
# tot_input_time, input_self_time, tot_output_time, output_self_time = concatenate_data(body_io_time[0], body_io_time[1],self_io_time[0],self_io_time[1])
# video_list_time, video_list_od_time = load_video_data_mm_abs()
# print(tot_input_list.shape,tot_input_time.shape, tot_output_time.shape, video_list_time.shape, tot_input_self.shape)
"""

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

# vel_tier = add_velocity_tier(tot_input_list)
# #laterality_bootstrap_flies(tot_input_list, tot_output_list)
# laterality_error_bar_flies(tot_input_list, tot_output_list, idx_leg=1, bool_save=False, figname='seaborn_regressions')
# #plot_scatter_laterality(tot_input_list, tot_output_list, video_list, bool_plot=False)
# regression_small, regression_large = regression_fr_laterality_velidp(tot_input_list, tot_output_list, video_list, 1)
# print(regression_small.shape)
# print(regression_large.shape)
# # plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_final')

# velocity_plot = np.nanmean(tot_input_list_od[:,:,2],1)
# amplitude_plot = tot_input_list_od[:,-1,0] - tot_input_list_od[:,0,0]

# list_self_with_horizons = [] 
# list_self_with_horizons.append(tot_input_self)
# for horizon in tqdm(range(2,6)):
#     body_io_h, self_io_h = load_all_data_horizon(horizon)
#     tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_output_self_h = concatenate_data(body_io_h[0], body_io_h[1], self_io_h[0], self_io_h[1])
#     tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_ouptut_self_h = flip_data(tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_output_self_h)
#     list_self_with_horizons.append(tot_input_self_h)


# plot_successive_contacts_fly(list_self_with_horizons, video_list, bool_plot=True, bool_save=True, figname='successive_contacts')


###################################################
### PART 1 - Velocity-dependent ffwd controller ###
###################################################
plot_feedforward_controller_final_time(tot_input_time, tot_output_time, video_list_time, bool_plot=False, bool_save=False, figname='time_ffwd_output_fly')
plot_feedforward_controller_final(tot_input_list, tot_output_list, video_list, bool_plot=False, bool_save=False, figname='feedforward_outputs_fly')
plot_feedforward_body_flies_final_outliers(tot_input_list, tot_output_list, video_list, bool_plot=False, bool_save=False, figname='tmp')
plot_feedforward_body_flies_final(tot_input_list, video_list, bool_plot=False, bool_save=False, figname='ffwd_body_flies')   
plt.close('all')

###############################################################
### PART 2 - Controller inference of the remaining variance ###
###############################################################

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
"""
rsquare_leg2_front,_ = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 1, 0)
rsquare_leg2_lat,fb_gains = get_rsquare_matrix_normalization(tot_input_list, tot_output_list, video_list, 1, 1)
tmp_output_path = os.path.join(os.getcwd(),'round_2_analysis')
os.makedirs(tmp_output_path, exist_ok=True)
np.save(os.path.join(tmp_output_path,'matrix_gains_flies.npy'), fb_gains)
sys.exit()
"""
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

print(type(rsquare_leg1_front))
print(rsquare_leg1_front.shape)

ind_difference = np.zeros((rsquare_leg1_front.shape[0],6))
for ii in range(ind_difference.shape[0]):
    ind_difference[ii,0] = np.max(rsquare_leg2_front[ii,:]-rsquare_leg2_front_self[ii,:])
    ind_difference[ii,1] = np.max(rsquare_leg2_lat[ii,:]-rsquare_leg2_lat_self[ii,:])
    ind_difference[ii,2] = np.max(rsquare_leg1_front[ii,:]-rsquare_leg1_front_self[ii,:])
    ind_difference[ii,3] = np.max(rsquare_leg1_lat[ii,:]-rsquare_leg1_lat_self[ii,:])
    ind_difference[ii,4] = np.max(rsquare_leg3_front[ii,:]-rsquare_leg3_front_self[ii,:])
    ind_difference[ii,5] = np.max(rsquare_leg3_lat[ii,:]-rsquare_leg3_lat_self[ii,:])

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
vec_fly_x = np.linspace(2.8,3.2,8)
for animal in range(8):
    axs.scatter(vec_fly_x[animal], ind_difference[animal,1],s=5, color='b', alpha=0.2)
axs.scatter(3, np.nanmean(ind_difference[:,1]),color='b',s=20)
axs.plot([3,3],[np.nanmean(ind_difference[:,1])+np.nanstd(ind_difference[:,1]), np.nanmean(ind_difference[:,1])-np.nanstd(ind_difference[:,1])],color='b',lw=2)
axs.set_ylim([-0.05,1.05]), axs.set_xlim([0,4])
print(np.nanmean(ind_difference[:,1]))
print(np.nanstd(ind_difference[:,1]))
plt.tight_layout()
# fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','flyimprovements.png'),bbox_inches='tight')
# fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','flyimprovements.svg'),bbox_inches='tight')


plt.close('all')
plot_rsquares_timing(list_body_time, list_self_time, bool_plot=False, bool_save=False, figname='final_rsquares_time')
plot_rsquares_final(list_body, list_self, bool_plot=False, bool_save=False, figname='final_rsquares_flies_video')


# Temporary location for the laterality plots
vel_tier = add_velocity_tier(tot_input_list)
regression_small, regression_large = regression_fr_laterality_velidp(tot_input_list, tot_output_list, video_list, 1)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_final')

matrix_rsquares_ffwd = compute_feedforward_rsquare(tot_input_list, tot_output_list, video_list)
matrix_contribution_variances = compute_variance_contributions(matrix_rsquares_ffwd, rsquare_leg1_front, rsquare_leg1_lat, rsquare_leg2_front, rsquare_leg2_lat, rsquare_leg3_front, rsquare_leg3_lat)
matrix_contribution_variances_time = compute_variance_contributions_time(matrix_rsquares_ffwd, rsquare_leg1_front, rsquare_leg1_lat, rsquare_leg2_front, rsquare_leg2_lat, rsquare_leg3_front, rsquare_leg3_lat)

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,0,:],0),'k',label='central foreaft')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,1,:],0),'k:',label='central lateral')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,0,3,:],0),'r',label='front lateral ffwd')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,3,:],0),'r:',label='front lateral fb')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,4,:],0),'b',label='hind foreaft')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,5,:],0),'b:',label='hind lateral')
axs.set_ylim([-0.05,1.05])
axs.set_xticks([0,5,10,15,20])
axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
axs.set_xlim([-0.5,20.5])
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','timegains.svg'),bbox_inches='tight')
fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','timegains.png'),bbox_inches='tight')


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_xlim([0,4])
axs.set_ylim([-0.05,1.05])
axs.scatter(3,np.nanmean(matrix_contribution_variances_time[:,1,3,15],0),color='k',s=20)
axs.plot([3,3],[np.nanmean(matrix_contribution_variances_time[:,1,3,15],0)+np.nanstd(matrix_contribution_variances_time[:,1,3,15],0),np.nanmean(matrix_contribution_variances_time[:,1,3,15],0)-np.nanstd(matrix_contribution_variances_time[:,1,3,15],0)],color='k',lw=2)
plt.tight_layout()
fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','gainsmidstance.svg'),bbox_inches='tight')
fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','gainsmidstance.png'),bbox_inches='tight')


"""

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

print(len(list_input_horizon), list_input_horizon[0].shape)

diff_input_1 = tot_input_list[:,17,5] - np.nanmean(tot_input_list[:,17,5])
diff_output_1 = tot_output_list[:,4] - np.nanmean(tot_output_list[:,4])

diff_input_2 = list_input_horizon[0][:,17,5] - np.nanmean(list_input_horizon[0][:,17,5])
diff_output_2 = list_output_horizon[0][:,4] - np.nanmean(list_output_horizon[0][:,4])

idx_nans_1 = np.where(~np.isnan(diff_input_1))[0]
idx_nans_2 = np.where(~np.isnan(diff_input_2))[0]

reg_1 = scipy.stats.linregress(diff_input_1[idx_nans_1], diff_output_1[idx_nans_1])
reg_2 = scipy.stats.linregress(diff_input_1[idx_nans_2], diff_output_2[idx_nans_2])
print(reg_1, reg_2)

print('oh')
plot_successive_contacts_fly(list_self_with_horizons, video_list, bool_plot=False, bool_save=False, figname='successive_contacts')
sys.exit()


############################################################
### PART 3 - Characterization of the feedback controller ###
############################################################

# 3.0 Histograms of the total variance captured on average for each leg


# 3.1 Total variance captured by the front limbs 
matrix_rsquares_ffwd = compute_feedforward_rsquare(tot_input_list, tot_output_list, video_list)
matrix_contribution_variances = compute_variance_contributions(matrix_rsquares_ffwd, rsquare_leg1_front, rsquare_leg1_lat, rsquare_leg2_front, rsquare_leg2_lat, rsquare_leg3_front, rsquare_leg3_lat)
matrix_contribution_variances_time = compute_variance_contributions_time(matrix_rsquares_ffwd, rsquare_leg1_front, rsquare_leg1_lat, rsquare_leg2_front, rsquare_leg2_lat, rsquare_leg3_front, rsquare_leg3_lat)

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,0,:],0),'k',label='centra foreaft')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,1,:],0),'k:',label='central lateral')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,2,:],0),'r',label='front foreaft')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,3,:],0),'r:',label='front lateral')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,4,:],0),'b',label='hind foreaft')
axs.plot(np.nanmean(matrix_contribution_variances_time[:,1,5,:],0),'b:',label='hind lateral')
axs.set_ylim([-0.05,1.05])
axs.legend(frameon=False)
plt.tight_layout()


plot_histograms_total_variance(matrix_contribution_variances, bool_plot=False, bool_save=False, figname='total_variance')
plot_variance_contributions_final(matrix_contribution_variances, bool_plot=False, bool_save=False, figname='variance_contribution')



# 3.2 Correlation between successives contacts

list_self_with_horizons = [] 
list_self_with_horizons.append(tot_input_self)
for horizon in tqdm(range(2,5)):
    body_io_h, self_io_h = load_all_data_horizon(horizon)
    tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_output_self_h = concatenate_data(body_io_h[0], body_io_h[1], self_io_h[0], self_io_h[1])
    tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_ouptut_self_h = flip_data(tot_input_list_h, tot_input_self_h, tot_output_list_h, tot_output_self_h)
    list_self_with_horizons.append(tot_input_self_h)


plot_successive_contacts_fly(list_self_with_horizons, video_list, bool_plot=False, bool_save=False, figname='successive_contacts')


# 3.3 Normalized feedback gains for the lateral deviation
vel_tier = add_velocity_tier(tot_input_list)
regression_matrix_diag_leg2 = regression_fr_body(vel_tier, tot_input_list, tot_output_list, video_list, 1)

plot_normalized_feedback_gains(regression_matrix_diag_leg2, bool_plot=False, bool_save=False, figname='fb_gains_lat_lat')

# 3.4 Contribution of the orientation for the central legs



###########################################################################
### PART 4 - Investigation of the laterality for foot placement control ###
###########################################################################


regression_small, regression_large = regression_fr_laterality_velidp(tot_input_list, tot_output_list, video_list, 1)
plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname='laterality_final')
