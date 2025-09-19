import os, sys 
import pickle 
import scipy.stats 
from tqdm import tqdm
import numpy as np 
import warnings
import pandas as pd
import statsmodels.api as sm 
# from rdd import rdd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore') 

input_path = os.path.join(os.getcwd(),'Datasets','Fly','stability')

rdd_path = os.path.join(os.getcwd(),'rdd_figures')
os.makedirs(rdd_path, exist_ok=True)

# Loading the data 
with open(os.path.join(input_path,'list_input_fr_stability.pkl'),'rb') as f1:
    tot_input_fr = pickle.load(f1)
with open(os.path.join(input_path,'list_output_fr_stability.pkl'),'rb') as f2:
    tot_output_fr = pickle.load(f2)
with open(os.path.join(input_path,'list_video_stability.pkl'),'rb') as f3:
    tot_video = np.array(pickle.load(f3))



total_input_fr = tot_input_fr[0]
print(len(tot_input_fr))
total_output_fr = tot_output_fr[0]
total_chunk = np.zeros((tot_input_fr[0].shape[0],))
for ii in tqdm(range(1,len(tot_input_fr))):
    total_input_fr = np.concatenate((total_input_fr, tot_input_fr[ii]),0)
    total_output_fr = np.concatenate((total_output_fr, tot_output_fr[ii]),0)
    total_chunk = np.concatenate((total_chunk, ii*np.ones(tot_input_fr[ii].shape[0],)))


# sys.exit()

idx_flip = np.where(np.nanmean(total_input_fr[:,:,6],1)<0)[0]
total_input_fr[idx_flip,:,:] = - total_input_fr[idx_flip,:,:]
col_flip = [0,1,3,4,6,7]
for col in col_flip:
    total_output_fr[idx_flip,col] = - total_output_fr[idx_flip,col]
total_video = tot_video[total_chunk.astype(int)]
print(np.unique(total_video))
matrix_individual_eigenvalues = np.zeros((8,4))
local_reg = np.zeros((8,1))
for video in range(8):
    idx_video = np.where((total_video.flatten()==video+1) & (~np.isnan(total_input_fr[:,0,0])))[0]
    print(total_input_fr.shape)
    local_input = total_input_fr[idx_video,15,4:8]
    local_output = total_input_fr[idx_video,26,4:8]
    print(len(idx_video))
    A_local = np.linalg.inv(local_input.T @ local_input) @ (local_input.T @ local_output)
    local_eig, _ = np.linalg.eig(A_local)
    tmp_slope = scipy.stats.linregress(local_input[:,1], local_output[:,1])
    local_reg[video] = tmp_slope.slope
    matrix_individual_eigenvalues[video, :] = local_eig
print(np.unique(total_chunk))
tmp_output_path = os.path.join(os.getcwd(),'round_2_analysis')
np.save(os.path.join(tmp_output_path,'matrix_eig_flies.npy'), matrix_individual_eigenvalues)
np.save(os.path.join(tmp_output_path,'matrix_slope_flies.npy'), local_reg)
fig, axs = plt.subplots(1,1)
for ii in range(50):
    axs.plot(total_input_fr[ii,:,4],lw=0.5,color='k')
sys.exit()
plt.show()

print(total_output_fr.shape)
deviations_fp = np.zeros((total_input_fr.shape[0],))
deviations_fp_fwd = np.zeros((total_input_fr.shape[0],))
for chunk in range(len(tot_input_fr)):
    idx_chunk = np.where((total_chunk==chunk))[0]
    tmp_vel_1 = np.nanmean(total_input_fr[idx_chunk,:21,6],1)
    tmp_vel_2 = np.nanmean(total_input_fr[idx_chunk,21:,6],1)
    total_input_fr[idx_chunk,:21,5] = total_input_fr[idx_chunk,:21,5] - np.nanmean(total_input_fr[idx_chunk,:21,5],0)
    total_input_fr[idx_chunk,:21,6] = total_input_fr[idx_chunk,:21,6] - np.expand_dims(tmp_vel_1,-1)
    total_input_fr[idx_chunk,:21,7] = total_input_fr[idx_chunk,:21,7] - np.nanmean(total_input_fr[idx_chunk,:21,7],0)
    total_input_fr[idx_chunk,21:,5] = total_input_fr[idx_chunk,21:,5] - np.nanmean(total_input_fr[idx_chunk,21:,5],0)
    total_input_fr[idx_chunk,21:,6] = total_input_fr[idx_chunk,21:,6] - np.expand_dims(tmp_vel_2,-1)
    total_input_fr[idx_chunk,21:,7] = total_input_fr[idx_chunk,21:,7] - np.nanmean(total_input_fr[idx_chunk,21:,7],0)
    deviations_fp[idx_chunk] = total_output_fr[idx_chunk,4] - np.nanmean(total_output_fr[idx_chunk,4])
    deviations_fp_fwd[idx_chunk] = total_output_fr[idx_chunk,5] - np.nanmean(total_output_fr[idx_chunk,5])
    for line in range(total_input_fr[idx_chunk,:,:].shape[0]):
        time_vector = np.arange(21)
        if np.isnan(total_input_fr[idx_chunk[line],0,4]):
            continue
        else:
            linelin1 = scipy.stats.linregress(time_vector, total_input_fr[idx_chunk[line],:21,4])
            linelin2 = scipy.stats.linregress(time_vector, total_input_fr[idx_chunk[line],21:,4])
            total_input_fr[idx_chunk[line],:21,4] = total_input_fr[idx_chunk[line],:21,4] - (time_vector*linelin1.slope+linelin1.intercept)
            total_input_fr[idx_chunk[line],21:,4] = total_input_fr[idx_chunk[line],21:,4] - (time_vector*linelin2.slope+linelin2.intercept)


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(total_input_fr[line,:,4], color='k')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(total_input_fr[line,:,5], color='k')
plt.tight_layout()


matrix_inputs = np.zeros((total_input_fr.shape[0],4))
matrix_outputs = np.zeros((total_input_fr.shape[0],4))
matrix_inputs_avg = np.zeros((total_input_fr.shape[0],4))
matrix_outputs_avg = np.zeros((total_input_fr.shape[0],4))


matrix_inputs = total_input_fr[:,15,[4,5,6,7]]
matrix_outputs = total_input_fr[:,26,[4,5,6,7]]
matrix_inputs_avg = np.nanmean(total_input_fr[:,11:21,[4,5,6,7]],1)
matrix_outputs_avg = np.nanmean(total_input_fr[:,21:32,[4,5,6,7]],1)

idx_nans = np.where((~np.isnan(matrix_inputs[:,0])) & (~np.isnan(matrix_outputs[:,0])))[0]

matrix_inputs_final = matrix_inputs[idx_nans,:]
matrix_outputs_final = matrix_outputs[idx_nans,:]
matrix_inputs_final_avg = matrix_inputs_avg[idx_nans,:]
matrix_outputs_final_avg = matrix_outputs_avg[idx_nans,:]
print(matrix_inputs_final.shape)
sys.exit()

deviation_fp = deviations_fp[idx_nans]
deviation_fp_fwd = deviations_fp_fwd[idx_nans]

error_reduction = np.zeros((matrix_inputs_final.shape[0],))
for line in range(error_reduction.shape[0]):
    error_reduction[line] = matrix_inputs_final[line,1] - matrix_outputs_final[line,1]

error_reduction_avg = np.zeros((matrix_inputs_final.shape[0],))
for line in range(error_reduction.shape[0]):
    error_reduction_avg[line] = matrix_inputs_final_avg[line,1] - matrix_outputs_final_avg[line,1]


idx_large = np.where((np.abs(deviation_fp))>(np.nanmedian(np.abs(deviation_fp))))[0]
idx_small = np.where((np.abs(deviation_fp))<(np.nanmedian(np.abs(deviation_fp))))[0]

error_reduction_avg_l = np.zeros((len(idx_large),))
for line in range(len(idx_large)):
    error_reduction_avg_l[line] = matrix_inputs_final_avg[idx_large[line],1] - matrix_outputs_final_avg[idx_large[line],1]


error_reduction_avg_s = np.zeros((len(idx_small),))
for line in range(len(idx_small)):
    error_reduction_avg_s[line] = matrix_inputs_final_avg[idx_small[line],1] - matrix_outputs_final_avg[idx_small[line],1]


error_reduction_avg_fwd = np.zeros((matrix_inputs_final.shape[0],))
for line in range(error_reduction_avg_fwd.shape[0]):
    error_reduction_avg_fwd[line] = matrix_inputs_final_avg[line,0] - matrix_outputs_final_avg[line,0]



error_reduction_vel = np.zeros((matrix_inputs_final.shape[0],))
for line in range(error_reduction.shape[0]):
    error_reduction_vel[line] = matrix_inputs_final[line,3] - matrix_outputs_final[line,3]

error_reduction_avg_vel = np.zeros((matrix_inputs_final.shape[0],))
for line in range(error_reduction.shape[0]):
    error_reduction_avg_vel[line] = matrix_inputs_final_avg[line,3] - matrix_outputs_final_avg[line,3]


plt.close('all')

idx_pos = np.where(deviation_fp>0)[0]
idx_neg = np.where(deviation_fp<0)[0]

pos_deviations = deviation_fp[idx_pos]
pos_errors = error_reduction_avg[idx_pos]

neg_deviations = deviation_fp[idx_neg]
neg_errors = error_reduction_avg[idx_neg]

idx_below_median = np.where(pos_deviations<np.nanmedian(pos_deviations))[0]
idx_above_median = np.where(pos_deviations>np.nanmedian(pos_deviations))[0]
idx_below_median_neg = np.where(neg_deviations<np.nanmedian(neg_deviations))[0]
idx_above_median_neg = np.where(neg_deviations>np.nanmedian(neg_deviations))[0]
global_reg_below = scipy.stats.linregress(pos_deviations[idx_below_median], pos_errors[idx_below_median])
global_reg_above = scipy.stats.linregress(pos_deviations[idx_above_median], pos_errors[idx_above_median])
global_reg_below_neg = scipy.stats.linregress(neg_deviations[idx_below_median_neg], neg_errors[idx_below_median_neg])
global_reg_above_neg = scipy.stats.linregress(neg_deviations[idx_above_median_neg], neg_errors[idx_above_median_neg])


from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, 269-2)
print(f"slope below (95%) : {global_reg_below.slope:.6f} +/- {ts*global_reg_below.stderr:.6f}", f"rsquare: {global_reg_below.rvalue**2:.6f}")
print(f"slope above (95%) : {global_reg_above.slope:.6f} +/- {ts*global_reg_above.stderr:.6f}", f"rsquare: {global_reg_above.rvalue**2:.6f}")

slope_below, slope_above = [], [] 
intercept_below, intercept_above = [], [] 
slope_below_neg, slope_above_neg = [], [] 
intercept_below_neg, intercept_above_neg = [], [] 
diff_list = []
n_boot = 10000
for _ in range(n_boot):
    idx_local_below = np.random.choice(idx_below_median, len(idx_below_median), replace=True)
    idx_local_above = np.random.choice(idx_above_median, len(idx_above_median), replace=True)
    idx_local_below_neg = np.random.choice(idx_below_median_neg, len(idx_below_median_neg), replace=True)
    idx_local_above_neg = np.random.choice(idx_above_median_neg, len(idx_above_median_neg), replace=True)
    local_reg_below = scipy.stats.linregress(pos_deviations[idx_local_below], pos_errors[idx_local_below])
    local_reg_above = scipy.stats.linregress(pos_deviations[idx_local_above], pos_errors[idx_local_above])
    local_reg_below_neg = scipy.stats.linregress(neg_deviations[idx_local_below_neg], neg_errors[idx_local_below_neg])
    local_reg_above_neg = scipy.stats.linregress(neg_deviations[idx_local_above_neg], neg_errors[idx_local_above_neg])
    slope_below.append(local_reg_below.slope), slope_above.append(local_reg_above.slope)
    intercept_below.append(local_reg_below.intercept), intercept_above.append(local_reg_above.intercept)
    slope_below_neg.append(local_reg_below_neg.slope), slope_above_neg.append(local_reg_above_neg.slope)
    intercept_below_neg.append(local_reg_below_neg.intercept), intercept_above_neg.append(local_reg_above_neg.intercept)
    diff_list.append(local_reg_below.slope - local_reg_above.slope)

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.violinplot([np.array(diff_list)], widths=0.5, showextrema=False, showmeans=True, points=60)
# axs.axhline(np.nanpercentile(np.array(diff_list),5),color='k',ls=':',lw=2)
# axs.axhline(np.nanpercentile(np.array(diff_list),95),color='k',ls=':',lw=2)
# axs.axhline(0, color='k',lw=2)
# axs.set_ylabel('Differences below-above')
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'flies_slopes_diff.png'),bbox_inches='tight')


# data = pd.DataFrame({'y':pos_errors, 'x':pos_deviations})
# bandwidth_opt = rdd.optimal_bandwidth(data['y'], data['x'], cut=np.nanmedian(pos_deviations))
# print(bandwidth_opt)
# data_rdd = rdd.truncated_data(data, 'x', bandwidth_opt, cut=np.nanmedian(pos_deviations))
# model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(pos_deviations))
# print(model.fit().summary())



# fig, axs = plt.subplots(1,2,figsize=(6,3))
# axs[0].spines[['top','right']].set_visible(False)
# axs[1].spines[['top','right']].set_visible(False)
# vp1_0 = axs[0].violinplot([np.array(slope_below_neg)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
# vp2_0 = axs[0].violinplot([np.array(slope_above_neg)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
# vp1_1 = axs[1].violinplot([np.array(slope_below)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
# vp2_1 = axs[1].violinplot([np.array(slope_above)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
# axs[0].set_ylabel('Slope')
# vp1_0['cmeans'].set_color('blue')
# vp2_0['cmeans'].set_color('red')
# vp1_1['cmeans'].set_color('blue')
# vp2_1['cmeans'].set_color('red')
# for pc in vp1_0['bodies']:
#     pc.set_facecolor('blue')
#     pc.set_edgecolor('blue')
# for pc in vp2_0['bodies']:
#     pc.set_facecolor('red')
#     pc.set_edgecolor('red')
# for pc in vp1_1['bodies']:
#     pc.set_facecolor('blue')
#     pc.set_edgecolor('blue')
# for pc in vp2_1['bodies']:
#     pc.set_facecolor('red')
#     pc.set_edgecolor('red')
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'slopes_violinplot_flies.png'),bbox_inches='tight')
# fig.savefig(os.path.join(rdd_path,'slopes_violinplot_flies.svg'),bbox_inches='tight')


# fig, axs = plt.subplots(1,2,figsize=(6,3))
# axs[0].spines[['top','right']].set_visible(False)
# axs[1].spines[['top','right']].set_visible(False)
# axs[0].set_xlabel('Foot placement'), axs[0].set_ylabel('Error reduction')
# axs[1].set_xlabel('Foot placement')
# axs[0].scatter(neg_deviations[idx_below_median_neg], neg_errors[idx_below_median_neg], color='b', s=5)
# axs[0].scatter(neg_deviations[idx_above_median_neg], neg_errors[idx_above_median_neg], color='r', s=5)
# x_plot_below = np.linspace(min(neg_deviations),np.nanmedian(neg_deviations))
# x_plot_above = np.linspace(np.nanmedian(neg_deviations),0)
# axs[0].plot(x_plot_below, x_plot_below*global_reg_below_neg.slope+global_reg_below_neg.intercept,'k',lw=2)
# axs[0].plot(x_plot_above, x_plot_above*global_reg_above_neg.slope+global_reg_above_neg.intercept,'k',lw=2)
# axs[0].plot(x_plot_below, x_plot_below*np.nanmedian(np.array(slope_below_neg))+np.nanmedian(np.array(intercept_below_neg)),'k:',lw=2)
# axs[0].plot(x_plot_above, x_plot_above*np.nanmedian(np.array(slope_above_neg))+np.nanmedian(np.array(intercept_above_neg)),'k:',lw=2)
# # Rightmost subplot
# axs[1].scatter(pos_deviations[idx_below_median], pos_errors[idx_below_median], color='b', s=5)
# axs[1].scatter(pos_deviations[idx_above_median], pos_errors[idx_above_median], color='r', s=5)
# x_plot_below = np.linspace(0,np.nanmedian(pos_deviations))
# x_plot_above = np.linspace(np.nanmedian(pos_deviations), max(pos_deviations))
# axs[1].plot(x_plot_below, x_plot_below*global_reg_below.slope+global_reg_below.intercept,'k',lw=2)
# axs[1].plot(x_plot_above, x_plot_above*global_reg_above.slope+global_reg_above.intercept,'k',lw=2)
# axs[1].plot(x_plot_below, x_plot_below*np.nanmedian(np.array(slope_below))+np.nanmedian(np.array(intercept_below)),'k:',lw=2)
# axs[1].plot(x_plot_above, x_plot_above*np.nanmedian(np.array(slope_above))+np.nanmedian(np.array(intercept_above)),'k:',lw=2)
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'flies_points_rdd.png'),bbox_inches='tight')
# fig.savefig(os.path.join(rdd_path,'flies_points_rdd.svg'),bbox_inches='tight')

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
# axs.scatter(pos_deviations[idx_below_median], pos_errors[idx_below_median], color='b', s=5)
# axs.scatter(pos_deviations[idx_above_median], pos_errors[idx_above_median], color='r', s=5)
# x_plot_below = np.linspace(0,np.nanmedian(pos_deviations))
# x_plot_above = np.linspace(np.nanmedian(pos_deviations), max(pos_deviations))
# axs.plot(x_plot_below, x_plot_below*global_reg_below.slope+global_reg_below.intercept,'k',lw=2)
# axs.plot(x_plot_above, x_plot_above*global_reg_above.slope+global_reg_above.intercept,'k',lw=2)
# axs.plot(x_plot_below, x_plot_below*np.nanmedian(np.array(slope_below))+np.nanmedian(np.array(intercept_below)),'k:',lw=2)
# axs.plot(x_plot_above, x_plot_above*np.nanmedian(np.array(slope_above))+np.nanmedian(np.array(intercept_above)),'k:',lw=2)
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'flies_points.png'),bbox_inches='tight')


x_plot = np.linspace(0,np.max(pos_deviations))
print(pos_deviations.shape, pos_errors.shape)
df = pd.DataFrame({'deviations':pd.Series(pos_deviations), 'errors':pd.Series(pos_errors), 'x_plot':pd.Series(np.sort(pos_deviations))})
X = sm.add_constant(df['deviations'].values)
X_plot = sm.add_constant(df['x_plot'].values)
ols_model = sm.OLS(df['errors'].values, X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)

tot_reg_lin = scipy.stats.linregress(pos_deviations, pos_errors)

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# pred = est.get_prediction(X_plot).summary_frame()
# axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
# axs.scatter(pos_deviations, pos_errors , color='k', s=5)
# axs.plot(np.sort(pos_deviations), pred['mean_ci_upper'],'b--')
# axs.plot(np.sort(pos_deviations), pred['mean_ci_lower'],'b--')
# axs.plot(x_plot, x_plot*tot_reg_lin.slope+tot_reg_lin.intercept,color='b',lw=2)
# axs.fill_between(np.sort(pos_deviations), pred['mean_ci_upper'], pred['mean_ci_lower'],color='b',alpha=0.5)
# axs.set_xlim([0, 0.1]), axs.set_ylim([-0.10, 0.15])
# fig.savefig(os.path.join(rdd_path,'blackdots_lateral_flies.png'),bbox_inches='tight')
# fig.savefig(os.path.join(rdd_path,'blackdots_lateral_flies.svg'),bbox_inches='tight')
# plt.tight_layout()

# plt.show()



#########################
### Forward direction ###
#########################
idx_pos = np.where(deviation_fp_fwd>0)[0]
pos_deviations = deviation_fp_fwd[idx_pos]
pos_errors = error_reduction_avg_fwd[idx_pos]
bounds_pos = np.nanpercentile(pos_deviations, [0,10,20,30,40,50,60,70,80,90,100])


plt.close('all')
# Do the statistics around the median ... 
idx_below_median = np.where(pos_deviations<np.nanmedian(pos_deviations))[0]
idx_above_median = np.where(pos_deviations>np.nanmedian(pos_deviations))[0]
global_reg_below = scipy.stats.linregress(pos_deviations[idx_below_median], pos_errors[idx_below_median])
global_reg_above = scipy.stats.linregress(pos_deviations[idx_above_median], pos_errors[idx_above_median])
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, 269-2)
print(f"slope below (95%) : {global_reg_below.slope:.6f} +/- {ts*global_reg_below.stderr:.6f}", f"rsquare: {global_reg_below.rvalue**2:.6f}")
print(f"slope above (95%) : {global_reg_above.slope:.6f} +/- {ts*global_reg_above.stderr:.6f}", f"rsquare: {global_reg_above.rvalue**2:.6f}")

slope_below, slope_above = [], [] 
diff_list = []
n_boot = 10000
for _ in range(n_boot):
    idx_local_below = np.random.choice(idx_below_median, len(idx_below_median), replace=True)
    idx_local_above = np.random.choice(idx_above_median, len(idx_above_median), replace=True)
    local_reg_below = scipy.stats.linregress(pos_deviations[idx_local_below], pos_errors[idx_local_below])
    local_reg_above = scipy.stats.linregress(pos_deviations[idx_local_above], pos_errors[idx_local_above])
    slope_below.append(np.nanmean(pos_errors[idx_local_below])), slope_above.append(np.nanmean(pos_errors[idx_local_above]))
    diff_list.append(np.nanmean(pos_errors[idx_local_below])-np.nanmean(pos_errors[idx_local_above]))
print(len(slope_below), len(slope_above))
print(np.nanmean(slope_below))
print(np.nanmean(slope_above))

print(np.array(slope_below).shape, np.array(slope_above).shape)

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.violinplot([np.array(diff_list)], widths=0.5, showextrema=False, showmeans=True, points=60)
# axs.axhline(np.nanpercentile(np.array(diff_list),5),color='k',ls=':',lw=2)
# axs.axhline(np.nanpercentile(np.array(diff_list),95),color='k',ls=':',lw=2)
# axs.axhline(0, color='k',lw=2)
# axs.set_ylabel('Differences below-above')
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'flies_slopes_diff_fwd.png'),bbox_inches='tight')



# data = pd.DataFrame({'y':pos_errors, 'x':pos_deviations})
# bandwidth_opt =36# rdd.optimal_bandwidth(data['y'], data['x'], cut=np.nanmedian(pos_deviations))
# print(bandwidth_opt)
# data_rdd = rdd.truncated_data(data, 'x', bandwidth_opt, cut=np.nanmedian(pos_deviations))
# model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(pos_deviations))
# print(model.fit().summary())


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# fig.suptitle('Slope below and above median - Flies', fontweight='bold')
# axs.spines[['top','right']].set_visible(False)
# axs.violinplot([np.array(slope_below)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
# axs.violinplot([np.array(slope_above)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
# axs.set_ylabel('Slope')
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'flies_slopes_fwd.png'),bbox_inches='tight')


x_plot = np.linspace(0,np.max(pos_deviations))
print(pos_deviations.shape, pos_errors.shape)
df = pd.DataFrame({'deviations':pd.Series(pos_deviations), 'errors':pd.Series(pos_errors), 'x_plot':pd.Series(np.sort(pos_deviations))})
X = sm.add_constant(df['deviations'].values)
X_plot = sm.add_constant(df['x_plot'].values)
ols_model = sm.OLS(df['errors'].values, X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)

tot_reg_lin = scipy.stats.linregress(pos_deviations, pos_errors)
print(tot_reg_lin)
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
pred = est.get_prediction(X_plot).summary_frame()
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
axs.scatter(pos_deviations, pos_errors , color='k', s=5)
axs.plot(np.sort(pos_deviations), pred['mean_ci_upper'],'b--')
axs.plot(np.sort(pos_deviations), pred['mean_ci_lower'],'b--')
axs.plot(x_plot, x_plot*tot_reg_lin.slope+tot_reg_lin.intercept,color='b',lw=2)
axs.fill_between(np.sort(pos_deviations), pred['mean_ci_upper'], pred['mean_ci_lower'],color='b',alpha=0.5)
axs.set_xlim([0, 0.3]), axs.set_ylim([-0.005, 0.005])
fig.savefig(os.path.join(rdd_path,'blackdots_forward_flies.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'blackdots_forward_flies.svg'),bbox_inches='tight')
plt.tight_layout()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(pos_deviations[idx_below_median], pos_errors[idx_below_median], color='b', s=5)
# axs.scatter(pos_deviations[idx_above_median], pos_errors[idx_above_median], color='r', s=5)
# x_plot_below = np.linspace(0,np.nanmedian(pos_deviations))
# axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
# x_plot_above = np.linspace(np.nanmedian(pos_deviations), max(pos_deviations))
# axs.plot([min(x_plot_below), max(x_plot_below)], [np.nanmedian(pos_errors[idx_local_below]), np.nanmedian(pos_errors[idx_local_below])], 'k', lw=2)
# axs.plot([min(x_plot_above), max(x_plot_above)], [np.nanmedian(pos_errors[idx_local_above]), np.nanmedian(pos_errors[idx_local_above])], 'k:', lw=2)
# axs.fill_between([min(x_plot_below), max(x_plot_below)],[np.nanpercentile(np.array(slope_below),5), np.nanpercentile(np.array(slope_below),95)],color='k',alpha=0.5)
# axs.fill_between([min(x_plot_above), max(x_plot_above)],[np.nanpercentile(np.array(slope_above),5), np.nanpercentile(np.array(slope_above),95)],color='k',alpha=0.5)
# # axs.fill_between([min(x_plot_above), max(x_plot_above)],[np.nanmedian(pos_errors[idx_local_above])-scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_above])-scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2],[np.nanmedian(pos_errors[idx_local_above])+scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_above])+scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2],color='k',alpha=0.5)
# # axs.fill_between([min(x_plot_below), max(x_plot_below)],[np.nanmedian(pos_errors[idx_local_below])-scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_below])-scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2],[np.nanmedian(pos_errors[idx_local_below])+scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_below])+scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2],color='k',alpha=0.5)
# axs.set_xlim([0,0.3])
# plt.tight_layout()
# fig.savefig(os.path.join(rdd_path,'flies_points_fwd.png'),bbox_inches='tight')
# plt.show()
# sys.exit()
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

idx_pos = np.where(deviation_fp>0)[0]
x_local = deviation_fp[idx_pos]
y_local = error_reduction_avg[idx_pos]

idx_neg = np.where(deviation_fp<0)[0]
x_local_n = deviation_fp[idx_neg]
y_local_n = error_reduction_avg[idx_neg]
x_plot = np.linspace(0,0.10)
x_plot_n = np.linspace(-0.10,0)


reg_local_pos = scipy.stats.linregress(x_local,y_local)
reg_local_neg = scipy.stats.linregress(x_local_n,y_local_n)


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.axvline(0,color='k',lw=2)
# axs.scatter(x_local_n,y_local_n,color='b',s=5,alpha=0.5)
# axs.scatter(x_local,y_local,color='r',s=5,alpha=0.5)
# axs.plot(x_plot, x_plot*reg_local_pos.slope + reg_local_pos.intercept, color='k', lw=2,ls=':')
# axs.plot(x_plot_n, x_plot_n*reg_local_neg.slope + reg_local_neg.intercept, color='k', lw=2,ls=':')
# plt.tight_layout()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(x_local, y_local, color='k',s=5,alpha=0.5)
# axs.plot(x_plot, sigmoid(x_plot,*popt),color='r')
# axs.plot(x_plot, x_plot*reg_local_pos.slope+reg_local_pos.intercept, color='b')
# plt.tight_layout()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)

# idx_neg = np.where(deviation_fp<0)[0]
# axs.scatter(np.abs(deviation_fp[idx_pos]), np.abs(error_reduction_avg[idx_pos]), s=5, color='b', alpha=0.2)
# axs.scatter(np.abs(deviation_fp[idx_neg]), np.abs(error_reduction_avg[idx_neg]), s=5, color='r', alpha=0.2)
# plt.tight_layout()

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(np.abs(deviation_fp_fwd), np.abs(error_reduction_avg_fwd), s=5, color='k')
# plt.tight_layout()


reg_whole_window = scipy.stats.linregress(deviation_fp, error_reduction_avg)
print(reg_whole_window)
reg_whole_window_fwd = scipy.stats.linregress(deviation_fp_fwd, error_reduction_avg_fwd)
print(reg_whole_window_fwd)
print('Large then small')
reg_whole_window_l = scipy.stats.linregress(deviation_fp[idx_large], error_reduction_avg_l)
print(reg_whole_window_l)
reg_whole_window_s = scipy.stats.linregress(deviation_fp[idx_small], error_reduction_avg_s)
print(reg_whole_window_s)

# test_lat = scipy.stats.chatterjeexi(deviation_fp, error_reduction_avg)
# test_fwd= scipy.stats.chatterjeexi(deviation_fp_fwd, error_reduction_avg_fwd)

print(test_lat, test_fwd)
# sys.exit()

time_vector = np.arange(21,42)
vec_rsquare = np.zeros((len(time_vector)))
for ii in range(len(time_vector)):
    local_output = total_input_fr[idx_nans,time_vector[ii],5]
    local_error = matrix_inputs_final_avg[:,1] - local_output
    reg_local = scipy.stats.linregress(deviation_fp, local_error)
    vec_rsquare[ii] = reg_local.slope

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.plot(time_vector, vec_rsquare,color='k',lw=2)
# axs.set_xlabel('Time'), axs.set_ylabel('Explained variance')
# plt.tight_layout()


# idx_random = np.random.choice(total_input_fr.shape[0],25,replace=True)
# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# for ii in range(len(idx_random)):
#     axs.plot(np.arange(21), total_input_fr[idx_random[ii],:21,5],'k')
#     axs.plot(np.arange(21,42), total_input_fr[idx_random[ii],21:42,5],'k')
# plt.tight_layout()


# # Represent the body deviations as a function of foot contact deviations 
# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, matrix_inputs_final[:,3],color='k',s=5,alpha=0.5)
# axs.set_xlabel('Foot deviations'), axs.set_ylabel('Previous body deviations')
# plt.tight_layout()

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, matrix_outputs_final[:,3],color='k',s=5,alpha=0.5)
# axs.set_xlabel('Foot deviations'), axs.set_ylabel('Next body deviations')
# plt.tight_layout()

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, matrix_inputs_final_avg[:,3],color='k',s=5,alpha=0.5)
# axs.set_xlabel('Foot deviations'), axs.set_ylabel('Previous body deviations')
# plt.tight_layout()

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, matrix_outputs_final_avg[:,3],color='k',s=5,alpha=0.5)
# axs.set_xlabel('Foot deviations'), axs.set_ylabel('Next body deviations')
# plt.tight_layout()

idx_low = np.where(np.abs(deviation_fp)<np.nanmedian(np.abs(deviation_fp)))[0]
idx_high = np.where(np.abs(deviation_fp)>np.nanmedian(np.abs(deviation_fp)))[0]

print(np.nanmedian(error_reduction[idx_low]), np.nanmedian(error_reduction[idx_high]))

# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, error_reduction, color='k',s=5, alpha=0.2)
# axs.set_xlabel('Foot deviation'), axs.set_ylabel('Instantaneous error reduction - pos')
# plt.tight_layout()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, error_reduction_avg, color='k',s=5, alpha=0.2)
# axs.set_xlabel('Foot deviation'), axs.set_ylabel('Average error reduction - pos')
# plt.tight_layout()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, error_reduction_vel, color='k',s=5, alpha=0.2)
# axs.set_xlabel('Foot deviation'), axs.set_ylabel('Instantaneous error reduction - vel')
# plt.tight_layout()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(deviation_fp, error_reduction_avg_vel, color='k',s=5, alpha=0.2)
# axs.set_xlabel('Foot deviation'), axs.set_ylabel('Average error reduction - vel')
# plt.tight_layout()

A_low = np.linalg.inv(matrix_inputs_final[idx_low,:].T @ matrix_inputs_final[idx_low,:]) @ matrix_inputs_final[idx_low,:].T @ matrix_outputs_final[idx_low,:]
A_high = np.linalg.inv(matrix_inputs_final[idx_high,:].T @ matrix_inputs_final[idx_high,:]) @ matrix_inputs_final[idx_high,:].T @ matrix_outputs_final[idx_high,:]
A = np.linalg.inv(matrix_inputs_final.T @ matrix_inputs_final) @ matrix_inputs_final.T @ matrix_outputs_final
eigenvalue, eigenvectors = np.linalg.eig(A)
eig_low, _ = np.linalg.eig(A_low)
eig_high, _ = np.linalg.eig(A_high)
print(abs(eigenvalue))
print(abs(eig_high))
print(abs(eig_low))

plt.show()
