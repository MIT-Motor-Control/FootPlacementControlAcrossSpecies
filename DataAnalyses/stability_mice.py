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

# def fit_sigmoid(x,a):

input_path = os.path.join(os.getcwd(),'Datasets','Mouse','Klibaite','stability')
rdd_path = os.path.join(os.getcwd(),'rdd_figures')
os.makedirs(rdd_path, exist_ok=True)
# Loading the data
with open(os.path.join(input_path,'input_list_stability.pkl'),'rb') as f1:
    tot_input_fr = pickle.load(f1)
with open(os.path.join(input_path,'output_list_stability.pkl'),'rb') as f2:
    tot_output_fr = pickle.load(f2)


total_input_fr = tot_input_fr[0]
total_output_fr = tot_output_fr[0]
total_animal = np.zeros((tot_input_fr[0].shape[0],1))
for ii in tqdm(range(1,len(tot_input_fr))):
    total_input_fr = np.concatenate((total_input_fr, tot_input_fr[ii]),0)
    total_output_fr = np.concatenate((total_output_fr, tot_output_fr[ii]),0)
    total_animal = np.concatenate((total_animal, ii*np.ones((tot_input_fr[ii].shape[0],1))),0)

idx_flip = np.where(np.nanmean(total_input_fr[:,:,2],1)<0)[0]
total_input_fr[idx_flip,:,:] = - total_input_fr[idx_flip,:,:]
col_flip = [0,1,3,4]
for col in col_flip:
    total_output_fr[idx_flip,col] = - total_output_fr[idx_flip,col]

# End of the data processing
print(total_input_fr.shape)
tmp_mat = total_input_fr[:,:,[4,5,6,7]]
idx_nans = np.where(~np.isnan(total_input_fr[:,0,0]))[0]
matrix_inputs_early = tmp_mat[idx_nans,15,:]
matrix_outputs_early = tmp_mat[idx_nans,36,:]
A_early = np.linalg.inv(matrix_inputs_early.T @ matrix_inputs_early) @ matrix_inputs_early.T @ matrix_outputs_early
eig_early,_ = np.linalg.eig(A_early)
print('The eigenvalues of the non-normalized poincarre map are', np.abs(eig_early))
import copy
# Computing the individual eigenvalues 
matrix_eigenvalues = np.zeros((80, 4))
local_reg  = np.zeros((80,1))
for animal in range(matrix_eigenvalues.shape[0]):
    idx_animal = np.where((total_animal.flatten()==animal) & (~np.isnan(total_input_fr[:,0,0])))[0]
    local_input = tmp_mat[idx_animal,15,:]
    tmp_input = copy.deepcopy(local_input)
    tmp_input[:,1] = local_input[:,0]
    tmp_input[:,0] = local_input[:,1]
    local_output = tmp_mat[idx_animal,36,:]
    tmp_output = copy.deepcopy(local_output)
    tmp_output[:,1] = local_output[:,0]
    tmp_output[:,0] = local_output[:,1]
    reg_tmp = scipy.stats.linregress(local_input[:,1], local_output[:,2])
    local_reg[animal] = reg_tmp.slope
    A_local = np.linalg.inv(tmp_input.T @ tmp_input) @ (tmp_input.T @ tmp_output)
    print(A_local)
    local_eig, _ = np.linalg.eig(A_local)
    matrix_eigenvalues[animal, :] = local_eig
tmp_output_path = os.path.join(os.getcwd(),'round_2_analysis')
np.save(os.path.join(tmp_output_path,'matrix_eig_mice_flipped.npy'), matrix_eigenvalues)
np.save(os.path.join(tmp_output_path,'matrix_eig_mice_single.npy'), local_reg)
sys.exit()

matrix_outputs_stab = np.zeros((total_output_fr.shape[0],))
matrix_outputs_stab_fwd = np.zeros((total_output_fr.shape[0],))
for animal in range(80):
    idx_animal = np.where((total_animal).flatten()==animal)[0]
    matrix_outputs_stab[idx_animal] = total_output_fr[idx_animal,1] - np.nanmean(total_output_fr[idx_animal,1],0)
    matrix_outputs_stab_fwd[idx_animal] = total_output_fr[idx_animal,0] - np.nanmean(total_output_fr[idx_animal,0],0)



# Normalization inputs
for animal in tqdm(range(80)):
    idx_animal = np.where((total_animal).flatten()==animal)[0]
    tmp_vel1 = np.nanmean(total_input_fr[idx_animal,:21,6],1)
    tmp_vel2 = np.nanmean(total_input_fr[idx_animal,21:,6],1)
    # print(type(tmp_vel1), type(tot_output_fr[idx_animal,0]))
    regvel = scipy.stats.linregress(tmp_vel1, total_output_fr[idx_animal,0])
    total_output_fr[idx_animal,0] = total_output_fr[idx_animal,0] - (tmp_vel1*regvel.slope+regvel.intercept)
    total_output_fr[idx_animal,1] = total_output_fr[idx_animal,1] - np.nanmean(total_output_fr[idx_animal,1])
    total_input_fr[idx_animal,:21,5] = total_input_fr[idx_animal,:21,5] - np.nanmean(total_input_fr[idx_animal,:21,5],0)
    total_input_fr[idx_animal,:21,6] = total_input_fr[idx_animal,:21,6] - np.expand_dims(tmp_vel1,-1)
    total_input_fr[idx_animal,:21,7] = total_input_fr[idx_animal,:21,7] - np.nanmean(total_input_fr[idx_animal,:21,7],0)
    total_input_fr[idx_animal,21:,5] = total_input_fr[idx_animal,21:,5] - np.nanmean(total_input_fr[idx_animal,21:,5],0)
    total_input_fr[idx_animal,21:,6] = total_input_fr[idx_animal,21:,6] - np.expand_dims(tmp_vel2,-1)
    total_input_fr[idx_animal,21:,7] = total_input_fr[idx_animal,21:,7] - np.nanmean(total_input_fr[idx_animal,21:,7],0)
    for line in range(total_input_fr.shape[0]):
        time_vector = np.arange(21)
        if np.isnan(total_input_fr[line,0,4]):
            continue
        else:
            linelin1 = scipy.stats.linregress(time_vector, total_input_fr[line,:21,4])
            linelin2 = scipy.stats.linregress(time_vector, total_input_fr[line,21:,4])
            total_input_fr[line,:21,4] = total_input_fr[line,:21,4] - (time_vector*linelin1.slope+linelin1.intercept)
            total_input_fr[line,21:,4] = total_input_fr[line,21:,4] - (time_vector*linelin2.slope+linelin2.intercept)


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(total_input_fr[line,:,4],'k',lw=0.5)
plt.tight_layout()
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(total_input_fr[line,:,5],'k',lw=0.5)
plt.tight_layout()


matrix_inputs = total_input_fr[:,15,[4,5,6,7]]
matrix_outputs = total_input_fr[:,35,[4,5,6,7]]
matrix_inputs_averaged = np.nanmean(total_input_fr[:,10:21,[4,5,6,7]],1)
matrix_outputs_averaged = np.nanmean(total_input_fr[:,21:32,[4,5,6,7]],1)

idx_nans = np.where((~np.isnan(matrix_inputs[:,0])) & (~np.isnan(matrix_outputs[:,0])))[0]

matrix_inputs_final = matrix_inputs[idx_nans,:]
matrix_outputs_final = matrix_outputs[idx_nans,:]
matrix_inputs_averaged_final = matrix_inputs_averaged[idx_nans,:]
matrix_outputs_averaged_final = matrix_outputs_averaged[idx_nans,:]
matrix_outputs_stab_final = matrix_outputs_stab[idx_nans]
matrix_outputs_stab_fwd_final = matrix_outputs_stab_fwd[idx_nans]

error_reduction = matrix_inputs_final[:,1] - matrix_outputs_final[:,1]
error_reduction_averaged = matrix_inputs_averaged_final[:,1] - matrix_outputs_averaged_final[:,1]
error_reduction_averaged_fwd = matrix_inputs_averaged_final[:,0] - matrix_outputs_averaged_final[:,0]

time_vector = np.arange(21,42)
vec_rsquare = np.zeros((len(time_vector),))
for ii in range(len(time_vector)):
    local_output = total_input_fr[:,time_vector[ii],5]
    local_error = matrix_inputs_averaged[:,1] - local_output
    reg_local = scipy.stats.linregress(matrix_outputs_stab_final, local_error[idx_nans])
    vec_rsquare[ii] = reg_local.slope

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(time_vector, vec_rsquare,'k',lw=2)
axs.set_xlabel('Time'), axs.set_ylabel('Explained variance')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(matrix_outputs_stab_final, error_reduction_averaged,color='k',s=5,alpha=0.5)
axs.set_xlim([-50,50])
axs.set_xlabel('Contact deviation'), axs.set_ylabel('Error reduction')
plt.tight_layout()
print(scipy.stats.chatterjeexi(matrix_outputs_stab_final, error_reduction_averaged))

idx_pos = np.where(matrix_outputs_stab_final>0)[0]
idx_neg = np.where(matrix_outputs_stab_final<0)[0]

pos_deviations = matrix_outputs_stab_final[idx_pos]
pos_errors = error_reduction_averaged[idx_pos]
neg_deviations = matrix_outputs_stab_final[idx_neg]
neg_errors = error_reduction_averaged[idx_neg]

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


data = pd.DataFrame({'y':pos_errors, 'x':pos_deviations})
bandwidth_opt =36# rdd.optimal_bandwidth(data['y'], data['x'], cut=np.nanmedian(pos_deviations))
print(bandwidth_opt)
data_rdd = rdd.truncated_data(data, 'x', bandwidth_opt, cut=np.nanmedian(pos_deviations))
model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(pos_deviations))
print(model.fit().summary())

fig, axs = plt.subplots(1,2,figsize=(6,3))
axs[0].spines[['top','right']].set_visible(False)
axs[1].spines[['top','right']].set_visible(False)
vp1_0 = axs[0].violinplot([np.array(slope_below_neg)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
vp2_0 = axs[0].violinplot([np.array(slope_above_neg)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
vp1_1 = axs[1].violinplot([np.array(slope_below)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
vp2_1 = axs[1].violinplot([np.array(slope_above)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
axs[0].set_ylabel('Slope')
vp1_0['cmeans'].set_color('blue')
vp2_0['cmeans'].set_color('red')
vp1_1['cmeans'].set_color('blue')
vp2_1['cmeans'].set_color('red')
for pc in vp1_0['bodies']:
    pc.set_facecolor('blue')
    pc.set_edgecolor('blue')
for pc in vp2_0['bodies']:
    pc.set_facecolor('red')
    pc.set_edgecolor('red')
for pc in vp1_1['bodies']:
    pc.set_facecolor('blue')
    pc.set_edgecolor('blue')
for pc in vp2_1['bodies']:
    pc.set_facecolor('red')
    pc.set_edgecolor('red')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'slopes_violinplot_mice.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'slopes_violinplot_mice.svg'),bbox_inches='tight')


fig, axs = plt.subplots(1,2,figsize=(6,3))
axs[0].spines[['top','right']].set_visible(False)
axs[1].spines[['top','right']].set_visible(False)
# Leftmost subplot
axs[0].set_xlabel('Foot placement'), axs[0].set_ylabel('Error reduction')
axs[1].set_xlabel('Foot placement')
axs[0].scatter(neg_deviations[idx_below_median_neg], neg_errors[idx_below_median_neg], color='b', s=5)
axs[0].scatter(neg_deviations[idx_above_median_neg], neg_errors[idx_above_median_neg], color='r', s=5)
x_plot_below = np.linspace(min(neg_deviations),np.nanmedian(neg_deviations))
x_plot_above = np.linspace(np.nanmedian(neg_deviations),0)
axs[0].plot(x_plot_below, x_plot_below*global_reg_below_neg.slope+global_reg_below_neg.intercept,'k',lw=2)
axs[0].plot(x_plot_above, x_plot_above*global_reg_above_neg.slope+global_reg_above_neg.intercept,'k',lw=2)
axs[0].plot(x_plot_below, x_plot_below*np.nanmedian(np.array(slope_below_neg))+np.nanmedian(np.array(intercept_below_neg)),'k:',lw=2)
axs[0].plot(x_plot_above, x_plot_above*np.nanmedian(np.array(slope_above_neg))+np.nanmedian(np.array(intercept_above_neg)),'k:',lw=2)
# Rightmost subplot
axs[1].scatter(pos_deviations[idx_below_median], pos_errors[idx_below_median], color='b', s=5)
axs[1].scatter(pos_deviations[idx_above_median], pos_errors[idx_above_median], color='r', s=5)
x_plot_below = np.linspace(0,np.nanmedian(pos_deviations))
x_plot_above = np.linspace(np.nanmedian(pos_deviations), max(pos_deviations))
axs[1].plot(x_plot_below, x_plot_below*global_reg_below.slope+global_reg_below.intercept,'k',lw=2)
axs[1].plot(x_plot_above, x_plot_above*global_reg_above.slope+global_reg_above.intercept,'k',lw=2)
axs[1].plot(x_plot_below, x_plot_below*np.nanmedian(np.array(slope_below))+np.nanmedian(np.array(intercept_below)),'k:',lw=2)
axs[1].plot(x_plot_above, x_plot_above*np.nanmedian(np.array(slope_above))+np.nanmedian(np.array(intercept_above)),'k:',lw=2)
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'mice_points_rdd.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'mice_points_rdd.svg'),bbox_inches='tight')

x_plot = np.linspace(0, np.max(pos_deviations))
df = pd.DataFrame({'deviations':pd.Series(pos_deviations), 'errors':pd.Series(pos_errors), 'x_plot':pd.Series(np.sort(pos_deviations))})
X = sm.add_constant(df['deviations'].values)
X_plot = sm.add_constant(df['x_plot'].values)
ols_model = sm.OLS(df['errors'].values, X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)

tot_reg_lin = scipy.stats.linregress(pos_deviations, pos_errors)

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
pred = est.get_prediction(X_plot).summary_frame()
axs.scatter(pos_deviations, pos_errors, color='k', s=5)
axs.plot(np.sort(pos_deviations), pred['mean_ci_upper'],'b--')
axs.plot(np.sort(pos_deviations), pred['mean_ci_lower'],'b--')
axs.plot(x_plot, x_plot*tot_reg_lin.slope+tot_reg_lin.intercept,color='b',lw=2)
axs.fill_between(np.sort(pos_deviations), pred['mean_ci_upper'], pred['mean_ci_lower'],color='b',alpha=0.5)
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
axs.set_xlim([0, 60]), axs.set_ylim([-25, 45])
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'blackdots_lateral_mice.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'blackdots_lateral_mice.svg'),bbox_inches='tight')


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_deviations[idx_below_median], pos_errors[idx_below_median], color='b', s=5)
axs.scatter(pos_deviations[idx_above_median], pos_errors[idx_above_median], color='r', s=5)
x_plot_below = np.linspace(0,np.nanmedian(pos_deviations))
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
x_plot_above = np.linspace(np.nanmedian(pos_deviations), max(pos_deviations))
axs.plot(x_plot_below, x_plot_below*global_reg_below.slope+global_reg_below.intercept,'k',lw=2)
axs.plot(x_plot_above, x_plot_above*global_reg_above.slope+global_reg_above.intercept,'k',lw=2)
axs.plot(x_plot_below, x_plot_below*np.nanmedian(np.array(slope_below))+np.nanmedian(np.array(intercept_below)),'k:',lw=2)
axs.plot(x_plot_above, x_plot_above*np.nanmedian(np.array(slope_above))+np.nanmedian(np.array(intercept_above)),'k:',lw=2)
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'mice_points.png'),bbox_inches='tight')
plt.show()



#########################
### Forward direction ###
#########################idx_pos = np.where(matrix_outputs_stab_final>0)[0]
idx_pos = np.where(matrix_outputs_stab_fwd_final>0)[0]
pos_deviations = matrix_outputs_stab_fwd_final[idx_pos]
pos_errors = error_reduction_averaged_fwd[idx_pos]
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

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(diff_list)], widths=0.5, showextrema=False, showmeans=True, points=60)
axs.axhline(np.nanpercentile(np.array(diff_list),5),color='k',ls=':',lw=2)
axs.axhline(np.nanpercentile(np.array(diff_list),95),color='k',ls=':',lw=2)
axs.axhline(0, color='k',lw=2)
axs.set_ylabel('Differences below-above')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'mice_slopes_diff_fwd.png'),bbox_inches='tight')



# data = pd.DataFrame({'y':pos_errors, 'x':pos_deviations})
# bandwidth_opt =36# rdd.optimal_bandwidth(data['y'], data['x'], cut=np.nanmedian(pos_deviations))
# print(bandwidth_opt)
# data_rdd = rdd.truncated_data(data, 'x', bandwidth_opt, cut=np.nanmedian(pos_deviations))
# model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(pos_deviations))
# print(model.fit().summary())


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Slope below and above median - Mice', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(slope_below)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
axs.violinplot([np.array(slope_above)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
axs.set_ylabel('Slope')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'mice_slopes_fwd.png'),bbox_inches='tight')


x_plot = np.linspace(0,np.max(pos_deviations))
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
axs.scatter(pos_deviations, pos_errors, color='k', s=5)
axs.plot(np.sort(pos_deviations), pred['mean_ci_upper'],'b--')
axs.plot(np.sort(pos_deviations), pred['mean_ci_lower'],'b--')
axs.plot(x_plot, x_plot*tot_reg_lin.slope+tot_reg_lin.intercept,color='b',lw=2)
axs.fill_between(np.sort(pos_deviations), pred['mean_ci_upper'], pred['mean_ci_lower'],color='b',alpha=0.5)
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
axs.set_xlim([0, 150]), axs.set_ylim([-2.5, 5])
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'blackdots_foreaft_mice.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'blackdots_foreaft_mice.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_deviations[idx_below_median], pos_errors[idx_below_median], color='b', s=5)
axs.scatter(pos_deviations[idx_above_median], pos_errors[idx_above_median], color='r', s=5)
x_plot_below = np.linspace(0,np.nanmedian(pos_deviations))
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
x_plot_above = np.linspace(np.nanmedian(pos_deviations), max(pos_deviations))
axs.plot([min(x_plot_below), max(x_plot_below)], [np.nanmedian(pos_errors[idx_local_below]), np.nanmedian(pos_errors[idx_local_below])], 'k', lw=2)
axs.plot([min(x_plot_above), max(x_plot_above)], [np.nanmedian(pos_errors[idx_local_above]), np.nanmedian(pos_errors[idx_local_above])], 'k:', lw=2)
axs.fill_between([min(x_plot_below), max(x_plot_below)],[np.nanpercentile(np.array(slope_below),5), np.nanpercentile(np.array(slope_below),95)],color='k',alpha=0.5)
axs.fill_between([min(x_plot_above), max(x_plot_above)],[np.nanpercentile(np.array(slope_above),5), np.nanpercentile(np.array(slope_above),95)],color='k',alpha=0.5)
# axs.fill_between([min(x_plot_above), max(x_plot_above)],[np.nanmedian(pos_errors[idx_local_above])-scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_above])-scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2],[np.nanmedian(pos_errors[idx_local_above])+scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_above])+scipy.stats.iqr(pos_errors[idx_local_above],nan_policy='omit')/2],color='k',alpha=0.5)
# axs.fill_between([min(x_plot_below), max(x_plot_below)],[np.nanmedian(pos_errors[idx_local_below])-scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_below])-scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2],[np.nanmedian(pos_errors[idx_local_below])+scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2, np.nanmedian(pos_errors[idx_local_below])+scipy.stats.iqr(pos_errors[idx_local_below],nan_policy='omit')/2],color='k',alpha=0.5)
axs.set_xlim([0,150])
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'mice_points_fwd.png'),bbox_inches='tight')
plt.show()
sys.exit()

sys.exit()
pos_deviations = matrix_outputs_stab_final[idx_pos]
pos_errors = error_reduction_averaged[idx_pos]
bounds_pos = np.nanpercentile(pos_deviations, [0,10,20,30,40,50,60,70,80,90,100])

def sigmoid(x, A, B, C, D):
    y = A / (1 + np.exp(-C*(x-B))) + D
    return y
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# fit a sigmoid to the first percentiles
list_fitted_params, list_rsquare = [], []
for ii in range(10):
    x_fit = np.linspace(bounds_pos[0],bounds_pos[ii+1])
    idx_local = np.where((pos_deviations>0) & (pos_deviations<bounds_pos[ii+1]))[0]
    x_local = pos_deviations[idx_local]
    y_local = pos_errors[idx_local]

    p0 = [max(y_local), np.nanmedian(x_local), 1, min(y_local)]
    popt, _ = curve_fit(sigmoid, x_local, y_local, p0, method='lm', maxfev=1000000)
    list_fitted_params.append(popt)
    y_pred = sigmoid(x_local, *popt)
    r_square_local = r2_score(y_pred, y_local)
    list_rsquare.append(r_square_local)



window_width = 2
idx_local_low = np.where((matrix_outputs_stab_final>4.75-window_width) & (matrix_outputs_stab_final<4.75))[0]
idx_local_high = np.where((matrix_outputs_stab_final>4.75) & (matrix_outputs_stab_final<4.75+window_width))[0]
idx_local_all = np.where((matrix_outputs_stab_final>4.75-window_width) & (matrix_outputs_stab_final<4.75+window_width))[0]

reg_local_low = scipy.stats.linregress(matrix_outputs_stab_final[idx_local_low], error_reduction_averaged[idx_local_low])
reg_local_high = scipy.stats.linregress(matrix_outputs_stab_final[idx_local_high], error_reduction_averaged[idx_local_high])
reg_local_all = scipy.stats.linregress(matrix_outputs_stab_final[idx_local_all], error_reduction_averaged[idx_local_all])
x_p1 = np.linspace(4.75-window_width, 4.75)
x_p2 = np.linspace(4.75,4.75+window_width)
x_p = np.linspace(4.75-window_width, 4.75+window_width)



fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(matrix_outputs_stab_final[idx_local_low], error_reduction_averaged[idx_local_low], color='k', s=5, alpha=0.5)
axs.scatter(matrix_outputs_stab_final[idx_local_high], error_reduction_averaged[idx_local_high], color='r', s=5, alpha=0.5)
axs.plot(x_p1, x_p1*reg_local_low.slope+reg_local_low.intercept, color='k', lw=2)
axs.plot(x_p2, x_p2*reg_local_high.slope+reg_local_high.intercept,color='r',lw=2)
axs.plot(x_p, x_p*reg_local_all.slope+reg_local_all.intercept,color='b',lw=2, ls=':')
axs.set_xlabel('Foot placement deviation'), axs.set_ylabel('Error reduction')
axs.set_xlim([0,150])
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'mice_points_fwd.png'),bbox_inches='tight')
plt.show()



sys.exit()


import matplotlib.cm as cm
color_sigmo = cm.grey(np.linspace(0,1,len(list_fitted_params)))
fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Lateral direction',fontweight='bold')
for ii in range(len(bounds_pos)-1):
    axs.axvline(bounds_pos[ii+1], color='r',lw=1)
    axs.plot(x_fit, sigmoid(x_fit, *list_fitted_params[ii]), color=color_sigmo[ii], lw=1)
axs.spines[['top','right']].set_visible(False)
axs.scatter(x_local,y_local,color='b')
axs.scatter(matrix_outputs_stab_final, error_reduction_averaged, color='k',s=5,alpha=0.5)
axs.set_xlabel('Contact deviation'), axs.set_ylabel('Error reduction')
plt.tight_layout()


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(matrix_outputs_stab_fwd_final, error_reduction_averaged_fwd,color='k',s=5,alpha=0.5)
axs.set_xlim([-50,50])
axs.set_xlabel('Contact deviation'), axs.set_ylabel('Error reduction')
plt.tight_layout()
print(scipy.stats.chatterjeexi(matrix_outputs_stab_fwd_final, error_reduction_averaged_fwd))
reg_whole_window = scipy.stats.linregress(matrix_outputs_stab_final, error_reduction_averaged)
print(reg_whole_window)


A = np.linalg.inv(matrix_inputs_final.T @ matrix_inputs_final) @ matrix_inputs_final.T @ matrix_outputs_final
eigenvalue, eigenvectors = np.linalg.eig(A)
print(abs(eigenvalue))
plt.show()
