import os, sys 
import pickle 
import scipy.stats
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import pandas as pd 

input_path = os.path.join(os.getcwd(),'Datasets','Humans','Camargo','processed_stability')
rdd_path = os.path.join(os.getcwd(),'rdd_figures')
os.makedirs(rdd_path, exist_ok=True)
# Loading the data 
with open(os.path.join(input_path,'tot_input_fr.pkl'),'rb') as f1:
    tot_input_fr = pickle.load(f1)
with open(os.path.join(input_path,'tot_output_fr.pkl'),'rb') as f2:
    tot_output_fr = pickle.load(f2)
with open(os.path.join(input_path,'tot_subject1.pkl'),'rb') as f3:
    tot_subject1 = pickle.load(f3)

with open(os.path.join(input_path,'tot_input_fl.pkl'),'rb') as f1:
    tot_input_fl = pickle.load(f1)
with open(os.path.join(input_path,'tot_output_fl.pkl'),'rb') as f2:
    tot_output_fl = pickle.load(f2)
with open(os.path.join(input_path,'tot_subject2.pkl'),'rb') as f3:
    tot_subject2 = pickle.load(f3)
n_sub = int(np.max(tot_subject1)+1)
matrix_inputs_early = tot_input_fr[:,75,:]
matrix_outputs_early = tot_input_fr[:,176,:]

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Foreaft direction', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(tot_input_fl[line,:,0],color='k',lw=1)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Lateral direction',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(tot_input_fl[line,:,1],color='k',lw=1)
plt.tight_layout()


A_early = np.linalg.inv(matrix_inputs_early.T @ matrix_inputs_early) @ matrix_inputs_early.T @ matrix_outputs_early
eig_early,_ = np.linalg.eig(A_early)
print('The eigenvalues of the non-normalized poincarre map are', np.abs(eig_early))

matrix_eig = np.zeros((4,n_sub))
A_early_array = np.zeros((4,4,n_sub))
for sub in range(n_sub):
    idx_subject = np.where(tot_subject1==sub)[0]
    local_input = tot_input_fr[idx_subject,75,:]
    local_output = tot_input_fr[idx_subject,176,:]
    if local_input.shape[0]<4:
        continue
    else:
        A_early_array[:,:,sub] = np.linalg.inv(local_input.T @ local_input) @ local_input.T @ local_output
        local_eig, _ = np.linalg.eig(A_early_array[:,:,sub])
        matrix_eig[:,sub] = np.abs(local_eig)
max_vec = np.zeros((n_sub,))
for ii in range(sub):
    if np.max(matrix_eig[:,ii])!=0:
        max_vec[ii] = np.max(matrix_eig[:,ii])
    else:
        max_vec[ii] = np.nan

print(matrix_eig[1,:])
print(np.nanmean(max_vec))
print(scipy.stats.iqr(max_vec,nan_policy='omit'))
# sys.exit()
tot_input_fr_intermediate = tot_input_fr - np.nanmean(tot_input_fr,0)
matrix_inputs_int = tot_input_fr_intermediate[:,75,:]
matrix_outputs_int = tot_input_fr_intermediate[:,176,:]
A_int = np.linalg.inv(matrix_inputs_int.T @ matrix_inputs_int) @ matrix_inputs_int.T @ matrix_outputs_int
eig_int,_ = np.linalg.eig(A_int)
print('The eigenvalues of the normalized poincarre map are', np.abs(eig_int))

matrix_eig_int = np.zeros((4,n_sub))
A_early_array_int = np.zeros((4,4,n_sub))
local_reg= np.zeros((n_sub,1))
for sub in range(n_sub):
    idx_subject = np.where(tot_subject1==sub)[0]
    local_input = tot_input_fr[idx_subject,75,:] - np.nanmean(tot_input_fr[idx_subject,75,:],0)
    local_output = tot_input_fr[idx_subject,176,:] - np.nanmean(tot_input_fr[idx_subject,176,:],0)
    if local_input.shape[0]<4:
        continue
    else:
        A_early_array_int[:,:,sub] = np.linalg.inv(local_input.T @ local_input) @ local_input.T @ local_output
        local_eig, _ = np.linalg.eig(A_early_array[:,:,sub])
        matrix_eig_int[:,sub] = np.abs(local_eig)
        tmp_reg = scipy.stats.linregress(local_input[:,1], local_output[:,1])
        local_reg[sub] = tmp_reg.slope
max_vec_int = np.zeros((n_sub,))
for ii in range(sub):
    if np.max(matrix_eig_int[:,ii])!=0:
        max_vec_int[ii] = np.max(matrix_eig_int[:,ii])
    else:
        max_vec_int[ii] = np.nan
print(matrix_eig_int[1,:])
tmp_output_path = os.path.join(os.getcwd(),'round_2_analysis')
os.makedirs(tmp_output_path, exist_ok=True)
np.save(os.path.join(tmp_output_path,'eig_humans.npy'), matrix_eig_int)
np.save(os.path.join(tmp_output_path,'slope_humans.npy'), local_reg)
sys.exit()
print(np.nanmean(max_vec_int))
print(scipy.stats.iqr(max_vec_int,nan_policy='omit'))


# Input output normalization
for subject in range(n_sub):
    idx_subject = np.where((tot_subject1)==subject)[0]
    tmp_vel1 = np.nanmean(tot_input_fr[idx_subject,:101,2],1)
    tmp_vel2 = np.nanmean(tot_input_fr[idx_subject,101:,2],1)
    regvel = scipy.stats.linregress(tmp_vel1, tot_output_fr[idx_subject,0])
    tot_output_fr[idx_subject,0] = tot_output_fr[idx_subject,0] - (tmp_vel1*regvel.slope + regvel.intercept)
    tot_output_fr[idx_subject,1] = tot_output_fr[idx_subject,1] - np.nanmean(tot_output_fr[idx_subject,1])
    tot_input_fr[idx_subject,:101,1] = tot_input_fr[idx_subject,:101,1] - np.nanmean(tot_input_fr[idx_subject,:101,1],0)
    tot_input_fr[idx_subject,:101,3] = tot_input_fr[idx_subject,:101,3] - np.nanmean(tot_input_fr[idx_subject,:101,3],0) 
    tot_input_fr[idx_subject,:101,2] = tot_input_fr[idx_subject,:101,2] - np.expand_dims(tmp_vel1,-1)
    tot_input_fr[idx_subject,101:,1] = tot_input_fr[idx_subject,101:,1] - np.nanmean(tot_input_fr[idx_subject,101:,1],0)
    tot_input_fr[idx_subject,101:,3] = tot_input_fr[idx_subject,101:,3] - np.nanmean(tot_input_fr[idx_subject,101:,3],0) 
    tot_input_fr[idx_subject,101:,2] = tot_input_fr[idx_subject,101:,2] - np.expand_dims(tmp_vel2,-1)
    for line in range(len(idx_subject)):
        x_vec = np.arange(101)
        reglin_1 = scipy.stats.linregress(x_vec, tot_input_fr[idx_subject[line],:101,0])
        reglin_2 = scipy.stats.linregress(x_vec, tot_input_fr[idx_subject[line],101:,0])
        tot_input_fr[idx_subject[line],:101,0] = tot_input_fr[idx_subject[line],:101,0] - (x_vec*reglin_1.slope+reglin_1.intercept)
        tot_input_fr[idx_subject[line],101:,0] = tot_input_fr[idx_subject[line],101:,0] - (x_vec*reglin_2.slope+reglin_2.intercept)

for subject in range(n_sub):
    idx_subject = np.where((tot_subject2)==subject)[0]
    tmp_vel1 = np.nanmean(tot_input_fl[idx_subject,:101,2],1)
    tmp_vel2 = np.nanmean(tot_input_fl[idx_subject,101:,2],1)
    tot_output_fl[idx_subject,1] = tot_output_fl[idx_subject,1] - np.nanmean(tot_output_fl[idx_subject,1])
    tot_input_fl[idx_subject,:101,1] = tot_input_fl[idx_subject,:101,1] - np.nanmean(tot_input_fl[idx_subject,:101,1],0)
    tot_input_fl[idx_subject,:101,3] = tot_input_fl[idx_subject,:101,3] - np.nanmean(tot_input_fl[idx_subject,:101,3],0) 
    tot_input_fl[idx_subject,:101,2] = tot_input_fl[idx_subject,:101,2] - np.expand_dims(tmp_vel1,-1)
    tot_input_fl[idx_subject,101:,1] = tot_input_fl[idx_subject,101:,1] - np.nanmean(tot_input_fl[idx_subject,101:,1],0)
    tot_input_fl[idx_subject,101:,3] = tot_input_fl[idx_subject,101:,3] - np.nanmean(tot_input_fl[idx_subject,101:,3],0) 
    tot_input_fl[idx_subject,101:,2] = tot_input_fl[idx_subject,101:,2] - np.expand_dims(tmp_vel2,-1)
    for line in range(len(idx_subject)):
        x_vec = np.arange(101)
        reglin_1 = scipy.stats.linregress(x_vec, tot_input_fl[idx_subject[line],:101,0])
        reglin_2 = scipy.stats.linregress(x_vec, tot_input_fl[idx_subject[line],101:,0])
        tot_input_fl[idx_subject[line],:101,0] = tot_input_fl[idx_subject[line],:101,0] - (x_vec*reglin_1.slope+reglin_1.intercept)
        tot_input_fl[idx_subject[line],101:,0] = tot_input_fl[idx_subject[line],101:,0] - (x_vec*reglin_2.slope+reglin_2.intercept)


# Compute the deviations at a specific point (a quarter of gait cycles preceeding or following the contact)
matrix_inputs = tot_input_fr[:,75,:]
matrix_outputs = tot_input_fr[:,126,:]

fig,axs= plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(tot_input_fr[line,:,1],'k',lw=0.5)
plt.tight_layout()


fig,axs= plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for line in range(25):
    axs.plot(tot_input_fr[line,:,0],'k',lw=0.5)
plt.tight_layout()


# Compute the average deviations during the half gait cycle preceeding or following the contact 
matrix_inputs_averaged = np.nanmean(tot_input_fr[:,50:101,:],1)
matrix_outputs_averaged = np.nanmean(tot_input_fr[:,101:152,:],1)
matrix_inputs_averaged_od = np.nanmean(tot_input_fl[:,50:101,:],1)
matrix_outputs_averaged_od = np.nanmean(tot_input_fl[:,101:152,:],1)


error_reduction = matrix_inputs[:,1] - matrix_outputs[:,1]
error_reduction_avg = matrix_inputs_averaged[:,1] - matrix_outputs_averaged[:,1]
error_reduction_avg_fwd = matrix_inputs_averaged[:,0] - matrix_outputs_averaged[:,0]
error_reduction_avg_od = matrix_inputs_averaged_od[:,1] - matrix_outputs_averaged_od[:,1]

reg_whole_window = scipy.stats.linregress(tot_output_fr[:,1], error_reduction_avg)
reg_whole_window_fwd = scipy.stats.linregress(tot_output_fr[:,0], error_reduction_avg_fwd)
reg_whole_window_od = scipy.stats.linregress(tot_output_fl[:,1], error_reduction_avg_od)
print('Information concerning the regressions')
print(reg_whole_window.rvalue**2, reg_whole_window.slope, reg_whole_window.pvalue)
print(reg_whole_window_od.rvalue**2, reg_whole_window_od.slope, reg_whole_window_od.pvalue)

idx_large = np.where(np.abs(tot_output_fr[:,1])>np.nanmedian(np.abs(tot_output_fr[:,1])))[0]
idx_small = np.where(np.abs(tot_output_fr[:,1])<np.nanmedian(np.abs(tot_output_fr[:,1])))[0]

reg_large = scipy.stats.linregress(tot_output_fr[idx_large,1], error_reduction_avg[idx_large])
reg_small = scipy.stats.linregress(tot_output_fr[idx_small,1], error_reduction_avg[idx_small])
print('============================================')
print(reg_large.rvalue**2, reg_small.rvalue**2)
print(reg_large.slope, reg_small.slope)
print(reg_large.pvalue, reg_small.pvalue)

time_vector = np.arange(101,202)
vec_rsquare = np.zeros((len(time_vector),))
vec_rsquare_od = np.zeros((len(time_vector),))
for ii in range(len(time_vector)):
    local_output = tot_input_fr[:,time_vector[ii],1]
    local_error = matrix_inputs_averaged[:,1] - local_output
    reg_local = scipy.stats.linregress(tot_output_fr[:,1],local_error)
    vec_rsquare[ii] = reg_local.slope
    local_output = tot_input_fl[:,time_vector[ii],1]
    local_error = matrix_inputs_averaged_od[:,1] - local_output
    reg_local = scipy.stats.linregress(tot_output_fl[:,1],local_error)
    vec_rsquare_od[ii] = reg_local.slope

print(reg_whole_window)
plt.close('all')
# Representing the stuff for the correlation



fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(time_vector, vec_rsquare,color='k',lw=2)
axs.plot(time_vector, vec_rsquare_od,color='r',lw=2)
axs.set_ylim([-0.05,1.05])
axs.set_ylabel('Explained variance')
axs.set_xlabel('Maximal window size')
plt.tight_layout()
# fig.savefig('stability_humans.png',bbox_inches='tight')
# fig.savefig('stability_humans.svg',bbox_inches='tight')
print('Reached the end')


# Doing the analyzes for the inflection points
idx_pos = np.where(tot_output_fr[:,1]>0)[0]
idx_neg = np.where(tot_output_fr[:,1]<0)[0]
print(len(idx_pos) / tot_output_fr.shape[0])

pos_deviations = tot_output_fr[idx_pos,1]
pos_errors = error_reduction_avg[idx_pos]
bounds_pos = np.nanpercentile(pos_deviations, [0,10,20,30,40,50,60,70,80,90,100])

neg_deviations = tot_output_fr[idx_neg,1]
neg_errors = error_reduction_avg[idx_neg]



plt.close('all')
# Do the statistics around the median ... 
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
slope_below_neg, slope_above_neg = [], []
intercept_below, intercept_above = [], [] 
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
    slope_below_neg.append(local_reg_below_neg.slope), slope_above_neg.append(local_reg_above_neg.slope)
    intercept_below.append(local_reg_below.intercept), intercept_above.append(local_reg_above.intercept)
    intercept_below_neg.append(local_reg_below_neg.intercept), intercept_above.append(local_reg_above_neg.intercept)
    diff_list.append(local_reg_below.slope - local_reg_above.slope)




data = pd.DataFrame({'y':pos_errors, 'x':pos_deviations})
bandwidth_opt =36# rdd.optimal_bandwidth(data['y'], data['x'], cut=np.nanmedian(pos_deviations))
print(bandwidth_opt)
data_rdd = rdd.truncated_data(data, 'x', bandwidth_opt, cut=np.nanmedian(pos_deviations))
model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(pos_deviations))
print(model.fit().summary())


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(diff_list)], widths=0.5, showextrema=False, showmeans=True, points=60)
axs.axhline(np.nanpercentile(np.array(diff_list),5),color='k',ls=':',lw=2)
axs.axhline(np.nanpercentile(np.array(diff_list),95),color='k',ls=':',lw=2)
axs.axhline(0, color='k',lw=2)
axs.set_ylabel('Differences below-above')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'humans_slopes_diff.png'),bbox_inches='tight')


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
fig.savefig(os.path.join(rdd_path,'slopes_violinplot_humans.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'slopes_violinplot_humans.svg'),bbox_inches='tight')



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
fig.savefig(os.path.join(rdd_path,'humans_points_rdd.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path,'humans_points_rdd.svg'),bbox_inches='tight')


# Let us fit a line and add the confidence interval of that line on the plot below
tot_reg_lin = scipy.stats.linregress(pos_deviations, pos_errors)
n_boot = 10000
slope_distro = []

x_plot = np.linspace(0,np.max(pos_deviations))
print(pos_deviations.shape, pos_errors.shape)
df = pd.DataFrame({'deviations':pd.Series(pos_deviations), 'errors':pd.Series(pos_errors), 'x_plot':pd.Series(np.sort(pos_deviations))})
X = sm.add_constant(df['deviations'].values)
X_plot = sm.add_constant(df['x_plot'].values)
ols_model = sm.OLS(df['errors'].values, X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)
print(scipy.stats.chatterjeexi(tot_output_fr[idx_pos,0], pos_errors))
print(scipy.stats.chatterjeexi(tot_output_fr[idx_pos,1], pos_errors))
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(tot_output_fr[idx_pos,2], pos_errors, color='k', s=5)
print(scipy.stats.chatterjeexi(tot_output_fr[idx_pos,2], pos_errors))
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(tot_output_fr[idx_pos,3], pos_errors, color='r', s=5)
print(scipy.stats.chatterjeexi(tot_output_fr[idx_pos,3], pos_errors))
plt.tight_layout()


# do the part where we split the data into n bins for each side and compare left and right part of the subpart with ttest

bounds_pos = np.nanpercentile(pos_deviations, [0,10,20,30,40,50,60,70,80,90,100])
print(len(bounds_pos))
for bin in range(len(bounds_pos)-1):
    idx_local_bin = np.where((pos_deviations>bounds_pos[bin]) & (pos_deviations<bounds_pos[bin+1]))[0]
    local_deviations = pos_deviations[idx_local_bin]
    local_errors = pos_errors[idx_local_bin]
    median_dev = np.nanmedian(local_deviations)
    idx_small_local = np.where(local_deviations<median_dev)[0]
    idx_large_local = np.where(local_deviations>median_dev)[0]
    local_Test = scipy.stats.ranksums(local_deviations[idx_small_local], local_deviations[idx_large_local])
    print(local_Test, np.nanmedian(local_deviations[idx_small_local]), np.nanmedian(local_deviations[idx_large_local]))


plt.show()


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_deviations, pos_errors, color='k', s=5)
axs.set_xlim([0, 250]), axs.set_ylim([-100,150])
pred = est.get_prediction(X_plot).summary_frame()
axs.plot(np.sort(pos_deviations), pred['mean_ci_upper'],'b--')
axs.plot(np.sort(pos_deviations), pred['mean_ci_lower'],'b--')
axs.plot(x_plot, x_plot*tot_reg_lin.slope+tot_reg_lin.intercept,color='b',lw=2)
axs.fill_between(np.sort(pos_deviations), pred['mean_ci_upper'], pred['mean_ci_lower'],color='b',alpha=0.5)
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path, 'blackdots_lateral_humans.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path, 'blackdots_lateral_humans.svg'),bbox_inches='tight')
plt.show()


#########################
### Forward direction ###
#########################

idx_pos = np.where(tot_output_fr[:,0]>0)[0]

pos_deviations = tot_output_fr[idx_pos,0]
pos_errors = error_reduction_avg[idx_pos]
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
fig.savefig(os.path.join(rdd_path,'humans_slopes_diff_fwd.png'),bbox_inches='tight')



# data = pd.DataFrame({'y':pos_errors, 'x':pos_deviations})
# bandwidth_opt =36# rdd.optimal_bandwidth(data['y'], data['x'], cut=np.nanmedian(pos_deviations))
# print(bandwidth_opt)
# data_rdd = rdd.truncated_data(data, 'x', bandwidth_opt, cut=np.nanmedian(pos_deviations))
# model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(pos_deviations))
# print(model.fit().summary())


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Slope below and above median - Humans', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(slope_below)], side='low', showextrema=False, showmeans=True, widths=0.5, points=60)
axs.violinplot([np.array(slope_above)], side='high', showextrema=False, showmeans=True, widths=0.5, points=60)
axs.set_ylabel('Slope')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path,'humans_slopes_fwd.png'),bbox_inches='tight')


df = pd.DataFrame({'deviations':pd.Series(pos_deviations), 'errors':pd.Series(pos_errors), 'x_plot':pd.Series(np.sort(pos_deviations))})
X = sm.add_constant(df['deviations'].values)
X_plot = sm.add_constant(df['x_plot'].values)
ols_model = sm.OLS(df['errors'].values, X)
est = ols_model.fit()
out = est.conf_int(alpha=0.05, cols=None)


tot_reg_lin = scipy.stats.linregress(pos_deviations, pos_errors)

print(tot_reg_lin)

x_plot = np.linspace(0,np.max(pos_deviations))
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_deviations, pos_errors, color='k', s=5)
axs.set_xlim([0, 150]), axs.set_ylim([-150,150])
pred = est.get_prediction(X_plot).summary_frame()
axs.plot(np.sort(pos_deviations), pred['mean_ci_upper'],'b--')
axs.plot(np.sort(pos_deviations), pred['mean_ci_lower'],'b--')
axs.plot(x_plot, x_plot*tot_reg_lin.slope+tot_reg_lin.intercept, 'b')
axs.fill_between(np.sort(pos_deviations), pred['mean_ci_upper'], pred['mean_ci_lower'],color='b',alpha=0.5)
axs.set_xlabel('Foot placement'), axs.set_ylabel('Error reduction')
plt.tight_layout()
fig.savefig(os.path.join(rdd_path, 'blackdots_foreaft_humans.png'),bbox_inches='tight')
fig.savefig(os.path.join(rdd_path, 'blackdots_foreaft_humans.svg'),bbox_inches='tight')
plt.show()




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
fig.savefig(os.path.join(rdd_path,'humans_points_fwd.png'),bbox_inches='tight')
plt.show()
sys.exit()






plt.close('all')
def sigmoid(x, A, B, C, D):
    y = A / (1 + np.exp(-C*(x-B))) + D
    return y
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# fit a sigmoid to the first percentiles
list_fitted_params, list_rsquare = [], []
list_rsquare_line = []
for ii in range(len(bounds_pos)-1):
    x_fit = np.linspace(bounds_pos[0],bounds_pos[ii+1])
    idx_local = np.where((pos_deviations>0) & (pos_deviations<bounds_pos[ii+1]))[0]
    x_local = pos_deviations[idx_local]
    y_local = pos_errors[idx_local]
    tmp_reg = scipy.stats.linregress(x_local,y_local)
    list_rsquare_line.append(tmp_reg.rvalue**2)

    p0 = [max(y_local), np.nanmedian(x_local), 1, min(y_local)]
    popt, _ = curve_fit(sigmoid, x_local, y_local, p0, method='lm', maxfev=1000000)
    list_fitted_params.append(popt)
    y_pred = sigmoid(x_local, *popt)
    r_square_local = r2_score(y_pred, y_local)
    print(r_square_local, tmp_reg.rvalue**2)
    list_rsquare.append(r_square_local)

# print(list_rsquare)
# print(list_rsquare_line)
mat_params = np.array(list_fitted_params)
# sys.exit()
#print(mat_params.shape)
# print(mat_params[:,2])
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.arange(len(list_rsquare)), list_rsquare)
axs.set_ylim([0,1])
plt.tight_layout()



fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Foreaft direction',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.scatter(tot_output_fr[:,0], error_reduction_avg_fwd, color='k',s=5,alpha=0.5)
axs.set_xlabel('Contact deviation'), axs.set_ylabel('Error reduction')
plt.tight_layout()
import matplotlib.cm as cm
color_sigmo = cm.plasma(np.linspace(0,1,len(list_fitted_params)))
fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Lateral direction',fontweight='bold')
for ii in range(len(bounds_pos)-1):
    axs.axvline(bounds_pos[ii+1], color='r',lw=1)
    axs.plot(x_fit, sigmoid(x_fit, *list_fitted_params[ii]), color=color_sigmo[ii], lw=1)
axs.spines[['top','right']].set_visible(False)
axs.scatter(x_local,y_local,color='b')
axs.scatter(tot_output_fr[:,1], error_reduction_avg, color='k',s=5,alpha=0.5)
axs.set_xlabel('Contact deviation'), axs.set_ylabel('Error reduction')
plt.tight_layout()


# All the models predict something around 27.5 ... let's do the analyses there :) 
window_width = 17.5
idx_local_low = np.where((tot_output_fr[:,1]>27.5-window_width) & (tot_output_fr[:,1]<27.5))[0]
idx_local_high = np.where((tot_output_fr[:,1]>27.5) & (tot_output_fr[:,1]<27.5+window_width))[0]
idx_local_all = np.where((tot_output_fr[:,1]>27.5-window_width) & (tot_output_fr[:,1]<27.5+window_width))[0]
from tqdm import tqdm

n_boot = 10000
list_slope_left, list_slope_right, list_intercept_left, list_intercept_right = [], [], [], []
for _ in tqdm(range(n_boot)):
    idx_local_low_ = np.random.choice(idx_local_low, len(idx_local_low), replace=True)
    idx_local_high_ = np.random.choice(idx_local_high, len(idx_local_high), replace=True)
    reg_local_low = scipy.stats.linregress(tot_output_fr[idx_local_low_,1], error_reduction_avg[idx_local_low_])
    reg_local_high = scipy.stats.linregress(tot_output_fr[idx_local_high_,1], error_reduction_avg[idx_local_high_])
    reg_total = scipy.stats.linregress(tot_output_fr[idx_local_all,1], error_reduction_avg[idx_local_all])
    list_slope_left.append(reg_local_low.slope), list_intercept_left.append(reg_local_low.intercept), list_slope_right.append(reg_local_high.slope), list_intercept_right.append(reg_local_high.intercept)
    x_p1 = np.linspace(27.5-window_width, 27.5)
    x_p2 = np.linspace(27.5, 27.5+window_width)
    x_p = np.linspace(27.5-window_width, 27.5+window_width)

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Slope',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(list_slope_left)] , side='low',  points=60, showextrema=False, showmeans=True, widths=0.5)
axs.violinplot([np.array(list_slope_right)], side='high', points=60, showextrema=False, showmeans=True, widths=0.5)
plt.tight_layout()


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Intercept',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(list_intercept_left)] , side='low',  points=60, showextrema=False, showmeans=True, widths=0.5)
axs.violinplot([np.array(list_intercept_right)], side='high', points=60, showextrema=False, showmeans=True, widths=0.5)
plt.tight_layout()
plt.show()


# fig, axs = plt.subplots(1,1,figsize=(3,3))
# axs.spines[['top','right']].set_visible(False)
# axs.scatter(tot_output_fr[idx_local_low,1], error_reduction_avg[idx_local_low], color='k', s=5, alpha=0.5)
# axs.scatter(tot_output_fr[idx_local_high,1], error_reduction_avg[idx_local_high], color='r', s=5, alpha=0.5)
# axs.plot(x_p1, x_p1*reg_local_low.slope+reg_local_low.intercept, color='k',lw=2)
# axs.plot(x_p2, x_p2*reg_local_high.slope+reg_local_high.intercept, color='r',lw=2)
# axs.plot(x_p, x_p*reg_total.slope+reg_total.intercept, color='b',lw=2)
# axs.set_xlabel('Foot placement deviation'), axs.set_ylabel('Error reduction')
# plt.tight_layout()
# plt.show()

# print('Local regression low', reg_local_low)
# print('Local regression high', reg_local_high)

# print(len(idx_local_low), len(idx_local_high))

plt.close('all')

idx_pos = np.where(tot_output_fr[:,1]>0)[0]
plt.show()