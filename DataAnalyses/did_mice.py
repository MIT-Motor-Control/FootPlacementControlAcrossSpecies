import os, sys 
import pickle 
import scipy.stats 
from tqdm import tqdm
import numpy as np 
import warnings
import pandas as pd 
import statsmodels.api as sm
from rdd import rdd
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

input_deviations_0 = total_input_fr[:,17,:]
input_deviations_1 = total_input_fr[:,22,:]

idx_pos = np.where(input_deviations_0[:,5]>0)[0]
pos_deviations_0 = input_deviations_0[idx_pos,5]
pos_deviations_1 = input_deviations_1[idx_pos,5]
pos_fp_deviations = total_output_fr[idx_pos,1]

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(total_input_fr[:,15,5], total_output_fr[:,1], color='k',s=5)
axs.set_xlabel('Input deviation'), axs.set_ylabel('Output deviation')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_deviations_0, pos_fp_deviations, color='r',s=5)
axs.set_xlabel('Input deviation'), axs.set_ylabel('Output deviation')
plt.tight_layout()
idx_small_dev = np.where(pos_deviations_0<np.nanmedian(pos_deviations_0))[0]
norm_pos_fp_deviations = np.abs(pos_fp_deviations)
begin_end = np.nanpercentile(norm_pos_fp_deviations[idx_small_dev],[0,100])
idx_inter = np.where((norm_pos_fp_deviations[idx_small_dev]>begin_end[0]) & (norm_pos_fp_deviations[idx_small_dev]<begin_end[1]))[0]
bounds_vec = np.sort(norm_pos_fp_deviations[idx_small_dev[idx_inter]])
list_pvalue, list_stats = [], []
list_pvaluer, list_statsr = [], []
for b in bounds_vec:
    idx_smaller = np.where(norm_pos_fp_deviations[idx_small_dev]<b)[0]
    local_test = scipy.stats.ttest_1samp(norm_pos_fp_deviations[idx_small_dev[idx_smaller]],popmean=0)
    local_test_r = scipy.stats.wilcoxon(norm_pos_fp_deviations[idx_small_dev[idx_smaller]])
    list_pvalue.append(local_test.pvalue)
    list_pvaluer.append(local_test_r.pvalue)
    list_stats.append(local_test.statistic)
    list_statsr.append(local_test_r.statistic)

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.axhline(np.log10(0.05),color='r',lw=2)
axs.plot(bounds_vec, np.log10(np.array(list_pvalue)),'k', lw=2,label='ttest')
axs.plot(bounds_vec, np.log10(np.array(list_pvaluer)),'b', lw=2,label='wilcoxon')
axs.legend(frameon=False)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.axhline(2.25, color='r', lw=2)
axs.plot(bounds_vec, list_stats,'k', lw=2,label='ttest')
axs.plot(bounds_vec, list_statsr,'r', lw=2,label='wilcoxon')
axs.legend(frameon=False)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_xlabel('error t0')
axs.set_ylabel('fp error')
axs.scatter(pos_deviations_0[idx_small_dev], pos_fp_deviations[idx_small_dev],color='k', s=5)
plt.tight_layout()
print('=============================')
print(scipy.stats.ttest_1samp(pos_fp_deviations[idx_small_dev],popmean=0))



# Get the median 
idx_large_fp = np.where(pos_fp_deviations[idx_small_dev]>0)[0]#np.nanmedian(pos_fp_deviations[idx_small_dev]))[0]
idx_small_fp = np.where(pos_fp_deviations[idx_small_dev]<0)[0]#np.nanmedian(pos_fp_deviations[idx_small_dev]))[0]
print(len(idx_large_fp), len(idx_small_fp), np.nanmedian(pos_fp_deviations[idx_small_dev]))

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_xlabel('error t0')
axs.set_ylabel('fp error')
axs.scatter(pos_deviations_0[idx_small_dev][idx_large_fp], pos_fp_deviations[idx_small_dev][idx_large_fp], color='r',s=5)
axs.scatter(pos_deviations_0[idx_small_dev][idx_small_fp], pos_fp_deviations[idx_small_dev][idx_small_fp], color='b',s=5)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_xlabel('fp error')
axs.set_ylabel('error t1')
axs.scatter(pos_fp_deviations[idx_small_dev][idx_large_fp], pos_deviations_1[idx_small_dev][idx_large_fp], color='r',s=5)
axs.scatter(pos_fp_deviations[idx_small_dev][idx_small_fp], pos_deviations_1[idx_small_dev][idx_small_fp], color='b',s=5)
plt.tight_layout()

# Figure for error reduction 
data_1 = pos_deviations_0[idx_small_dev][idx_small_fp] - pos_deviations_1[idx_small_dev][idx_small_fp]
data_2 = pos_deviations_0[idx_small_dev][idx_large_fp] - pos_deviations_1[idx_small_dev][idx_large_fp]
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.boxplot(data_1, positions=[0])
axs.boxplot(data_2, positions=[1])
axs.set_xlabel('jitter'), axs.set_ylabel('error reduction')
plt.tight_layout()

output_figures = os.path.join(os.getcwd(), 'FiguresCausality')
os.makedirs(output_figures, exist_ok=True)
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right','bottom']].set_visible(False)
vp1 = axs.violinplot([data_1], side='low',showextrema=False, showmeans=True, widths=0.5, points=60)
vp2 = axs.violinplot([data_2], side='high',showextrema=False, showmeans=True, widths=0.5, points=60)
print(scipy.stats.ttest_ind(data_1, data_2))
print(scipy.stats.ttest_ind(data_1, data_2,alternative='less'))
print(scipy.stats.ttest_ind(data_1, data_2,alternative='greater'))
axs.set_ylabel('error reduction')
print(scipy.stats.ttest_1samp(data_1, popmean=0))
print(scipy.stats.ttest_1samp(data_2, popmean=0))
print(scipy.stats.ttest_1samp(data_1, alternative='greater', popmean=0))
print(scipy.stats.ttest_1samp(data_1, alternative='less', popmean=0))
print('=====================')
print(scipy.stats.ttest_1samp(data_2, alternative='greater', popmean=0))
print(scipy.stats.ttest_1samp(data_2, alternative='less', popmean=0))
axs.set_xticks([])
axs.set_ylabel('error reduction')
for i, body in enumerate(vp1['bodies']):
    body.set_facecolor('red')
vp1['cmeans'].set_color('red')
for i, body in enumerate(vp2['bodies']):
    body.set_facecolor('blue')
vp2['cmeans'].set_color('blue')
labels=['No treatment','Treatment']
handles = [vp1['cmeans'], vp2['cmeans']]
axs.legend(handles, labels, frameon=False)
axs.axhline(0, color='k', ls=':',lw=2)
plt.tight_layout()
fig.savefig(os.path.join(output_figures, 'DiDMice.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figures, 'DiDMice.svg'),bbox_inches='tight')
plt.show()