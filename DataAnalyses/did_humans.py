import os, sys 
import pickle 
import scipy.stats
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import pandas as pd 
from rdd import rdd

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
n_sub = 21
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

matrix_inputs_averaged = np.nanmean(tot_input_fr[:,50:101,:],1)
matrix_outputs_averaged = np.nanmean(tot_input_fr[:,101:152,:],1)

input_deviations_0 = tot_input_fr[:,75,:]
input_deviations_1 = tot_input_fr[:,125,:]

idx_pos = np.where(input_deviations_0[:,1]>0)[0]

pos_deviations_0 = input_deviations_0[idx_pos,1]
pos_deviations_1 = input_deviations_1[idx_pos,1]
pos_fp_deviations = tot_output_fr[idx_pos,1]
pos_subject = tot_subject1[idx_pos]

# Application of the difference in difference approaches
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(tot_input_fr[:,75,1], tot_output_fr[:,1], color='k',s=5)
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
vp1 = axs.violinplot([data_1], side='low',showextrema=False, showmeans=True)
vp2 = axs.violinplot([data_2], side='high',showextrema=False, showmeans=True)
print(scipy.stats.ttest_ind(data_1, data_2))
print(scipy.stats.ttest_ind(data_1, data_2,alternative='less'))
print(scipy.stats.ttest_ind(data_1, data_2,alternative='greater'))
axs.set_ylabel('error reduction')
plt.tight_layout()
print(scipy.stats.ttest_1samp(data_1, popmean=0))
print(scipy.stats.ttest_1samp(data_2, popmean=0))
print(scipy.stats.ttest_1samp(data_1, alternative='greater', popmean=0))
print(scipy.stats.ttest_1samp(data_1, alternative='less', popmean=0))
print('=======================')
print(scipy.stats.ttest_1samp(data_2, alternative='greater',popmean=0))
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
fig.savefig(os.path.join(output_figures, 'DiDHumans.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figures, 'DiDHumans.svg'),bbox_inches='tight')
plt.show()