import os, sys
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
import scipy.stats


# Loading the data 
path_data = os.path.join(os.getcwd(),'Datasets','Humans','Yang','processed')
path_figure = os.path.join(os.getcwd(),'humans_results','figures')

list_file = []
for file in os.listdir(path_data):
    if file.endswith('.npy'):
        tmp_file = np.load(os.path.join(path_data,file))
        list_file.append(tmp_file)

# Get contat location for all these data 
n_subjects = len(list_file)
horizon_max = 5
corr_matrix = np.zeros((n_subjects,horizon_max,2))
slope_matrix = np.zeros((n_subjects,horizon_max,2))
for horizon in range(horizon_max):
    print(horizon)
    for subject in range(n_subjects):
        local_data = list_file[subject]
        idx_right = np.arange(1,local_data.shape[0],2)
        idx_left = np.arange(0,local_data.shape[0],2)
        data_left = local_data[idx_left,-1,3]
        data_right = local_data[idx_right,-1,6]
        left_reg = scipy.stats.linregress(data_left[:-(horizon+1)], data_left[horizon+1:])
        right_reg = scipy.stats.linregress(data_right[:-(horizon+1)], data_right[horizon+1:])
        corr_matrix[subject,horizon,0] = left_reg.rvalue
        corr_matrix[subject,horizon,1] = right_reg.rvalue
        slope_matrix[subject,horizon,0] = left_reg.slope
        slope_matrix[subject,horizon,1] = right_reg.slope

print('Regression right')
print(right_reg)

print('Regression left')
print(left_reg)


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.set_title('Slopes')
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.arange(horizon_max), np.nanmean(slope_matrix[:,:,0],0),color='r',label='Left foot')
axs.scatter(np.arange(horizon_max)+0.15, np.nanmean(slope_matrix[:,:,1],0),color='b',label='Right foot')
for horizon in range(horizon_max):
    axs.plot([horizon, horizon],[np.nanmean(slope_matrix[:,horizon,0],0)+np.nanstd(slope_matrix[:,horizon,0],0),np.nanmean(slope_matrix[:,horizon,0],0)-np.nanstd(slope_matrix[:,horizon,0],0)],color='r',lw=2)
    axs.plot([horizon+0.15, horizon+0.15],[np.nanmean(slope_matrix[:,horizon,1],0)+np.nanstd(slope_matrix[:,horizon,1],0),np.nanmean(slope_matrix[:,horizon,1],0)-np.nanstd(slope_matrix[:,horizon,1],0)],color='b',lw=2)

axs.set_ylim([-1,1])
axs.axhline(0,color='k',lw=2,ls=':')
plt.tight_layout()


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.set_title('Pearson correlation coefficient')
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.arange(horizon_max)+1, np.nanmean(corr_matrix[:,:,0],0),color='r',label='Left foot')
axs.scatter(np.arange(horizon_max)+1.15, np.nanmean(corr_matrix[:,:,1],0),color='b',label='Right foot')
for horizon in range(horizon_max):
    axs.plot([horizon+1, horizon+1],[np.nanmean(corr_matrix[:,horizon,0],0)+np.nanstd(corr_matrix[:,horizon,0],0),np.nanmean(corr_matrix[:,horizon,0],0)-np.nanstd(corr_matrix[:,horizon,0],0)],color='r',lw=2)
    axs.plot([horizon+1.15, horizon+1.15],[np.nanmean(corr_matrix[:,horizon,1],0)+np.nanstd(corr_matrix[:,horizon,1],0),np.nanmean(corr_matrix[:,horizon,1],0)-np.nanstd(corr_matrix[:,horizon,1],0)],color='b',lw=2)
axs.set_ylim([-1.05,1.05])
axs.axhline(0,color='k',lw=2,ls=':')
plt.tight_layout()

print(np.nanmean(corr_matrix[:,:,0]),0)

tot_matrix = np.concatenate((np.squeeze(corr_matrix[:,:,0]), np.squeeze(corr_matrix[:,:,1])),0)
# Fit an exponential curve through the data
import scipy.optimize
def func(x,a,b,c):
    return a*np.exp(-b*x)+c

x_data = np.arange(horizon_max)
y_data = np.nanmean(tot_matrix,0)

from tqdm import tqdm 
n_boot = 10_000
mat_decays = np.zeros((n_boot,1))
mat_param = np.zeros((n_boot,3))
for iter in tqdm(range(n_boot)):
    idx_local = np.random.choice(21,21,replace=True)
    y_data_local = np.nanmean(slope_matrix[idx_local,:,1],0)
    popt_local, _ = scipy.optimize.curve_fit(func, x_data, y_data_local, maxfev = 50000)
    mat_param[iter,:] = popt_local
    mat_decays[iter,0] = popt_local[1]


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot(mat_param[:,0],showextrema=False, showmeans=True,points=100)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot(mat_param[:,1],showextrema=False, showmeans=True,points=100)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot(mat_param[:,2],showextrema=False, showmeans=True,points=100)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_xlim([-0.5,2.5])
axs.set_ylim([-0.5,2.5])
tmp = 1/mat_decays
print('Parameters to put in papers')
print(np.nanmedian(tmp))
print(scipy.stats.iqr(tmp,0,nan_policy='omit'))
axs.scatter(1, np.nanmedian(tmp),color='k',s=20)
axs.plot([1,1], [np.nanmedian(tmp)+scipy.stats.iqr(tmp,nan_policy='omit')/2,np.nanmedian(tmp)-scipy.stats.iqr(tmp,nan_policy='omit')/2],color='k',lw=2)
#axs.plot([1,1], [np.nanpercentile(tmp,25),np.nanpercentile(tmp,75)],color='k',lw=2)
#axs.boxplot(tmp[~np.isnan(tmp)],showcaps=False,whis=(5,95),boxprops=dict(color='k'),whiskerprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,autorange=False,widths=0.25)
plt.tight_layout()
plt.show()


popt, pcov = scipy.optimize.curve_fit(func, x_data, y_data, maxfev=5000)
print(1/popt[1])
y_pred = func(x_data+1,popt[0],popt[1],popt[2])
print('Parameters')
print(popt)
print(pcov)



fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.arange(horizon_max)+1, np.nanmean(tot_matrix,0), color='k', s=10)
for horizon in range(horizon_max):
    axs.plot([horizon+1, horizon+1],[np.nanmean(tot_matrix[:,horizon],0) + np.nanstd(tot_matrix[:,horizon],0), np.nanmean(tot_matrix[:,horizon],0) - np.nanstd(tot_matrix[:,horizon],0)],color='k',lw=2)
axs.plot(x_data+1,y_pred,color='k',lw=1)
axs.set_ylim([-1.05,1.05])
axs.set_xlim([0,6])
axs.set_yticks([-1,-0.5,0,0.5,1])
axs.set_yticklabels(['-1','-0.5','0','0.5','1'])
axs.axhline(0, color='k', lw=2, ls=':')
plt.tight_layout()
fig.savefig(os.path.join(path_figure,'horizon_correlation_treadmill.svg'),bbox_inches='tight')
fig.savefig(os.path.join(path_figure,'horizon_correlation_treadmill.png'),bbox_inches='tight')

# Figure valeur initale

fig,axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(1, np.nanmean(tot_matrix[:,0]),color='k',s=20)
axs.plot([1,1], [np.nanmean(tot_matrix[:,0])+np.nanstd(tot_matrix[:,0]),np.nanmean(tot_matrix[:,0])-np.nanstd(tot_matrix[:,0])],color='k',lw=2)
xpos_vec = np.linspace(0.8,1.2,30)
for subject in range(int(tot_matrix.shape[0]/2)):
    axs.scatter(xpos_vec[subject],tot_matrix[subject,0],color='k',s=5, alpha=0.2)
axs.set_xlim([0,4])
axs.set_ylim([-1.05,1.05])
axs.set_yticks([-1,0,1])
plt.tight_layout()
fig.savefig(os.path.join(path_figure,'time0_correlation.svg'),bbox_inches='tight')
fig.savefig(os.path.join(path_figure,'time0_correlation.png'),bbox_inches='tight')

plt.show()
