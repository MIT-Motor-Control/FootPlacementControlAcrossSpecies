import os, sys
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
import scipy.optimize
from tqdm import tqdm 
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


def func(x,a,b,c):
    return a*np.exp(-b*x)+c

x_data = np.arange(horizon_max)
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
axs.set_xlim([-0.5,2.5])
axs.set_ylim([-0.5,2.5])
tmp = 1/mat_decays
print('Parameters to put in papers')
print(np.nanmedian(tmp))
print(scipy.stats.iqr(tmp,0,nan_policy='omit'))
axs.scatter(1, np.nanmedian(tmp),color='k',s=20)
axs.plot([1,1], [np.nanmedian(tmp)+scipy.stats.iqr(tmp,nan_policy='omit')/2,np.nanmedian(tmp)-scipy.stats.iqr(tmp,nan_policy='omit')/2],color='k',lw=2)
plt.tight_layout()

plt.show()
