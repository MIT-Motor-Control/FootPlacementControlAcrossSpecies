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
from utils.utils_mice import *
from utils.plot_functions_mice import *
warnings.filterwarnings('ignore') 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def calculate_aic(y_true, y_predicted, k):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_predicted)
    aic = n * np.log(mse) + 2 * k
    return aic 

def log_likelihood(residuals, n, k):
    ssr = np.nansum(residuals**2)
    sigma2 = ssr / (n-k)
    return -n/2*np.log(2*np.pi)-n/2*np.log(sigma2)-ssr/(2*sigma2)

def piecewise_linear(x, a1, b1, a2, b2, x0):
    return np.where(x<x0, a1*x+b1, a2*x+b2)

def residuals(params, x, y):
    a1, b1, a2, b2, x0 = params
    return y - piecewise_linear(x, a1, b1, a2, b2, x0)


# Loading the data
body_io, self_io, timing_io = load_all_data()

tmp = timing_io[0][0]
idx_nnans = np.where(~np.isnan(tmp[:,0,0]))[0]
tot_input_list, tot_input_self, tot_output_list, tot_output_self, tot_animal = concatenate_data(body_io[0], body_io[1], self_io[0], self_io[1])
tot_input_list_od, tot_input_self_od, tot_output_list_od, tot_output_self_od, tot_animal_od = concatenate_data(body_io[2], body_io[3], self_io[2], self_io[3])


print(tot_input_list.shape, tot_input_self.shape)
n_animal = 80
list_kink, aic_lin, aic_bilin, bic_lin, bic_bilin = [], [], [], [], []
list_kink_b, aic_lin_b, aic_bilin_b, bic_lin_b, bic_bilin_b = [], [], [], [], []
list_slope_1, list_slope_2 = [], []
for animal in tqdm(range(n_animal)):
    idx_local_an = np.where(tot_animal==animal)[0]
    local_input_data = tot_input_list[idx_local_an,:,:]
    local_output_data = tot_output_list[idx_local_an,:]
    local_input = local_input_data[:,17,5] - np.nanmean(local_input_data[:,17,5],0)
    local_output = local_output_data[:,1] - np.nanmean(local_output_data[:,1],0)
    
    idx_pos = np.where(local_input>0)[0]
    pos_errors = local_input[idx_pos]
    pos_deviations = local_output[idx_pos]
    linear_model = LinearRegression()
    linear_model.fit(pos_errors.reshape(-1,1), pos_deviations)
    begin_end = np.nanpercentile(pos_errors,[5,95])
    idx_inter = np.where((pos_errors>begin_end[0]) & (pos_errors<begin_end[1]))[0]
    kink_values = np.sort(pos_errors[idx_inter])
    list_residuals = []
    for k_value in kink_values:
        initial_params = [7,-8,2,-4,k_value]
        result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors, pos_deviations)**2), initial_params, method = 'L-BFGS-B')
        list_residuals.append(np.sum(residuals(result_bilinear.x, pos_errors, pos_deviations)**2))
    idx_argmin = np.argmin(np.array(list_residuals))
    list_kink.append(kink_values[idx_argmin])
    initial_params = [7,-8,2,-4,kink_values[idx_argmin]]
    idx_before = np.where(pos_errors<kink_values[idx_argmin])[0]
    idx_after = np.where(pos_errors>kink_values[idx_argmin])[0]
    n_min = min(len(idx_before), len(idx_after))
    if len(idx_before)<=len(idx_after):
        idx_new_before = idx_before
        sort_after = np.argsort(pos_errors[idx_after])
        idx_new_after = idx_after[sort_after[:n_min]]
    else:
        idx_new_after = idx_after
        sort_before = np.argsort(pos_errors[idx_before])[::-1]
        idx_new_before = idx_before[sort_before[:n_min]]
    idx_tot = np.hstack((idx_new_before, idx_new_after))
    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.scatter(pos_errors[idx_tot], pos_deviations[idx_tot],s=10,color='k',alpha=0.5)
    # axs.scatter(pos_errors, pos_deviations, color='r',s=5,alpha=0.5)
    # plt.tight_layout()
    model_b = LinearRegression()
    model_b.fit(pos_errors[idx_tot].reshape(-1,1), pos_deviations[idx_tot])
    result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors, pos_deviations)**2), initial_params, method='L-BFGS-B')
    result_bilinear_b = minimize(lambda p: np.sum(residuals(p, pos_errors[idx_tot], pos_deviations[idx_tot])**2), initial_params, method='L-BFGS-B')
    list_slope_1.append(result_bilinear_b.x[0]), list_slope_2.append(result_bilinear_b.x[2])
    y_pred = linear_model.predict(pos_errors.reshape(-1,1))
    y_pred_b = model_b.predict(pos_errors[idx_tot].reshape(-1,1))
    ll_full = log_likelihood(pos_deviations-y_pred, len(pos_deviations),2)
    ll_full_b = log_likelihood(pos_deviations[idx_tot]-y_pred_b, len(pos_deviations[idx_tot]),2)
    aic_lin.append(4-2*ll_full)
    bic_lin.append(-2*ll_full + 2*np.log(len(pos_deviations)))
    aic_lin_b.append(4-2*ll_full_b)
    bic_lin_b.append(-2*ll_full_b + 2*np.log(len(pos_deviations[idx_tot])))

    y_pred_bilin = piecewise_linear(pos_errors, *result_bilinear.x)
    y_pred_bilin_b = piecewise_linear(pos_errors[idx_tot], *result_bilinear_b.x)
    ll_full_bilin = log_likelihood(pos_deviations-y_pred_bilin, len(pos_deviations), 5)
    ll_full_bilin_b = log_likelihood(pos_deviations[idx_tot]-y_pred_bilin_b, len(pos_deviations[idx_tot]), 5)
    aic_bilin.append(10-2*ll_full_bilin)
    bic_bilin.append(-2*ll_full_bilin + 5*np.log(len(pos_deviations)))
    aic_bilin_b.append(10-2*ll_full_bilin_b)
    bic_bilin_b.append(-2*ll_full_bilin_b + 5*np.log(len(pos_deviations[idx_tot])))

print(scipy.stats.ttest_rel(list_slope_1, list_slope_2))

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right','bottom']].set_visible(False)
axs.scatter(np.zeros((len(list_slope_1,))), list_slope_1, color='r', s=5, alpha=0.5,label='Left slope')
axs.scatter(np.zeros((len(list_slope_2,))), list_slope_2, color='b', s=5, alpha=0.5,label='Right slope')
for sub in range(n_animal):
    axs.plot([0,1],[list_slope_1[sub], list_slope_2[sub]],color='k', lw=1)
axs.set_ylabel('Slopes')
axs.legend(frameon=False)
plt.tight_layout()

dif_aic = np.array(aic_lin) - np.array(aic_bilin)
dif_bic = np.array(bic_lin) - np.array(bic_bilin)
idx_sort = np.argsort(dif_aic)
sort_diff = np.sort(dif_aic)

dif_aic_b = np.array(aic_lin_b) - np.array(aic_bilin_b)
dif_bic_b = np.array(bic_lin_b) - np.array(bic_bilin_b)
idx_sort_b = np.argsort(dif_aic_b)
sort_diff_b = np.sort(dif_aic_b)

fig, axs = plt.subplots(1,2,figsize=(6,3), sharex=True, sharey=True)
fig.suptitle('Models comparison - Mice')
axs[0].set_title('AIC comparion')
axs[1].set_title('BIC comparison')
axs[0].spines[['top','right']].set_visible(False)
axs[0].axhline(0, color='k', lw=2)
axs[1].spines[['top','right']].set_visible(False)
axs[1].axhline(0, color='k', lw=2)
for bin in range(n_animal):
    if sort_diff[bin]<0:
        axs[0].bar(bin, sort_diff[bin], width=0.8, facecolor='r')
    else:
        axs[0].bar(bin, sort_diff[bin], width=0.8, facecolor='b')
    if dif_bic[idx_sort[bin]]<0:
        axs[1].bar(bin+0.1, dif_bic[idx_sort[bin]], width=0.8, facecolor='r')
    else:
        axs[1].bar(bin+0.1, dif_bic[idx_sort[bin]], width=0.8, facecolor='b')
# axs.bar(np.arange(21), sort_diff, width=0.8)
axs[0].set_xlabel('Subjects'), axs[1].set_xlabel('Subjects')
axs[0].set_ylabel(r"$\Delta$ AIC"), axs[1].set_ylabel(r"$\Delta$ BIC") 
plt.tight_layout()


fig, axs = plt.subplots(1,2,figsize=(6,3), sharex=True, sharey=True)
fig.suptitle('Local models comparison - Mice')
axs[0].set_title('AIC comparion')
axs[1].set_title('BIC comparison')
axs[0].spines[['top','right']].set_visible(False)
axs[0].axhline(0, color='k', lw=2)
axs[1].spines[['top','right']].set_visible(False)
axs[1].axhline(0, color='k', lw=2)
for bin in range(n_animal):
    if sort_diff_b[bin]<0:
        axs[0].bar(bin, sort_diff_b[bin], width=0.8, facecolor='r')
    else:
        axs[0].bar(bin, sort_diff_b[bin], width=0.8, facecolor='b')
    if dif_bic_b[idx_sort_b[bin]]<0:
        axs[1].bar(bin, dif_bic_b[idx_sort_b[bin]], width=0.8, facecolor='r')
    else:
        axs[1].bar(bin, dif_bic_b[idx_sort_b[bin]], width=0.8, facecolor='b')
# axs.bar(np.arange(21), sort_diff, width=0.8)
axs[0].set_xlabel('Subjects'), axs[1].set_xlabel('Subjects')
axs[0].set_ylabel(r"$\Delta$ AIC"), axs[1].set_ylabel(r"$\Delta$ BIC") 
plt.tight_layout()
plt.show()

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



idx_pos = np.where(total_input_fr[:,15,5]>0)[0]
pos_deviations = total_output_fr[idx_pos,1]
pos_errors = total_input_fr[idx_pos,15,5]

pval_list = []
bounds_percentile = np.nanpercentile(pos_errors,[0,10,20,30,40,50,60,70,80,90,100])
for ii in range(len(bounds_percentile)-1):
    print(f'Results for the bin number {ii}')
    idx_local = np.where((pos_errors>bounds_percentile[ii]) & (pos_errors<bounds_percentile[ii+1]))[0]
    local_med = np.nanmedian(pos_errors[idx_local])
    idx_local_small = np.where(pos_errors[idx_local]<local_med)[0]
    idx_local_large = np.where(pos_errors[idx_local]>local_med)[0]
    local_small = pos_deviations[idx_local][idx_local_small]
    local_large = pos_deviations[idx_local][idx_local_large]
    idx_nans = np.where((~np.isnan(local_small)) & (~np.isnan(local_large)))[0]
    local_test = scipy.stats.ttest_ind(local_small[idx_nans], local_large[idx_nans],alternative='less')
    print(local_test)
    pval_list.append(local_test.pvalue)
    data_rdd = pd.DataFrame({'y':pos_deviations[idx_local],'x':pos_errors[idx_local]})
    data_binned = rdd.bin_data(data_rdd,'y','x',100)
    if local_test.pvalue < 0.05:
        model = rdd.rdd(data_rdd, 'x','y',cut=np.nanmedian(pos_errors[idx_local]))
        print(model.fit().summary())
    # model_bis = rdd.rdd(data_binned, 'x','y',cut=np.nanmedian(pos_deviations))
    # print(model_bis.fit().summary())
print('========================================================')

data_tot = pd.DataFrame({'y':pos_deviations,'x':pos_errors})
model_tot = rdd.rdd(data_tot,'x','y',cut=np.nanmedian(pos_errors))
print(model_tot.fit().summary())
results_tot = model_tot.fit()
print(results_tot.summary())
print(results_tot.params.keys())


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_errors, pos_deviations,color='k',s=5,alpha=0.5)
axs.axvline(bounds_percentile[8],color='r',lw=2)
axs.axvline(bounds_percentile[9],color='r',lw=2)
plt.tight_layout()


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('pvalue mice')
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.arange(len(bounds_percentile)-1), np.log10(np.array(pval_list)), color='k', s=5, alpha=0.5)
axs.axhline(np.log10(0.05), color='r',lw=2)
axs.set_xlabel('Decile of body erros'), axs.set_ylabel('log10(p)')
plt.tight_layout()
plt.show()