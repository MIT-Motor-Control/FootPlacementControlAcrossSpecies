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
from utils.utils_humans import *
from utils.plot_functions_humans import *
from utils.test_plots_humans import *

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




print(tot_input_fr.shape, tot_output_fr.shape)

idx_pos = np.where(tot_input_fr[:,75,1]>0)[0]
pos_errors = tot_input_fr[idx_pos,75,1]
pos_deviations = tot_output_fr[idx_pos,1]

# plt.show()

bins_bounds = np.nanpercentile(pos_errors,[0,10,20,30,40,50,60,70,80,90,100])
list_pvalue = []
for ii in range(len(bins_bounds)-1):
    idx_local = np.where((pos_errors>bins_bounds[ii]) & (pos_errors<bins_bounds[ii+1]))[0]
    errors_local = pos_errors[idx_local]
    deviations_local = pos_deviations[idx_local]
    idx_larger =  np.where(errors_local>np.nanmedian(errors_local))[0]
    idx_smaller=  np.where(errors_local<np.nanmedian(errors_local))[0]
    local_test = scipy.stats.ttest_ind(deviations_local[idx_larger], deviations_local[idx_smaller],alternative='greater')
    print(local_test)
    list_pvalue.append(local_test.pvalue)
    if local_test.pvalue<0.05:
        print(ii, bins_bounds[ii], bins_bounds[ii+1])
        data_rdd = pd.DataFrame({'y':deviations_local, 'x':errors_local})
        local_model = rdd.rdd(data_rdd, 'x', 'y', cut=np.nanmedian(errors_local))
        print(local_model.fit().summary())
        print('=========================================')

data_tot = pd.DataFrame({'y':pos_deviations,'x':pos_errors})
model_tot = rdd.rdd(data_tot, 'x', 'y', cut=np.nanmedian(pos_errors))
print(model_tot.fit().summary())

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
idx_local = np.where((pos_errors>bins_bounds[6]) & (pos_errors<bins_bounds[7]))[0]
axs.scatter(pos_errors[idx_local], pos_deviations[idx_local], color='k', s=5, alpha=1)
axs.set_xlabel('Body state errors'), axs.set_ylabel('Foot placement errors')
axs.axvline(bins_bounds[6], color='r', lw=2)
axs.axvline(bins_bounds[7],color='r', lw=2)
axs.axvline(np.nanmedian(pos_errors[idx_local]), color='k', ls=':')
idx_left = np.where((pos_errors>bins_bounds[6]) & (pos_errors<np.nanmedian(pos_errors[idx_local])))[0]
idx_right = np.where((pos_errors>np.nanmedian(pos_errors[idx_local])) & (pos_errors<bins_bounds[7]))[0]
reg_left = scipy.stats.linregress(pos_errors[idx_left], pos_errors[idx_left])
reg_right = scipy.stats.linregress(pos_errors[idx_right], pos_errors[idx_right])
x_plot_left = np.linspace(bins_bounds[6],np.nanmedian(pos_errors[idx_local]))
x_plot_right = np.linspace(np.nanmedian(pos_errors[idx_local]), bins_bounds[7])
axs.plot(x_plot_left, x_plot_left*reg_left.slope+reg_left.intercept, 'k', lw=2)
axs.plot(x_plot_right, x_plot_right*reg_right.slope+reg_right.intercept, 'k', lw=2)
axs.plot(x_plot_left, np.nanmean(pos_deviations[idx_left])*np.ones((50,)),'r',lw=2)
axs.plot(x_plot_right, np.nanmean(pos_deviations[idx_right])*np.ones((50,)),'r',lw=2)
axs.set_xlim([10, 20])
plt.tight_layout()

## Investigation of the RKD models 


PATH_CAMARGO_DATASET = os.path.join(os.getcwd(), 'Datasets','Humans','Camargo','processed')
list_io1, list_io2, list_sub = load_data_camargo()
tot_input_leg1, tot_self_input_leg1, tot_output_leg1, tot_self_output_leg1 = list_io1[0], list_io1[1], list_io1[2], list_io1[3]
tot_input_leg2, tot_self_input_leg2, tot_output_leg2, tot_self_output_leg2 = list_io2[0], list_io2[1], list_io2[2], list_io2[3]
tot_sub1, tot_sub2 = list_sub[0], list_sub[1]
print(tot_input_leg1.shape, tot_output_leg1.shape)

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

# Investigate the RKD locally
list_kink, aic_lin, aic_bilin, bic_lin, bic_bilin = [], [], [], [], []
list_kink_b, aic_lin_b, aic_bilin_b, bic_lin_b, bic_bilin_b = [], [], [], [], []
list_slope_1, list_slope_2 = [], []
for sub in tqdm(range(21)):
    idx_local_sub = np.where((tot_sub1==sub))[0]
    local_input = tot_input_leg1[idx_local_sub,75,1] - np.nanmean(tot_input_leg1[idx_local_sub,75,1],0)
    local_output = tot_output_leg1[idx_local_sub,1] - np.nanmean(tot_output_leg1[idx_local_sub,1],0)
    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.scatter(local_input, local_output, color='r', s=5)
    # axs.axvline(0, color='k', lw=2, ls=':')
    
    idx_pos = np.where((local_input>0))[0]
    pos_errors = local_input[idx_pos]
    pos_deviations = local_output[idx_pos]
    linear_model = LinearRegression()
    linear_model.fit(pos_errors.reshape(-1,1), pos_deviations)
    # axs.scatter(pos_errors, pos_deviations, color='k', s=15, alpha=0.5)
    # plt.tight_layout()
    begin_end = np.nanpercentile(pos_errors,[5,95])
    idx_inter = np.where((pos_errors>begin_end[0]) & (pos_errors<begin_end[1]))[0]
    kink_values = np.sort(pos_errors[idx_inter])
    list_residuals = []
    for k_value in kink_values:
        initial_params = [7,-8,2,-4,k_value]
        result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors, pos_deviations)**2), initial_params, method='L-BFGS-B')
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
for sub in range(21):
    axs.plot([0,1],[list_slope_1[sub], list_slope_2[sub]],color='k', lw=1)
axs.set_ylabel('Slopes')
axs.legend(frameon=False)
plt.tight_layout()
plt.show()

dif_aic = np.array(aic_lin) - np.array(aic_bilin)
dif_bic = np.array(bic_lin) - np.array(bic_bilin)
idx_sort = np.argsort(dif_aic)
sort_diff = np.sort(dif_aic)

dif_aic_b = np.array(aic_lin_b) - np.array(aic_bilin_b)
dif_bic_b = np.array(bic_lin_b) - np.array(bic_bilin_b)
idx_sort_b = np.argsort(dif_aic_b)
sort_diff_b = np.sort(dif_aic_b)

fig, axs = plt.subplots(1,2,figsize=(6,3), sharex=True, sharey=True)
fig.suptitle('Models comparison - Humans')
axs[0].set_title('AIC comparion')
axs[1].set_title('BIC comparison')
axs[0].spines[['top','right']].set_visible(False)
axs[0].axhline(0, color='k', lw=2)
axs[1].spines[['top','right']].set_visible(False)
axs[1].axhline(0, color='k', lw=2)
for bin in range(21):
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
fig.suptitle('Local models comparison - Humans')
axs[0].set_title('AIC comparion')
axs[1].set_title('BIC comparison')
axs[0].spines[['top','right']].set_visible(False)
axs[0].axhline(0, color='k', lw=2)
axs[1].spines[['top','right']].set_visible(False)
axs[1].axhline(0, color='k', lw=2)
for bin in range(21):
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

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.set_xlim([-1,1])
axs.set_ylabel('Kink position')
axs.scatter(np.random.uniform(-0.1,0.1,21), list_kink, color='r',s=5)
axs.scatter(0, np.nanmedian(np.array(list_kink)),color='r',s=25)
axs.plot([0,0],[np.nanmedian(np.array(list_kink))+scipy.stats.iqr(np.array(list_kink))/2, np.nanmedian(np.array(list_kink))-scipy.stats.iqr(np.array(list_kink))/2],'r', lw=2)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.arange(21), np.array(aic_lin) - np.array(aic_bilin), color='k', s=5, alpha=0.5)
plt.tight_layout()
plt.show()



# RKD - Regression kink discontinuity
# 0. Loop over the entire x-vector (take the 80th central percentile)
model = LinearRegression()
model.fit(pos_errors.reshape(-1,1), pos_deviations)
y_predicted = model.predict(pos_errors.reshape(-1,1))

begin_end = np.nanpercentile(pos_errors,[5,95])
idx_inter = np.where((pos_errors>begin_end[0]) & (pos_errors<begin_end[1]))[0]
kink_values = np.sort(pos_errors[idx_inter])
list_residuals = []

x_0_list = []
n_boot = 10_000
for _ in tqdm(range(n_boot)):
    for k_value in tqdm(kink_values):
        initial_params = [7,-8,2,-4,k_value]
        result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors, pos_deviations)**2), initial_params, method='L-BFGS-B')
        list_residuals.append(np.sum(residuals(result_bilinear.x, pos_errors, pos_deviations)**2))
    idx_argmin = np.argmin(np.array(list_residuals))
    x_0_list.append(kink_values[idx_argmin])

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.violinplot([x_0_list], width=0.5, showmeans=True, showextremas=False, points=60)
sys.exit()

for k_value in kink_values:
    initial_params = [7,-8,2,-4,k_value]
    result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors, pos_deviations)**2), initial_params, method='L-BFGS-B')
    list_residuals.append(np.sum(residuals(result_bilinear.x, pos_errors, pos_deviations)**2))
idx_argmin = np.argmin(np.array(list_residuals))
initial_params = [7, -8,2,-4,kink_values[idx_argmin]]
result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors, pos_deviations)**2), initial_params, method='L-BFGS-B')

boot_aic_lin, boot_aic_bilin = [], [] 
n_boot = 10_000
for _ in tqdm(range(n_boot)):
    idx_rnd = np.random.choice(len(pos_errors), len(pos_errors), replace=True)
    result_bilinear = minimize(lambda p: np.sum(residuals(p, pos_errors[idx_rnd], pos_deviations[idx_rnd])**2), initial_params, method='L-BFGS-B')
    y_pred = model.predict(pos_errors[idx_rnd].reshape(-1,1))
    ll_full = log_likelihood(pos_deviations[idx_rnd]-y_pred, len(pos_deviations),2)
    boot_aic_lin.append(4-2*ll_full)

    y_pred_bilin = piecewise_linear(pos_errors[idx_rnd], *result_bilinear.x)
    ll_full_bilin = log_likelihood(pos_deviations[idx_rnd]-y_pred_bilin, len(pos_deviations), 5)
    boot_aic_bilin.append(10-2*ll_full_bilin)


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Individual bootstrapped distros',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([boot_aic_lin], widths=0.5, showextrema=False, showmeans=True, points=60, side='low')
axs.violinplot([boot_aic_bilin], widths=0.5, showextrema=False, showmeans=True, points=60, side='high')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Difference bootstrapped distros',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(boot_aic_lin)-np.array(boot_aic_bilin)], widths=0.5, showextrema=False, showmeans=True, points=60)
axs.axhline(0, color='k', lw=2)
axs.axhline(np.nanpercentile(np.array(boot_aic_lin)-np.array(boot_aic_bilin),5),color='k',ls=':')
axs.axhline(np.nanpercentile(np.array(boot_aic_lin)-np.array(boot_aic_bilin),95),color='k',ls=':')
plt.tight_layout()

slope_left, slope_right, inter_left, inter_right = [], [] ,[] ,[]
for _ in tqdm(range(n_boot)):
    idx_rnd = np.random.choice(len(pos_errors), len(pos_errors), replace=True)
    results_local = minimize(lambda p: np.sum(residuals(p, pos_errors[idx_rnd], pos_deviations[idx_rnd])**2), initial_params, method='L-BFGS-B')
    opt_local = results_local.x
    slope_left.append(opt_local[0]), slope_right.append(opt_local[2])
    inter_left.append(opt_local[1]), inter_right.append(opt_local[3])

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot([slope_left], widths=0.5, showextrema=False, showmeans=True, points=60, side='low')
axs.violinplot([slope_right], widths=0.5, showextrema=False, showmeans=True, points=60, side='high')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Difference in slope', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(slope_left)-np.array(slope_right)], widths=0.5, showextrema=False, showmeans=True, points=60)
axs.axhline(np.nanpercentile(np.array(slope_left)-np.array(slope_right),5), color='k', lw=2, ls=':')
axs.axhline(np.nanpercentile(np.array(slope_left)-np.array(slope_right),95), color='k', lw=2, ls=':')
axs.axhline(0, color='k', lw=2)
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.violinplot([inter_left], widths=0.5, showextrema=False, showmeans=True, points=60, side='low')
axs.violinplot([inter_right], widths=0.5, showextrema=False, showmeans=True, points=60, side='high')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Difference in intercept', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.violinplot([np.array(inter_left)-np.array(inter_right)], widths=0.5, showextrema=False, showmeans=True, points=60)
axs.axhline(np.nanpercentile(np.array(inter_left)-np.array(inter_right),5), color='k', lw=2, ls=':')
axs.axhline(np.nanpercentile(np.array(inter_left)-np.array(inter_right),95), color='k', lw=2, ls=':')
axs.axhline(0, color='k', lw=2)
plt.tight_layout()
plt.show()


k = pos_errors.reshape(-1,1).shape[1] + 1 
ll_full = log_likelihood(pos_deviations-y_predicted, len(pos_deviations),2)
aic = 2*2 - 2*ll_full

opt_prediction_bilinear = piecewise_linear(pos_errors, *result_bilinear.x)
ll_bilinear = log_likelihood(pos_deviations-opt_prediction_bilinear, len(pos_deviations),5)
aic_bi = 10 - 2 * ll_bilinear
opt_prediction_bilinear = piecewise_linear(np.sort(pos_errors), *result_bilinear.x)


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_errors, pos_deviations, color='k', s=5, alpha=0.5)
axs.plot(np.sort(pos_errors), opt_prediction_bilinear, color='b', lw=2)
axs.axvline(kink_values[idx_argmin], color='k', ls=':', lw=2)
axs.set_xlabel('Body state errors'), axs.set_ylabel('Foot placement errors')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('AIC comparison', fontweight='bold')
axs.spines[['top','right']].set_visible(False)
axs.axhline(aic_bi, color='m', lw=2, label='Optimal bilinear model')
axs.axhline(aic, color='r', lw=2, label='Simple linear model')
axs.set_xlabel('Location of kink'), axs.set_ylabel('AIC')
axs.legend(frameon=False)
plt.tight_layout()
plt.show()


list_bottom, list_top = [], [] 
for subject in range(n_sub):
    idx_sub = np.where(tot_subject1==subject)[0]
    local_input = tot_input_leg1[idx_sub, :,1] - np.nanmean(tot_input_leg1[idx_sub, :, 1],0)
    local_output = tot_output_leg1[idx_sub,1] - np.nanmean(tot_output_leg1[idx_sub,1],0)
    idx_pos = np.where(local_input[:,75]>0)[0]
    local_errors = local_input[idx_pos,75]
    local_foot = local_output[idx_pos]
    idx_low = np.where(local_errors<np.nanmedian(local_errors))[0]
    idx_high = np.where(local_errors>np.nanmedian(local_errors))[0]
    if len(idx_low)==0 or len(idx_high)==0:
        list_bottom.append(np.nan), list_top.append(np.nan)
    else:
        reg_high = scipy.stats.linregress(local_errors[idx_high], local_foot[idx_high])
        reg_low = scipy.stats.linregress(local_errors[idx_low], local_foot[idx_low])
        list_bottom.append(reg_low.slope), list_top.append(reg_high.slope)



idx_below = np.where(pos_errors<np.nanmedian(pos_errors))[0]
idx_above = np.where(pos_errors>np.nanmedian(pos_errors))[0]
reg_below = scipy.stats.linregress(pos_errors[idx_below], pos_deviations[idx_below])
reg_above = scipy.stats.linregress(pos_errors[idx_above], pos_deviations[idx_above])
df_below = pd.DataFrame({'errors':pd.Series(pos_errors[idx_below]),'deviations':pd.Series(pos_deviations[idx_below]),'x_plot':pd.Series(np.sort(pos_deviations[idx_below]))})
df_above = pd.DataFrame({'errors':pd.Series(pos_errors[idx_above]),'deviations':pd.Series(pos_deviations[idx_above]),'x_plot':pd.Series(np.sort(pos_deviations[idx_above]))})
X_below, X_above = sm.add_constant(df_below['deviations'].values), sm.add_constant(df_above['deviations'].values)
X_plot_below, X_plot_above = sm.add_constant(df_below['x_plot'].values), sm.add_constant(df_above['x_plot'].values)
ols_below = sm.OLS(df_below['deviations'].values, X_below)
ols_above = sm.OLS(df_above['deviations'].values, X_above)
est_below = ols_below.fit()
est_above = ols_above.fit()
out_below = est_below.conf_int(alpha=0.05, cols=None)
out_above = est_above.conf_int(alpha=0.05, cols=None)

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(pos_errors, pos_deviations, color='k', alpha=0.5, s=5)
pred_below = est_below.get_prediction(X_plot_below).summary_frame()
pred_above = est_above.get_prediction(X_plot_above).summary_frame()
axs.fill_between(np.sort(pos_errors[idx_below]), pred_below['mean_ci_upper'], pred_below['mean_ci_lower'],color='b',alpha=0.5)
axs.fill_between(np.sort(pos_errors[idx_above]), pred_above['mean_ci_upper'], pred_above['mean_ci_lower'],color='r',alpha=0.5)
axs.plot(pos_errors[idx_below], pos_errors[idx_below]*reg_below.slope+reg_below.intercept,color='b',lw=2)
axs.plot(pos_errors[idx_above], pos_errors[idx_above]*reg_above.slope+reg_above.intercept,color='r',lw=2)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.random.uniform(-0.1,0.1,len(list_bottom)), list_bottom, color='r', s=5)
axs.scatter(np.random.uniform(0.9,1.1,len(list_top)), list_top, color='b', s=5)
axs.scatter(0, np.nanmedian(np.array(list_bottom)),color='b',s=20)
axs.scatter(1, np.nanmedian(np.array(list_top)),color='r',s=20)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
fig.suptitle('pvalue humans')
axs.scatter(np.arange(10), np.log10(np.array(list_pvalue)), color='k', s=5)
axs.axhline(np.log10(0.05), color='r', lw=2)
axs.set_xlabel('Decile of body erros'), axs.set_ylabel('log10(p)')
plt.tight_layout()
plt.show()