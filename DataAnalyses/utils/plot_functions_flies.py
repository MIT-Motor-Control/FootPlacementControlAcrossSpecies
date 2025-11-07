import os, sys
import pickle
import scipy
import scipy.optimize
import copy
from tqdm import tqdm
import scikit_posthocs as spph
import plotly
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
OUTPUT_FIGURES = os.path.join(os.getcwd(),'fly_results','figures')
OUTPUT_STATS = os.path.join(os.getcwd(), 'fly_results','stats')


def plot_rsquares_final(list_body, list_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plotting the rsquares for the fly data
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[0],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[0],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[0],0)+scipy.stats.iqr(list_body[0],0,nan_policy='omit')/2, np.nanmedian(list_body[0],0)-scipy.stats.iqr(list_body[0],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[0],0)+scipy.stats.iqr(list_self[0],0,nan_policy='omit')/2, np.nanmedian(list_self[0],0)-scipy.stats.iqr(list_self[0],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_front_foreaft.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_front_foreaft.svg'),bbox_inches='tight')

    list_pvalue_11 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[0][:,time])) & (~np.isnan(list_self[0][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[0][idx_stats_nan,time], list_self[0][idx_stats_nan,time], alternative='greater')
        list_pvalue_11.append(stat_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[1],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[1],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[1],0)+scipy.stats.iqr(list_body[1],0,nan_policy='omit')/2, np.nanmedian(list_body[1],0)-scipy.stats.iqr(list_body[1],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[1],0)+scipy.stats.iqr(list_self[1],0,nan_policy='omit')/2, np.nanmedian(list_self[1],0)-scipy.stats.iqr(list_self[1],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_front_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_front_lateral.svg'),bbox_inches='tight')

    list_pvalue_12 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[1][:,time])) & (~np.isnan(list_self[1][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[1][idx_stats_nan,time], list_self[1][idx_stats_nan,time], alternative='greater')
        list_pvalue_12.append(stat_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[2],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[2],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[2],0)+scipy.stats.iqr(list_body[2],0,nan_policy='omit')/2, np.nanmedian(list_body[2],0)-scipy.stats.iqr(list_body[2],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[2],0)+scipy.stats.iqr(list_self[2],0,nan_policy='omit')/2, np.nanmedian(list_self[2],0)-scipy.stats.iqr(list_self[2],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_center_foreaft.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_center_foreaft.svg'),bbox_inches='tight')

    list_pvalue_21 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[2][:,time])) & (~np.isnan(list_self[2][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[2][idx_stats_nan,time], list_self[2][idx_stats_nan,time], alternative='greater')
        list_pvalue_21.append(stat_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[3],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[3],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[3],0)+scipy.stats.iqr(list_body[3],0,nan_policy='omit')/2, np.nanmedian(list_body[3],0)-scipy.stats.iqr(list_body[3],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[3],0)+scipy.stats.iqr(list_self[3],0,nan_policy='omit')/2, np.nanmedian(list_self[3],0)-scipy.stats.iqr(list_self[3],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_center_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_center_lateral.svg'),bbox_inches='tight')

    list_pvalue_22 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[3][:,time])) & (~np.isnan(list_self[3][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[3][idx_stats_nan,time], list_self[3][idx_stats_nan,time], alternative='greater')
        list_pvalue_22.append(stat_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[4],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[4],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[4],0)+scipy.stats.iqr(list_body[4],0,nan_policy='omit')/2, np.nanmedian(list_body[4],0)-scipy.stats.iqr(list_body[4],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[4],0)+scipy.stats.iqr(list_self[4],0,nan_policy='omit')/2, np.nanmedian(list_self[4],0)-scipy.stats.iqr(list_self[4],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_hind_foreaft.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_hind_foreaft.svg'),bbox_inches='tight')

    list_pvalue_31 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[4][:,time])) & (~np.isnan(list_self[4][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[4][idx_stats_nan,time], list_self[4][idx_stats_nan,time], alternative='greater')
        list_pvalue_31.append(stat_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[5],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[5],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[5],0)+scipy.stats.iqr(list_body[5],0,nan_policy='omit')/2, np.nanmedian(list_body[5],0)-scipy.stats.iqr(list_body[5],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[5],0)+scipy.stats.iqr(list_self[5],0,nan_policy='omit')/2, np.nanmedian(list_self[5],0)-scipy.stats.iqr(list_self[5],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_hind_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_hind_lateral.svg'),bbox_inches='tight')

    list_pvalue_32 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[5][:,time])) & (~np.isnan(list_self[5][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[5][idx_stats_nan,time], list_self[5][idx_stats_nan,time], alternative='greater')
        list_pvalue_32.append(stat_test.pvalue)
    if bool_plot:
        plt.show()

def plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the laterality information for the fly datasets
    """

    diff_gains = regression_small[:,:,1,1,0] - regression_large[:,:,1,1,0]

    # Plotting for each velocity bins
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(regression_small[:,:,1,1,2]**2,0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,:,1,1,2]**2,0),'b',lw=2)
    axs.set_ylim([-0.5,1.5])
    plt.tight_layout()



    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(diff_gains,0),color='k')
    axs.fill_between(np.arange(21), np.nanmean(diff_gains,0)+scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,np.nanmean(diff_gains,0)-scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,color='k', alpha=0.3)
    axs.plot(np.nanmean(regression_small[:,:,1,1,0],0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,:,1,1,0],0),'b',lw=2)
    axs.fill_between(np.arange(21), np.nanmean(regression_small[:,:,1,1,0],0)+scipy.stats.iqr(regression_small[:,:,1,1,0],0,nan_policy='omit')/2,np.nanmean(regression_small[:,:,1,1,0],0)-scipy.stats.iqr(regression_small[:,:,1,1,0],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmean(regression_large[:,:,1,1,0],0)+scipy.stats.iqr(regression_large[:,:,1,1,0],0,nan_policy='omit')/2,np.nanmean(regression_large[:,:,1,1,0],0)-scipy.stats.iqr(regression_large[:,:,1,1,0],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.set_ylim([-1.5,1.5])
    axs.axhline(0,color='k',lw=2,ls=':')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_gains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_gains.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(regression_small[:,:,1,1,2]**2,0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,:,1,1,2]**2,0),'b',lw=2)
    axs.fill_between(np.arange(21), np.nanmean(regression_small[:,:,1,1,2]**2,0)+scipy.stats.iqr(regression_small[:,:,1,1,2]**2,0,nan_policy='omit')/2,np.nanmean(regression_small[:,:,1,1,2]**2,0)-scipy.stats.iqr(regression_small[:,:,1,1,2]**2,0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmean(regression_large[:,:,1,1,2]**2,0)+scipy.stats.iqr(regression_large[:,:,1,1,2]**2,0,nan_policy='omit')/2,np.nanmean(regression_large[:,:,1,1,2]**2,0)-scipy.stats.iqr(regression_large[:,:,1,1,2]**2,0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_rsquares.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_rsquares.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(diff_gains,0),color='k')
    axs.fill_between(np.arange(21), np.nanmean(diff_gains,0)+scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,np.nanmean(diff_gains,0)-scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,color='k', alpha=0.5)
    axs.axhline(0, color='k', lw=2, ls=':')
    axs.set_ylim([-1,1])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_diffgains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_diffgains.svg'),bbox_inches='tight')

    # Plot the laterality at midstance 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(1, np.nanmean(regression_small[:,15,1,1,0]),color='b',s=20)    
    axs.scatter(2, np.nanmean(regression_large[:,15,1,1,0]),color='r',s=20)
    axs.plot([1,1], [np.nanmean(regression_small[:,15,1,1,0])+scipy.stats.iqr(regression_small[:,15,1,1,0],0,nan_policy='omit')/2,np.nanmean(regression_small[:,15,1,1,0])-scipy.stats.iqr(regression_small[:,15,1,1,0],0,nan_policy='omit')/2],color='b',lw=2)
    axs.plot([2,2], [np.nanmean(regression_large[:,15,1,1,0])+scipy.stats.iqr(regression_large[:,15,1,1,0],0,nan_policy='omit')/2,np.nanmean(regression_large[:,15,1,1,0])-scipy.stats.iqr(regression_large[:,15,1,1,0],0,nan_policy='omit')/2],color='r',lw=2)
    x_vec1 = np.linspace(0.8,1.2,8)
    x_vec2 = np.linspace(1.8,2.2,8)
    for video in range(8):
        axs.scatter(x_vec1[video], regression_small[video,15,1,1,0],color='b',alpha=0.2,s=5)
        axs.scatter(x_vec2[video], regression_large[video,15,1,1,0],color='r',alpha=0.2,s=5)
    axs.set_xlim([0,3])
    axs.set_ylim([-1,1])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_midstance_gains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_midstance_gains.svg'),bbox_inches='tight')
    test1 = scipy.stats.wilcoxon(regression_small[:,15,1,1,0], regression_large[:,15,1,1,0],nan_policy='omit')
    test_ttest = scipy.stats.ttest_ind(regression_small[:,15,1,1,0], regression_large[:,15,1,1,0],nan_policy='omit')
    mean_1, std_1 = np.nanmean(regression_small[:,15,1,1,0]), np.nanstd(regression_small[:,15,1,1,0]) 
    mean_2, std_2 = np.nanmean(regression_large[:,15,1,1,0]), np.nanstd(regression_large[:,15,1,1,0]) 
    std_tot = np.nanstd(np.concatenate((regression_large[:,15,1,1,0],regression_small[:,15,1,1,0]),0))
    print('Cohen d', (mean_1-mean_2)/(std_tot))

    if bool_plot:
        plt.show()


def plot_successive_contacts_fly(input_list, video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the correlations between successive contacts as a function of the lookup horizon for the fly data
    """

    n_videos = int(np.max(video_list))
    n_horizons = len(input_list)
    corr_matrix = np.zeros((n_videos, n_horizons,3))
    slope_matrix = np.zeros((n_videos, n_horizons,3))
    for vid in range(1,n_videos+1):
        for hor in range(n_horizons):
            idx_vid = np.where(video_list==vid)[0]
            local_input = input_list[hor][idx_vid,:,:]
            idx_good_1 = np.where((local_input[:,-1,4]>0) & (~np.isnan(local_input[:,-1,4])))[0]
            idx_good_2 = np.where((local_input[:,-1,0]>-32) & (~np.isnan(local_input[:,-1,0])))[0]
            idx_good_3 = np.where((local_input[:,-1,8]>-62) & (~np.isnan(local_input[:,-1,8])))[0]
            reg1 = scipy.stats.linregress(local_input[idx_good_1,0,5], local_input[idx_good_1,-1,5])
            reg2 = scipy.stats.linregress(local_input[idx_good_2,0,1], local_input[idx_good_2,-1,1])
            reg3 = scipy.stats.linregress(local_input[idx_good_3,0,9], local_input[idx_good_3,-1,9])
            corr_matrix[vid-1, hor, 0], corr_matrix[vid-1, hor, 1], corr_matrix[vid-1, hor, 2] = reg1.rvalue**2, reg2.rvalue**2, reg3.rvalue**2
            slope_matrix[vid-1,hor,0], slope_matrix[vid-1,hor,1], slope_matrix[vid-1,hor,2] = reg1.slope, reg2.slope, reg3.slope
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,6),np.nanmean(corr_matrix[:,:,0],0),color='k',s=10)
    axs.scatter(np.arange(1,6),np.nanmean(corr_matrix[:,:,1],0),color='k',s=10)
    axs.scatter(np.arange(1,6),np.nanmean(corr_matrix[:,:,2],0),color='k',s=10)
    for hor in range(n_horizons):
        axs.plot([hor+1,hor+1], [np.nanmean(corr_matrix[:,hor,0],0)+scipy.stats.iqr(corr_matrix[:,hor,0],0,nan_policy='omit'), np.nanmean(corr_matrix[:,hor,0],0)-scipy.stats.iqr(corr_matrix[:,hor,0],0,nan_policy='omit')],color='k',lw=2)
        axs.plot([hor+1,hor+1], [np.nanmean(corr_matrix[:,hor,1],0)+scipy.stats.iqr(corr_matrix[:,hor,1],0,nan_policy='omit'), np.nanmean(corr_matrix[:,hor,1],0)-scipy.stats.iqr(corr_matrix[:,hor,1],0,nan_policy='omit')],color='k',lw=2)
        axs.plot([hor+1,hor+1], [np.nanmean(corr_matrix[:,hor,2],0)+scipy.stats.iqr(corr_matrix[:,hor,2],0,nan_policy='omit'), np.nanmean(corr_matrix[:,hor,2],0)-scipy.stats.iqr(corr_matrix[:,hor,2],0,nan_policy='omit')],color='k',lw=2)
    axs.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_rvalue.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_rvalue.png'),bbox_inches='tight')

    def func(x,a,b,c):
        """
        function called by the curvefit for exponential fit 
        """
        return a*np.exp(-b*x)+c

    x_data = np.arange(1,6)-1
    y_data1 = np.nanmean(slope_matrix[:,:,0],0)
    y_data2 = np.nanmean(slope_matrix[:,:,1],0)
    y_data3 = np.nanmean(slope_matrix[:,:,2],0)
    popt1,pcov1 = scipy.optimize.curve_fit(func,x_data,y_data1,maxfev=5000)
    popt2,pcov2 = scipy.optimize.curve_fit(func,x_data,y_data2,maxfev=5000)
    popt3,pcov3 = scipy.optimize.curve_fit(func,x_data,y_data3,maxfev=5000)
    from tqdm import tqdm
    # Fitting the exponential decays with bootstrap resampling 
    n_boot = 10_000 
    mat_decays = np.zeros((n_boot,3))
    for iter in tqdm(range(n_boot)):
        idx_local = np.random.choice(8,8,replace=True)
        y_data_1_local = np.nanmean(slope_matrix[idx_local,:,0],0)
        y_data_2_local = np.nanmean(slope_matrix[idx_local,:,1],0)
        y_data_3_local = np.nanmean(slope_matrix[idx_local,:,2],0)
        popt_1_local, _ = scipy.optimize.curve_fit(func, x_data, y_data_1_local, maxfev=5000)
        popt_2_local, _ = scipy.optimize.curve_fit(func, x_data, y_data_2_local, maxfev=5000)
        popt_3_local, _ = scipy.optimize.curve_fit(func, x_data, y_data_3_local, maxfev=5000)
        mat_decays[iter,0], mat_decays[iter,1], mat_decays[iter,2] = popt_1_local[1], popt_2_local[1], popt_3_local[1]


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    tmp = 1/mat_decays
    axs.scatter([1,2,3],np.nanmedian(tmp,0),color='k',s=20)
    axs.set_xlim([-0.5,3.5]), axs.set_ylim([-0.5,2.5])
    axs.plot([3,3],[np.nanmedian(tmp[:,2])+scipy.stats.iqr(tmp[:,2],nan_policy='omit')/2,np.nanmedian(tmp[:,2])-scipy.stats.iqr(tmp[:,2],nan_policy='omit')/2],color='k',lw=2)
    plt.tight_layout()
    


    print(popt1, popt2, popt3)
    y_pred1 = func(x_data,popt1[0],popt1[1],popt1[2])
    y_pred2 = func(x_data,popt2[0],popt2[1],popt2[2])
    y_pred3 = func(x_data,popt3[0],popt3[1],popt3[2])
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,6),np.nanmean(slope_matrix[:,:,0],0),color='b',s=20,marker='^')
    axs.scatter(np.arange(1,6),np.nanmean(slope_matrix[:,:,1],0),color='b',s=20)
    axs.scatter(np.arange(1,6),np.nanmean(slope_matrix[:,:,2],0),color='b',s=20,marker='v')
    for hor in range(n_horizons):
        axs.plot([hor+1,hor+1], [np.nanmean(slope_matrix[:,hor,0],0)+scipy.stats.iqr(slope_matrix[:,hor,0],0,nan_policy='omit'), np.nanmean(slope_matrix[:,hor,0],0)-scipy.stats.iqr(slope_matrix[:,hor,0],0,nan_policy='omit')],color='b',lw=2)
        axs.plot([hor+1,hor+1], [np.nanmean(slope_matrix[:,hor,1],0)+scipy.stats.iqr(slope_matrix[:,hor,1],0,nan_policy='omit'), np.nanmean(slope_matrix[:,hor,1],0)-scipy.stats.iqr(slope_matrix[:,hor,1],0,nan_policy='omit')],color='b',lw=2)
        axs.plot([hor+1,hor+1], [np.nanmean(slope_matrix[:,hor,2],0)+scipy.stats.iqr(slope_matrix[:,hor,2],0,nan_policy='omit'), np.nanmean(slope_matrix[:,hor,2],0)-scipy.stats.iqr(slope_matrix[:,hor,2],0,nan_policy='omit')],color='b',lw=2)
    axs.plot(x_data+1,y_pred1,color='b',lw=1)
    axs.plot(x_data+1,y_pred2,color='b',lw=1)
    axs.plot(x_data+1,y_pred3,color='b',lw=1)
    axs.set_ylim([-1.05, 1.05]), axs.set_xlim([0,6])
    axs.set_yticks([-1,-0.5,0,0.5,1])
    axs.set_yticklabels(['-1','-0.5','0','0.5','1'])
    axs.axhline(0,color='k',lw=2,ls=':')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_slopes.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_slopes.svg'),bbox_inches='tight')

    # Illustrate the initial values 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(1,np.nanmean(slope_matrix[:,0,1]),color='b',s=20,marker='^')
    axs.scatter(2,np.nanmean(slope_matrix[:,0,0]),color='b',s=20)
    axs.scatter(3,np.nanmean(slope_matrix[:,0,2]),color='b',s=20,marker='v')
    axs.plot([1,1],[np.nanmean(slope_matrix[:,0,1])+np.nanstd(slope_matrix[:,0,1]),np.nanmean(slope_matrix[:,0,1])-np.nanstd(slope_matrix[:,0,1])],color='b',lw=2)
    axs.plot([2,2],[np.nanmean(slope_matrix[:,0,0])+np.nanstd(slope_matrix[:,0,0]),np.nanmean(slope_matrix[:,0,0])-np.nanstd(slope_matrix[:,0,0])],color='b',lw=2)
    axs.plot([3,3],[np.nanmean(slope_matrix[:,0,2])+np.nanstd(slope_matrix[:,0,2]),np.nanmean(slope_matrix[:,0,2])-np.nanstd(slope_matrix[:,0,2])],color='b',lw=2)
    x_vec1 = np.linspace(0.8,1.2,8)
    x_vec2 = np.linspace(1.8,2.2,8)
    x_vec3 = np.linspace(2.8,3.2,8)
    for video in range(8):
        axs.scatter(x_vec1[video], slope_matrix[video,0,1], color='b',s=5,alpha=0.2)
        axs.scatter(x_vec2[video], slope_matrix[video,0,0], color='b',s=5,alpha=0.2)
        axs.scatter(x_vec3[video], slope_matrix[video,0,2], color='b',s=5,alpha=0.2)
    axs.set_ylim([-1.05,1.05])
    axs.set_xlim([0,4])
    axs.set_yticks([-1,0,1])
    plt.tight_layout()
    test1 = scipy.stats.wilcoxon(np.squeeze(slope_matrix[:,0,0]),np.squeeze(slope_matrix[:,0,1]),alternative='less')
    test2 = scipy.stats.wilcoxon(np.squeeze(slope_matrix[:,0,0]),np.squeeze(slope_matrix[:,0,2]),alternative='less')
    print(test1, test2)
    print(scipy.stats.ttest_rel(np.squeeze(slope_matrix[:,0,0]),np.squeeze(slope_matrix[:,0,1]),alternative='less'))
    print(scipy.stats.ttest_rel(np.squeeze(slope_matrix[:,0,0]),np.squeeze(slope_matrix[:,0,2]),alternative='less'))
    print(scipy.stats.shapiro(np.squeeze(slope_matrix[:,0,0])))
    print(scipy.stats.shapiro(np.squeeze(slope_matrix[:,0,1])))
    print(scipy.stats.shapiro(np.squeeze(slope_matrix[:,0,2])))
    tmp_1 = np.squeeze(slope_matrix[:,0,0])
    tmp_2 = np.squeeze(slope_matrix[:,0,1])
    tmp_3 = np.squeeze(slope_matrix[:,0,2])
    cohend1 = (np.nanmean(tmp_1) - np.nanmean(tmp_2))/(np.nanstd(np.concatenate((tmp_1,tmp_2))))
    cohend2 = (np.nanmean(tmp_1) - np.nanmean(tmp_3))/(np.nanstd(np.concatenate((tmp_1,tmp_3))))
    print(cohend1)
    print(cohend2)
    tmp_kruskal = scipy.stats.kruskal(np.squeeze(slope_matrix[:,0,0]), np.squeeze(slope_matrix[:,0,1]), np.squeeze(slope_matrix[:,0,2]))
    print(tmp_kruskal)
    data_dunn = [np.squeeze(slope_matrix[:,0,0]), np.squeeze(slope_matrix[:,0,1]), np.squeeze(slope_matrix[:,0,2])]
    print(spph.posthoc_dunn(data_dunn, p_adjust='fdr_bh'))
    print('done for stats')

    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_slopes_initials.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_slopes_initials.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()

def exp_func(x,A,B,C,D):
    return A*np.exp(-B*(x-C))+D
    

def plot_feedforward_controller_final(input_list, output_list, video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the velocity dependency behavior for the fly data
    """
    list_slope, list_pvalue = [], []
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        n_per_animal = int(np.floor(len(idx_video)/12))
        for ii in range(12):
            tmp_len_1 = output_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],0]
            tmp_len_2 = output_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],3]
            tmp_len_3 = output_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],6]
            tmp_wid_1 = output_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],1]
            tmp_wid_2 = output_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],4]
            tmp_wid_3 = output_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],7]
            tmp_vel = np.nanmean(input_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],:,2],1)
            idx_to_keep1 = np.where((tmp_len_1>-32))[0]
            idx_to_keep2 = np.where((tmp_len_2>-0))[0]
            idx_to_keep3 = np.where((tmp_len_3>-62))[0]
            idx_nans_1 = np.where(~np.isnan(tmp_len_1[idx_to_keep1]))[0]
            idx_nans_2 = np.where(~np.isnan(tmp_len_2[idx_to_keep2]))[0]
            idx_nans_3 = np.where(~np.isnan(tmp_len_3[idx_to_keep3]))[0]
            lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1]], tmp_len_1[idx_to_keep1[idx_nans_1]])
            lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2]], tmp_len_2[idx_to_keep2[idx_nans_2]])
            lin_reg_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3]], tmp_len_3[idx_to_keep3[idx_nans_3]])
            lin_wid_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1]], tmp_wid_1[idx_to_keep1[idx_nans_1]])
            lin_wid_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2]], tmp_wid_2[idx_to_keep2[idx_nans_2]])
            lin_wid_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3]], tmp_wid_3[idx_to_keep3[idx_nans_3]])
            list_slope.append(np.array([lin_reg_1.slope, lin_wid_1.slope, lin_reg_2.slope, lin_wid_2.slope, lin_reg_3.slope, lin_wid_3.slope]))
            list_pvalue.append(np.array([lin_reg_1.pvalue, lin_wid_1.pvalue, lin_reg_2.pvalue, lin_wid_2.pvalue, lin_reg_3.pvalue, lin_wid_3.pvalue]))

    mat_slope = np.array(list_slope)
    mat_pvalue = np.array(list_pvalue)


    print(f'{len(np.where(mat_pvalue[:,0]<0.05)[0])}/{mat_pvalue.shape[0]}')
    print(f'{len(np.where(mat_pvalue[:,1]<0.05)[0])}/{mat_pvalue.shape[0]}')
    print(f'{len(np.where(mat_pvalue[:,2]<0.05)[0])}/{mat_pvalue.shape[0]}')
    print(f'{len(np.where(mat_pvalue[:,3]<0.05)[0])}/{mat_pvalue.shape[0]}')
    print(f'{len(np.where(mat_pvalue[:,4]<0.05)[0])}/{mat_pvalue.shape[0]}')
    print(f'{len(np.where(mat_pvalue[:,5]<0.05)[0])}/{mat_pvalue.shape[0]}')
    tot_len_1 = output_list[(video_list==0).flatten(),0]
    tot_len_2 = output_list[(video_list==0).flatten(),3]
    tot_len_3 = output_list[(video_list==0).flatten(),6]
    tot_wid_1 = output_list[(video_list==0).flatten(),1]
    tot_wid_2 = output_list[(video_list==0).flatten(),4]
    tot_wid_3 = output_list[(video_list==0).flatten(),7]
    tot_vel1 = np.nanmean(input_list[(video_list==0).flatten(),:,2],1)
    tot_vel2 = np.nanmean(input_list[(video_list==0).flatten(),:,6],1)
    tot_vel3 = np.nanmean(input_list[(video_list==0).flatten(),:,10],1)

    # Figure for individual video
    idx_video1 = np.where((video_list==1).flatten())[0]
    local_output = output_list[idx_video1,:]
    local_velocity_1 = np.nanmean(input_list[idx_video1,:,2],1)
    local_velocity_2 = np.nanmean(input_list[idx_video1,:,6],1)
    local_velocity_3 = np.nanmean(input_list[idx_video1,:,10],1)
    idx_forward_1 = np.where(local_output[:,0]>-0.15)[0]
    idx_forward_2 = np.where(local_output[:,3]>-0)[0]
    idx_forward_3 = np.where(local_output[:,6]>-0.3)[0]
    idx_nans_fwd1 = np.where(~np.isnan(local_output[idx_forward_1,0]))[0]
    idx_nans_fwd2 = np.where(~np.isnan(local_output[idx_forward_2,3]))[0]
    idx_nans_fwd3 = np.where(~np.isnan(local_output[idx_forward_3,6]))[0]
    reg_len1 = scipy.stats.linregress(local_velocity_1[idx_forward_1[idx_nans_fwd1]], local_output[idx_forward_1[idx_nans_fwd1],0])
    reg_len2 = scipy.stats.linregress(local_velocity_2[idx_forward_2[idx_nans_fwd2]], local_output[idx_forward_2[idx_nans_fwd2],3])
    reg_len3 = scipy.stats.linregress(local_velocity_3[idx_forward_3[idx_nans_fwd3]], local_output[idx_forward_3[idx_nans_fwd3],6])
    reg_wid1 = scipy.stats.linregress(local_velocity_1[idx_forward_1[idx_nans_fwd1]], local_output[idx_forward_1[idx_nans_fwd1],1])
    reg_wid2 = scipy.stats.linregress(local_velocity_2[idx_forward_2[idx_nans_fwd2]], local_output[idx_forward_2[idx_nans_fwd2],4])
    reg_wid3 = scipy.stats.linregress(local_velocity_3[idx_forward_3[idx_nans_fwd3]], local_output[idx_forward_3[idx_nans_fwd3],7])
    idx_plot1 = np.random.choice(idx_nans_fwd1,250,replace=False)
    idx_plot2 = np.random.choice(idx_nans_fwd2,250,replace=False)
    idx_plot3 = np.random.choice(idx_nans_fwd3,250,replace=False)
    print(reg_len1.rvalue**2, reg_len2.rvalue**2, reg_len3.rvalue**2)
    print(reg_wid1.rvalue**2, reg_wid2.rvalue**2, reg_wid3.rvalue**2)
    xlin = np.linspace(0,4)

    
    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],0],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(local_velocity_2[idx_forward_2[idx_nans_fwd2[idx_plot2]]], local_output[idx_forward_2[idx_nans_fwd2[idx_plot2]],3],color='g',alpha=0.3,s=10)
    axs.scatter(local_velocity_3[idx_forward_3[idx_nans_fwd3[idx_plot3]]], local_output[idx_forward_3[idx_nans_fwd3[idx_plot3]],6],color='m',alpha=0.3,s=10)
    axs.plot(xlin, xlin*reg_len1.slope + reg_len1.intercept, color=[115/255,147/255,179/255],lw=2)
    axs.plot(xlin, xlin*reg_len2.slope + reg_len2.intercept, color='g',lw=2)
    axs.plot(xlin, xlin*reg_len3.slope + reg_len3.intercept, color='m',lw=2)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step length')
    axs.set_ylim([-0.3,0.3])
    axs.set_xticks([1,2,3])
    axs.set_yticks([-0.3,0,0.3])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_foreaft_scatter.svg'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_foreaft_scatter.png'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],1],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(local_velocity_2[idx_forward_2[idx_nans_fwd2[idx_plot2]]], local_output[idx_forward_2[idx_nans_fwd2[idx_plot2]],4],color='g',alpha=0.3,s=10)
    axs.scatter(local_velocity_3[idx_forward_3[idx_nans_fwd3[idx_plot3]]], local_output[idx_forward_3[idx_nans_fwd3[idx_plot3]],7],color='m',alpha=0.3,s=10)
    axs.plot(xlin, xlin*reg_wid1.slope + reg_wid1.intercept, color=[115/255,147/255,179/255],lw=2)
    axs.plot(xlin, xlin*reg_wid2.slope + reg_wid2.intercept, color='g',lw=2)
    axs.plot(xlin, xlin*reg_wid3.slope + reg_wid3.intercept, color='m',lw=2)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step width')
    axs.set_ylim([-0.3,0.3])
    axs.set_xticks([1,2,3])
    axs.set_yticks([-0.3,0,0.3])
    plt.tight_layout()  
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_lateral_scatter.svg'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_lateral_scatter.png'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[0,0.2,0.4,0.6,0.8,1], height=np.nanmean(mat_slope,0), yerr=np.nanstd(mat_slope,0), color=['b','b','g','g','m','m'], width=0.2)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()
