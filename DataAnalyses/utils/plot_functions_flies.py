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

def plot_difference_comparison(ind_difference, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the individual differences between baseline and body-prediction
    """
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    vec_fly_x = np.linspace(2.8,3.2,8)
    for animal in range(8):
        axs.scatter(vec_fly_x[animal], ind_difference[animal,1],s=5, color='b', alpha=0.2)
    axs.scatter(3, np.nanmean(ind_difference[:,1]),color='b',s=20)
    axs.plot([3,3],[np.nanmean(ind_difference[:,1])+np.nanstd(ind_difference[:,1]),
                    np.nanmean(ind_difference[:,1])-np.nanstd(ind_difference[:,1])],color='b', lw=2)
    axs.set_ylim([-1.05,1.05]), axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()

def plot_rsquares_timing(list_body, list_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plotting the rsquares for the fly time data
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[0],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[0],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[0],0)+scipy.stats.iqr(list_body[0],0,nan_policy='omit')/2, np.nanmedian(list_body[0],0)-scipy.stats.iqr(list_body[0],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[0],0)+scipy.stats.iqr(list_self[0],0,nan_policy='omit')/2, np.nanmedian(list_self[0],0)-scipy.stats.iqr(list_self[0],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg1_time.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg1_time.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[1],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[1],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[1],0)+scipy.stats.iqr(list_body[1],0,nan_policy='omit')/2, np.nanmedian(list_body[1],0)-scipy.stats.iqr(list_body[1],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[1],0)+scipy.stats.iqr(list_self[1],0,nan_policy='omit')/2, np.nanmedian(list_self[1],0)-scipy.stats.iqr(list_self[1],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg2_time.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg2_time.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_body[2],0),'b',lw=2)
    axs.plot(np.nanmedian(list_self[2],0),'r',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list_body[2],0)+scipy.stats.iqr(list_body[2],0,nan_policy='omit')/2, np.nanmedian(list_body[2],0)-scipy.stats.iqr(list_body[2],0,nan_policy='omit')/2, color='b', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list_self[2],0)+scipy.stats.iqr(list_self[2],0,nan_policy='omit')/2, np.nanmedian(list_self[2],0)-scipy.stats.iqr(list_self[2],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg3_time.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg3_time.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()

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
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_hind_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_hind_lateral.svg'),bbox_inches='tight')

    list_pvalue_32 = []
    for time in range(21):
        idx_stats_nan = np.where((~np.isnan(list_body[5][:,time])) & (~np.isnan(list_self[5][:,time])))[0]
        stat_test = scipy.stats.wilcoxon(list_body[5][idx_stats_nan,time], list_self[5][idx_stats_nan,time], alternative='greater')
        list_pvalue_32.append(stat_test.pvalue)
    """
    print(np.array(list_pvalue_11))
    print(np.array(list_pvalue_12))
    print(np.array(list_pvalue_21))
    print(np.array(list_pvalue_22))
    print(np.array(list_pvalue_31))
    print(np.array(list_pvalue_32))
    """
    if bool_plot:
        plt.show()


def plot_scatter_laterality(tot_input_list, tot_output_list, tot_video, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the scatterplot for the laterality in the flies 
    """
    idx_nan = np.where(~np.isnan(tot_input_list[:,0,0]))[0]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_input_list[idx_nan,15,5]-np.nanmean(tot_input_list[idx_nan,15,5]),
                tot_output_list[idx_nan,4]-np.nanmean(tot_output_list[idx_nan,4]),color='k',alpha=0.2,s=5)
    tmp_inp = tot_input_list[idx_nan,15,5]-np.nanmean(tot_input_list[idx_nan,15,5])
    tmp_out = tot_output_list[idx_nan,4]-np.nanmean(tot_output_list[idx_nan,4])
    xpos, xneg = np.linspace(0,0.05), np.linspace(-0.05,0)
    idx_pos, idx_neg = np.where(tmp_inp>0)[0], np.where(tmp_inp<0)[0]
    reg_pos = scipy.stats.linregress(tmp_inp[idx_pos],tmp_out[idx_pos])
    reg_neg = scipy.stats.linregress(tmp_inp[idx_neg],tmp_out[idx_neg])
    axs.plot(xpos, xpos*reg_pos.slope + reg_pos.intercept, color='r', lw=2)
    axs.plot(xneg, xneg*reg_neg.slope + reg_neg.intercept, color='b', lw=2)
    axs.set_xlim([-0.05,0.05])
    axs.set_ylim([-0.10,0.10])
    plt.tight_layout() 
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()

def plot_laterality_gains(gains_small, gains_large, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the lateral gains at midstance for the multilinear regression 
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(1, np.nanmean(gains_small[:,15,1,1,1]),color='r',s=10)
    axs.scatter(2, np.nanmean(gains_large[:,15,1,1,1]),color='b',s=10)
    axs.plot([1,1], [np.nanmean(gains_small[:,15,1,1,1])-np.nanstd(gains_small[:,15,1,1,1]),np.nanmean(gains_small[:,15,1,1,1])+np.nanstd(gains_small[:,15,1,1,1])],color='r',lw=2)
    axs.plot([2,2], [np.nanmean(gains_large[:,15,1,1,1])-np.nanstd(gains_large[:,15,1,1,1]),np.nanmean(gains_large[:,15,1,1,1])+np.nanstd(gains_large[:,15,1,1,1])],color='b',lw=2)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}.svg'),bbox_inches='tight')
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

    print(test1)
    print(test_ttest)

    if bool_plot:
        plt.show()

def plot_normalized_feedback_gains(regression_matrix, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final normalized feedback gains for the lateral deviaiton
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(3, np.nanmean(regression_matrix[:,15,2,1,1,0],0),color='k',s=20)
    axs.plot([3,3],[np.nanmean(regression_matrix[:,15,2,1,1,0],0)+np.nanstd(regression_matrix[:,15,2,1,1,0],0),np.nanmean(regression_matrix[:,15,2,1,1,0],0)-np.nanstd(regression_matrix[:,15,2,1,1,0],0)],color='k',lw=2)
    axs.set_xlim([0,4])
    axs.set_ylim([0,3])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def plot_histograms_total_variance(input_variance, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the histograms for the total_variance captured
    """
    
    fig,axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[1,1.3,1.6,1.9,2.1,2.4],height=np.nanmean(input_variance[:,0,:],0),width=0.2,color='b')
    axs.bar(x=[1,1.3,1.6,1.9,2.1,2.4],bottom=np.nanmean(input_variance[:,0,:],0), height=np.nanmean(input_variance[:,1,:],0),width=0.2,color='r')
    axs.set_xlim([0.9,3.9])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

def plot_variance_contributions_final(input_variance, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final subplots for the variance contribution
    """
    # Compute the sum of ffwd and fb variance
    sum_variance = np.sum(input_variance,1)
    prop_variance = input_variance[:,0,:] / sum_variance

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter([2.95], [np.nanmean(sum_variance[:,2])], color='k',s=20)
    axs.scatter([3.05], [np.nanmean(sum_variance[:,3])], color='r',s=20)
    axs.plot([2.95,2.95],[np.nanmean(sum_variance[:,2])-np.nanstd(sum_variance[:,2]),np.nanmean(sum_variance[:,2])+np.nanstd(sum_variance[:,2])],color='k',lw=2)
    axs.plot([3.05,3.05],[np.nanmean(sum_variance[:,3])-np.nanstd(sum_variance[:,3]),np.nanmean(sum_variance[:,3])+np.nanstd(sum_variance[:,3])],color='r',lw=2)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_total_var.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_total_var.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter([2.95], [np.nanmean(prop_variance[:,2])], color='k',s=20)
    axs.scatter([3.05], [np.nanmean(prop_variance[:,3])], color='r',s=20)
    axs.plot([2.95,2.95],[np.nanmean(prop_variance[:,2])-np.nanstd(prop_variance[:,2]),np.nanmean(prop_variance[:,2])+np.nanstd(prop_variance[:,2])],color='k',lw=2)
    axs.plot([3.05,3.05],[np.nanmean(prop_variance[:,3])-np.nanstd(prop_variance[:,3]),np.nanmean(prop_variance[:,3])+np.nanstd(prop_variance[:,3])],color='r',lw=2)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_proportion_var.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_proportion_var.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()

def plot_rsquare_figure(list_leg1, list_leg2, list_leg3, bool_show=False, bool_save=False, figname=None):

    fig, axs = plt.subplots(2 ,3, figsize=(9,5), sharex=True, sharey=True)
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[0,2].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right']].set_visible(False)
    axs[1,2].spines[['top','right']].set_visible(False)

    axs[0,0].plot(np.nanmean(list_leg2[0],0),'k',lw=2,label='body')
    axs[0,0].plot(np.nanmean(list_leg2[2],0),'r',lw=2,label='self')
    axs[0,0].legend(frameon=False)
    axs[0,0].fill_between(np.arange(21), np.nanmean(list_leg2[0],0)+np.nanstd(list_leg2[0],0),np.nanmean(list_leg2[0],0)-np.nanstd(list_leg2[0],0),color='k',alpha=0.5)
    axs[0,0].fill_between(np.arange(21), np.nanmean(list_leg2[2],0)+np.nanstd(list_leg2[2],0),np.nanmean(list_leg2[2],0)-np.nanstd(list_leg2[2],0),color='r',alpha=0.5)
    axs[1,0].plot(np.nanmean(list_leg2[1],0),'k',lw=2)
    axs[1,0].plot(np.nanmean(list_leg2[3],0),'r',lw=2)
    axs[1,0].fill_between(np.arange(21), np.nanmean(list_leg2[1],0)+np.nanstd(list_leg2[1],0),np.nanmean(list_leg2[1],0)-np.nanstd(list_leg2[1],0),color='k',alpha=0.5)
    axs[1,0].fill_between(np.arange(21), np.nanmean(list_leg2[3],0)+np.nanstd(list_leg2[3],0),np.nanmean(list_leg2[3],0)-np.nanstd(list_leg2[3],0),color='r',alpha=0.5)
    axs[0,1].plot(np.nanmean(list_leg1[0],0),'k',lw=2)
    axs[0,1].plot(np.nanmean(list_leg1[2],0),'r',lw=2)
    axs[0,1].fill_between(np.arange(21), np.nanmean(list_leg1[0],0)+np.nanstd(list_leg1[0],0),np.nanmean(list_leg1[0],0)-np.nanstd(list_leg1[0],0),color='k',alpha=0.5)
    axs[0,1].fill_between(np.arange(21), np.nanmean(list_leg1[2],0)+np.nanstd(list_leg1[2],0),np.nanmean(list_leg1[2],0)-np.nanstd(list_leg1[2],0),color='r',alpha=0.5)
    axs[1,1].plot(np.nanmean(list_leg1[1],0),'k',lw=2)
    axs[1,1].plot(np.nanmean(list_leg1[3],0),'r',lw=2)
    axs[1,1].fill_between(np.arange(21), np.nanmean(list_leg1[1],0)+np.nanstd(list_leg1[1],0),np.nanmean(list_leg1[1],0)-np.nanstd(list_leg1[1],0),color='k',alpha=0.5)
    axs[1,1].fill_between(np.arange(21), np.nanmean(list_leg1[3],0)+np.nanstd(list_leg1[3],0),np.nanmean(list_leg1[3],0)-np.nanstd(list_leg1[3],0),color='r',alpha=0.5)
    axs[0,2].plot(np.nanmean(list_leg3[0],0),'k',lw=2)
    axs[0,2].plot(np.nanmean(list_leg3[2],0),'r',lw=2)
    axs[0,2].fill_between(np.arange(21), np.nanmean(list_leg3[0],0)+np.nanstd(list_leg3[0],0),np.nanmean(list_leg3[0],0)-np.nanstd(list_leg3[0],0),color='k',alpha=0.5)
    axs[0,2].fill_between(np.arange(21), np.nanmean(list_leg3[2],0)+np.nanstd(list_leg3[2],0),np.nanmean(list_leg3[2],0)-np.nanstd(list_leg3[2],0),color='r',alpha=0.5)
    axs[1,2].plot(np.nanmean(list_leg3[1],0),'k',lw=2)
    axs[1,2].plot(np.nanmean(list_leg3[3],0),'r',lw=2)
    axs[1,2].fill_between(np.arange(21), np.nanmean(list_leg3[1],0)+np.nanstd(list_leg3[1],0),np.nanmean(list_leg3[1],0)-np.nanstd(list_leg3[1],0),color='k',alpha=0.5)
    axs[1,2].fill_between(np.arange(21), np.nanmean(list_leg3[3],0)+np.nanstd(list_leg3[3],0),np.nanmean(list_leg3[3],0)-np.nanstd(list_leg3[3],0),color='r',alpha=0.5)
                  
    axs[0,2].set_xlabel('Relative gait fraction')
    axs[1,2].set_xlabel('Relative gait fraction')
    axs[0,0].set_ylabel('Coefficient of determination')
    axs[1,0].set_ylabel('Coefficient of determination')
    axs[0,1].set_ylabel('Coefficient of determination')
    axs[0,0].set_title('Front limbs - Foreaft direction')
    axs[0,1].set_title('Central limbs - Foreaft direction')
    axs[0,2].set_title('Hind limbs - Foreaft direction')
    axs[1,0].set_title('Front limbs - Lateral direction')
    axs[1,1].set_title('Central limbs - Lateral direction')
    axs[1,2].set_title('Hind limbs - Lateral direction')
    axs[0,2].set_xticks([0,5,10,15,20])
    axs[0,2].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Compute the stats associated with that figure 
    list_00, list_10, list_01, list_11, list_02, list_12 = [], [], [], [], [], []
    pval00, pval10, pval01, pval11, pval02, pval12 = np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1))
    for time in range(21):
        list_00.append(scipy.stats.ks_2samp(list_leg2[0][:,time] ,list_leg2[2][:,time] ,alternative='less'))
        list_10.append(scipy.stats.ks_2samp(list_leg2[1][:,time] ,list_leg2[3][:,time] ,alternative='less'))
        list_01.append(scipy.stats.ks_2samp(list_leg1[0][:,time] ,list_leg1[2][:,time] ,alternative='less'))
        list_11.append(scipy.stats.ks_2samp(list_leg1[1][:,time] ,list_leg1[3][:,time] ,alternative='less'))
        list_02.append(scipy.stats.ks_2samp(list_leg3[0][:,time] ,list_leg3[2][:,time] ,alternative='less'))
        list_12.append(scipy.stats.ks_2samp(list_leg3[1][:,time] ,list_leg3[3][:,time] ,alternative='less'))
        pval00[time] = list_00[time].pvalue
        pval10[time] = list_10[time].pvalue
        pval01[time] = list_01[time].pvalue
        pval11[time] = list_11[time].pvalue
        pval02[time] = list_02[time].pvalue
        pval12[time] = list_12[time].pvalue
    ptot = np.concatenate((pval00,pval10,pval01,pval11,pval02, pval12),1)
    np.savetxt(os.path.join(OUTPUT_STATS,'stats_rsquare.csv'),ptot, delimiter=',')

def plot_feedback_gains(list_regression, bool_show=False, bool_save=False, figname=None):
    """Plot the feedback gains """
    fig, axs = plt.subplots(2,3,figsize=(9,5), sharex=True, sharey=True)
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[0,2].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right']].set_visible(False)
    axs[1,2].spines[['top','right']].set_visible(False)
    scale_change00 = copy.deepcopy(list_regression[1][:,15,2,0,:,0])
    scale_change01 = copy.deepcopy(list_regression[1][:,15,2,0,:,0])
    scale_change02 = copy.deepcopy(list_regression[1][:,15,2,0,:,0])
    scale_change10 = copy.deepcopy(list_regression[1][:,15,2,0,:,0])
    scale_change11 = copy.deepcopy(list_regression[1][:,15,2,0,:,0])
    scale_change12 = copy.deepcopy(list_regression[1][:,15,2,0,:,0])
    idx_nans_0 = np.where(~np.isnan(list_regression[1][:,15,2,0,0,0]))[0]
    idx_nans_1 = np.where(~np.isnan(list_regression[0][:,15,2,0,0,0]))[0]
    idx_nans_2 = np.where(~np.isnan(list_regression[2][:,15,2,0,0,0]))[0]
    axs[0,0].boxplot(list_regression[1][idx_nans_0,15,2,0,:,0],positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[0,1].boxplot(list_regression[0][idx_nans_1,15,2,0,:,0],positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[0,2].boxplot(list_regression[2][idx_nans_2,15,2,0,:,0],positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[1,0].boxplot(list_regression[1][idx_nans_0,15,2,1,:,0],positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[1,1].boxplot(list_regression[0][idx_nans_1,15,2,1,:,0],positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[1,2].boxplot(list_regression[2][idx_nans_2,15,2,1,:,0],positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[0,0].axhline(0, color='k', ls=':')
    axs[0,1].axhline(0, color='k', ls=':')
    axs[0,2].axhline(0, color='k', ls=':')
    axs[1,0].axhline(0, color='k', ls=':')
    axs[1,1].axhline(0, color='k', ls=':')
    axs[1,2].axhline(0, color='k', ls=':')
    axs[1,0].set_xlabel('Input variable')
    axs[1,1].set_xlabel('Input variable')
    axs[1,2].set_xlabel('Input variable')
    axs[0,0].set_ylabel('Slope')
    axs[1,0].set_ylabel('Slope')
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

def plot_laterality_front(list_laterality, bool_show=False, bool_save=False,figname=None):
    """
    Plots the figure for the laterality (only front limbs)
    """
    fig, axs = plt.subplots(2,1,figsize=(3,5), sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)

    axs[0].plot(np.nanmean(list_laterality[2][:,:,2,1,1,2]**2,0),'k')
    axs[0].plot(np.nanmean(list_laterality[3][:,:,2,1,1,2]**2,0),color=[0.5,0.5,0.5])
    axs[0].fill_between(np.arange(21), np.nanmean(list_laterality[2][:,:,2,1,1,2]**2,0)-scipy.stats.iqr(list_laterality[2][:,:,2,1,1,2]**2,0,nan_policy='omit')/2, np.nanmean(list_laterality[2][:,:,2,1,1,2]**2,0)+scipy.stats.iqr(list_laterality[2][:,:,2,1,1,2]**2,0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs[0].fill_between(np.arange(21), np.nanmean(list_laterality[3][:,:,2,1,1,2]**2,0)-scipy.stats.iqr(list_laterality[3][:,:,2,1,1,2]**2,0,nan_policy='omit')/2, np.nanmean(list_laterality[3][:,:,2,1,1,2]**2,0)+scipy.stats.iqr(list_laterality[3][:,:,2,1,1,2]**2,0,nan_policy='omit')/2, color=[0.5,0.5,0.5], alpha=0.5)
    axs[1].axhline(0, color='k', ls=':')
    axs[1].plot(np.nanmean(list_laterality[2][:,:,2,1,1,0],0),'k')
    axs[1].plot(np.nanmean(list_laterality[3][:,:,2,1,1,0],0),color=[0.5,0.5,0.5])
    axs[1].fill_between(np.arange(21), np.nanmean(list_laterality[2][:,:,2,1,1,0],0)+scipy.stats.iqr(list_laterality[2][:,:,2,1,1,0],0,nan_policy='omit')/2,np.nanmean(list_laterality[2][:,:,2,1,1,0],0)-scipy.stats.iqr(list_laterality[2][:,:,2,1,1,0],0,nan_policy='omit')/2, color='k',alpha=0.5)
    axs[1].fill_between(np.arange(21), np.nanmean(list_laterality[3][:,:,2,1,1,0],0)+scipy.stats.iqr(list_laterality[3][:,:,2,1,1,0],0,nan_policy='omit')/2,np.nanmean(list_laterality[3][:,:,2,1,1,0],0)-scipy.stats.iqr(list_laterality[3][:,:,2,1,1,0],0,nan_policy='omit')/2, color=[0.5,0.5,0.5],alpha=0.5)
    axs[1].set_xticks([0,5,10,15,20])
    axs[1].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[1].set_xlabel('Relative gait fraction')
    axs[0].set_ylabel('Coefficient of determination')
    axs[1].set_ylabel('Slope')
    axs[0].set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    #Run the stats for the figure above 
    list_00, list_01 = [], []
    pval00, pval01 = np.zeros((21,1)), np.zeros((21,1))
    for time in range(21):
        list_00.append(scipy.stats.ks_2samp(list_laterality[2][:,time,2,1,1,2]**2, list_laterality[3][:,time,2,1,1,2]**2,alternative='less'))
        list_01.append(scipy.stats.ks_2samp(list_laterality[2][:,time,2,1,1,0], list_laterality[3][:,time,2,1,1,0],alternative='less'))
        pval00[time], pval01[time] = list_00[time].pvalue, list_01[time].pvalue
    ptot = np.concatenate((pval00, pval01),1)
    np.savetxt(os.path.join(OUTPUT_STATS, 'stas_laterality.csv'), ptot, delimiter=',')

def plot_laterality(list_laterality, bool_show=False, bool_save=False, figname=None):
    """
    Plots the figure for the laterality question
    """
    fig, axs = plt.subplots(3,2,figsize=(5,3), sharex=True, sharey=True)
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right']].set_visible(False)
    axs[2,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right']].set_visible(False)
    axs[2,1].spines[['top','right']].set_visible(False)

    axs[0,0].plot(np.nanmean(list_laterality[0][:,:,2,1,1,2]**2,0),'k')
    axs[0,0].plot(np.nanmean(list_laterality[1][:,:,2,1,1,2]**2,0),'k:')
    axs[0,0].fill_between(np.arange(21), np.nanmean(list_laterality[0][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[0][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[0][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[0][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[0,0].fill_between(np.arange(21), np.nanmean(list_laterality[1][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[1][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[1][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[1][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[1,0].plot(np.nanmean(list_laterality[2][:,:,2,1,1,2]**2,0),'k')
    axs[1,0].plot(np.nanmean(list_laterality[3][:,:,2,1,1,2]**2,0),'k:')
    axs[1,0].fill_between(np.arange(21), np.nanmean(list_laterality[2][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[2][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[2][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[2][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[1,0].fill_between(np.arange(21), np.nanmean(list_laterality[3][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[3][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[3][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[3][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[2,0].plot(np.nanmean(list_laterality[4][:,:,2,1,1,2]**2,0),'k')
    axs[2,0].plot(np.nanmean(list_laterality[5][:,:,2,1,1,2]**2,0),'k:')
    axs[2,0].fill_between(np.arange(21), np.nanmean(list_laterality[4][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[4][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[4][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[4][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[2,0].fill_between(np.arange(21), np.nanmean(list_laterality[5][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[5][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[5][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[5][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[0,1].plot(np.nanmean(list_laterality[6][:,:,2,1,1,2]**2,0),'k')
    axs[0,1].plot(np.nanmean(list_laterality[7][:,:,2,1,1,2]**2,0),'k:')
    axs[0,1].fill_between(np.arange(21), np.nanmean(list_laterality[6][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[6][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[6][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[6][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[0,1].fill_between(np.arange(21), np.nanmean(list_laterality[7][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[7][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[7][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[7][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[1,1].plot(np.nanmean(list_laterality[8][:,:,2,1,1,2]**2,0),'k')
    axs[1,1].plot(np.nanmean(list_laterality[9][:,:,2,1,1,2]**2,0),'k:')
    axs[1,1].fill_between(np.arange(21), np.nanmean(list_laterality[8][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[8][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[8][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[8][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[1,1].fill_between(np.arange(21), np.nanmean(list_laterality[9][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[9][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[9][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[9][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[2,1].plot(np.nanmean(list_laterality[10][:,:,2,1,1,2]**2,0),'k')
    axs[2,1].plot(np.nanmean(list_laterality[11][:,:,2,1,1,2]**2,0),'k:')
    axs[2,1].fill_between(np.arange(21), np.nanmean(list_laterality[10][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[10][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[10][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[10][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[2,1].fill_between(np.arange(21), np.nanmean(list_laterality[11][:,:,2,1,1,2]**2,0)-np.nanstd(list_laterality[11][:,:,2,1,1,2]**2,0),np.nanmean(list_laterality[11][:,:,2,1,1,2]**2,0)+np.nanstd(list_laterality[11][:,:,2,1,1,2]**2,0),color='k',alpha=0.5)
    axs[0,0].set_xticks([0,5,10,15,20])
    axs[0,0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[1,0].set_ylabel('Coefficient of determination')
    axs[2,0].set_xlabel('Relative gait fraction')
    axs[2,1].set_xlabel('Relative gait fraction')
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')


def plot_individual_contact_fly(tot_input_list, tot_output_list, video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the individual contact for the fly data
    """
    idx_video = np.where(video_list==1)[0]
    rnd_vid = np.random.choice(len(idx_video),150)
    tmp_len_1 = tot_output_list[idx_video[rnd_vid],0]
    tmp_len_2 = tot_output_list[idx_video[rnd_vid],3]
    tmp_len_3 = tot_output_list[idx_video[rnd_vid],6]
    tmp_wid_1 = tot_output_list[idx_video[rnd_vid],1]
    tmp_wid_2 = tot_output_list[idx_video[rnd_vid],4]
    tmp_wid_3 = tot_output_list[idx_video[rnd_vid],7]

    fig, axs = plt.subplots(1,1,figsize=(3,5))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(0, 0, s=30, color='r', alpha=0.5)
    axs.scatter(tmp_wid_1, tmp_len_1, color='g', s=10, alpha=0.5)
    axs.scatter(tmp_wid_2, tmp_len_2, color=[115/255,147/255,179/255], s=10, alpha=0.5)
    axs.scatter(tmp_wid_3, tmp_len_3, color='m', s=10, alpha=0.5)
    axs.set_xlabel('Lateral direction')
    axs.set_ylabel('Foreaft direction')
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,'individual_contact_fly.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,'individual_contact_fly.svg'),bbox_inches='tight')

def plot_feedforward_controller_inputs(input_list, input_self, bool_plot=False, bool_save=False, bool_fb=False, figname=None):
    """
    Plots the feedforward behavior on the input variables of the model
    """
    tot_velocity = np.nanmean(input_list[:,:,2],1)
    pctl_velocity = np.nanpercentile(tot_velocity,[33,66,100])
    tot_vel_bin = np.zeros((len(tot_velocity),))
    for line in range(len(tot_velocity)):
        if tot_velocity[line]< pctl_velocity[0]: 
            tot_vel_bin[line] = 0
        elif tot_velocity[line]<pctl_velocity[1]:
            tot_vel_bin[line] = 1
        elif tot_velocity[line] < pctl_velocity[2]:
            tot_vel_bin[line] = 2
        else:
            tot_vel_bin[line] = np.nan

    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_list[tot_vel_bin==0,:,0],0) - np.nanmean(input_list[tot_vel_bin==0,-1,0]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_list[tot_vel_bin==1,:,0],0) - np.nanmean(input_list[tot_vel_bin==1,-1,0]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_list[tot_vel_bin==2,:,0],0) - np.nanmean(input_list[tot_vel_bin==2,-1,0]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==0,:,0],0) - np.nanmean(input_list[tot_vel_bin==0,-1,0]) + scipy.stats.iqr(input_list[tot_vel_bin==0,:,0],0,nan_policy='omit')/2,np.nanmean(input_list[tot_vel_bin==0,:,0],0) - np.nanmean(input_list[tot_vel_bin==0,-1,0]) - scipy.stats.iqr(input_list[tot_vel_bin==0,:,0],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==1,:,0],0) - np.nanmean(input_list[tot_vel_bin==1,-1,0]) + scipy.stats.iqr(input_list[tot_vel_bin==1,:,0],0,nan_policy='omit')/2,np.nanmean(input_list[tot_vel_bin==1,:,0],0) - np.nanmean(input_list[tot_vel_bin==1,-1,0]) - scipy.stats.iqr(input_list[tot_vel_bin==1,:,0],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==2,:,0],0) - np.nanmean(input_list[tot_vel_bin==2,-1,0]) + scipy.stats.iqr(input_list[tot_vel_bin==2,:,0],0,nan_policy='omit')/2,np.nanmean(input_list[tot_vel_bin==2,:,0],0) - np.nanmean(input_list[tot_vel_bin==2,-1,0]) - scipy.stats.iqr(input_list[tot_vel_bin==2,:,0],0,nan_policy='omit')/2, color='k', alpha=0.7)
    

    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_body.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_body.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_list[tot_vel_bin==0,:,1],0) - np.nanmean(input_list[tot_vel_bin==0,:,1]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_list[tot_vel_bin==1,:,1],0) - np.nanmean(input_list[tot_vel_bin==1,:,1]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_list[tot_vel_bin==2,:,1],0) - np.nanmean(input_list[tot_vel_bin==2,:,1]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==0,:,1],0) - np.nanmean(input_list[tot_vel_bin==0,:,1]) + scipy.stats.iqr(input_list[tot_vel_bin==0,:,1],0,nan_policy='omit')/2,np.nanmean(input_list[tot_vel_bin==0,:,1],0) - np.nanmean(input_list[tot_vel_bin==0,:,1]) - scipy.stats.iqr(input_list[tot_vel_bin==0,:,1],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==1,:,1],0) - np.nanmean(input_list[tot_vel_bin==1,:,1]) + scipy.stats.iqr(input_list[tot_vel_bin==1,:,1],0,nan_policy='omit')/2,np.nanmean(input_list[tot_vel_bin==1,:,1],0) - np.nanmean(input_list[tot_vel_bin==1,:,1]) - scipy.stats.iqr(input_list[tot_vel_bin==1,:,1],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==2,:,1],0) - np.nanmean(input_list[tot_vel_bin==2,:,1]) + scipy.stats.iqr(input_list[tot_vel_bin==2,:,1],0,nan_policy='omit')/2,np.nanmean(input_list[tot_vel_bin==2,:,1],0) - np.nanmean(input_list[tot_vel_bin==2,:,1]) - scipy.stats.iqr(input_list[tot_vel_bin==2,:,1],0,nan_policy='omit')/2, color='k', alpha=0.7)
    
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_body.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_body.svg'),bbox_inches='tight')
    # Foreaft leg 1 
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,0],0) - np.nanmean(input_self[tot_vel_bin==0,-1,0]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,0],0) - np.nanmean(input_self[tot_vel_bin==1,-1,0]) ,'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,0],0) - np.nanmean(input_self[tot_vel_bin==2,-1,0]) ,'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,0],0) - np.nanmean(input_self[tot_vel_bin==0,-1,0]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,0],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==0,:,0],0) - np.nanmean(input_self[tot_vel_bin==0,-1,0]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,0],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,0],0) - np.nanmean(input_self[tot_vel_bin==1,-1,0]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,0],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==1,:,0],0) - np.nanmean(input_self[tot_vel_bin==1,-1,0]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,0],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,0],0) - np.nanmean(input_self[tot_vel_bin==2,-1,0]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,0],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==2,:,0],0) - np.nanmean(input_self[tot_vel_bin==2,-1,0]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,0],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_central_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_central_self.svg'),bbox_inches='tight')
    # Lateral leg 1
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,1],0) - np.nanmean(input_self[tot_vel_bin==0,:,1]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,1],0) - np.nanmean(input_self[tot_vel_bin==1,:,1]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,1],0) - np.nanmean(input_self[tot_vel_bin==2,:,1]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,1],0) - np.nanmean(input_self[tot_vel_bin==0,:,1]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,1],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==0,:,1],0) - np.nanmean(input_self[tot_vel_bin==0,:,1]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,1],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,1],0) - np.nanmean(input_self[tot_vel_bin==1,:,1]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,1],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==1,:,1],0) - np.nanmean(input_self[tot_vel_bin==1,:,1]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,1],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,1],0) - np.nanmean(input_self[tot_vel_bin==2,:,1]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,1],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==2,:,1],0) - np.nanmean(input_self[tot_vel_bin==2,:,1]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,1],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_central_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_central_self.svg'),bbox_inches='tight')
    # Foreaft leg 2
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,4],0) - np.nanmean(input_self[tot_vel_bin==0,-1,4]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,4],0) - np.nanmean(input_self[tot_vel_bin==1,-1,4]) ,'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,4],0) - np.nanmean(input_self[tot_vel_bin==2,-1,4]) ,'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,4],0) - np.nanmean(input_self[tot_vel_bin==0,-1,4]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,4],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==0,:,4],0) - np.nanmean(input_self[tot_vel_bin==0,-1,4]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,4],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,4],0) - np.nanmean(input_self[tot_vel_bin==1,-1,4]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,4],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==1,:,4],0) - np.nanmean(input_self[tot_vel_bin==1,-1,4]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,4],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,4],0) - np.nanmean(input_self[tot_vel_bin==2,-1,4]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,4],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==2,:,4],0) - np.nanmean(input_self[tot_vel_bin==2,-1,4]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,4],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_front_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_front_self.svg'),bbox_inches='tight')
    # Lateral leg 2
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,5],0) - np.nanmean(input_self[tot_vel_bin==0,:,5]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,5],0) - np.nanmean(input_self[tot_vel_bin==1,:,5]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,5],0) - np.nanmean(input_self[tot_vel_bin==2,:,5]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,5],0) - np.nanmean(input_self[tot_vel_bin==0,:,5]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,5],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==0,:,5],0) - np.nanmean(input_self[tot_vel_bin==0,:,5]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,5],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,5],0) - np.nanmean(input_self[tot_vel_bin==1,:,5]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,5],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==1,:,5],0) - np.nanmean(input_self[tot_vel_bin==1,:,5]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,5],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,5],0) - np.nanmean(input_self[tot_vel_bin==2,:,5]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,5],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==2,:,5],0) - np.nanmean(input_self[tot_vel_bin==2,:,5]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,5],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_front_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_front_self.svg'),bbox_inches='tight')
    # Foreaft leg 3
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,8],0) - np.nanmean(input_self[tot_vel_bin==0,-1,8]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,8],0) - np.nanmean(input_self[tot_vel_bin==1,-1,8]) ,'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,8],0) - np.nanmean(input_self[tot_vel_bin==2,-1,8]) ,'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,8],0) - np.nanmean(input_self[tot_vel_bin==0,-1,8]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,8],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==0,:,8],0) - np.nanmean(input_self[tot_vel_bin==0,-1,8]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,8],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,8],0) - np.nanmean(input_self[tot_vel_bin==1,-1,8]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,8],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==1,:,8],0) - np.nanmean(input_self[tot_vel_bin==1,-1,8]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,8],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,8],0) - np.nanmean(input_self[tot_vel_bin==2,-1,8]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,8],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==2,:,8],0) - np.nanmean(input_self[tot_vel_bin==2,-1,8]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,8],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_hind_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_foreaft_hind_self.svg'),bbox_inches='tight')
    # Lateral leg 3
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,9],0) - np.nanmean(input_self[tot_vel_bin==0,:,9]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,9],0) - np.nanmean(input_self[tot_vel_bin==1,:,9]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,9],0) - np.nanmean(input_self[tot_vel_bin==2,:,9]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,9],0) - np.nanmean(input_self[tot_vel_bin==0,:,9]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,9],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==0,:,9],0) - np.nanmean(input_self[tot_vel_bin==0,:,9]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,9],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,9],0) - np.nanmean(input_self[tot_vel_bin==1,:,9]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,9],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==1,:,9],0) - np.nanmean(input_self[tot_vel_bin==1,:,9]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,9],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,9],0) - np.nanmean(input_self[tot_vel_bin==2,:,9]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,9],0,nan_policy='omit')/2,np.nanmean(input_self[tot_vel_bin==2,:,9],0) - np.nanmean(input_self[tot_vel_bin==2,:,9]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,9],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_hind_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_lateral_hind_self.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()


def plot_successive_contacts_fly(input_list, video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the correlations between successive contacts as a function of the lookup horizon for the fly data
    """

    n_videos = int(np.max(video_list))
    n_horizons = len(input_list)
    print(n_horizons)
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

    print(np.nanmedian(mat_decays,0))
    print(np.nanpercentile(mat_decays,[5,95],0))
    print(scipy.stats.iqr(mat_decays,0))


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    tmp = 1/mat_decays
    axs.scatter([1,2,3],np.nanmedian(tmp,0),color='k',s=20)
    print('Parameters to put in the paper')
    print(np.nanmedian(tmp,0))
    print(scipy.stats.iqr(tmp,0,nan_policy='omit'))
    axs.set_xlim([-0.5,3.5]), axs.set_ylim([-0.5,2.5])
    #axs.plot([1,1],[np.nanmedian(tmp[:,0])+scipy.stats.iqr(tmp[:,0],nan_policy='omit')/2,np.nanmedian(tmp[:,0])-scipy.stats.iqr(tmp[:,0],nan_policy='omit')/2],color='k',lw=2)
    #axs.plot([2,2],[np.nanmedian(tmp[:,1])+scipy.stats.iqr(tmp[:,1],nan_policy='omit')/2,np.nanmedian(tmp[:,1])-scipy.stats.iqr(tmp[:,1],nan_policy='omit')/2],color='k',lw=2)
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


def plot_normalized_feedback_final(gains_matrix, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the normalized feedback gains for the fly data
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

def plot_all_successive_contacts_fly(input_list, video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the information for all the successive contacts as a function of the horizon
    """
    n_videos = int(np.max(video_list))
    n_horizons = len(input_list)
    corr_matrix = np.zeros((n_videos, n_horizons,3))
    slope_matrix = np.zeros((n_videos, n_horizons, 3))
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
            corr_matrix[vid-1, hor, 0], corr_matrix[vid-1, hor, 1], corr_matrix[vid-1, hor, 2] = reg1.rvalue, reg2.rvalue, reg3.rvalue
            slope_matrix[vid-1,hor,0], slope_matrix[vid-1,hor,1], slope_matrix[vid-1,hor,2] = reg1.slope, reg2.slope, reg3.slope
    
    np.save(os.path.join(OUTPUT_STATS,'slope_correlation'), slope_matrix)
    np.save(os.path.join(OUTPUT_STATS,'rsquare_correlation'), corr_matrix)
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,11),np.nanmean(corr_matrix[:,:,0],0),color='g',s=10)
    axs.scatter(np.arange(1,11),np.nanmean(corr_matrix[:,:,1],0),color=[115/255,147/255,179/255],s=10)
    axs.scatter(np.arange(1,11),np.nanmean(corr_matrix[:,:,2],0),color='m',s=10)
    for hor in range(n_horizons):
        axs.plot([hor+1,hor+1], [np.nanmean(corr_matrix[:,:,0],0)+scipy.stats.iqr(corr_matrix[:,:,0],0,nan_policy='omit'), np.nanmean(corr_matrix[:,:,0],0)+scipy.stats.iqr(corr_matrix[:,:,0],0,nan_policy='omit')],color='g',lw=1)
        axs.plot([hor+1,hor+1], [np.nanmean(corr_matrix[:,:,1],0)+scipy.stats.iqr(corr_matrix[:,:,1],0,nan_policy='omit'), np.nanmean(corr_matrix[:,:,1],0)+scipy.stats.iqr(corr_matrix[:,:,1],0,nan_policy='omit')],color=[115/255,147/255,179/255],lw=1)
        axs.plot([hor+1,hor+1], [np.nanmean(corr_matrix[:,:,2],0)+scipy.stats.iqr(corr_matrix[:,:,2],0,nan_policy='omit'), np.nanmean(corr_matrix[:,:,2],0)+scipy.stats.iqr(corr_matrix[:,:,2],0,nan_policy='omit')],color='m',lw=1)
    axs.set_xlabel('Horizon')
    axs.set_ylabel('R²')
    axs.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}.svg'),bbox_inches='tight')

    


    if bool_plot:
        plt.show()

def plot_successive_contacts(input_list,video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the correlation between successive contacts for the fly data
    """
    idx_local = np.where(video_list==1)[0]
    local_input = input_list[idx_local,:,:]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_input[local_input[:,-1,4]>0,0,4],local_input[local_input[:,-1,4]>0,-1,4],color='g',s=5, alpha=0.5)
    axs.scatter(local_input[local_input[:,-1,0]>-32,0,0],local_input[local_input[:,-1,0]>-32,-1,0],color=[115/255,147/255,179/255],s=5, alpha=0.5)
    axs.scatter(local_input[local_input[:,-1,8]>-62,0,8],local_input[local_input[:,-1,8]>-62,-1,8],color='m',s=5, alpha=0.5)
    axs.set_xlabel('Step length (t)')
    axs.set_ylabel('Step length (t+1)')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_input[local_input[:,-1,4]>0,0,5],local_input[local_input[:,-1,4]>0,-1,5],color='g',s=5, alpha=0.5,label='Front')
    axs.scatter(local_input[local_input[:,-1,0]>-32,0,1],local_input[local_input[:,-1,0]>-32,-1,1],color=[115/255,147/255,179/255],s=5, alpha=0.5,label='Central')
    axs.scatter(local_input[local_input[:,-1,8]>-62,0,9],local_input[local_input[:,-1,8]>-62,-1,9],color='m',s=5, alpha=0.5,label='Hind')
    idx_good_1 = np.where((local_input[:,-1,4]>0) & (~np.isnan(local_input[:,-1,4])))[0]
    idx_good_2 = np.where((local_input[:,-1,0]>-32) & (~np.isnan(local_input[:,-1,0])))[0]
    idx_good_3 = np.where((local_input[:,-1,8]>-62) & (~np.isnan(local_input[:,-1,8])))[0]
    print(len(idx_good_1))
    reg1 = scipy.stats.linregress(local_input[idx_good_1,0,5], local_input[idx_good_1,-1,5])
    reg2 = scipy.stats.linregress(local_input[idx_good_2,0,1], local_input[idx_good_2,-1,1])
    reg3 = scipy.stats.linregress(local_input[idx_good_3,0,9], local_input[idx_good_3,-1,9])
    print(reg1.rvalue**2, reg2.rvalue**2, reg3.rvalue**2)
    axs.legend(frameon=False)
    axs.set_xlabel('Step width (t)')
    axs.set_ylabel('Step width (t+1)')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')


    if bool_plot:
        plt.show()



def plot_rsquare_normalized(lists_body, lists_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the rsquare metrics for the fly data
    """
    npoints = lists_body[0].shape[1]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[0],0),'k',lw=2)
    axs.plot(np.nanmedian(lists_self[0],0),'r',lw=2)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_body[0],0)+scipy.stats.iqr(lists_body[0],0,nan_policy='omit')/2, np.nanmedian(lists_body[0],0)-scipy.stats.iqr(lists_body[0],0,nan_policy='omit')/2, color='k', alpha=0.5)
    print(lists_self[0].shape)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_self[0],0)+scipy.stats.iqr(lists_self[0],0,nan_policy='omit')/2, np.nanmedian(lists_self[0],0)-scipy.stats.iqr(lists_self[0],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_xticks(np.arange(0,npoints,5))
    #axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_1.svg'),bbox_inches='tight')

    # Stats for the above subfigure 
    #list_subfig1_pvalue, list_subfig1_stats  = [], []
    #for time in range(lists_body[0].shape[1]):
    #    tmp_test = scipy.stats.ks_2samp(lists_body[0][:,time], lists_self[0][:,time], alternative='less')
    #    list_subfig1_pvalue.append(tmp_test.pvalue)
    #    list_subfig1_stats.append(tmp_test.statistic)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[1],0),'k',lw=2)
    axs.plot(np.nanmedian(lists_self[1],0),'r',lw=2)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_body[1],0)+scipy.stats.iqr(lists_body[1],0,nan_policy='omit')/2, np.nanmedian(lists_body[1],0)-scipy.stats.iqr(lists_body[1],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_self[1],0)+scipy.stats.iqr(lists_self[1],0,nan_policy='omit')/2, np.nanmedian(lists_self[1],0)-scipy.stats.iqr(lists_self[1],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_xticks(np.arange(0,npoints,5))
    #axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_2.svg'),bbox_inches='tight')

     # Stats for the above subfigure 
    #list_subfig2_pvalue, list_subfig2_stats  = [], []
    #for time in range(lists_body[0].shape[1]):
    #    tmp_test = scipy.stats.ks_2samp(lists_body[1][:,time], lists_self[1][:,time], alternative='less')
    #    list_subfig2_pvalue.append(tmp_test.pvalue)
    #    list_subfig2_stats.append(tmp_test.statistic)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[2],0),'k',lw=2)
    axs.plot(np.nanmedian(lists_self[2],0),'r',lw=2)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_body[2],0)+scipy.stats.iqr(lists_body[2],0,nan_policy='omit')/2, np.nanmedian(lists_body[2],0)-scipy.stats.iqr(lists_body[2],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_self[2],0)+scipy.stats.iqr(lists_self[2],0,nan_policy='omit')/2, np.nanmedian(lists_self[2],0)-scipy.stats.iqr(lists_self[2],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_xticks(np.arange(0,npoints,5))
    #axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_3.svg'),bbox_inches='tight')
    
     # Stats for the above subfigure 
    #list_subfig3_pvalue, list_subfig3_stats  = [], []
    #for time in range(lists_body[0].shape[1]):
    #    tmp_test = scipy.stats.ks_2samp(lists_body[2][:,time], lists_self[2][:,time], alternative='less')
    #    list_subfig3_pvalue.append(tmp_test.pvalue)
    #    list_subfig3_stats.append(tmp_test.statistic)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[3],0),'k',lw=2)
    axs.plot(np.nanmedian(lists_self[3],0),'r',lw=2)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_body[3],0)+scipy.stats.iqr(lists_body[3],0,nan_policy='omit')/2, np.nanmedian(lists_body[3],0)-scipy.stats.iqr(lists_body[3],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_self[3],0)+scipy.stats.iqr(lists_self[3],0,nan_policy='omit')/2, np.nanmedian(lists_self[3],0)-scipy.stats.iqr(lists_self[3],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_xticks(np.arange(0,npoints,5))
    #axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_4.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_4.svg'),bbox_inches='tight')
    
     # Stats for the above subfigure 
    #list_subfig4_pvalue, list_subfig4_stats  = [], []
    #for time in range(lists_body[0].shape[1]):
    #    tmp_test = scipy.stats.ks_2samp(lists_body[3][:,time], lists_self[3][:,time], alternative='less')
    #    list_subfig4_pvalue.append(tmp_test.pvalue)
    #    list_subfig4_stats.append(tmp_test.statistic)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[4],0),'k',lw=2)
    axs.plot(np.nanmedian(lists_self[4],0),'r',lw=2)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_body[4],0)+scipy.stats.iqr(lists_body[4],0,nan_policy='omit')/2, np.nanmedian(lists_body[4],0)-scipy.stats.iqr(lists_body[4],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_self[4],0)+scipy.stats.iqr(lists_self[4],0,nan_policy='omit')/2, np.nanmedian(lists_self[4],0)-scipy.stats.iqr(lists_self[4],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_xticks(np.arange(0,npoints,5))
    #axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_5.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_5.svg'),bbox_inches='tight')

     # Stats for the above subfigure 
    #list_subfig5_pvalue, list_subfig5_stats  = [], []
    #for time in range(lists_body[0].shape[1]):
    #    tmp_test = scipy.stats.ks_2samp(lists_body[4][:,time], lists_self[4][:,time], alternative='less')
    #    list_subfig5_pvalue.append(tmp_test.pvalue)
    #    list_subfig5_stats.append(tmp_test.statistic)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[5],0),'k',lw=2)
    axs.plot(np.nanmedian(lists_self[5],0),'r',lw=2)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_body[5],0)+scipy.stats.iqr(lists_body[5],0,nan_policy='omit')/2, np.nanmedian(lists_body[5],0)-scipy.stats.iqr(lists_body[5],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(npoints), np.nanmedian(lists_self[5],0)+scipy.stats.iqr(lists_self[5],0,nan_policy='omit')/2, np.nanmedian(lists_self[5],0)-scipy.stats.iqr(lists_self[5],0,nan_policy='omit')/2, color='r', alpha=0.5)
    axs.set_xticks(np.arange(0,npoints,5))
    #axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_6.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES, f'{figname}_6.svg'),bbox_inches='tight')

     # Stats for the above subfigure 
    #list_subfig6_pvalue, list_subfig6_stats  = [], []
    #for time in range(lists_body[0].shape[1]):
    #    tmp_test = scipy.stats.ks_2samp(lists_body[5][:,time], lists_self[5][:,time], alternative='less')
    #    list_subfig6_pvalue.append(tmp_test.pvalue)
    #    list_subfig6_stats.append(tmp_test.statistic)

    #print(np.array(list_subfig1_pvalue).shape)
    # Saving all the stats in the appropriate folder
    #tot_array_pvalues = np.hstack((np.expand_dims(np.array(list_subfig1_pvalue),-1), np.expand_dims(np.array(list_subfig2_pvalue),-1), np.expand_dims(np.array(list_subfig3_pvalue),-1), np.expand_dims(np.array(list_subfig4_pvalue),-1), np.expand_dims(np.array(list_subfig5_pvalue),-1), np.expand_dims(np.array(list_subfig6_pvalue),-1)))
    #np.savetxt(os.path.join(OUTPUT_STATS,'stats_rsquares_revised_pvalue.csv'),tot_array_pvalues, delimiter=',')
        
    if bool_plot:
        plt.show()


def plot_rsquare_normalized_modified(lists_body, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the rsquare metrics for the fly data
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[0],0),'k',lw=2)
    axs.fill_between(np.arange(41), np.nanmedian(lists_body[0],0)+scipy.stats.iqr(lists_body[0],0,nan_policy='omit')/2, np.nanmedian(lists_body[0],0)-scipy.stats.iqr(lists_body[0],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.set_xticks([0,5,10,15,20,25,30,35,40])
    axs.set_xticklabels(['-2','-1.75','-1.5','-1.25','-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[1],0),'k',lw=2)
    axs.fill_between(np.arange(41), np.nanmedian(lists_body[1],0)+scipy.stats.iqr(lists_body[1],0,nan_policy='omit')/2, np.nanmedian(lists_body[1],0)-scipy.stats.iqr(lists_body[1],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.set_xticks([0,5,10,15,20,25,30,35,40])
    axs.set_xticklabels(['-2','-1.75','-1.5','-1.25','-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[2],0),'k',lw=2)
    axs.fill_between(np.arange(41), np.nanmedian(lists_body[2],0)+scipy.stats.iqr(lists_body[2],0,nan_policy='omit')/2, np.nanmedian(lists_body[2],0)-scipy.stats.iqr(lists_body[2],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.set_xticks([0,5,10,15,20,25,30,35,40])
    axs.set_xticklabels(['-2','-1.75','-1.5','-1.25','-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[3],0),'k',lw=2)
    axs.fill_between(np.arange(41), np.nanmedian(lists_body[3],0)+scipy.stats.iqr(lists_body[3],0,nan_policy='omit')/2, np.nanmedian(lists_body[3],0)-scipy.stats.iqr(lists_body[3],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[4],0),'k',lw=2)
    axs.fill_between(np.arange(41), np.nanmedian(lists_body[4],0)+scipy.stats.iqr(lists_body[4],0,nan_policy='omit')/2, np.nanmedian(lists_body[4],0)-scipy.stats.iqr(lists_body[4],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])



    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(lists_body[5],0),'k',lw=2)
    axs.fill_between(np.arange(41), np.nanmedian(lists_body[5],0)+scipy.stats.iqr(lists_body[5],0,nan_policy='omit')/2, np.nanmedian(lists_body[5],0)-scipy.stats.iqr(lists_body[5],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.set_xticks([0,5,10,15,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()

        
    if bool_plot:
        plt.show()

def plot_feedforward_body_flies_final_outliers(input_list, output_list, video_list, bool_plot=False, bool_save=False, figname='None'):
    """
    Same as below but with the removal of outliers ...
    """
    local_input = input_list[(video_list==1).flatten(),:,:]
    local_velocity = np.nanmean(local_input[:,:,6],1)
    local_amplitude_x = np.nanmax(local_input[:,:,4],1) - np.nanmin(local_input[:,:,4],1)
    local_amplitude_y = np.nanmax(local_input[:,:,5],1) - np.nanmin(local_input[:,:,5],1)
    local_output = output_list[(video_list==1).flatten(),:]
    idx_to_keep = np.where((local_output[:,0]>-0.15) & (local_output[:,3]>0) & (local_output[:,6]>-0.3))[0]
    cmap = plt.cm.plasma


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.set_title('Position')
    color_vector = (local_velocity - np.nanmin(local_velocity)) / (np.nanmax(local_velocity) - np.nanmin(local_velocity))
    for line in range(local_input.shape[0]):
        axs.plot(local_input[line,:,0],lw=0.2, color=cmap(color_vector[line]))
    plt.tight_layout()



    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.set_title('Velocity')
    for line in range(local_input.shape[0]):
        axs.plot(local_input[line,:,2],lw=0.2, color=cmap(color_vector[line]))
    
    plt.tight_layout()
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    print(len(idx_to_keep))
    axs.set_title('Scatter outliers')
    axs.scatter(local_velocity, local_amplitude_x,color='k',s=10,alpha=0.2)
    plt.tight_layout()


def plot_feedforward_body_flies_final(input_list, video_list, bool_plot=False, bool_save=False, figname='None'):
    """
    Plots the impact of the velocity on the head position for the fly data
    """

    list_slope, list_pvalue = [], []
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        n_per_animal = int(np.floor(len(idx_video)/12))
        for ii in range(12):
            tot_input = input_list[idx_video[ii*n_per_animal:(ii+1)*n_per_animal],:,:]
            tot_velocity = np.nanmean(tot_input[:,:,2],1)

            amplitude_x = tot_input[:,-1,0] - tot_input[:,0,0]
            amplitude_y = np.nanmax(tot_input[:,:,1],1) - np.nanmin(tot_input[:,:,1],1)
            idx_nans = np.where(~np.isnan(amplitude_y))[0]
            reg_ampl_x = scipy.stats.linregress(tot_velocity[idx_nans], amplitude_x[idx_nans])
            reg_ampl_y = scipy.stats.linregress(tot_velocity[idx_nans], amplitude_y[idx_nans])

            list_slope.append(np.array([reg_ampl_x.slope, reg_ampl_y.slope]))
            list_pvalue.append(np.array([reg_ampl_x.pvalue, reg_ampl_y.pvalue]))

    mat_slopes = np.array(list_slope)
    mat_pvalues = np.array(list_pvalue)
    print(f'{len(np.where(mat_pvalues[:,0]<0.05)[0])}/{mat_pvalues.shape[0]}')
    print(f'{len(np.where(mat_pvalues[:,1]<0.05)[0])}/{mat_pvalues.shape[0]}')


    local_input = input_list[(video_list==2).flatten(),:,:]
    local_velocity = np.nanmedian(local_input[:,:,2],1)
    local_velocity2 = np.nanmedian(local_input[:,:,6],1)
    local_velocity3 = np.nanmedian(local_input[:,:,10],1)
    print(local_velocity.shape)
    local_amplitude_x = local_input[:,-1,0] - local_input[:,0,0]
    local_amplitude_x2= local_input[:,-1,4] - local_input[:,0,4]
    local_amplitude_x3= local_input[:,-1,8] - local_input[:,0,8]
    local_amplitude_y = np.nanmax(local_input[:,:,1],1) - np.nanmin(local_input[:,:,1],1)

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity, local_amplitude_x, color='k', s=10, alpha=0.2)
    axs.scatter(local_velocity2, local_amplitude_x2, color='k', s=10, alpha=0.2)
    axs.scatter(local_velocity3, local_amplitude_x3, color='k', s=10, alpha=0.2)
    axs.scatter(local_velocity, local_amplitude_y, color='r', s=10, alpha=0.2)
    axs.set_ylabel('Amplitude'), axs.set_xlabel('Velocity')
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_amplitude_sample.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_amplitude_sample.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[0,1], width=0.2, height=np.nanmean(mat_slopes,0), yerr=np.nanstd(mat_slopes,0),color='b')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()

def plot_feedforward_controller_final_vel(input_list, output_list, video_list, vel_vector, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the velocity dependency behavior for the fly data 
    """
    avg_velocity = np.abs(vel_vector)
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
    idx_plot1 = np.random.choice(idx_nans_fwd1,250,replace=False)
    idx_plot2 = np.random.choice(idx_nans_fwd2,250,replace=False)
    idx_plot3 = np.random.choice(idx_nans_fwd3,250,replace=False)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], avg_velocity[idx_forward_1[idx_nans_fwd1[idx_plot1]]],color=[115/255,147/255,179/255],s=10)
    axs.scatter(local_velocity_2[idx_forward_2[idx_nans_fwd2[idx_plot2]]], avg_velocity[idx_forward_2[idx_nans_fwd2[idx_plot2]]],color='g',s=10)
    axs.scatter(local_velocity_3[idx_forward_3[idx_nans_fwd3[idx_plot3]]], avg_velocity[idx_forward_3[idx_nans_fwd3[idx_plot3]]],color='m',s=10)
    axs.set_xlabel('Cycle-averaged velocity')
    axs.set_ylabel('Bout-averaged velocity')
    axs.plot([1,4],[1,4],color='k',lw=2)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_comparison.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_comparison.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.set_title('Cycle averaged velocity',fontweight='bold',fontsize=10)
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],0],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(local_velocity_2[idx_forward_2[idx_nans_fwd2[idx_plot2]]], local_output[idx_forward_2[idx_nans_fwd2[idx_plot2]],3],color='g',alpha=0.3,s=10)
    axs.scatter(local_velocity_3[idx_forward_3[idx_nans_fwd3[idx_plot3]]], local_output[idx_forward_3[idx_nans_fwd3[idx_plot3]],6],color='m',alpha=0.3,s=10)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step length')
    axs.set_ylim([-0.3,0.3])
    axs.set_xticks([1,2,3])
    axs.set_yticks([-0.3,0,0.3])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_cycle_averaged.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_cycle_averaged.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.set_title('Bout averaged velocity',fontweight='bold',fontsize=10)
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(avg_velocity[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],0],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(avg_velocity[idx_forward_2[idx_nans_fwd2[idx_plot2]]], local_output[idx_forward_2[idx_nans_fwd2[idx_plot2]],3],color='g',alpha=0.3,s=10)
    axs.scatter(avg_velocity[idx_forward_3[idx_nans_fwd3[idx_plot3]]], local_output[idx_forward_3[idx_nans_fwd3[idx_plot3]],6],color='m',alpha=0.3,s=10)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step length')
    axs.set_ylim([-0.3,0.3])
    axs.set_xticks([1,2,3])
    axs.set_yticks([-0.3,0,0.3])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_bout_averaged.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_bout_averaged.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def exp_func(x,A,B,C,D):
    return A*np.exp(-B*(x-C))+D

from scipy.optimize import curve_fit
    

def plot_feedforward_controller_final_time(input_list, output_list, video_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the velocity dependency bheavior for the fly data
    """
    print(input_list.shape, output_list.shape)
    idx_video1 = np.where((video_list==1).flatten())[0]
    local_output = output_list[idx_video1,:]
    local_velocity_1 = np.abs(np.nanmean(input_list[idx_video1,:,2],1))
    local_velocity_2 = np.abs(np.nanmean(input_list[idx_video1,:,6],1))
    local_velocity_3 = np.abs(np.nanmean(input_list[idx_video1,:,10],1))

    idx_nans_tot = np.where((~np.isnan(local_velocity_1)) & (~np.isnan(local_velocity_2)) & (~np.isnan(local_velocity_3)))[0]
    popt1, _ = curve_fit(exp_func, local_velocity_1[idx_nans_tot], local_output[idx_nans_tot,2], bounds=([0,0,0,0],[10,1,200,1]))
    popt2, _ = curve_fit(exp_func, local_velocity_2[idx_nans_tot], local_output[idx_nans_tot,5], bounds=([0,0,0,0],[10,1,200,1]))
    popt3, _ = curve_fit(exp_func, local_velocity_3[idx_nans_tot], local_output[idx_nans_tot,8], bounds=([0,0,0,0],[10,1,200,1]))
    xplot = np.linspace(1,3)
    print(popt1)
    print(popt2)
    print(popt3)

    idx_forward_1 = np.where(local_output[:,0]>-0.15)[0]
    idx_forward_2 = np.where(local_output[:,3]>-0)[0]
    idx_forward_3 = np.where(local_output[:,6]>-0.3)[0]
    idx_nans_fwd1 = np.where(~np.isnan(local_output[idx_forward_1,0]))[0]
    idx_nans_fwd2 = np.where(~np.isnan(local_output[idx_forward_2,3]))[0]
    idx_nans_fwd3 = np.where(~np.isnan(local_output[idx_forward_3,6]))[0]
    idx_plot1 = np.random.choice(idx_nans_fwd1,250,replace=False)
    idx_plot2 = np.random.choice(idx_nans_fwd2,250,replace=False)
    idx_plot3 = np.random.choice(idx_nans_fwd3,250,replace=False)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],0],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],3],color='g',alpha=0.3,s=10)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],6],color='m',alpha=0.3,s=10)
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],0],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],3],color='g',alpha=0.3,s=10)
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],6],color='m',alpha=0.3,s=10)
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],1],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],4],color='g',alpha=0.3,s=10)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],7],color='m',alpha=0.3,s=10)
    plt.tight_layout()


    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    print(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]].shape)
    print(local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],2].shape)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],2],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.scatter(local_velocity_2[idx_forward_2[idx_nans_fwd2[idx_plot2]]], local_output[idx_forward_2[idx_nans_fwd2[idx_plot2]],5],color='g',alpha=0.3,s=10)
    axs.scatter(local_velocity_3[idx_forward_3[idx_nans_fwd3[idx_plot3]]], local_output[idx_forward_3[idx_nans_fwd3[idx_plot3]],8],color='m',alpha=0.3,s=10)
    axs.plot(xplot, exp_func(xplot,*popt1),color=[115/255,147/255,179/255],lw=2)
    axs.plot(xplot, exp_func(xplot,*popt2),'g',lw=2)
    axs.plot(xplot, exp_func(xplot,*popt3),'m',lw=2)
    plt.tight_layout()
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    axs.set_ylim([0,0.25]), axs.set_xlim([0,4])
    axs.set_xticks([1,2,3])
    # axs.set_yticks([-0.3,0,0.3])
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_group_level_figure_timing.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_group_level_figure_timing.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_1[idx_forward_1[idx_nans_fwd1[idx_plot1]]], local_output[idx_forward_1[idx_nans_fwd1[idx_plot1]],2],color=[115/255,147/255,179/255],alpha=0.3,s=10)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    axs.set_ylim([0,0.25]), axs.set_xlim([0,4])
    axs.plot(xplot, exp_func(xplot,*popt1),color=[115/255,147/255,179/255],lw=2)
    axs.set_xticks([1,2,3])
    plt.tight_layout()
    # axs.set_yticks([-0.3,0,0.3])
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg1_figure_timing.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg1_figure_timing.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_2[idx_forward_2[idx_nans_fwd2[idx_plot2]]], local_output[idx_forward_2[idx_nans_fwd2[idx_plot2]],5],color='g',alpha=0.3,s=10)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    axs.set_ylim([0,0.25]), axs.set_xlim([0,4])
    axs.plot(xplot, exp_func(xplot,*popt2),'g',lw=2)
    axs.set_xticks([1,2,3])
    plt.tight_layout()
    # axs.set_yticks([-0.3,0,0.3])
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg2_figure_timing.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg2_figure_timing.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity_3[idx_forward_3[idx_nans_fwd3[idx_plot3]]], local_output[idx_forward_3[idx_nans_fwd3[idx_plot3]],8],color='m',alpha=0.3,s=10)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    axs.plot(xplot, exp_func(xplot,*popt3),'m',lw=2)
    axs.set_ylim([0,0.25]), axs.set_xlim([0,4])
    axs.set_xticks([1,2,3])
    plt.tight_layout()
    # axs.set_yticks([-0.3,0,0.3])
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg3_figure_timing.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_leg3_figure_timing.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()

    

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



def plot_feedforward_controller(output_list, output_list_od, input_list, input_list_od, video_list, video_list_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the feedforward behaviors
    """

    # FIGURE 1 - Variances in the foreaft and lateral directions
    list_var, list_var_od = [],[]
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        idx_video_od = np.where(video_list_od==video_id)[0]
        tmp_len_1 = output_list[idx_video,0]
        tmp_len_2 = output_list[idx_video,3]
        tmp_len_3 = output_list[idx_video,6]
        tmp_wid_1 = output_list[idx_video,1]
        tmp_wid_2 = output_list[idx_video,4]
        tmp_wid_3 = output_list[idx_video,7]
        tmp_len_1_od = output_list_od[idx_video_od,0]
        tmp_len_2_od = output_list_od[idx_video_od,3]
        tmp_len_3_od = output_list_od[idx_video_od,6]
        tmp_wid_1_od = output_list_od[idx_video_od,1]
        tmp_wid_2_od = output_list_od[idx_video_od,4]
        tmp_wid_3_od = output_list_od[idx_video_od,7]
        list_var.append(np.array([scipy.stats.iqr(tmp_len_2,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_len_1,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_len_3,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_2,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_1,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_3,nan_policy='omit')]))
        list_var_od.append(np.array([scipy.stats.iqr(tmp_len_2_od,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_len_1_od,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_len_3_od,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_2_od,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_1_od,nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_3_od,nan_policy='omit')]))
    array_var, array_var_od = np.array(list_var), np.array(list_var_od)
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar([0,0.5,1.5,2,3,3.5],np.nanmedian(array_var,0), width=0.4, yerr=scipy.stats.iqr(array_var,0,nan_policy='omit'),color=[[115/255, 147/255, 179/255],[115/255,147/255,179/255],'g','g','m','m'],ecolor='k')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_1.svg'),bbox_inches='tight') 
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_diag_1.csv'), array_var, delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_odiag_1tats_odiag.csv'),array_var_od, delimiter=',')
    

    # FIGURES 2/3 - Step lengths/widths as a function of velocity
    idx_video = np.where(video_list==1)[0]
    tmp_len_1 = output_list[idx_video,0]
    tmp_len_2 = output_list[idx_video,3]
    tmp_len_3 = output_list[idx_video,6]
    tmp_wid_1 = output_list[idx_video,1]
    tmp_wid_2 = output_list[idx_video,4]
    tmp_wid_3 = output_list[idx_video,7]
    tmp_vel = np.nanmean(input_list[idx_video,:,2],1)

    idx_to_keep1 = np.where((tmp_len_1>-32))[0]
    idx_to_keep2 = np.where((tmp_len_2>-0))[0]
    idx_to_keep3 = np.where((tmp_len_3>-62))[0]
    idx_nans_1 = np.where((~np.isnan(tmp_len_1[idx_to_keep1])))[0]
    idx_nans_2 = np.where((~np.isnan(tmp_len_2[idx_to_keep2])))[0]
    idx_nans_3 = np.where((~np.isnan(tmp_len_3[idx_to_keep3])))[0]
    idx_nans_1_ = np.where((~np.isnan(tmp_wid_1[idx_to_keep1])))[0]
    idx_nans_2_ = np.where((~np.isnan(tmp_wid_2[idx_to_keep2])))[0]
    idx_nans_3_ = np.where((~np.isnan(tmp_wid_3[idx_to_keep3])))[0]
    lin_reg_len_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1]], tmp_len_1[idx_to_keep1[idx_nans_1]])
    lin_reg_len_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2]], tmp_len_2[idx_to_keep2[idx_nans_2]])
    lin_reg_len_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3]], tmp_len_3[idx_to_keep3[idx_nans_3]])
    lin_reg_wid_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1_]], tmp_wid_1[idx_to_keep1[idx_nans_1_]])
    lin_reg_wid_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2_]], tmp_wid_2[idx_to_keep2[idx_nans_2_]])
    lin_reg_wid_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3_]], tmp_wid_3[idx_to_keep3[idx_nans_3_]])
    x_axis = np.linspace(200,700)
    y_pred_len_1 = lin_reg_len_1.slope * x_axis + lin_reg_len_1.intercept
    y_pred_len_2 = lin_reg_len_2.slope * x_axis + lin_reg_len_2.intercept
    y_pred_len_3 = lin_reg_len_3.slope * x_axis + lin_reg_len_3.intercept
    y_pred_wid_1 = lin_reg_wid_1.slope * x_axis + lin_reg_wid_1.intercept
    y_pred_wid_2 = lin_reg_wid_2.slope * x_axis + lin_reg_wid_2.intercept
    y_pred_wid_3 = lin_reg_wid_3.slope * x_axis + lin_reg_wid_3.intercept

    fig, axs = plt.subplots(1,3,figsize=(8,3), sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[2].spines[['top','right']].set_visible(False)
    axs[0].set_xlabel('Velocity'), axs[0].set_ylabel('Step length')
    axs[1].set_xlabel('Velocity'), axs[1].set_ylabel('Step length')
    axs[2].set_xlabel('Velocity'), axs[2].set_ylabel('Step length')
    axs[0].set_xlim([0,800])
    axs[0].set_ylim([-65,40]), axs[1].set_ylim([-65,40]), axs[2].set_ylim([-65,40])
    axs[0].scatter(tmp_vel[idx_to_keep2], tmp_len_2[idx_to_keep2], color=[115/255, 147/255, 179/255], alpha=0.5)
    axs[1].scatter(tmp_vel[idx_to_keep1], tmp_len_1[idx_to_keep1], color='g', alpha=0.5)
    axs[2].scatter(tmp_vel[idx_to_keep3], tmp_len_3[idx_to_keep3], color='m', alpha=0.5)
    axs[0].plot(x_axis, y_pred_len_2, color='k', lw=3)
    axs[1].plot(x_axis, y_pred_len_1, color='k', lw=3)
    axs[2].plot(x_axis, y_pred_len_3, color='k', lw=3)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,3,figsize=(8,3), sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[2].spines[['top','right']].set_visible(False)
    axs[0].set_xlabel('Velocity'), axs[0].set_ylabel('Step width')
    axs[1].set_xlabel('Velocity'), axs[1].set_ylabel('Step width')
    axs[2].set_xlabel('Velocity'), axs[2].set_ylabel('Step width')
    axs[0].set_xlim([0,800])
    axs[0].set_ylim([-55,50]), axs[1].set_ylim([-55,50]), axs[2].set_ylim([-55,50])
    axs[0].scatter(tmp_vel[idx_to_keep2], tmp_wid_2[idx_to_keep2], color=[115/255, 147/255, 179/255], alpha=0.5)
    axs[1].scatter(tmp_vel[idx_to_keep1], tmp_wid_1[idx_to_keep1], color='g', alpha=0.5)
    axs[2].scatter(tmp_vel[idx_to_keep3], tmp_wid_3[idx_to_keep3], color='m', alpha=0.5)
    axs[0].plot(x_axis, y_pred_wid_2, color='k', lw=3)
    axs[1].plot(x_axis, y_pred_wid_1, color='k', lw=3)
    axs[2].plot(x_axis, y_pred_wid_3, color='k', lw=3)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_3.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

    # Saving the statistics for the lengths and widths ... 
    list_pvalue, list_slope, list_intercept = [], [], []
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        tmp_len_1 = output_list[idx_video,0]
        tmp_len_2 = output_list[idx_video,3]
        tmp_len_3 = output_list[idx_video,6]
        tmp_vel = np.nanmean(input_list[idx_video,:,2],1)
        idx_to_keep1 = np.where((tmp_len_1>-32))[0]
        idx_to_keep2 = np.where((tmp_len_2>-0))[0]
        idx_to_keep3 = np.where((tmp_len_3>-62))[0]
        idx_nans_1 = np.where(~np.isnan(tmp_len_1[idx_to_keep1]))[0]
        idx_nans_2 = np.where(~np.isnan(tmp_len_2[idx_to_keep2]))[0]
        idx_nans_3 = np.where(~np.isnan(tmp_len_3[idx_to_keep3]))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1]], tmp_len_1[idx_to_keep1[idx_nans_1]])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2]], tmp_len_2[idx_to_keep2[idx_nans_2]])
        lin_reg_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3]], tmp_len_3[idx_to_keep3[idx_nans_3]])
        idx_video_od = np.where(video_list_od==video_id)[0]
        tmp_len_1_od = output_list_od[idx_video_od,0]
        tmp_len_2_od = output_list_od[idx_video_od,3]
        tmp_len_3_od = output_list_od[idx_video_od,6]
        tmp_vel_od = np.nanmean(input_list_od[idx_video_od,:,2],1)
        idx_to_keep1_od = np.where((tmp_len_1_od>-0))[0]
        idx_to_keep2_od = np.where((tmp_len_2_od>-62))[0]
        idx_to_keep3_od = np.where((tmp_len_3_od>-32))[0]
        idx_nans_1_od = np.where(~np.isnan(tmp_len_1_od[idx_to_keep1_od]))[0]
        idx_nans_2_od = np.where(~np.isnan(tmp_len_2_od[idx_to_keep2_od]))[0]
        idx_nans_3_od = np.where(~np.isnan(tmp_len_3_od[idx_to_keep3_od]))[0]
        lin_reg_1_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep1_od[idx_nans_1_od]], tmp_len_1_od[idx_to_keep1_od[idx_nans_1_od]])
        lin_reg_2_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep2_od[idx_nans_2_od]], tmp_len_2_od[idx_to_keep2_od[idx_nans_2_od]])
        lin_reg_3_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep3_od[idx_nans_3_od]], tmp_len_3_od[idx_to_keep3_od[idx_nans_3_od]])
        list_pvalue.append(np.array([lin_reg_1.pvalue, lin_reg_2.pvalue, lin_reg_3.pvalue, lin_reg_1_od.pvalue, lin_reg_2_od.pvalue, lin_reg_3_od.pvalue]))
        list_slope.append(np.array([lin_reg_1.slope, lin_reg_2.slope, lin_reg_3.slope, lin_reg_1_od.slope, lin_reg_2_od.slope, lin_reg_3_od.slope]))
        list_intercept.append(np.array([lin_reg_1.intercept, lin_reg_2.intercept, lin_reg_3.intercept, lin_reg_1_od.intercept, lin_reg_2_od.intercept, lin_reg_3_od.intercept]))
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_2_pvalues.csv'),np.array(list_pvalue), delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_2_slopes.csv'),np.array(list_slope),delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_2_intercepts.csv'),np.array(list_intercept),delimiter=',')
    
   

    mat_slopes = np.array(list_slope)
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    x_vec_flies = np.linspace(0.9,1.1,8)
    for line in range(len(x_vec_flies)):
        axs.scatter(x_vec_flies[line]+0.5, mat_slopes[line,0], color='k', s=5, alpha=0.5,marker='^')
        axs.scatter(x_vec_flies[line], mat_slopes[line,1], color='k', s=5, alpha=0.5)
        axs.scatter(x_vec_flies[line]+1, mat_slopes[line,2], color='k', s=5, alpha=0.5,marker='v')
    axs.scatter(1.5, np.nanmean(mat_slopes[:,0]),color='k',s=20,marker='^')
    axs.scatter(1, np.nanmean(mat_slopes[:,1]),color='k',s=20)
    axs.scatter(2, np.nanmean(mat_slopes[:,2]),color='k',s=20,marker='v')
    axs.set_xlim([0.5,2.5])
    axs.set_ylim([-0.005,0.05])
    axs.axhline(0, color='k', lw=2, ls=':')
    plt.tight_layout()

    tmp_kruskal = scipy.stats.kruskal(mat_slopes[:,0], mat_slopes[:,1], mat_slopes[:,2])
    print(tmp_kruskal)
    data_dunn = [np.squeeze(mat_slopes[:,0]), np.squeeze(mat_slopes[:,1]), np.squeeze(mat_slopes[:,2])]
    print(spph.posthoc_dunn(data_dunn, p_adjust='bonferroni'))

    if True:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_ffwd_slopes.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}_ffwd_slopes.svg'),bbox_inches='tight')

    list_pvalue, list_slope, list_intercept = [], [], []
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        tmp_len_1 = output_list[idx_video,1]
        tmp_len_2 = output_list[idx_video,4]
        tmp_len_3 = output_list[idx_video,7]
        tmp_vel = np.nanmean(input_list[idx_video,:,2],1)
        idx_nans_1 = np.where(~np.isnan(tmp_len_1))[0]
        idx_nans_2 = np.where(~np.isnan(tmp_len_2))[0]
        idx_nans_3 = np.where(~np.isnan(tmp_len_3))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_nans_1], tmp_len_1[idx_nans_1])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_nans_2], tmp_len_2[idx_nans_2])
        lin_reg_3 = scipy.stats.linregress(tmp_vel[idx_nans_3], tmp_len_3[idx_nans_3])
        idx_video_od = np.where(video_list_od==video_id)[0]
        tmp_len_1_od = output_list_od[idx_video_od,1]
        tmp_len_2_od = output_list_od[idx_video_od,4]
        tmp_len_3_od = output_list_od[idx_video_od,7]
        tmp_vel_od = np.nanmean(input_list_od[idx_video_od,:,2],1)
        idx_nans_1_od = np.where(~np.isnan(tmp_len_1_od))[0]
        idx_nans_2_od = np.where(~np.isnan(tmp_len_2_od))[0]
        idx_nans_3_od = np.where(~np.isnan(tmp_len_3_od))[0]
        lin_reg_1_od = scipy.stats.linregress(tmp_vel_od[idx_nans_1_od], tmp_len_1_od[idx_nans_1_od])
        lin_reg_2_od = scipy.stats.linregress(tmp_vel_od[idx_nans_2_od], tmp_len_2_od[idx_nans_2_od])
        lin_reg_3_od = scipy.stats.linregress(tmp_vel_od[idx_nans_3_od], tmp_len_3_od[idx_nans_3_od])
        list_pvalue.append(np.array([lin_reg_1.pvalue, lin_reg_2.pvalue, lin_reg_3.pvalue, lin_reg_1_od.pvalue, lin_reg_2_od.pvalue, lin_reg_3_od.pvalue]))
        list_slope.append(np.array([lin_reg_1.slope, lin_reg_2.slope, lin_reg_3.slope, lin_reg_1_od.slope, lin_reg_2_od.slope, lin_reg_3_od.slope]))
        list_intercept.append(np.array([lin_reg_1.intercept, lin_reg_2.intercept, lin_reg_3.intercept, lin_reg_1_od.intercept, lin_reg_2_od.intercept, lin_reg_3_od.intercept]))
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_3_pvalues.csv'),np.array(list_pvalue), delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_3_slopes.csv'),np.array(list_slope),delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,f'{figname}_3_intercepts.csv'),np.array(list_intercept),delimiter=',')
    

def plot_contact_variability(output_list, output_list_od, video_list, video_list_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the contacts variability and save the associated statistics
    """
    list_var, list_var_od = [], []
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        idx_video_od = np.where(video_list_od==video_id)[0]
        tmp_len_1 = output_list[idx_video,0]
        tmp_len_2 = output_list[idx_video,3]
        tmp_len_3 = output_list[idx_video,6]
        tmp_wid_1 = output_list[idx_video,1]
        tmp_wid_2 = output_list[idx_video,4]
        tmp_wid_3 = output_list[idx_video,7]
        tmp_len_1_od = output_list_od[idx_video_od,0]
        tmp_len_2_od = output_list_od[idx_video_od,3]
        tmp_len_3_od = output_list_od[idx_video_od,6]
        tmp_wid_1_od = output_list_od[idx_video_od,1]
        tmp_wid_2_od = output_list_od[idx_video_od,4]
        tmp_wid_3_od = output_list_od[idx_video_od,7]
        list_var.append(np.array([scipy.stats.iqr(tmp_len_2,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_len_1,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_len_3,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_wid_2,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_wid_1,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_wid_3,nan_policy='omit') ]))
        list_var_od.append(np.array([scipy.stats.iqr(tmp_len_1_od,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_len_3_od,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_len_2_od,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_wid_1_od,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_wid_3_od,nan_policy='omit'),
                                    scipy.stats.iqr(tmp_wid_2_od,nan_policy='omit') ]))
    array_var, array_var_od = np.array(list_var), np.array(list_var_od)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar([0,0.5,1.5,2,3,3.5], np.nanmedian(array_var,0), width=0.4, yerr=scipy.stats.iqr(array_var,0,nan_policy='omit'),color=[[115/255,147/255,179/255],[115/255,147/255,179/255],'g','g','m','m'], ecolor='k')
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    
    #Save the associated statistics
    np.savetxt(os.path.join(OUTPUT_STATS,'stats_diag_contact_dimensions.csv'), array_var, delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,'stats_odiag_contact_dimensions.csv'),array_var_od, delimiter=',')

def plot_variability_contact(tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, video_list, video_list_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the fly variability of foot contact for an exemplar animal
    """
    idx_video = np.where(video_list==1)[0]
    rnd_vid = np.random.choice(len(idx_video),len(idx_video))
    tmp_len_1 = tot_output_list[idx_video[rnd_vid],0]
    tmp_len_2 = tot_output_list[idx_video[rnd_vid],3]
    tmp_len_3 = tot_output_list[idx_video[rnd_vid],6]
    tmp_wid_1 = tot_output_list[idx_video[rnd_vid],1]
    tmp_wid_2 = tot_output_list[idx_video[rnd_vid],4]
    tmp_wid_3 = tot_output_list[idx_video[rnd_vid],7]
    tmp_vel = np.nanmean(tot_input_list[idx_video[rnd_vid],:,2],1)

    idx_to_keep1 = np.where((tmp_len_1>-32))[0]
    idx_to_keep2 = np.where((tmp_len_2>-0))[0]
    idx_to_keep3 = np.where((tmp_len_3>-62))[0]
    idx_nans_1 = np.where((~np.isnan(tmp_wid_1[idx_to_keep1])))
    idx_nans_2 = np.where((~np.isnan(tmp_wid_2[idx_to_keep2])))
    idx_nans_3 = np.where((~np.isnan(tmp_wid_3[idx_to_keep3])))
    lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1]], tmp_wid_1[idx_to_keep1[idx_nans_1]])
    lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2]], tmp_wid_2[idx_to_keep2[idx_nans_2]])
    lin_reg_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3]], tmp_wid_3[idx_to_keep3[idx_nans_3]])
    x_axis = np.linspace(200,700)
    fig, axs = plt.subplots(1,3,figsize=(6,3),sharex=True, sharey=True)
    axs[0].set_xlabel('Velocity'), axs[1].set_xlabel('Velocity'), axs[2].set_xlabel('Velocity')
    axs[1].set_ylabel('Step width')
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].spines[['top','right']].set_visible(False)
    axs[2].spines[['top','right']].set_visible(False)
    axs[1].scatter(tmp_vel[idx_to_keep1], tmp_wid_1[idx_to_keep1], color='g', s=10,alpha=0.5)
    axs[0].scatter(tmp_vel[idx_to_keep2], tmp_wid_2[idx_to_keep2], color=[115/225,147/225,179/255],s=10,alpha=0.5)
    axs[2].scatter(tmp_vel[idx_to_keep3], tmp_wid_3[idx_to_keep3], color='m', s=10,alpha=0.5)
    axs[1].plot(x_axis, lin_reg_1.slope * x_axis + lin_reg_1.intercept,color='k',lw=2)
    axs[0].plot(x_axis, lin_reg_2.slope * x_axis + lin_reg_2.intercept,color='k',lw=2)
    axs[2].plot(x_axis, lin_reg_3.slope * x_axis + lin_reg_3.intercept,color='k',lw=2)
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Get the statistics for the above
    list_pvalue, list_slope = [], []
    for video_id in range(1,9):
        idx_video = np.where(video_list==video_id)[0]
        tmp_len_1 = tot_output_list[idx_video,0]
        tmp_len_2 = tot_output_list[idx_video,3]
        tmp_len_3 = tot_output_list[idx_video,6]
        tmp_vel = np.nanmean(tot_input_list[idx_video,:,2],1)
        idx_to_keep1 = np.where((tmp_len_1>-32))[0]
        idx_to_keep2 = np.where((tmp_len_2>-0))[0]
        idx_to_keep3 = np.where((tmp_len_3>-62))[0]
        idx_nans_1 = np.where(~np.isnan(tmp_len_1[idx_to_keep1]))[0]
        idx_nans_2 = np.where(~np.isnan(tmp_len_2[idx_to_keep2]))[0]
        idx_nans_3 = np.where(~np.isnan(tmp_len_3[idx_to_keep3]))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep1[idx_nans_1]], tmp_len_1[idx_to_keep1[idx_nans_1]])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep2[idx_nans_2]], tmp_len_2[idx_to_keep2[idx_nans_2]])
        lin_reg_3 = scipy.stats.linregress(tmp_vel[idx_to_keep3[idx_nans_3]], tmp_len_3[idx_to_keep3[idx_nans_3]])
        idx_video_od = np.where(video_list_od==video_id)[0]
        tmp_len_1_od = tot_output_list_od[idx_video_od,0]
        tmp_len_2_od = tot_output_list_od[idx_video_od,3]
        tmp_len_3_od = tot_output_list_od[idx_video_od,6]
        tmp_vel_od = np.nanmean(tot_input_list_od[idx_video_od,:,2],1)
        idx_to_keep1_od = np.where((tmp_len_1_od>-0))[0]
        idx_to_keep2_od = np.where((tmp_len_2_od>-62))[0]
        idx_to_keep3_od = np.where((tmp_len_3_od>-32))[0]
        idx_nans_1_od = np.where(~np.isnan(tmp_len_1_od[idx_to_keep1_od]))[0]
        idx_nans_2_od = np.where(~np.isnan(tmp_len_2_od[idx_to_keep2_od]))[0]
        idx_nans_3_od = np.where(~np.isnan(tmp_len_3_od[idx_to_keep3_od]))[0]
        lin_reg_1_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep1_od[idx_nans_1_od]], tmp_len_1_od[idx_to_keep1_od[idx_nans_1_od]])
        lin_reg_2_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep2_od[idx_nans_2_od]], tmp_len_2_od[idx_to_keep2_od[idx_nans_2_od]])
        lin_reg_3_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep3_od[idx_nans_3_od]], tmp_len_3_od[idx_to_keep3_od[idx_nans_3_od]])
        list_pvalue.append(np.array([lin_reg_1.pvalue, lin_reg_2.pvalue, lin_reg_3.pvalue, lin_reg_1_od.pvalue, lin_reg_2_od.pvalue, lin_reg_3_od.pvalue]))
        list_slope.append(np.array([lin_reg_1.slope, lin_reg_2.slope, lin_reg_3.slope, lin_reg_1_od.slope, lin_reg_2_od.slope, lin_reg_3_od.slope]))
    np.savetxt(os.path.join(OUTPUT_STATS,'pvalues_vel_length.csv'),np.array(list_pvalue), delimiter=',')
    np.savetxt(os.path.join(OUTPUT_STATS,'slopes_vel_length.csv'),np.array(list_slope),delimiter=',')


def plot_contribution_variance_humans(bool_show=False, bool_save=False, figname=None):
    """
    Plots the contribution of individual models to the total variance
    """
    path_data = OUTPUT_STATS
    ffwd_rsquare = np.loadtxt(os.path.join(path_data,'rsquares_ffwd.csv'),delimiter=',')
    fb_rsquare_front_foreaft = np.loadtxt(os.path.join(path_data,'rsquares_fb_front_foreaft.csv'),delimiter=',')
    fb_rsquare_front_lateral = np.loadtxt(os.path.join(path_data,'rsquares_fb_front_lateral.csv'),delimiter=',')
    fb_rsquare_central_foreaft = np.loadtxt(os.path.join(path_data,'rsquares_fb_central_foreaft.csv'),delimiter=',')
    fb_rsquare_central_lateral = np.loadtxt(os.path.join(path_data,'rsquares_fb_central_lateral.csv'),delimiter=',')
    fb_rsquare_central_foreaft = np.loadtxt(os.path.join(path_data,'rsquares_fb_hind_foreaft.csv'),delimiter=',')
    fb_rsquare_central_lateral = np.loadtxt(os.path.join(path_data,'rsquares_fb_hind_lateral.csv'),delimiter=',')
    ffwd_means = np.nanmean(ffwd_rsquare[:,[2,3,0,1,4,5]],0)
    fb_means = np.array([np.nanmean(fb_rsquare_front_foreaft[:,15],0)*(1-ffwd_means[0]),
                         np.nanmean(fb_rsquare_front_lateral[:,15],0)*(1-ffwd_means[1]),
                         np.nanmean(fb_rsquare_central_foreaft[:,15],0)*(1-ffwd_means[2]),
                         np.nanmean(fb_rsquare_central_lateral[:,15],0)*(1-ffwd_means[3]),
                         np.nanmean(fb_rsquare_central_foreaft[:,15],0)*(1-ffwd_means[4]),
                         np.nanmean(fb_rsquare_central_lateral[:,15],0)*(1-ffwd_means[5])])
    positions = [0,0.4,1.1,1.5,2.2,2.6]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    width = 0.2
    axs.bar(positions, ffwd_means, width=width,color='k',edgecolor=None)
    axs.bar(positions, fb_means, width=width, bottom=ffwd_means,color=[0.7,0.7,0.7],edgecolor=None)
    axs.set_ylabel('Explained variance')
    axs.set_xticks(positions)
    axs.set_xlim([-0.5,3.6])
    axs.set_ylim([-0.05,1.05])
    axs.plot([positions[0]-width/2, positions[-1]+width/2], [1,1], color='k', lw=2, ls=':')
    axs.set_yticks([0,0.25,0.5,0.75,1])
    axs.set_yticklabels(['0','25','50','75','100'])
    axs.set_xticklabels(['Foreaft','Lateral','Foreaft','Lateral','Foreaft','Lateral'],rotation=90)
    plt.tight_layout()

    if bool_save:
        fig.savefig(os.path.join(OUTPUT_STATS,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_STATS,f'{figname}.svg'),bbox_inches='tight')

    if bool_show:
        plt.show()