import matplotlib.pyplot as plt
import numpy as np 
import os, sys
import warnings
import pandas as pd
from matplotlib import patches
import copy
import scipy
import scipy.optimize
from scipy.optimize import curve_fit
from sklearn import linear_model
warnings.filterwarnings('ignore')
PATH_FIGURES = os.path.join(os.getcwd(), 'mice_results', 'figures')
PATH_STATS = os.path.join(os.getcwd(), 'mice_results', 'stats')
list_nopartial = [r'$py_{tail}$',r'$px_{tail}$',r'$vy_{tail}$',r'$vx_{tail}$',r'$py_{nose}$',r'$px_{nose}$',r'$vy_{nose}$',r'$vx_{nose}$']

def plot_regression_gains(regression_matrix, bool_front=True, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the figure representing the feedback gains
    """
    pos1 = np.arange(8)
    fig, axs = plt.subplots(1,2,figsize=(10,3),sharex=True)
    fig.suptitle('Front limbs',fontweight='bold')
    axs[0].boxplot((regression_matrix[:,17,2,0,:-1,0]), positions=pos1, widths=0.25, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False, whis=(5,95), autorange=False)
    axs[1].boxplot((regression_matrix[:,17,2,1,:-1,0]), positions=pos1, widths=0.25, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False, whis=(5,95), autorange=False)
    axs[0].axhline(0, color='k', lw=2, ls=':')
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].axhline(0, color='k', lw=2, ls=':')
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].set_title('Fore-aft position',fontweight='bold')
    axs[1].set_title('Lateral position',fontweight='bold')
    axs[0].set_ylabel('Slope')
    axs[0].set_xlabel('Predictor')
    axs[1].set_xlabel('Predictor')
    axs[0].set_xticks(np.arange(8))
    axs[0].set_xticklabels(list_nopartial)
    axs[0].text(0.2, np.nanmedian(regression_matrix[:,17,2,0,0,0]),'*',fontsize=14)
    axs[0].text(4.2, np.nanmedian(regression_matrix[:,17,2,0,4,0]),'*',fontsize=14)
    axs[0].text(6.2, np.nanmedian(regression_matrix[:,17,2,0,6,0]),'*',fontsize=14)

    axs[1].text(5.2, np.nanmedian(regression_matrix[:,17,2,1,5,0]),'*',fontsize=14)
    axs[1].text(7.2, np.nanmedian(regression_matrix[:,17,2,1,7,0]),'*',fontsize=14)
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Additional figures for the individual gains ... 
    if bool_front:
        fig, axs = plt.subplots(1,3,figsize=(10,3),sharex=True)
        axs[0].plot(np.nanmedian(regression_matrix[:,:,2,0,0,0],0),color='k')
        axs[1].plot(np.nanmedian(regression_matrix[:,:,2,0,4,0],0),color='k')
        axs[2].plot(np.nanmedian(regression_matrix[:,:,2,0,6,0],0),color='k')
        axs[0].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,0,0,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,0,0,0],0,rng=(25,75))/2, np.nanmedian(regression_matrix[:,:,2,0,0,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,0,0,0],0,rng=(25,75))/2,color='k',alpha=0.5)
        axs[1].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,0,4,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,0,4,0],0,rng=(25,75))/2, np.nanmedian(regression_matrix[:,:,2,0,4,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,0,4,0],0,rng=(25,75))/2,color='k',alpha=0.5)
        axs[2].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,0,6,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,0,6,0],0,rng=(25,75))/2, np.nanmedian(regression_matrix[:,:,2,0,6,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,0,6,0],0,rng=(25,75))/2,color='k',alpha=0.5)
        axs[0].spines[['top','right']].set_visible(False)
        axs[1].spines[['top','right']].set_visible(False)
        axs[2].spines[['top','right']].set_visible(False)
        axs[0].set_xlabel('Relative gait fraction')
        axs[1].set_xlabel('Relative gait fraction')
        axs[2].set_xlabel('Relative gait fraction')
        axs[0].set_xticks([10,12.5,15,17.5,20])
        axs[0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
        axs[0].set_xlim([10,21])
        axs[0].set_ylabel('Slope')
        axs[0].set_title('fore-aft tail position')
        axs[1].set_title('fore-aft nose position')
        axs[2].set_title('fore-aft nose velocity')
        axs[0].axhline(0,color='k',lw=2,ls=':')
        axs[1].axhline(0,color='k',lw=2,ls=':')
        axs[2].axhline(0,color='k',lw=2,ls=':')
        axs[0].axvline(17,color='k',lw=2,ls=':')
        axs[1].axvline(17,color='k',lw=2,ls=':')
        axs[2].axvline(17,color='k',lw=2,ls=':')
        fig.suptitle('Fore-aft position of the front limbs predicted by...')
        if bool_plot:
            plt.show()
        if bool_save:
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_front_for_{figname}.png'),bbox_inches='tight')
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_front_for_{figname}.svg'),bbox_inches='tight')

        # Compute the statistics for the figures above
        pval_fig00, pval_fig01, pval_fig02 = np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1))
        for time in range(21):
            tmp_stats00 = scipy.stats.wilcoxon(regression_matrix[:,time,2,0,0,0],alternative='greater')
            tmp_stats01 = scipy.stats.wilcoxon(regression_matrix[:,time,2,0,4,0],alternative='greater')
            tmp_stats02 = scipy.stats.wilcoxon(regression_matrix[:,time,2,0,6,0],alternative='greater')
            pval_fig00[time], pval_fig01[time], pval_fig02[time] = tmp_stats00.pvalue, tmp_stats01.pvalue, tmp_stats02.pvalue 
        ptot = np.concatenate((pval_fig00, pval_fig01, pval_fig02),1)
        np.savetxt(os.path.join(PATH_STATS,f'gains_front_for{figname}.csv'),ptot,delimiter=',')



        fig, axs = plt.subplots(1,2,figsize=(6.66,3),sharex=True)
        axs[0].spines[['top','right']].set_visible(False)
        axs[1].spines[['top','right']].set_visible(False)
        axs[0].plot(np.nanmedian(regression_matrix[:,:,2,1,5,0],0),color='k')
        axs[1].plot(np.nanmedian(regression_matrix[:,:,2,1,7,0],0),color='k')
        axs[0].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,1,5,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,1,5,0],0)/2,np.nanmedian(regression_matrix[:,:,2,1,5,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,1,5,0],0)/2,color='k',alpha=0.5)
        axs[1].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,1,7,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,1,7,0],0)/2,np.nanmedian(regression_matrix[:,:,2,1,7,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,1,7,0],0)/2,color='k',alpha=0.5)
        axs[0].axhline(0,color='k',lw=2,ls=':')
        axs[1].axhline(0,color='k',lw=2,ls=':')
        axs[0].axvline(17,color='k',lw=2,ls=':')
        axs[1].axvline(17,color='k',lw=2,ls=':')
        axs[0].set_xlabel('Relative gait fraction')
        axs[1].set_xlabel('Relative gait fraction')
        axs[0].set_xticks([10,12.5,15,17.5,20])
        axs[0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
        axs[0].set_ylabel('Slope')
        axs[0].set_title('lateral nose position')
        axs[1].set_title('lateral nose velocity')
        axs[0].set_xlim([10,21])
        fig.suptitle('Lateral position of the front limbs predicted by ...')
        if bool_plot:
            plt.show()
        if bool_save:
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_front_lat_{figname}.png'),bbox_inches='tight')
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_front_lat_{figname}.svg'),bbox_inches='tight')

        # Compute the statistics for the figures above
        pval_fig00, pval_fig01 = np.zeros((21,1)), np.zeros((21,1))
        for time in range(21):
            tmp_stats00 = scipy.stats.wilcoxon(regression_matrix[:,time,2,1,5,0],alternative='greater')
            tmp_stats01 = scipy.stats.wilcoxon(regression_matrix[:,time,2,1,7,0],alternative='greater')
            pval_fig00[time], pval_fig01[time], pval_fig02[time] = tmp_stats00.pvalue, tmp_stats01.pvalue, tmp_stats02.pvalue 
        ptot = np.concatenate((pval_fig00, pval_fig01, pval_fig02),1)
        np.savetxt(os.path.join(PATH_STATS,f'gains_front_lat{figname}.csv'),ptot,delimiter=',')

    else:
        fig, axs = plt.subplots(1,2,figsize=(6.66,3),sharex=True)
        axs[0].plot(np.nanmedian(regression_matrix[:,:,2,0,0,0],0),color='k')
        axs[1].plot(np.nanmedian(regression_matrix[:,:,2,0,4,0],0),color='k')
        axs[0].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,0,0,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,0,0,0],0,rng=(25,75))/2, np.nanmedian(regression_matrix[:,:,2,0,0,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,0,0,0],0,rng=(25,75))/2,color='k',alpha=0.5)
        axs[1].fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,0,4,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,0,4,0],0,rng=(25,75))/2, np.nanmedian(regression_matrix[:,:,2,0,4,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,0,4,0],0,rng=(25,75))/2,color='k',alpha=0.5)
        axs[0].spines[['top','right']].set_visible(False)
        axs[1].spines[['top','right']].set_visible(False)
        axs[0].set_xlabel('Relative gait fraction')
        axs[1].set_xlabel('Relative gait fraction')
        axs[0].set_xticks([10,12.5,15,17.5,20])
        axs[0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
        axs[0].set_xlim([10, 21])
        axs[0].set_ylabel('Slope')
        axs[0].set_title('fore-aft tail position')
        axs[1].set_title('fore-aft nose position')
        axs[0].axhline(0,color='k',lw=2,ls=':')
        axs[1].axhline(0,color='k',lw=2,ls=':')
        axs[0].axvline(17,color='k',lw=2,ls=':')
        axs[1].axvline(17,color='k',lw=2,ls=':')
        fig.suptitle('Fore-aft position of the hind limbs predicted by...')
        if bool_plot:
            plt.show()
        if bool_save:
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_hind_for_{figname}.png'),bbox_inches='tight')
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_hind_for_{figname}.svg'),bbox_inches='tight')

        # Compute the statistics for the figures above
        pval_fig00, pval_fig01 = np.zeros((21,1)), np.zeros((21,1))
        for time in range(21):
            tmp_stats00 = scipy.stats.wilcoxon(regression_matrix[:,time,2,0,0,0],alternative='greater')
            tmp_stats01 = scipy.stats.wilcoxon(regression_matrix[:,time,2,0,4,0],alternative='greater')
            pval_fig00[time], pval_fig01[time] = tmp_stats00.pvalue, tmp_stats01.pvalue
        ptot = np.concatenate((pval_fig00, pval_fig01),1)
        np.savetxt(os.path.join(PATH_STATS,f'gains_hind_for{figname}.csv'),ptot,delimiter=',')

        fig, axs = plt.subplots(1,1,figsize=(3,3),sharex=True)
        axs.spines[['top','right']].set_visible(False)
        axs.plot(np.nanmedian(regression_matrix[:,:,2,1,1,0],0),color='k')
        axs.fill_between(np.arange(21),np.nanmedian(regression_matrix[:,:,2,1,1,0],0)+scipy.stats.iqr(regression_matrix[:,:,2,1,1,0],0)/2,np.nanmedian(regression_matrix[:,:,2,1,1,0],0)-scipy.stats.iqr(regression_matrix[:,:,2,1,1,0],0)/2,color='k',alpha=0.5)
        axs.axhline(0,color='k',lw=2,ls=':')
        axs.axvline(17,color='k',lw=2,ls=':')
        axs.set_xlabel('Relative gait fraction')
        axs.set_xticks([10,12.5,15,17.5,20])
        axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
        axs.set_xlim([10,21])
        axs.set_ylabel('Slope')
        axs.set_title('lateral tail position')
        fig.suptitle('Lateral position of the hind limbs predicted by ...')
        if bool_plot:
            plt.show()
        if bool_save:
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_hind_lat_{figname}.png'),bbox_inches='tight')
            fig.savefig(os.path.join(PATH_FIGURES,f'gains_hind_lat_{figname}.svg'),bbox_inches='tight')

            # Compute the statistics for the figures above
        pval_fig00 = np.zeros((21,1))
        for time in range(21):
            tmp_stats00 = scipy.stats.wilcoxon(regression_matrix[:,time,2,1,1,0],alternative='greater')
            pval_fig00[time] = tmp_stats00.pvalue
        ptot = pval_fig00
        np.savetxt(os.path.join(PATH_STATS,f'gains_hind_lat{figname}.csv'),ptot,delimiter=',')


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
    # Foreaft tail
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_list[tot_vel_bin==0,:,0],0) - np.nanmean(input_list[tot_vel_bin==0,-1,0]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_list[tot_vel_bin==1,:,0],0) - np.nanmean(input_list[tot_vel_bin==1,-1,0]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_list[tot_vel_bin==2,:,0],0) - np.nanmean(input_list[tot_vel_bin==2,-1,0]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.set_ylim([-100,0])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==0,:,0],0) - np.nanmean(input_list[tot_vel_bin==0,-1,0]) - scipy.stats.iqr(input_list[tot_vel_bin==0,:,0],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==0,:,0],0) - np.nanmean(input_list[tot_vel_bin==0,-1,0]) + scipy.stats.iqr(input_list[tot_vel_bin==0,:,0],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==1,:,0],0) - np.nanmean(input_list[tot_vel_bin==1,-1,0]) - scipy.stats.iqr(input_list[tot_vel_bin==1,:,0],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==1,:,0],0) - np.nanmean(input_list[tot_vel_bin==1,-1,0]) + scipy.stats.iqr(input_list[tot_vel_bin==1,:,0],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==2,:,0],0) - np.nanmean(input_list[tot_vel_bin==2,-1,0]) - scipy.stats.iqr(input_list[tot_vel_bin==2,:,0],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==2,:,0],0) - np.nanmean(input_list[tot_vel_bin==2,-1,0]) + scipy.stats.iqr(input_list[tot_vel_bin==2,:,0],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_tail_body.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_tail_body.svg'),bbox_inches='tight')
    # Foreaft nose
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_list[tot_vel_bin==0,:,4],0) - np.nanmean(input_list[tot_vel_bin==0,-1,4]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_list[tot_vel_bin==1,:,4],0) - np.nanmean(input_list[tot_vel_bin==1,-1,4]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_list[tot_vel_bin==2,:,4],0) - np.nanmean(input_list[tot_vel_bin==2,-1,4]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.set_ylim([-100,0])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==0,:,4],0) - np.nanmean(input_list[tot_vel_bin==0,-1,4]) - scipy.stats.iqr(input_list[tot_vel_bin==0,:,4],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==0,:,4],0) - np.nanmean(input_list[tot_vel_bin==0,-1,4]) + scipy.stats.iqr(input_list[tot_vel_bin==0,:,4],0,nan_policy='omit')/2,color='k',alpha=0.2)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==1,:,4],0) - np.nanmean(input_list[tot_vel_bin==1,-1,4]) - scipy.stats.iqr(input_list[tot_vel_bin==1,:,4],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==1,:,4],0) - np.nanmean(input_list[tot_vel_bin==1,-1,4]) + scipy.stats.iqr(input_list[tot_vel_bin==1,:,4],0,nan_policy='omit')/2,color='k',alpha=0.5)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==2,:,4],0) - np.nanmean(input_list[tot_vel_bin==2,-1,4]) - scipy.stats.iqr(input_list[tot_vel_bin==2,:,4],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==2,:,4],0) - np.nanmean(input_list[tot_vel_bin==2,-1,4]) + scipy.stats.iqr(input_list[tot_vel_bin==2,:,4],0,nan_policy='omit')/2,color='k',alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_nose_body.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_nose_body.svg'),bbox_inches='tight')

    #Stats for the part above ... 
    pval_list1 = []
    idx_nans1 = np.where(~np.isnan(input_list[tot_vel_bin==0,0,0]))[0]
    idx_nans2 = np.where(~np.isnan(input_list[tot_vel_bin==1,0,0]))[0]
    idx_nans3 = np.where(~np.isnan(input_list[tot_vel_bin==2,0,0]))[0]
    for time in range(21):
        anova = scipy.stats.f_oneway(input_list[tot_vel_bin==0,time,4][idx_nans1], input_list[tot_vel_bin==1,time,4][idx_nans2], input_list[tot_vel_bin==2,time,4][idx_nans3])
        pval_list1.append(anova.pvalue)
    print(pval_list1)
    # Lateral tail
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_list[tot_vel_bin==0,:,1],0) - np.nanmean(input_list[tot_vel_bin==0,:,1]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_list[tot_vel_bin==1,:,1],0) - np.nanmean(input_list[tot_vel_bin==1,:,1]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_list[tot_vel_bin==2,:,1],0) - np.nanmean(input_list[tot_vel_bin==2,:,1]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==0,:,1],0) - np.nanmean(input_list[tot_vel_bin==0,:,1]) - scipy.stats.iqr(input_list[tot_vel_bin==0,:,1],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==0,:,1],0) - np.nanmean(input_list[tot_vel_bin==0,:,1]) + scipy.stats.iqr(input_list[tot_vel_bin==0,:,1],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==1,:,1],0) - np.nanmean(input_list[tot_vel_bin==1,:,1]) - scipy.stats.iqr(input_list[tot_vel_bin==1,:,1],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==1,:,1],0) - np.nanmean(input_list[tot_vel_bin==1,:,1]) + scipy.stats.iqr(input_list[tot_vel_bin==1,:,1],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==2,:,1],0) - np.nanmean(input_list[tot_vel_bin==2,:,1]) - scipy.stats.iqr(input_list[tot_vel_bin==2,:,1],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==2,:,1],0) - np.nanmean(input_list[tot_vel_bin==2,:,1]) + scipy.stats.iqr(input_list[tot_vel_bin==2,:,1],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_tail_body.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_tail_body.svg'),bbox_inches='tight')
    # Lateral nose
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_list[tot_vel_bin==0,:,5],0) - np.nanmean(input_list[tot_vel_bin==0,:,5]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_list[tot_vel_bin==1,:,5],0) - np.nanmean(input_list[tot_vel_bin==1,:,5]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_list[tot_vel_bin==2,:,5],0) - np.nanmean(input_list[tot_vel_bin==2,:,5]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==0,:,5],0) - np.nanmean(input_list[tot_vel_bin==0,:,5]) - scipy.stats.iqr(input_list[tot_vel_bin==0,:,5],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==0,:,5],0) - np.nanmean(input_list[tot_vel_bin==0,:,5]) + scipy.stats.iqr(input_list[tot_vel_bin==0,:,5],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==1,:,5],0) - np.nanmean(input_list[tot_vel_bin==1,:,5]) - scipy.stats.iqr(input_list[tot_vel_bin==1,:,5],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==1,:,5],0) - np.nanmean(input_list[tot_vel_bin==1,:,5]) + scipy.stats.iqr(input_list[tot_vel_bin==1,:,5],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_list.shape[1]), np.nanmean(input_list[tot_vel_bin==2,:,5],0) - np.nanmean(input_list[tot_vel_bin==2,:,5]) - scipy.stats.iqr(input_list[tot_vel_bin==2,:,5],0,nan_policy='omit')/2, np.nanmean(input_list[tot_vel_bin==2,:,5],0) - np.nanmean(input_list[tot_vel_bin==2,:,5]) + scipy.stats.iqr(input_list[tot_vel_bin==2,:,5],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_nose_body.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_nose_body.svg'),bbox_inches='tight')
    
    
    pval_list1 = []
    idx_nans1 = np.where(~np.isnan(input_list[tot_vel_bin==0,0,0]))[0]
    idx_nans2 = np.where(~np.isnan(input_list[tot_vel_bin==1,0,0]))[0]
    idx_nans3 = np.where(~np.isnan(input_list[tot_vel_bin==2,0,0]))[0]
    for time in range(21):
        anova = scipy.stats.f_oneway(input_list[tot_vel_bin==0,time,5][idx_nans1], input_list[tot_vel_bin==1,time,5][idx_nans2], input_list[tot_vel_bin==2,time,5][idx_nans3])
        pval_list1.append(anova.pvalue)
    print(pval_list1)
    # Foreaft self - leg 1 
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,0],0) - np.nanmean(input_self[tot_vel_bin==0,-1,0]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,0],0) - np.nanmean(input_self[tot_vel_bin==1,-1,0]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,0],0) - np.nanmean(input_self[tot_vel_bin==2,-1,0]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.set_ylim([-120,0])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,0],0) - np.nanmean(input_self[tot_vel_bin==0,-1,0]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,0],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==0,:,0],0) - np.nanmean(input_self[tot_vel_bin==0,-1,0]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,0],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,0],0) - np.nanmean(input_self[tot_vel_bin==1,-1,0]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,0],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==1,:,0],0) - np.nanmean(input_self[tot_vel_bin==1,-1,0]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,0],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,0],0) - np.nanmean(input_self[tot_vel_bin==2,-1,0]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,0],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==2,:,0],0) - np.nanmean(input_self[tot_vel_bin==2,-1,0]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,0],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_leg1_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_leg1_self.svg'),bbox_inches='tight')

    # Foreaft self - leg 2 
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,4],0) - np.nanmean(input_self[tot_vel_bin==0,-1,4]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,4],0) - np.nanmean(input_self[tot_vel_bin==1,-1,4]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,4],0) - np.nanmean(input_self[tot_vel_bin==2,-1,4]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.set_ylim([-120,0])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,4],0) - np.nanmean(input_self[tot_vel_bin==0,-1,4]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,4],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==0,:,4],0) - np.nanmean(input_self[tot_vel_bin==0,-1,4]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,4],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,4],0) - np.nanmean(input_self[tot_vel_bin==1,-1,4]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,4],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==1,:,4],0) - np.nanmean(input_self[tot_vel_bin==1,-1,4]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,4],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,4],0) - np.nanmean(input_self[tot_vel_bin==2,-1,4]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,4],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==2,:,4],0) - np.nanmean(input_self[tot_vel_bin==2,-1,4]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,4],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_leg1_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_leg1_self.svg'),bbox_inches='tight')

    # Lateral self - leg 1 
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,1],0) - np.nanmean(input_self[tot_vel_bin==0,:,1]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,1],0) - np.nanmean(input_self[tot_vel_bin==1,:,1]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,1],0) - np.nanmean(input_self[tot_vel_bin==2,:,1]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,1],0) - np.nanmean(input_self[tot_vel_bin==0,:,1]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,1],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==0,:,1],0) - np.nanmean(input_self[tot_vel_bin==0,:,1]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,1],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,1],0) - np.nanmean(input_self[tot_vel_bin==1,:,1]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,1],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==1,:,1],0) - np.nanmean(input_self[tot_vel_bin==1,:,1]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,1],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,1],0) - np.nanmean(input_self[tot_vel_bin==2,:,1]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,1],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==2,:,1],0) - np.nanmean(input_self[tot_vel_bin==2,:,1]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,1],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_leg2_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_foreaft_leg2_self.svg'),bbox_inches='tight')

    # Lateral self - leg 2 
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_self[tot_vel_bin==0,:,5],0) - np.nanmean(input_self[tot_vel_bin==0,:,5]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(input_self[tot_vel_bin==1,:,5],0) - np.nanmean(input_self[tot_vel_bin==1,:,5]),'k',lw=1.25,label='intermediate')
    axs.plot(np.nanmean(input_self[tot_vel_bin==2,:,5],0) - np.nanmean(input_self[tot_vel_bin==2,:,5]),'k',lw=2,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    if bool_fb:
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==0,:,5],0) - np.nanmean(input_self[tot_vel_bin==0,:,5]) - scipy.stats.iqr(input_self[tot_vel_bin==0,:,5],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==0,:,5],0) - np.nanmean(input_self[tot_vel_bin==0,:,5]) + scipy.stats.iqr(input_self[tot_vel_bin==0,:,5],0,nan_policy='omit')/2, color='k', alpha=0.2)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==1,:,5],0) - np.nanmean(input_self[tot_vel_bin==1,:,5]) - scipy.stats.iqr(input_self[tot_vel_bin==1,:,5],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==1,:,5],0) - np.nanmean(input_self[tot_vel_bin==1,:,5]) + scipy.stats.iqr(input_self[tot_vel_bin==1,:,5],0,nan_policy='omit')/2, color='k', alpha=0.5)
        axs.fill_between(np.arange(input_self.shape[1]), np.nanmean(input_self[tot_vel_bin==2,:,5],0) - np.nanmean(input_self[tot_vel_bin==2,:,5]) - scipy.stats.iqr(input_self[tot_vel_bin==2,:,5],0,nan_policy='omit')/2, np.nanmean(input_self[tot_vel_bin==2,:,5],0) - np.nanmean(input_self[tot_vel_bin==2,:,5]) + scipy.stats.iqr(input_self[tot_vel_bin==2,:,5],0,nan_policy='omit')/2, color='k', alpha=0.7)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_leg2_self.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_lateral_leg2_self.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def plot_feedforward_input(velocity_vector, velocity_vector_od, tot_input_list, tot_input_list_od, tot_animal, tot_animal_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the impact of the feedforward controller on the input variables
    """
    # FIGURE 1 - Plotting the foreaft and lateral positions of the tail and nose markers
    # This is done for an exemplar animal
    idx_animal = 0
    tmp_vel = velocity_vector[(idx_animal==tot_animal).flatten()]
    tmp_input = tot_input_list[(idx_animal==tot_animal).flatten(),:,:]

    fig, axs = plt.subplots(1,1,figsize=(3,2))
    axs.spines[['top','right']].set_visible(False)
    for line in range(tmp_input.shape[0]):
        axs.plot(tmp_input[line,:,1],'k',lw=0.5)
    plt.tight_layout()
    plt.show()

    vel_pctl = np.nanpercentile(tmp_vel,[5,33,66,95])
    idx_vel_1 = np.where((tmp_vel<vel_pctl[1]) & (tmp_vel>vel_pctl[0]))[0]
    idx_vel_2 = np.where((tmp_vel<vel_pctl[2]) & (tmp_vel>vel_pctl[1]))[0]
    idx_vel_3 = np.where((tmp_vel<vel_pctl[3]) & (tmp_vel>vel_pctl[2]))[0]
    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(tmp_input[idx_vel_1,:,5],0),'k',lw=1)
    axs.plot(np.nanmean(tmp_input[idx_vel_2,:,5],0),'k',lw=2)
    axs.plot(np.nanmean(tmp_input[idx_vel_3,:,5],0),'k',lw=3)
    plt.tight_layout()
    plt.show()

def plot_feedforward_body_mice_final(tot_input_list, tot_animal, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the impact of the velocity on the head position for the mice data
    """
    n_animals = 80
    list_pvalues, list_slopes = [], []
    for idx_animal in range(n_animals):
        tot_input = tot_input_list[(idx_animal==tot_animal).flatten(),:,:]
        tot_velocity = np.nanmean(tot_input[:,:,6],1)

        amplitude_x = tot_input[:,-1,4] - tot_input[:,10,4]
        amplitude_y = np.nanmax(tot_input[:,:,5],1) - np.nanmin(tot_input[:,:,5],1)
        idx_nans = np.where((~np.isnan(amplitude_y)) & (tot_velocity<700) & (amplitude_x>100))[0]
        reg_ampl_x = scipy.stats.linregress(tot_velocity[idx_nans], amplitude_x[idx_nans])
        reg_ampl_y = scipy.stats.linregress(tot_velocity[idx_nans], amplitude_y[idx_nans])

        list_slopes.append(np.array([reg_ampl_x.slope, reg_ampl_y.slope]))
        list_pvalues.append(np.array([reg_ampl_x.pvalue, reg_ampl_y.pvalue]))

    mat_slopes = np.array(list_slopes)
    mat_pvalues = np.array(list_pvalues)
    print(f'{len(np.where(mat_pvalues[:,0]<0.005)[0])}/{mat_pvalues.shape[0]}')
    print(f'{len(np.where(mat_pvalues[:,1]<0.005)[0])}/{mat_pvalues.shape[0]}')
    local_input = tot_input_list[(tot_animal==0).flatten(),:,:]
    local_velocity = np.nanmean(local_input[:,:,6],1)
    local_amplitude_x = local_input[:,-1,4] - local_input[:,10,4]
    local_amplitude_y = np.nanmax(local_input[:,:,5],1) - np.nanmin(local_input[:,:,5],1)
    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity, local_amplitude_x, s=10, color='k', label='Foreaft - animal 0',alpha=0.5)
    axs.scatter(local_velocity, local_amplitude_y, s=10, color='r', label='Lateral',alpha=0.5)
    axs.set_xlim([50,550]), axs.set_xticks([100,500])
    axs.set_ylim([0,300])
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_sample_amplitude.svg'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_sample_amplitude.png'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, amplitude_x, color='r', alpha=0.5)
    axs.scatter(tot_velocity, amplitude_y, color='b', alpha=0.5)
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_amplitude.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_amplitude.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[0,1], width=0.2, height=np.nanmean(mat_slopes,0), yerr=np.nanstd(mat_slopes,0),color='b')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slopes.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slopes.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()

def plot_feedforward_control_hind(velocity_vector, tot_output_list, tot_animal, bool_plot=False, bool_save=False, figname=None):
    """
    Plos the relationship between movement velocity and foot placement for the hind paw as reference
    """
    n_animals = 80

    tot_output = tot_output_list[(tot_animal==0).flatten(),:]
    tot_velocity = velocity_vector[(tot_animal==0).flatten()]
    
    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,0], color='m', alpha=0.5, s=10)
    axs.scatter(tot_velocity, tot_output[:,3], color='c', s=10, alpha=0.5)
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Step length')
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_length_hind.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_length_hind.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,1], color='m', alpha=0.5, s=10)
    axs.scatter(tot_velocity, tot_output[:,4], color='c', s=10, alpha=0.5)
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Step length')
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_width_hind.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_width_hind.svg'),bbox_inches='tight')


def exp_func(x,A,B,C,D):
    return A*np.exp(-B*(x-C))+D


def plot_feedforward_controller_final(velocity_vector, tot_output_list, tot_animal, tot_input_abs, tot_output_abs, tot_animal_abs, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the relationship between movement velocity and foot placement
    """
    n_animals = 80
    points_per_animal = np.zeros((n_animals,))
    for animal in range(n_animals):
        idx_animal = np.where(tot_animal.flatten()==animal)[0]
        points_per_animal[animal] = len(idx_animal)
    idx_plot = 0#np.argmax(points_per_animal)-1
    # Stats for the step lengths & step widths 
    list_pval, list_slopes = [], []
    for idx_animal in range(n_animals):
        tmp_vel = velocity_vector[(idx_animal==tot_animal).flatten()]
        tmp_len_1 = tot_output_list[(idx_animal==tot_animal).flatten(),0]
        tmp_len_2 = tot_output_list[(idx_animal==tot_animal).flatten(),3]
        tmp_wid_1 = tot_output_list[(idx_animal==tot_animal).flatten(),1]
        tmp_wid_2 = tot_output_list[(idx_animal==tot_animal).flatten(),4]
        idx_to_keep = np.where((tmp_vel<700) & (tmp_len_1>0) & (tmp_len_1<70) & (tmp_len_2>-30) & (tmp_len_2<0))[0]
        idx_nan_1 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_1[idx_to_keep])))[0]
        idx_nan_2 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_2[idx_to_keep])))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_len_1[idx_to_keep[idx_nan_1]])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_len_2[idx_to_keep[idx_nan_2]])
        lin_wid_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_wid_1[idx_to_keep[idx_nan_1]])
        lin_wid_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_wid_2[idx_to_keep[idx_nan_2]])
        list_pval.append(np.array([lin_reg_1.pvalue,lin_reg_2.pvalue, lin_wid_1.pvalue, lin_wid_2.pvalue]))
        list_slopes.append(np.array([lin_reg_1.slope,lin_reg_2.slope, lin_wid_1.slope, lin_wid_2.slope]))

    mat_slopes = np.array(list_slopes)
    mat_pval = np.array(list_pval)
    str_len_front = f'{len(np.where(mat_pval[:,0]<0.05)[0])}/{n_animals}'
    str_len_hind = f'{len(np.where(mat_pval[:,1]<0.05)[0])}/{n_animals}'
    str_wid_front = f'{len(np.where(mat_pval[:,2]<0.05)[0])}/{n_animals}'
    str_wid_hind = f'{len(np.where(mat_pval[:,3]<0.05)[0])}/{n_animals}'
    np.savetxt('AAA_slopes.csv',mat_slopes)
    
    # Stats for the slopes of the mice
    print(np.nanmedian(mat_slopes,0))
    print(scipy.stats.iqr(mat_slopes,0))
    wilcox_slopes = scipy.stats.wilcoxon(mat_slopes[:,0],mat_slopes[:,1])
    print(wilcox_slopes)

    velocity_vector_abs = np.nanmean(tot_input_abs[:,:,2],1)

    idx_nans_tot = np.where((~np.isnan(velocity_vector)) & (~np.isnan(tot_output_abs[:,2])))[0]
    popt1, _ = curve_fit(exp_func,velocity_vector[idx_nans_tot], tot_output_abs[idx_nans_tot,2], bounds=([0,0,0,0],[10,1,200,1]))
    popt2, _ = curve_fit(exp_func,velocity_vector[idx_nans_tot], tot_output_abs[idx_nans_tot,5], bounds=([0,0,0,0],[10,1,200,1]))



    tot_output = tot_output_list[(tot_animal==idx_plot).flatten(),:]
    tot_velocity = velocity_vector[(tot_animal==idx_plot).flatten()]
    tot_output_abs_list = tot_output_abs[(tot_animal_abs==idx_plot).flatten(),:]
    tot_velocity_abs = velocity_vector_abs[(tot_animal_abs==idx_plot).flatten()]
    print(tot_velocity_abs.shape, tot_velocity.shape)
    idx_no_outlier = np.where((tot_velocity<700) & (tot_output[:,0]>0) & (tot_output[:,0]<70) & (tot_output[:,3]>-30) & (tot_output[:,3]<0))[0]
    idx_no_outlier_abs = np.where((tot_velocity_abs<700) & (tot_output_abs_list[:,0]>0) & (tot_output_abs_list[:,0]<70)  & (tot_output_abs_list[:,2]>0.05) & (tot_output_abs_list[:,5]>0.05))[0]
    
    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    xinput = np.linspace(200,500)
    idx_nans = np.where((~np.isnan(tot_velocity[idx_no_outlier])) & (~np.isnan(tot_output_abs_list[idx_no_outlier,2])) & (~np.isnan(tot_output_abs[idx_no_outlier,5])))[0]
    popt1_local, _ = curve_fit(exp_func, tot_velocity[idx_no_outlier[idx_nans]], tot_output_abs_list[idx_no_outlier[idx_nans],2], bounds=(0.9*popt1,1.1*popt1))
    yplot1 = exp_func(xinput, *popt1_local)
    popt2_local, _ = curve_fit(exp_func, tot_velocity[idx_no_outlier[idx_nans]], tot_output_abs_list[idx_no_outlier[idx_nans],2], bounds=(0.9*popt2,1.1*popt2))
    yplot2 = exp_func(xinput, *popt2_local)
    axs.scatter(tot_velocity_abs[idx_no_outlier_abs],tot_output_abs_list[idx_no_outlier_abs,2],color='m',s=10, alpha=0.5)
    axs.scatter(tot_velocity_abs[idx_no_outlier_abs],tot_output_abs_list[idx_no_outlier_abs,5],color='c',s=10, alpha=0.5)
    x_1 = tot_velocity_abs[idx_no_outlier_abs]
    x_2 = tot_output_abs_list[idx_no_outlier_abs,2]
    x_3 = tot_output_abs_list[idx_no_outlier_abs,5]
    df = pd.DataFrame({'x':x_1, 'y1':x_2, 'y2':x_3})
    df.to_csv('aaaaaa.csv')
    axs.plot(xinput, yplot1,'m',lw=3)
    axs.plot(xinput, yplot2,'c',lw=3)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    axs.set_ylim([0,0.45]), axs.set_xlim([50,750])
    axs.set_xticks([100,700])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_duration.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_duration.svg'),bbox_inches='tight')

    


    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity[idx_no_outlier],tot_output[idx_no_outlier,0],color='m',s=10, alpha=0.5)
    axs.scatter(tot_velocity[idx_no_outlier],tot_output[idx_no_outlier,3],color='c',s=10, alpha=0.5)
    local_lin1 = scipy.stats.linregress(tot_velocity[idx_no_outlier], tot_output[idx_no_outlier,0])
    local_lin2 = scipy.stats.linregress(tot_velocity[idx_no_outlier], tot_output[idx_no_outlier,3])
    xinput = np.linspace(200,500)
    print(local_lin1.rvalue**2, local_lin2.rvalue**2)
    print(local_lin1.pvalue, local_lin2.pvalue)
    axs.plot(xinput, local_lin1.slope*xinput + local_lin1.intercept,'m',lw=2)
    axs.plot(xinput, local_lin2.slope*xinput + local_lin2.intercept,'c',lw=2)
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Step length')
    axs.set_xlim([50,550]), axs.set_ylim([-35,75])
    axs.set_xticks([100,500])
    axs.set_yticks([-20,0,20,40,60]), axs.set_yticklabels(['-20','0','20','40','60'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_expemplar_length.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_expemplar_length.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity[idx_no_outlier],tot_output[idx_no_outlier,1],color='m',s=10, alpha=0.5)
    axs.scatter(tot_velocity[idx_no_outlier],tot_output[idx_no_outlier,4],color='c',s=10, alpha=0.5)
    local_lin1 = scipy.stats.linregress(tot_velocity[idx_no_outlier], tot_output[idx_no_outlier,1])
    local_lin2 = scipy.stats.linregress(tot_velocity[idx_no_outlier], tot_output[idx_no_outlier,4])
    xinput = np.linspace(200,500)
    print(local_lin1.rvalue**2, local_lin2.rvalue**2)
    print(local_lin1.pvalue, local_lin2.pvalue)
    axs.plot(xinput, local_lin1.slope*xinput + local_lin1.intercept,'m',lw=2)
    axs.plot(xinput, local_lin2.slope*xinput + local_lin2.intercept,'c',lw=2)
    axs.set_xlim([50,550]), axs.set_ylim([-40,40])
    axs.set_xticks([100,500])
    axs.set_yticks([-40,-20,0,20,40]), axs.set_yticklabels(['-40','-20','0','20','40'])
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Step width')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_expemplar_width.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_expemplar_width.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[0,0.33,0.66,1],height=[np.nanmean(mat_slopes[:,0]), np.nanmean(mat_slopes[:,1]), np.nanmean(mat_slopes[:,2]), np.nanmean(mat_slopes[:,3])], width=0.2,
            yerr=[np.nanstd(mat_slopes[:,0]), np.nanstd(mat_slopes[:,1]), np.nanstd(mat_slopes[:,2]), np.nanstd(mat_slopes[:,3])], color=['m','m','c','c'])
    axs.set_ylabel('Slope')
    axs.text(0.2, 0.03, str_len_front)
    axs.text(0.4, 0.03, str_len_hind)
    axs.text(0.6, 0.03, str_wid_front)
    axs.text(0.8, 0.03, str_wid_hind)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_final_ffwd.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_final_ffwd.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def plot_feedforward_controller(velocity_vector, velocity_vector_od, tot_output_list,tot_output_list_od, tot_animal, tot_animal_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the feedforward beahviors
    """

    # FIGURE 1 - Variances in the foreaft and lateral direction
    list_var, list_var_od = [], []
    for idx_animal in range(80):
        tmp_len_1 = tot_output_list[(idx_animal==tot_animal).flatten(),0]
        tmp_len_2 = tot_output_list[(idx_animal==tot_animal).flatten(),3]
        tmp_wid_1 = tot_output_list[(idx_animal==tot_animal).flatten(),1]
        tmp_wid_2 = tot_output_list[(idx_animal==tot_animal).flatten(),4]
        tmp_len_1_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),0]
        tmp_len_2_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),3]
        tmp_wid_1_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),1]
        tmp_wid_2_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),4]
        list_var.append([scipy.stats.iqr(tmp_len_1,nan_policy='omit'), scipy.stats.iqr(tmp_len_2,nan_policy='omit'), scipy.stats.iqr(tmp_len_1_od,nan_policy='omit'), scipy.stats.iqr(tmp_len_2_od,nan_policy='omit')])
        list_var_od.append([scipy.stats.iqr(tmp_wid_1,nan_policy='omit'), scipy.stats.iqr(tmp_wid_2,nan_policy='omit'), scipy.stats.iqr(tmp_wid_1_od,nan_policy='omit'), scipy.stats.iqr(tmp_wid_2_od,nan_policy='omit')])
    array_var, array_var_od = np.array(list_var), np.array(list_var_od)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar([0,0.5,1.5,2], np.nanmedian(array_var,0), width=0.4, yerr=scipy.stats.iqr(array_var,0,nan_policy='omit'), color=['m','m','c','c'], ecolor='k')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_1.svg'),bbox_inches='tight')
    np.savetxt(os.path.join(PATH_STATS, f'{figname}_diag_1.csv'), array_var, delimiter=',')
    np.savetxt(os.path.join(PATH_STATS, f'{figname}_odiag_1.csv'), array_var_od, delimiter=',')

    # FIGURE 2 - Step lengths as a function of velocity

    idx_animal = 55
    tmp_vel = velocity_vector[(idx_animal==tot_animal).flatten()]
    tmp_len_1 = tot_output_list[(idx_animal==tot_animal).flatten(),0]
    tmp_len_2 = tot_output_list[(idx_animal==tot_animal).flatten(),3]
    idx_to_keep = np.where((tmp_vel<700) & (tmp_len_1>0) & (tmp_len_1<70) & (tmp_len_2>-30) & (tmp_len_2<0))[0]
    idx_nan_1 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & (~np.isnan(tmp_len_1[idx_to_keep]))))[0]
    idx_nan_2 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & (~np.isnan(tmp_len_2[idx_to_keep]))))[0]
    idx_nan_1_ = np.where((~np.isnan(tmp_vel[idx_to_keep]) & (~np.isnan(tmp_wid_1[idx_to_keep]))))[0]
    idx_nan_2_ = np.where((~np.isnan(tmp_vel[idx_to_keep]) & (~np.isnan(tmp_wid_2[idx_to_keep]))))[0]
    lin_reg_len_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_len_1[idx_to_keep[idx_nan_1]])
    lin_reg_len_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_len_2[idx_to_keep[idx_nan_2]])
    lin_reg_wid_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1_]], tmp_wid_1[idx_to_keep[idx_nan_1_]])
    lin_reg_wid_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2_]], tmp_wid_2[idx_to_keep[idx_nan_2_]])
    x_axis = np.linspace(150,600)
    y_pred_len_1 = lin_reg_len_1.slope * x_axis + lin_reg_len_1.intercept
    y_pred_len_2 = lin_reg_len_2.slope * x_axis + lin_reg_len_2.intercept
    y_pred_wid_1 = lin_reg_wid_1.slope * x_axis + lin_reg_wid_1.intercept
    y_pred_wid_2 = lin_reg_wid_2.slope * x_axis + lin_reg_wid_2.intercept

    idx_an_abs = np.where((tot_animal_abs==idx_animal))[0]
    local_vel = np.nanmean(tot_input_abs[idx_an_abs, :,2],1)
    local_output_abs = tot_output_abs[idx_an_abs,:]
    fig, axs = plt.sublots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_vel, local_output_abs[:,2],color='m', alpha=0.5)
    axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    plt.tight_layout()
    


    fig, axs = plt.subplots(1,2,figsize=(5,3), sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].set_xlabel('Velocity'), axs[0].set_ylabel('Step width')
    axs[1].set_xlabel('Velocity'), axs[1].set_ylabel('Step width')
    axs[0].set_xlim([0,700])
    axs[0].set_ylim([-30,75]), axs[1].set_ylim([-30,75])
    axs[0].scatter(tmp_vel[idx_to_keep], tmp_len_1[idx_to_keep], color='m',alpha=0.5)
    axs[1].scatter(tmp_vel[idx_to_keep], tmp_len_2[idx_to_keep], color='c',alpha=0.5)
    axs[0].plot(x_axis, y_pred_len_1, color='k', lw=3)
    axs[1].plot(x_axis, y_pred_len_2, color='k', lw=3)

    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_2.svg'),bbox_inches='tight')

    # FIGURE 3 - Step widths as a function of velocity

    fig, axs = plt.subplots(1,2,figsize=(5,3), sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].set_xlabel('Velocity'), axs[0].set_ylabel('Step length')
    axs[1].set_xlabel('Velocity'), axs[1].set_ylabel('Step length')
    axs[0].set_xlim([0,700])
    axs[0].set_ylim([-50,50]), axs[1].set_ylim([-50,50])
    axs[0].scatter(tmp_vel[idx_to_keep], tmp_wid_1[idx_to_keep], color='m',alpha=0.5)
    axs[1].scatter(tmp_vel[idx_to_keep], tmp_wid_2[idx_to_keep], color='c',alpha=0.5)
    axs[0].plot(x_axis, y_pred_wid_1, color='k', lw=3)
    axs[1].plot(x_axis, y_pred_wid_2, color='k', lw=3)
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_3.svg'),bbox_inches='tight')


    # Stats for the step lengths & step widths 
    list_pval, list_slopes, list_intercept = [], [], []
    for idx_animal in range(80):
        tmp_vel = velocity_vector[(idx_animal==tot_animal).flatten()]
        tmp_len_1 = tot_output_list[(idx_animal==tot_animal).flatten(),0]
        tmp_len_2 = tot_output_list[(idx_animal==tot_animal).flatten(),3]
        tmp_vel_od = velocity_vector_od[(idx_animal==tot_animal_od).flatten()]
        tmp_len_1_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),0]
        tmp_len_2_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),3]
        idx_to_keep = np.where((tmp_vel<700) & (tmp_len_1>0) & (tmp_len_1<70) & (tmp_len_2>-30) & (tmp_len_2<0))[0]
        idx_to_keep_od = np.where((tmp_vel_od<700) & (tmp_len_1_od>0) & (tmp_len_1_od<70) & (tmp_len_2_od>-30) & (tmp_len_2_od<0))[0]
        idx_nan_1 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_1[idx_to_keep])))[0]
        idx_nan_2 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_2[idx_to_keep])))[0]
        idx_nan_1_od = np.where((~np.isnan(tmp_vel_od[idx_to_keep_od]) & ~np.isnan(tmp_len_1_od[idx_to_keep_od])))[0]
        idx_nan_2_od = np.where((~np.isnan(tmp_vel_od[idx_to_keep_od]) & ~np.isnan(tmp_len_2_od[idx_to_keep_od])))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_len_1[idx_to_keep[idx_nan_1]])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_len_2[idx_to_keep[idx_nan_2]])
        lin_reg_1_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep_od[idx_nan_1_od]], tmp_len_1_od[idx_to_keep_od[idx_nan_1_od]])
        lin_reg_2_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep_od[idx_nan_2_od]], tmp_len_2_od[idx_to_keep_od[idx_nan_2_od]])
        list_pval.append(np.array([lin_reg_1.pvalue,lin_reg_2.pvalue,lin_reg_1_od.pvalue, lin_reg_2_od.pvalue]))
        list_slopes.append(np.array([lin_reg_1.slope,lin_reg_2.slope,lin_reg_1_od.slope, lin_reg_2_od.slope]))
        list_intercept.append(np.array([lin_reg_1.intercept, lin_reg_2.intercept, lin_reg_1_od.intercept, lin_reg_2_od.intercept]))
    np.savetxt(os.path.join(PATH_STATS,f'{figname}_slopes_2.csv'), np.array(list_slopes), delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,f'{figname}_pvalues_2.csv'), np.array(list_pval), delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,f'{figname}_intercept_2.csv'), np.array(list_intercept), delimiter=',')

    list_pval, list_intercept, list_slopes = [], [], []
    for idx_animal in range(80):
        tmp_vel = velocity_vector[(idx_animal==tot_animal).flatten()]
        tmp_len_1 = tot_output_list[(idx_animal==tot_animal).flatten(),1]
        tmp_len_2 = tot_output_list[(idx_animal==tot_animal).flatten(),4]
        tmp_vel_od = velocity_vector_od[(idx_animal==tot_animal_od).flatten()]
        tmp_len_1_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),1]
        tmp_len_2_od = tot_output_list_od[(idx_animal==tot_animal_od).flatten(),4]
        idx_to_keep = np.where((tmp_vel<700))[0]
        idx_to_keep_od = np.where((tmp_vel_od<700))[0]
        idx_nan_1 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_1[idx_to_keep])))[0]
        idx_nan_2 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_2[idx_to_keep])))[0]
        idx_nan_1_od = np.where((~np.isnan(tmp_vel_od[idx_to_keep_od]) & ~np.isnan(tmp_len_1_od[idx_to_keep_od])))[0]
        idx_nan_2_od = np.where((~np.isnan(tmp_vel_od[idx_to_keep_od]) & ~np.isnan(tmp_len_2_od[idx_to_keep_od])))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_len_1[idx_to_keep[idx_nan_1]])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_len_2[idx_to_keep[idx_nan_2]])
        lin_reg_1_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep_od[idx_nan_1_od]], tmp_len_1_od[idx_to_keep_od[idx_nan_1_od]])
        lin_reg_2_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep_od[idx_nan_2_od]], tmp_len_2_od[idx_to_keep_od[idx_nan_2_od]])
        list_pval.append(np.array([lin_reg_1.pvalue,lin_reg_2.pvalue,lin_reg_1_od.pvalue, lin_reg_2_od.pvalue]))
        list_slopes.append(np.array([lin_reg_1.slope,lin_reg_2.slope,lin_reg_1_od.slope, lin_reg_2_od.slope]))
        list_intercept.append(np.array([lin_reg_1.intercept, lin_reg_2.intercept, lin_reg_1_od.intercept, lin_reg_2_od.intercept]))
    np.savetxt(os.path.join(PATH_STATS,f'{figname}_slopes_3.csv'), np.array(list_slopes), delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,f'{figname}_pvalues_3.csv'), np.array(list_pval), delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,f'{figname}_intercept_3.csv'),np.array(list_intercept))


def plot_contact_variability(output_list, output_list_od, animal_list, animal_list_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the contact variability and save the associated statistics
    """
    list_var, list_var_od = [], []
    for idx_animal in range(80):
        tmp_len_1 = output_list[(idx_animal==animal_list).flatten(),0]
        tmp_len_2 = output_list[(idx_animal==animal_list).flatten(),3]
        tmp_wid_1 = output_list[(idx_animal==animal_list).flatten(),1]
        tmp_wid_2 = output_list[(idx_animal==animal_list).flatten(),4]
        tmp_len_1_od = output_list_od[(idx_animal==animal_list_od).flatten(),0]
        tmp_len_2_od = output_list_od[(idx_animal==animal_list_od).flatten(),3]
        tmp_wid_1_od = output_list_od[(idx_animal==animal_list_od).flatten(),1]
        tmp_wid_2_od = output_list_od[(idx_animal==animal_list_od).flatten(),4]
        list_var.append(np.array([scipy.stats.iqr(tmp_len_1,nan_policy='omit'), scipy.stats.iqr(tmp_len_2,nan_policy='omit'), scipy.stats.iqr(tmp_wid_1,nan_policy='omit'), scipy.stats.iqr(tmp_wid_2,nan_policy='omit')]))
        list_var_od.append(np.array([scipy.stats.iqr(tmp_len_1_od,nan_policy='omit'), scipy.stats.iqr(tmp_len_2_od,nan_policy='omit'), scipy.stats.iqr(tmp_wid_1_od,nan_policy='omit'), scipy.stats.iqr(tmp_wid_2_od,nan_policy='omit')]))
    
    array_var, array_var_od = np.array(list_var), np.array(list_var_od)
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar([0,0.5,1.5,2], np.nanmedian(array_var,0), width=0.4, yerr=scipy.stats.iqr(array_var,0,nan_policy='omit'),color=['m','m','c','c'], ecolor='k')
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    np.savetxt(os.path.join(PATH_STATS,'individual_variances_diag_mouse.csv'),array_var, delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,'individual_variances_odiag_mouse.csv'),array_var_od, delimiter=',')

def plot_individual_contact(velocity_vector, output_list, animal_list, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the individual variabiabiliy in foot contact location for mouse data
    """
    idx_animal = 0 # np.random.choice(80,1)[0]
    tmp_vel = velocity_vector[(idx_animal==animal_list).flatten()]
    tmp_len_1 = output_list[(idx_animal==animal_list).flatten(),0]
    tmp_len_2 = output_list[(idx_animal==animal_list).flatten(),3]
    tmp_wid_1 = output_list[(idx_animal==animal_list).flatten(),1]
    tmp_wid_2 = output_list[(idx_animal==animal_list).flatten(),4]
    fig, axs = plt.subplots(1,1,figsize=(3,5))
    axs.scatter(tmp_wid_1, tmp_len_1, color='m', alpha=0.5, s=10)
    axs.scatter(tmp_wid_2, tmp_len_2, color='c', alpha=0.5, s=10)
    axs.scatter(0, 0, color='r', alpha=0.5, s=30)
    axs.spines[['top','right']].set_visible(False)
    axs.set_xlabel('Lateral direction')
    axs.set_xlim([-40,20])
    axs.set_ylim([-50,75])
    axs.set_ylabel('Foreaft direction')
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,'individual_contact_mouse.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,'individual_contact_mouse.svg'),bbox_inches='tight')
    

    

def plot_individual_variability(velocity_vector, velocity_vector_od, output_list, output_list_od, animal_list, animal_list_od, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the figure representing the individual variability
    """
    idx_animal = 55
    tmp_vel = velocity_vector[(idx_animal==animal_list).flatten()]
    tmp_len_1 = output_list[(idx_animal==animal_list).flatten(),0]
    tmp_len_2 = output_list[(idx_animal==animal_list).flatten(),3]
    idx_to_keep = np.where((tmp_vel<700) & (tmp_len_1>0) & (tmp_len_1<70) & (tmp_len_2>-30) & (tmp_len_2<0))[0]
    idx_nan_1 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_1[idx_to_keep])))[0]
    idx_nan_2 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_2[idx_to_keep])))[0]
    lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_len_1[idx_to_keep[idx_nan_1]])
    lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_len_2[idx_to_keep[idx_nan_2]])
    x_axis = np.linspace(150,600)
    y_pred_1 = lin_reg_1.slope * x_axis + lin_reg_1.intercept
    y_pred_2 = lin_reg_2.slope * x_axis + lin_reg_2.intercept

    fig, axs = plt.subplots(1,2,figsize=(5,3),sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].set_xlabel('Velocity'), axs[0].set_ylabel('Step length')
    axs[1].set_xlabel('Velocity'), axs[1].set_ylabel('Step length')
    axs[0].set_xlim([0,700])
    axs[0].set_ylim([-30,75]), axs[1].set_ylim([-30,75])
    axs[0].scatter(tmp_vel[idx_to_keep], tmp_len_1[idx_to_keep], color='m', alpha=0.5)
    axs[1].scatter(tmp_vel[idx_to_keep], tmp_len_2[idx_to_keep], color='c', alpha=0.5)
    axs[0].plot(x_axis, y_pred_1, color='k', lw=2)
    axs[1].plot(x_axis, y_pred_2, color='k', lw=2)
    
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Get the statistics associated with that metrics for all the individuals...
    list_pval, list_slopes = [], []
    for idx_animal in range(80):
        tmp_vel = velocity_vector[(idx_animal==animal_list).flatten()]
        tmp_len_1 = output_list[(idx_animal==animal_list).flatten(),0]
        tmp_len_2 = output_list[(idx_animal==animal_list).flatten(),3]
        tmp_vel_od = velocity_vector_od[(idx_animal==animal_list_od).flatten()]
        tmp_len_1_od = output_list_od[(idx_animal==animal_list_od).flatten(),0]
        tmp_len_2_od = output_list_od[(idx_animal==animal_list_od).flatten(),3]
        idx_to_keep = np.where((tmp_vel<700) & (tmp_len_1>0) & (tmp_len_1<70) & (tmp_len_2>-30) & (tmp_len_2<0))[0]
        idx_to_keep_od = np.where((tmp_vel_od<700) & (tmp_len_1_od>0) & (tmp_len_1_od<70) & (tmp_len_2_od>-30) & (tmp_len_2_od<0))[0]
        idx_nan_1 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_1[idx_to_keep])))[0]
        idx_nan_2 = np.where((~np.isnan(tmp_vel[idx_to_keep]) & ~np.isnan(tmp_len_2[idx_to_keep])))[0]
        idx_nan_1_od = np.where((~np.isnan(tmp_vel_od[idx_to_keep_od]) & ~np.isnan(tmp_len_1_od[idx_to_keep_od])))[0]
        idx_nan_2_od = np.where((~np.isnan(tmp_vel_od[idx_to_keep_od]) & ~np.isnan(tmp_len_2_od[idx_to_keep_od])))[0]
        lin_reg_1 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_1]], tmp_len_1[idx_to_keep[idx_nan_1]])
        lin_reg_2 = scipy.stats.linregress(tmp_vel[idx_to_keep[idx_nan_2]], tmp_len_2[idx_to_keep[idx_nan_2]])
        lin_reg_1_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep_od[idx_nan_1_od]], tmp_len_1_od[idx_to_keep_od[idx_nan_1_od]])
        lin_reg_2_od = scipy.stats.linregress(tmp_vel_od[idx_to_keep_od[idx_nan_2_od]], tmp_len_2_od[idx_to_keep_od[idx_nan_2_od]])
        list_pval.append(np.array([lin_reg_1.pvalue,lin_reg_2.pvalue,lin_reg_1_od.pvalue, lin_reg_2_od.pvalue]))
        list_slopes.append(np.array([lin_reg_1.slope,lin_reg_2.slope,lin_reg_1_od.slope, lin_reg_2_od.slope]))
    np.savetxt(os.path.join(PATH_STATS,'slopes_individual_contacts.csv'), np.array(list_slopes), delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,'pvalues_individual_contacts.csv'), np.array(list_pval), delimiter=',')

def plot_sample_laterality_mouse(tot_animal, tot_input_list, tot_output_list, time=17, bool_plot=False, bool_save=True, figname=None):
    """
    Plots a sample of the laterality specificity of the mouse data
    """

    idx_animal = 10
    idx_local = np.where(tot_animal[:,0]<idx_animal)[0]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.scatter(tot_input_list[idx_local,time,5] - np.nanmean(tot_input_list[idx_local,time,5],0),
                tot_output_list[idx_local,1] - np.nanmean(tot_output_list[idx_local,1],0), color='k', s=5, alpha=0.5)
    axs.set_xlabel('Pelvis lateral position')
    axs.set_ylabel('Lateral foot contact')
    axs.spines[['top','right']].set_visible(False)
    axs.axhline(0,color='k',lw=2,ls=':')
    axs.axvline(0,color='k',lw=2,ls=':')
    axs.set_xlim([-55,55])
    axs.set_ylim([-60,60])
    plt.tight_layout()
    if bool_plot:
        plt.show()

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')


def plot_contribution_variance_mice(bool_show=False, bool_save=False, figname=None):
    """
    Plots the contribution of individual models to the total variance
    """
    path_data = PATH_STATS
    ffwd_rsquare = np.loadtxt(os.path.join(path_data,'rsquares_ffwd.csv'),delimiter=',')
    fb_rsquare_front_foreaft = np.loadtxt(os.path.join(path_data,'rsquares_fb_front_foreaft.csv'),delimiter=',')
    fb_rsquare_front_lateral = np.loadtxt(os.path.join(path_data,'rsquares_fb_front_lateral.csv'),delimiter=',')
    fb_rsquare_hind_foreaft = np.loadtxt(os.path.join(path_data,'rsquares_fb_hind_foreaft.csv'),delimiter=',')
    fb_rsquare_hind_lateral = np.loadtxt(os.path.join(path_data,'rsquares_fb_hind_lateral.csv'),delimiter=',')

    ffwd_means = np.nanmean(ffwd_rsquare,0)
    fb_means = np.array([np.nanmean(fb_rsquare_front_foreaft[:,17],0)*(1-ffwd_means[0]),
                        np.nanmean(fb_rsquare_front_lateral[:,17],0)*(1-ffwd_means[1]),
                        np.nanmean(fb_rsquare_hind_foreaft[:,17],0)*(1-ffwd_means[2]),
                        np.nanmean(fb_rsquare_hind_lateral[:,17],0)*(1-ffwd_means[3])])
    positions = [0,0.4,1.1,1.5]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    width = 0.2
    axs.bar(positions, ffwd_means, width=width,color='k',edgecolor=None)
    axs.bar(positions, fb_means, width=width, bottom=ffwd_means,color=[0.7,0.7,0.7],edgecolor=None)
    axs.set_ylabel('Explained variance')
    axs.set_xticks([0,0.4,1.1,1.5])
    axs.set_xlim([-0.5,2])
    axs.set_ylim([-0.05,1.05])
    axs.plot([positions[0]-width/2, positions[-1]+width/2], [1,1], color='k', lw=2, ls=':')
    axs.set_yticks([0,0.25,0.5,0.75,1])
    axs.set_yticklabels(['0','25','50','75','100'])
    axs.set_xticklabels(['Foreaft','Lateral','Foreaft','Lateral'],rotation=90)
    plt.tight_layout()

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    if bool_show:
        plt.show()

def plot_rsquares_time(front_body, hind_body, front_self, hind_self, bool_plot=False, bool_save=False, figname=None):
    """
    Figures for the timing rsquares
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.arange(0,11), np.nanmedian(front_body[:,10:],0),'b',lw=2)
    axs.plot(np.arange(0,11), np.nanmedian(front_self[:80,10:],0),'r',lw=2)
    axs.fill_between(np.arange(11), np.nanmedian(front_body[:,10:],0)+scipy.stats.iqr(front_body[:,10:],0,nan_policy='omit')/2,np.nanmedian(front_body[:,10:],0)-scipy.stats.iqr(front_body[:,10:],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmedian(front_self[:80,10:],0)+scipy.stats.iqr(front_self[:80,10:],0,nan_policy='omit')/2,np.nanmedian(front_self[:80,10:],0)-scipy.stats.iqr(front_self[:80,10:],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_front.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_front.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.arange(0,11), np.nanmedian(hind_body[:,10:],0),'b',lw=2)
    axs.plot(np.arange(0,11), np.nanmedian(hind_self[:80,10:],0),'r',lw=2)
    axs.fill_between(np.arange(11), np.nanmedian(hind_body[:,10:],0)+scipy.stats.iqr(hind_body[:,10:],0,nan_policy='omit')/2,np.nanmedian(hind_body[:,10:],0)-scipy.stats.iqr(hind_body[:,10:],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmedian(hind_self[:80,10:],0)+scipy.stats.iqr(hind_self[:80,10:],0,nan_policy='omit')/2,np.nanmedian(hind_self[:80,10:],0)-scipy.stats.iqr(hind_self[:80,10:],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_hind.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_hind.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()


def plot_rsquares_final(front_foreaft_body, front_lateral_body, hind_foreaft_body, hind_lateral_body,
                        front_foreaft_self, front_lateral_self, hind_foreaft_self, hind_lateral_self,
                        bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final version of the rsquares for the mouse dataset
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.arange(0,11), np.nanmedian(front_foreaft_body[:,10:],0),'b',lw=2)
    axs.plot(np.arange(0,11), np.nanmedian(front_foreaft_self[:80,10:],0),'r',lw=2)
    axs.fill_between(np.arange(11), np.nanmedian(front_foreaft_body[:,10:],0)+scipy.stats.iqr(front_foreaft_body[:,10:],0,nan_policy='omit')/2,np.nanmedian(front_foreaft_body[:,10:],0)-scipy.stats.iqr(front_foreaft_body[:,10:],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmedian(front_foreaft_self[:80,10:],0)+scipy.stats.iqr(front_foreaft_self[:80,10:],0,nan_policy='omit')/2,np.nanmedian(front_foreaft_self[:80,10:],0)-scipy.stats.iqr(front_foreaft_self[:80,10:],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_front_foreaft.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_front_foreaft.svg'),bbox_inches='tight')

    list_pvalue_top_left = []
    time_vector = np.arange(11)
    for ii in range(len(time_vector)):
        stat_test = scipy.stats.wilcoxon(front_foreaft_body[:80,10+ii], front_foreaft_self[:80,10+ii], alternative='greater')
        list_pvalue_top_left.append(stat_test.pvalue)
    print(np.array(list_pvalue_top_left)<0.05)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.set_xlabel('Relative gait fraction')
    axs.plot(np.arange(0,11), np.nanmedian(front_lateral_body[:,10:],0),'b',lw=2)
    axs.plot(np.arange(0,11), np.nanmedian(front_lateral_self[:80,10:],0),'r',lw=2)
    axs.fill_between(np.arange(11), np.nanmedian(front_lateral_body[:,10:],0)+scipy.stats.iqr(front_lateral_body[:,10:],0,nan_policy='omit')/2,np.nanmedian(front_lateral_body[:,10:],0)-scipy.stats.iqr(front_lateral_body[:,10:],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmedian(front_lateral_self[:80,10:],0)+scipy.stats.iqr(front_lateral_self[:80,10:],0,nan_policy='omit')/2,np.nanmedian(front_lateral_self[:80,10:],0)-scipy.stats.iqr(front_lateral_self[:80,10:],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_front_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_front_lateral.svg'),bbox_inches='tight')

    list_pvalue_top_right = []
    time_vector = np.arange(11)
    for ii in range(len(time_vector)):
        stat_test = scipy.stats.wilcoxon(front_lateral_body[:80,10+ii], front_lateral_self[:80,10+ii], alternative='greater')
        list_pvalue_top_right.append(stat_test.pvalue)

    print(np.array(list_pvalue_top_right)<0.05)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.arange(0,11), np.nanmedian(hind_foreaft_body[:,10:],0),'b',lw=2)
    axs.plot(np.arange(0,11), np.nanmedian(hind_foreaft_self[:80,10:],0),'r',lw=2)
    axs.fill_between(np.arange(11), np.nanmedian(hind_foreaft_body[:,10:],0)+scipy.stats.iqr(hind_foreaft_body[:,10:],0,nan_policy='omit')/2,np.nanmedian(hind_foreaft_body[:,10:],0)-scipy.stats.iqr(hind_foreaft_body[:,10:],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmedian(hind_foreaft_self[:80,10:],0)+scipy.stats.iqr(hind_foreaft_self[:80,10:],0,nan_policy='omit')/2,np.nanmedian(hind_foreaft_self[:80,10:],0)-scipy.stats.iqr(hind_foreaft_self[:80,10:],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_hind_foreaft.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_hind_foreaft.svg'),bbox_inches='tight')

    list_pvalue_bottom_left = []
    time_vector = np.arange(11)
    for ii in range(len(time_vector)):
        stat_test = scipy.stats.wilcoxon(hind_foreaft_body[:80,10+ii], hind_foreaft_self[:80,10+ii], alternative='greater')
        list_pvalue_bottom_left.append(stat_test.pvalue)

    print(np.array(list_pvalue_bottom_left)<0.05)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.arange(0,11), np.nanmedian(hind_lateral_body[:,10:],0),'b',lw=2)
    axs.plot(np.arange(0,11), np.nanmedian(hind_lateral_self[:80,10:],0),'r',lw=2)
    axs.fill_between(np.arange(11), np.nanmedian(hind_lateral_body[:,10:],0)+scipy.stats.iqr(hind_lateral_body[:,10:],0,nan_policy='omit')/2,np.nanmedian(hind_lateral_body[:,10:],0)-scipy.stats.iqr(hind_lateral_body[:,10:],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmedian(hind_lateral_self[:80,10:],0)+scipy.stats.iqr(hind_lateral_self[:80,10:],0,nan_policy='omit')/2,np.nanmedian(hind_lateral_self[:80,10:],0)-scipy.stats.iqr(hind_lateral_self[:80,10:],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_hind_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_hind_lateral.svg'),bbox_inches='tight')

    list_pvalue_bottom_right = []
    time_vector = np.arange(11)
    for ii in range(len(time_vector)):
        stat_test = scipy.stats.wilcoxon(hind_lateral_body[:80,10+ii], hind_lateral_self[:80,10+ii], alternative='greater')
        list_pvalue_bottom_right.append(stat_test.pvalue)

    print(np.array(list_pvalue_bottom_right)<0.05)

    

    if bool_plot:
        plt.show()

def plot_histograms_total_variance(input_variance, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the histograms for the total variance captured
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[1,1.3,1.6,1.9], height=np.nanmean(input_variance[:,0,:],0), width=0.2, color='b')
    axs.bar(x=[1,1.3,1.6,1.9], height=np.nanmean(input_variance[:,1,:],0), bottom=np.nanmean(input_variance[:,0,:],0), width=0.2, color='r')
    axs.set_xlim([0.9,3.9])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def plot_normalized_feedback_gains(regression_matrix, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final normalized feedback gains for the lateral direction
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(2, np.nanmean(regression_matrix[:,17,2,1,5,0],0), color='k', s=20)
    axs.plot([2,2], [np.nanmean(regression_matrix[:,17,2,1,5,0],0)+np.nanstd(regression_matrix[:,17,2,1,5,0],0),np.nanmean(regression_matrix[:,17,2,1,5,0],0)-np.nanstd(regression_matrix[:,17,2,1,5,0],0)],color='k',lw=2)
    axs.set_xlim([0,4])
    axs.set_ylim([0,3])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()


def  plot_scatter_laterality(tot_animal, tot_input_list, tot_output_list, idx_leg, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the scatter plots for the laterality question ... 
    """
    n_animals = 80
    idx_animal_plot = np.random.choice(n_animals,1)[0]
    local_idx = np.where((tot_animal[:,0]!=134))[0]
    idx_nan = np.where(~np.isnan(tot_input_list[local_idx,0,0]))[0]
    local_input = tot_input_list[local_idx[idx_nan],:,8*idx_leg:8*idx_leg+8]
    local_output = tot_output_list[local_idx[idx_nan],3*idx_leg:3*idx_leg+3]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_input[:,17,5]-np.nanmean(local_input[:,17,5]),
                local_output[:,1]-np.nanmean(local_output[:,1]),color='k',s=5,alpha=0.05)
    tmp_inp = local_input[:,17,5] - np.nanmean(local_input[:,17,5])
    tmp_out = local_output[:,1] - np.nanmean(local_output[:,1])
    idx_pos = np.where(tmp_inp>0)[0]
    idx_neg = np.where(tmp_inp<0)[0]
    reg_neg = scipy.stats.linregress(tmp_inp[idx_neg],tmp_out[idx_neg])
    reg_pos = scipy.stats.linregress(tmp_inp[idx_pos],tmp_out[idx_pos])
    xpos, xneg = np.linspace(0,50), np.linspace(-50,0)
    axs.plot(xpos, xpos*reg_pos.slope + reg_pos.intercept,color='r',lw=2)
    axs.plot(xneg, xneg*reg_neg.slope + reg_neg.intercept,color='b',lw=2)
    axs.set_xlim([-50,50])
    axs.set_ylim([-40,40])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')
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
    Plots the laterality information for the mouse datasets 
    """

    diff_gains = regression_small[:,10:,1,5,0] - regression_large[:,10:,1,5,0]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(diff_gains,0),color='k')
    axs.fill_between(np.arange(11), np.nanmean(diff_gains,0)+scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,np.nanmean(diff_gains,0)-scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,color='k', alpha=0.3)
    axs.plot(np.nanmean(regression_small[:,10:,1,5,0],0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,10:,1,5,0],0),'b',lw=2)
    axs.fill_between(np.arange(11), np.nanmean(regression_small[:,10:,1,5,0],0)+scipy.stats.iqr(regression_small[:,10:,1,5,0],0,nan_policy='omit')/2,np.nanmean(regression_small[:,10:,1,5,0],0)-scipy.stats.iqr(regression_small[:,10:,1,5,0],0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmean(regression_large[:,10:,1,5,0],0)+scipy.stats.iqr(regression_large[:,10:,1,5,0],0,nan_policy='omit')/2,np.nanmean(regression_large[:,10:,1,5,0],0)-scipy.stats.iqr(regression_large[:,10:,1,5,0],0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.set_ylim([-1,1])
    axs.set_xticks([0,2.5,5,7.5,10])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.axhline(0,color='k',lw=2,ls=':')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_gains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_gains.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(regression_small[:,10:,1,5,2]**2,0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,10:,1,5,2]**2,0),'b',lw=2)
    axs.fill_between(np.arange(11), np.nanmean(regression_small[:,10:,1,5,2]**2,0)+scipy.stats.iqr(regression_small[:,10:,1,5,2]**2,0,nan_policy='omit')/2,np.nanmean(regression_small[:,10:,1,5,2]**2,0)-scipy.stats.iqr(regression_small[:,10:,1,5,2]**2,0,nan_policy='omit')/2,color='r',alpha=0.5)
    axs.fill_between(np.arange(11), np.nanmean(regression_large[:,10:,1,5,2]**2,0)+scipy.stats.iqr(regression_large[:,10:,1,5,2]**2,0,nan_policy='omit')/2,np.nanmean(regression_large[:,10:,1,5,2]**2,0)-scipy.stats.iqr(regression_large[:,10:,1,5,2]**2,0,nan_policy='omit')/2,color='b',alpha=0.5)
    axs.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_rsquares.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_rsquares.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(diff_gains,0),color='k')
    axs.fill_between(np.arange(11), np.nanmean(diff_gains,0)+scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,np.nanmean(diff_gains,0)-scipy.stats.iqr(diff_gains,0,nan_policy='omit')/2,color='k', alpha=0.5)
    axs.axhline(0, color='k', lw=2, ls=':')
    axs.set_ylim([-1,1])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_diffgains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_diffgains.svg'),bbox_inches='tight')

    #Compare the values at the contralateral midstance 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(1, np.nanmean(regression_small[:,17,1,5,0]), color='r',s=20)
    axs.scatter(2, np.nanmean(regression_large[:,17,1,5,0]), color='b',s=20)
    axs.plot([1,1], [np.nanmean(regression_small[:,17,1,5,0])+np.nanstd(regression_small[:,17,1,5,0]),np.nanmean(regression_small[:,17,1,5,0])-np.nanstd(regression_small[:,17,1,5,0])],color='r',lw=2)
    axs.plot([2,2], [np.nanmean(regression_large[:,17,1,5,0])+np.nanstd(regression_large[:,17,1,5,0]),np.nanmean(regression_large[:,17,1,5,0])-np.nanstd(regression_large[:,17,1,5,0])],color='b',lw=2)
    x_vec1 = np.linspace(0.8,1.2,80)
    x_vec2 = np.linspace(1.8,2.2,80)
    for animal in range(80):
        axs.scatter(x_vec1[animal], regression_small[animal,17,1,5,0],color='r',alpha=0.2,s=5)
        axs.scatter(x_vec2[animal], regression_large[animal,17,1,5,0],color='b',alpha=0.2,s=5)
    axs.set_xlim([0,3])
    axs.set_ylim([-1.05,1.05])
    axs.set_yticks([-1,0,1])
    plt.tight_layout()
    test1 = scipy.stats.wilcoxon(regression_small[:,17,1,5,0],regression_large[:,17,1,5,0])
    print(test1)

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_midstance_gains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_midstance_gains.svg'),bbox_inches='tight')
    

    if bool_plot:
        plt.show()

def plot_correlation_horizon_final(list_input_horizon, tot_animal, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the correlation between successive foot contacts
    """

    # Represent the correlations for one exemplar animals
    idx_animal = np.where(tot_animal[:,0]==0)[0]
    local_input_plot = list_input_horizon[0][idx_animal,:,:]


    matrix_correlation, matrix_slope = np.zeros((80,5,2)), np.zeros((80,5,2))
    for animal in range(80):
        idx_animal = np.where(tot_animal[:,0]==animal)[0]
        local_input0 = list_input_horizon[0][idx_animal,:,:]
        idx_nans = np.where(~np.isnan(local_input0[:,0,0]))[0]
        reg_0 = scipy.stats.linregress(local_input0[idx_nans,10,1], local_input0[idx_nans,-1,1])
        reg_1 = scipy.stats.linregress(local_input0[idx_nans,0,1], local_input0[idx_nans,-1,1])
        reg_0_ = scipy.stats.linregress(local_input0[idx_nans,10,5], local_input0[idx_nans,-1,5])
        reg_1_ = scipy.stats.linregress(local_input0[idx_nans,0,5], local_input0[idx_nans,-1,5])
        matrix_correlation[animal,0,0], matrix_correlation[animal,1,0] = reg_0.rvalue, reg_1.rvalue
        matrix_slope[animal,0,0], matrix_slope[animal,1,0] = reg_0.slope, reg_1.slope
        matrix_correlation[animal,0,1], matrix_correlation[animal,1,1] = reg_0_.rvalue, reg_1_.rvalue
        matrix_slope[animal,0,1], matrix_slope[animal,1,1] = reg_0_.slope, reg_1_.slope
        for horizon in range(1,len(list_input_horizon)):
            local_input = list_input_horizon[horizon][idx_animal,:,:]
            idx_nans = np.where(~np.isnan(local_input[:,0,0]))[0]
            reg = scipy.stats.linregress(local_input[idx_nans,0,1], local_input[idx_nans,-1,1])
            reg_ = scipy.stats.linregress(local_input[idx_nans,0,5], local_input[idx_nans,-1,5])
            matrix_correlation[animal,horizon+1,0] = reg.rvalue**2
            matrix_slope[animal,horizon+1,0] = reg.slope
            matrix_correlation[animal,horizon+1,1] = reg_.rvalue**2
            matrix_slope[animal,horizon+1,1] = reg_.slope

    def func(x,a,b,c):
        return a*np.exp(-b*x)+c

    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,6), np.nanmean(matrix_correlation[:,:,0],0),color='r',s=10)
    axs.scatter(np.arange(1,6), np.nanmean(matrix_correlation[:,:,1],0),color='b',s=10)
    for hor in range(5):
        axs.plot([hor+1, hor+1], [np.nanmean(matrix_correlation[:,hor,0],0)+scipy.stats.iqr(matrix_correlation[:,hor,0],0,nan_policy='omit')/2,np.nanmean(matrix_correlation[:,hor,0],0)-scipy.stats.iqr(matrix_correlation[:,hor,0],0,nan_policy='omit')/2], color='r',lw=2)
        axs.plot([hor+1, hor+1], [np.nanmean(matrix_correlation[:,hor,1],0)+scipy.stats.iqr(matrix_correlation[:,hor,1],0,nan_policy='omit')/2,np.nanmean(matrix_correlation[:,hor,1],0)-scipy.stats.iqr(matrix_correlation[:,hor,1],0,nan_policy='omit')/2], color='b',lw=2)
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_correlation.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_correlation.svg'),bbox_inches='tight')

    x_data = np.arange(1,6)-1
    y_data1 = np.nanmean(matrix_slope[:,:,0],0)
    y_data2 = np.nanmean(matrix_slope[:,:,1],0)
    popt1 ,pcov1 = scipy.optimize.curve_fit(func,x_data,y_data1,maxfev=5000)
    popt2 ,pcov2 = scipy.optimize.curve_fit(func,x_data,y_data2,maxfev=5000)
    from tqdm import tqdm
    # Fitting the exponential decays with bootstrap resampling 
    n_boot = 100_000 
    mat_decays = np.zeros((n_boot,2))
    mat_animals = np.zeros((80,2))
    for iter in tqdm(range(n_boot)):
        idx_local = np.random.choice(80,80,replace=True)
        y_data_1_local = np.nanmean(matrix_slope[idx_local,:,0],0)
        y_data_2_local = np.nanmean(matrix_slope[idx_local,:,1],0)
        popt_1_local, _ = scipy.optimize.curve_fit(func, x_data, y_data_1_local, maxfev=5000)
        popt_2_local, _ = scipy.optimize.curve_fit(func, x_data, y_data_2_local, maxfev=5000)
        mat_decays[iter,0], mat_decays[iter,1] = popt_1_local[1], popt_2_local[1]
    np.save('AAAAAAAAAAAAAAAAAAMICe.npy', mat_decays)
    print('SAVED ===============')
    print(np.nanmedian(mat_decays,0))
    print(scipy.stats.iqr(mat_decays,0))
    print(np.nanpercentile(mat_decays,[5,95],0))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.set_xlim([-0.5,2.5])
    axs.set_ylim([-0.5,2.5])
    tmp = 1 / mat_decays
    print('Parameters to put in paper')
    print(np.nanmedian(tmp,0))
    print(scipy.stats.iqr(tmp,0,nan_policy='omit'))
    axs.scatter([1,2], np.nanmedian(tmp,0),color='k',s=20)
    axs.plot([2,2],[np.nanpercentile(tmp[:,1],25),np.nanpercentile(tmp[:,1],75)],color='k',lw=2)
    #axs.boxplot(mat_decays, positions=[0,1],widths=[0.25,0.25],showfliers=False,whis=(5,95),medianprops=dict(color='k'),boxprops=dict(color='k'),whiskerprops=dict(color='k'))
    plt.tight_layout()

    print('Parameters')
    print(popt1)
    print(popt2)
    y_pred1 = func(x_data,popt1[0],popt1[1],popt1[2])
    y_pred2 = func(x_data,popt2[0],popt2[1],popt2[2])
    print(1/popt1[1])
    print(1/popt2[1])

    # do the same for all the data at once ... 
    x_tot_data = np.tile(x_data,80)
    y_tot_data1 = np.reshape(np.squeeze(matrix_slope[:,:,0]), (400,))
    y_tot_data2 = np.reshape(np.squeeze(matrix_slope[:,:,1]), (400,))

    popt_total1, pcov_total1 = scipy.optimize.curve_fit(func, x_tot_data, y_tot_data1, maxfev=5000)
    popt_total2, pcov_total2 = scipy.optimize.curve_fit(func, x_tot_data, y_tot_data2, maxfev=5000)
    perr1 = np.sqrt(np.diag(pcov_total1))[1]
    perr2 = np.sqrt(np.diag(pcov_total2))[1]
    print(1/(popt_total1[1]-perr1), 1/(popt_total1[1]+perr1))
    print(1/(popt_total2[1]-perr2), 1/(popt_total2[1]+perr2))
    print(popt_total1[1], popt_total2[1])

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,6), np.nanmean(matrix_slope[:,:,0],0),color='r',s=30)
    axs.scatter(np.arange(1,6)+0.15, np.nanmean(matrix_slope[:,:,1],0),color='r',s=30,marker='v')
    axs.plot(x_data+1,y_pred1,color='r',lw=1)
    axs.plot(x_data+1,y_pred2,color='r',lw=1)
    for hor in range(5):
        axs.plot([hor+1, hor+1], [np.nanmean(matrix_slope[:,hor,0],0)+scipy.stats.iqr(matrix_slope[:,hor,0],0,nan_policy='omit')/2,np.nanmean(matrix_slope[:,hor,0],0)-scipy.stats.iqr(matrix_slope[:,hor,0],0,nan_policy='omit')/2], color='r',lw=2)
        axs.plot([hor+1.15, hor+1.15], [np.nanmean(matrix_slope[:,hor,1],0)+scipy.stats.iqr(matrix_slope[:,hor,1],0,nan_policy='omit')/2,np.nanmean(matrix_slope[:,hor,1],0)-scipy.stats.iqr(matrix_slope[:,hor,1],0,nan_policy='omit')/2], color='r',lw=2)
    axs.set_ylim([-1.05,1.05])
    axs.set_xlim([0,6])
    axs.set_yticks([-1,-0.5,0,0.5,1])
    axs.set_yticklabels(['-1','-0.5','0','0.5','1'])
    axs.axhline(0,color='k',lw=2,ls=':')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slopes.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slopes.svg'),bbox_inches='tight')


    # Plot the values for the initial location
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(1,np.nanmean(matrix_slope[:,0,0]),color='r',s=20)
    axs.scatter(2,np.nanmean(matrix_slope[:,0,1]),color='r',s=20,marker='v')
    axs.plot([1,1], [np.nanmean(matrix_slope[:,0,0])+np.nanstd(matrix_slope[:,0,0]),np.nanmean(matrix_slope[:,0,0])-np.nanstd(matrix_slope[:,0,0])],color='r',lw=2)
    axs.plot([2,2], [np.nanmean(matrix_slope[:,0,1])+np.nanstd(matrix_slope[:,0,1]),np.nanmean(matrix_slope[:,0,1])-np.nanstd(matrix_slope[:,0,1])],color='r',lw=2)
    axs.set_xlim([0,4])
    xvec1 = np.linspace(0.8,1.2,80)
    xvec2 = np.linspace(1.8,2.2,80)
    for animal in range(80):
        axs.scatter(xvec1[animal], matrix_slope[animal,0,0],color='r',s=5,alpha=0.2)
        axs.scatter(xvec2[animal], matrix_slope[animal,0,1],color='r',s=5,alpha=0.2,marker='v')
    axs.set_ylim([-1.05,1.05])
    axs.set_yticks([-1,0,1])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_init_value.svg'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_init_value.png'),bbox_inches='tight')
    test_distros = scipy.stats.wilcoxon(np.squeeze(matrix_slope[:,0,0]),np.squeeze(matrix_slope[:,0,1]),alternative='less')
    print(test_distros)
    print(scipy.stats.ttest_rel(np.squeeze(matrix_slope[:,0,0]),np.squeeze(matrix_slope[:,0,1]),alternative='less'))
    print(scipy.stats.shapiro(np.squeeze(matrix_slope[:,0,0])))
    print(scipy.stats.shapiro(np.squeeze(matrix_slope[:,0,1])))
    tmp_1 = matrix_slope[:,0,0].flatten()
    tmp_2 = matrix_slope[:,0,1].flatten()
    cohend = (np.nanmean(tmp_1) - np.nanmean(tmp_2)) / (np.nanstd(np.concatenate((tmp_1,tmp_2))))
    print(cohend)
    if bool_plot:
        plt.show()

def plot_variance_contributions_final(input_variance, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final subplots for the variance contribution
    """
    # Compute the sum of ffwd and fb variance
    sum_variance = np.sum(input_variance,1)
    prop_variance = input_variance[:,0,:] / sum_variance

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter([1.95], [np.nanmean(sum_variance[:,0])], color='k',s=20)
    axs.scatter([2.05], [np.nanmean(sum_variance[:,1])], color='r',s=20)
    axs.plot([1.95,1.95],[np.nanmean(sum_variance[:,0])-np.nanstd(sum_variance[:,0]),np.nanmean(sum_variance[:,0])+np.nanstd(sum_variance[:,0])],color='k',lw=2)
    axs.plot([2.05,2.05],[np.nanmean(sum_variance[:,1])-np.nanstd(sum_variance[:,1]),np.nanmean(sum_variance[:,1])+np.nanstd(sum_variance[:,1])],color='r',lw=2)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_total_var.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_total_var.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter([1.95], [np.nanmean(prop_variance[:,0])], color='k',s=20)
    axs.scatter([2.05], [np.nanmean(prop_variance[:,1])], color='r',s=20)
    axs.plot([1.95,1.95],[np.nanmean(prop_variance[:,0])-np.nanstd(prop_variance[:,0]),np.nanmean(prop_variance[:,0])+np.nanstd(prop_variance[:,0])],color='k',lw=2)
    axs.plot([2.05,2.05],[np.nanmean(prop_variance[:,1])-np.nanstd(prop_variance[:,1]),np.nanmean(prop_variance[:,1])+np.nanstd(prop_variance[:,1])],color='r',lw=2)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_proportion_var.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_proportion_var.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()


def plot_rsquare_normalized(list1, list2, list3, list4, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the normalized version of the r-square figure for hte mouse data
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list1[0],0),'k',lw=2,label='body')
    axs.plot(np.nanmedian(list1[1],0),'r',lw=2,label='self')
    axs.fill_between(np.arange(21), np.nanmedian(list1[0],0)+scipy.stats.iqr(list1[0],0,nan_policy='omit')/2, np.nanmedian(list1[0],0)-scipy.stats.iqr(list1[0],0,nan_policy='omit')/2, color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list1[1],0)+scipy.stats.iqr(list1[1],0,nan_policy='omit')/2, np.nanmedian(list1[1],0)-scipy.stats.iqr(list1[1],0,nan_policy='omit')/2, color='r',alpha=0.5)
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylabel('Relative gait fraction')
    axs.set_xlabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.svg'),bbox_inches='tight')
    
    # Computing the stats for the subfigure above
    list_pvalue1 = []
    for time in range(list1[0].shape[1]):
        tmp_test = scipy.stats.ks_2samp(list1[0][~np.isnan(list1[0][:,time]),time],list1[1][~np.isnan(list1[1][:,time]),time],alternative='less')
        list_pvalue1.append(tmp_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list2[0],0),'k',lw=2,label='body')
    axs.plot(np.nanmedian(list2[1],0),'r',lw=2,label='self')
    axs.fill_between(np.arange(21), np.nanmedian(list2[0],0)+scipy.stats.iqr(list2[0],0,nan_policy='omit')/2, np.nanmedian(list2[0],0)-scipy.stats.iqr(list2[0],0,nan_policy='omit')/2, color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list2[1],0)+scipy.stats.iqr(list2[1],0,nan_policy='omit')/2, np.nanmedian(list2[1],0)-scipy.stats.iqr(list2[1],0,nan_policy='omit')/2, color='r',alpha=0.5)
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylabel('Relative gait fraction')
    axs.set_xlabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.svg'),bbox_inches='tight')

    # Computing the stats for the subfigure above
    list_pvalue2 = []
    for time in range(list2[0].shape[1]):
        tmp_test = scipy.stats.ks_2samp(list2[0][~np.isnan(list2[0][:,time]),time],list2[1][~np.isnan(list2[1][:,time]),time],alternative='less')
        list_pvalue2.append(tmp_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list3[0],0),'k',lw=2,label='body')
    axs.plot(np.nanmedian(list3[1],0),'r',lw=2,label='self')
    axs.fill_between(np.arange(21), np.nanmedian(list3[0],0)+scipy.stats.iqr(list3[0],0,nan_policy='omit')/2, np.nanmedian(list3[0],0)-scipy.stats.iqr(list3[0],0,nan_policy='omit')/2, color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list3[1],0)+scipy.stats.iqr(list3[1],0,nan_policy='omit')/2, np.nanmedian(list3[1],0)-scipy.stats.iqr(list3[1],0,nan_policy='omit')/2, color='r',alpha=0.5)
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_3.svg'),bbox_inches='tight')

    # Computing the stats for the subfigure above
    list_pvalue3 = []
    for time in range(list3[0].shape[1]):
        tmp_test = scipy.stats.ks_2samp(list3[0][~np.isnan(list3[0][:,time]),time],list3[1][~np.isnan(list3[1][:,time]),time],alternative='less')
        list_pvalue3.append(tmp_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list4[0],0),'k',lw=2,label='body')
    axs.plot(np.nanmedian(list4[1],0),'r',lw=2,label='self')
    axs.fill_between(np.arange(21), np.nanmedian(list4[0],0)+scipy.stats.iqr(list4[0],0,nan_policy='omit')/2, np.nanmedian(list4[0],0)-scipy.stats.iqr(list4[0],0,nan_policy='omit')/2, color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmedian(list4[1],0)+scipy.stats.iqr(list4[1],0,nan_policy='omit')/2, np.nanmedian(list4[1],0)-scipy.stats.iqr(list4[1],0,nan_policy='omit')/2, color='r',alpha=0.5)
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_4.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_4.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()

    # Computing the stats for the subfigure above
    list_pvalue4 = []
    for time in range(list4[0].shape[1]):
        tmp_test = scipy.stats.ks_2samp(list4[0][~np.isnan(list4[0][:,time]),time],list4[1][~np.isnan(list4[1][:,time]),time],alternative='less')
        list_pvalue4.append(tmp_test.pvalue)

    # Saving the stats 
    tot_array_pvalue = np.hstack((np.expand_dims(np.array(list_pvalue1),-1), np.expand_dims(np.array(list_pvalue2),-1), np.expand_dims(np.array(list_pvalue3),-1), np.expand_dims(np.array(list_pvalue4),-1)))
    np.savetxt(os.path.join(PATH_STATS,'stats_rsquares_normalized.csv'),tot_array_pvalue,delimiter=',')


def plot_correlation_horizon(list_input_horizon, tot_animal, bool_plot=False, bool_save=False, figname=None):
    """
    Inspect the correlation between successive foot contacts
    """
    matrix_correlation = np.zeros((80,5,2))
    matrix_slope = np.zeros((80,5,2))
    for animal in range(80):
        idx_animal = np.where(tot_animal[:,0]==animal)[0]
        local_input0 = list_input_horizon[0][idx_animal,:,:]
        idx_nans = np.where(~np.isnan(local_input0[:,0,0]))[0]
        reg_0 = scipy.stats.linregress(local_input0[idx_nans,10,1], local_input0[idx_nans,-1,1])
        reg_1 = scipy.stats.linregress(local_input0[idx_nans,0,1], local_input0[idx_nans,-1,1])
        reg_0_ = scipy.stats.linregress(local_input0[idx_nans,10,5], local_input0[idx_nans,-1,5])
        reg_1_ = scipy.stats.linregress(local_input0[idx_nans,0,5], local_input0[idx_nans,-1,5])
        matrix_correlation[animal,0,0], matrix_correlation[animal,1,0] = reg_0.rvalue, reg_1.rvalue
        matrix_slope[animal,0,0], matrix_slope[animal,1,0] = reg_0.slope, reg_1.slope
        matrix_correlation[animal,0,1], matrix_correlation[animal,1,1] = reg_0_.rvalue, reg_1_.rvalue
        matrix_slope[animal,0,1], matrix_slope[animal,1,1] = reg_0_.slope, reg_1_.slope
        for horizon in range(1,len(list_input_horizon)):
            local_input = list_input_horizon[horizon][idx_animal,:,:]
            idx_nans = np.where(~np.isnan(local_input[:,0,0]))[0]
            reg = scipy.stats.linregress(local_input[idx_nans,0,1], local_input[idx_nans,-1,1])
            reg_ = scipy.stats.linregress(local_input[idx_nans,0,5], local_input[idx_nans,-1,5])
            matrix_correlation[animal,horizon+1,0] = reg.rvalue
            matrix_slope[animal,horizon+1,0] = reg.slope
            matrix_correlation[animal,horizon+1,1] = reg_.rvalue
            matrix_slope[animal,horizon+1,1] = reg_.slope

    return matrix_correlation, matrix_slope


def plot_rsquare_normalized_diff(list1, list2, list3, list4, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the normalized version of the r-square figure for hte mouse data
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(list1[0]-list1[1][:80,:],0),color='k',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list1[0]-list1[1][:80,:],0)+scipy.stats.iqr(list1[0]-list1[1][:80,:],0,nan_policy='omit')/2,
                     np.nanmedian(list1[0]-list1[1][:80,:],0)-scipy.stats.iqr(list1[0]-list1[1][:80,:],0,nan_policy='omit')/2, color='k', alpha=0.2) 
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.axhline(0, color='k', ls=':', lw=2)
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylabel('Relative gait fraction')
    axs.set_xlabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-1.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(list2[0]-list2[1][:80,:],0),color='k',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list2[0]-list2[1][:80,:],0)+scipy.stats.iqr(list2[0]-list2[1][:80,:],0,nan_policy='omit')/2,
                     np.nanmedian(list2[0]-list2[1][:80,:],0)-scipy.stats.iqr(list2[0]-list2[1][:80,:],0,nan_policy='omit')/2, color='k', alpha=0.2) 
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.axhline(0, color='k', ls=':', lw=2)
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylabel('Relative gait fraction')
    axs.set_xlabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-1.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(list3[0]-list3[1][:80,:],0),color='k',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list3[0]-list3[1][:80,:],0)+scipy.stats.iqr(list3[0]-list3[1][:80,:],0,nan_policy='omit')/2,
                     np.nanmedian(list3[0]-list3[1][:80,:],0)-scipy.stats.iqr(list3[0]-list3[1][:80,:],0,nan_policy='omit')/2, color='k', alpha=0.2) 
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.axhline(0, color='k', ls=':', lw=2)
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-1.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_3.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(list4[0]-list4[1][:80,:],0),color='k',lw=2)
    axs.fill_between(np.arange(21), np.nanmedian(list4[0]-list4[1][:80,:],0)+scipy.stats.iqr(list4[0]-list4[1][:80,:],0,nan_policy='omit')/2,
                     np.nanmedian(list4[0]-list4[1][:80,:],0)-scipy.stats.iqr(list4[0]-list4[1][:80,:],0,nan_policy='omit')/2, color='k', alpha=0.2) 
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.axhline(0, color='k', ls=':', lw=2)
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_xlim([10,21])
    axs.set_ylim([-1.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_4.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_4.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()


def plot_rsquare_figure(rsquare_front_for, rsquare_front_lat, rsquare_hind_for, rsquare_hind_lat,
                         rs_self_front_for, rs_self_front_lat, rs_self_hind_for, rs_self_hind_lat, bool_show=False, bool_save=False, figname=None):
    """
    Generates the figure representing the rsquares for the different conditions
    """


    fig, axs = plt.subplots(2,2,figsize=(6,5),sharey='row',sharex='row',height_ratios=[3,3])
    fig.suptitle(figname)
    axs[0,0].plot(np.nanmedian(rsquare_front_for,0),color='k',lw=2,label='body')
    axs[0,0].plot(np.nanmedian(rs_self_front_for,0),color='r',lw=2,label='self')
    axs[0,0].legend(frameon=False)
    axs[0,0].fill_between(np.arange(21), np.nanmedian(rsquare_front_for,0)+scipy.stats.iqr(rsquare_front_for,0)/2, np.nanmedian(rsquare_front_for,0)-scipy.stats.iqr(rsquare_front_for,0)/2,color='k',alpha=0.5)
    axs[0,0].fill_between(np.arange(21), np.nanmedian(rs_self_front_for,0)+scipy.stats.iqr(rs_self_front_for,0)/2, np.nanmedian(rs_self_front_for,0)-scipy.stats.iqr(rs_self_front_for,0)/2,color='r',alpha=0.5)

    axs[1,0].plot(np.nanmedian(rsquare_front_lat,0),color='k',lw=2,label='body')
    axs[1,0].plot(np.nanmedian(rs_self_front_lat,0),color='r',lw=2,label='self')
    axs[1,0].fill_between(np.arange(21), np.nanmedian(rsquare_front_lat,0)+scipy.stats.iqr(rsquare_front_lat,0)/2, np.nanmedian(rsquare_front_lat,0)-scipy.stats.iqr(rsquare_front_lat,0)/2,color='k',alpha=0.5)
    axs[1,0].fill_between(np.arange(21), np.nanmedian(rs_self_front_lat,0)+scipy.stats.iqr(rs_self_front_lat,0)/2, np.nanmedian(rs_self_front_lat,0)-scipy.stats.iqr(rs_self_front_lat,0)/2,color='r',alpha=0.5)

    axs[0,1].plot(np.nanmedian(rsquare_hind_for,0),color='k',lw=2,label='body')
    axs[0,1].plot(np.nanmedian(rs_self_hind_for,0),color='r',lw=2,label='self')
    axs[0,1].fill_between(np.arange(21), np.nanmedian(rsquare_hind_for,0)+scipy.stats.iqr(rsquare_hind_for,0)/2, np.nanmedian(rsquare_hind_for,0)-scipy.stats.iqr(rsquare_hind_for,0)/2,color='k',alpha=0.5)
    axs[0,1].fill_between(np.arange(21), np.nanmedian(rs_self_hind_for,0)+scipy.stats.iqr(rs_self_hind_for,0)/2, np.nanmedian(rs_self_hind_for,0)-scipy.stats.iqr(rs_self_hind_for,0)/2,color='r',alpha=0.5)

    axs[1,1].plot(np.nanmedian(rsquare_hind_lat,0),color='k',lw=2,label='body')
    axs[1,1].plot(np.nanmedian(rs_self_hind_lat,0),color='r',lw=2,label='self')
    axs[1,1].fill_between(np.arange(21), np.nanmedian(rsquare_hind_lat,0)+scipy.stats.iqr(rsquare_hind_lat,0)/2, np.nanmedian(rsquare_hind_lat,0)-scipy.stats.iqr(rsquare_hind_lat,0)/2,color='k',alpha=0.5)
    axs[1,1].fill_between(np.arange(21), np.nanmedian(rs_self_hind_lat,0)+scipy.stats.iqr(rs_self_hind_lat,0)/2, np.nanmedian(rs_self_hind_lat,0)-scipy.stats.iqr(rs_self_hind_lat,0)/2,color='r',alpha=0.5)

    axs[0,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right']].set_visible(False)
    axs[0,0].set_xticks([10,12.5,15,17.5,20]), axs[0,0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[0,1].set_xticks([10,12.5,15,17.5,20]), axs[0,1].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[1,0].set_xticks([10,12.5,15,17.5,20]), axs[1,0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[1,1].set_xticks([10,12.5,15,17.5,20]), axs[1,1].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[0,0].set_ylabel('Coefficient of determination (R²)')#, axs[1,0].set_ylabel('Lateral direction',)
    axs[0,0].set_title('Front legs'), axs[0,1].set_title('Hind legs')
    axs[0,0].axvline(17,color='k',lw=2,ls=':'), axs[0,1].axvline(17,color='k',lw=2,ls=':')
    axs[1,0].axvline(17,color='k',lw=2,ls=':'), axs[1,1].axvline(17,color='k',lw=2,ls=':')
    axs[0,0].set_xlim([10,21]), axs[0,1].set_xlim([10,21]), axs[1,0].set_xlim([10,21]), axs[1,1].set_xlim([10,21]) 
    axs[0,0].set_ylim([-0.05, 1.05])
    axs[0,0].set_yticks([0,0.25,0.5,0.75,1])
    axs[1,0].set_ylim([-0.05, 1.05])
    axs[1,0].set_yticks([0,0.25,0.5,0.75,1])
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Compute the statistics associated with this figure
    list_00, list_01, list_10, list_11 = [], [], [], []
    pval00, pval01, pval10, pval11 = np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1)), np.zeros((21,1))
    for time in range(21):
        list_00.append(scipy.stats.ks_2samp(rsquare_front_for[:,time], rs_self_front_for[:,time],alternative='less'))
        list_01.append(scipy.stats.ks_2samp(rsquare_front_lat[:,time], rs_self_front_lat[:,time],alternative='less'))
        list_10.append(scipy.stats.ks_2samp(rsquare_hind_for[:,time], rs_self_hind_for[:,time],alternative='less'))
        list_11.append(scipy.stats.ks_2samp(rsquare_hind_lat[:,time], rs_self_hind_lat[:,time],alternative='less'))
        pval00[time], pval01[time], pval10[time], pval11[time] = list_00[time].pvalue, list_01[time].pvalue, list_10[time].pvalue, list_11[time].pvalue
    ptot = np.concatenate((pval00, pval01, pval10, pval11),1)
    np.savetxt(os.path.join(PATH_STATS,'stats_rsquares.csv'),ptot,delimiter=',')
    

def plot_laterality_front_mice(front_matrix_inward, front_matrix_outward, bool_show=False, bool_save=False, figname=None):
    """
    Plots the laterality for the mouse data
    """
    fig, axs = plt.subplots(2,1,figsize=(3,5),sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].plot(np.nanmedian(front_matrix_outward[:,:,2,1,5,2]**2,0),color='k', lw=2, label='inward')
    axs[0].plot(np.nanmedian(front_matrix_inward[:,:,2,1,5,2]**2,0),color=[0.5,0.5,0.5], lw=2, label='inward')
    axs[0].fill_between(np.arange(21), np.nanmedian(front_matrix_inward[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(front_matrix_inward[:,:,2,1,5,2]**2,0,nan_policy='omit')/2, np.nanmedian(front_matrix_inward[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(front_matrix_inward[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,color=[0.5,0.5,0.5],alpha=0.5)
    axs[0].fill_between(np.arange(21), np.nanmedian(front_matrix_outward[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(front_matrix_outward[:,:,2,1,5,2]**2,0,nan_policy='omit')/2, np.nanmedian(front_matrix_outward[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(front_matrix_outward[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,color='k',alpha=0.5)
    axs[1].axhline(0, color='k', ls=':')
    axs[1].plot(np.nanmedian(front_matrix_outward[:,:,2,1,5,0],0),color='k', lw=2, label='inward')
    axs[1].plot(np.nanmedian(front_matrix_inward[:,:,2,1,5,0],0),color=[0.5,0.5,0.5], lw=2, label='inward')
    axs[1].fill_between(np.arange(21), np.nanmedian(front_matrix_inward[:,:,2,1,5,0],0)+scipy.stats.iqr(front_matrix_inward[:,:,2,1,5,0],0,nan_policy='omit')/2, np.nanmedian(front_matrix_inward[:,:,2,1,5,0],0)-scipy.stats.iqr(front_matrix_inward[:,:,2,1,5,0],0,nan_policy='omit')/2,color=[0.5,0.5,0.5],alpha=0.5)
    axs[1].fill_between(np.arange(21), np.nanmedian(front_matrix_outward[:,:,2,1,5,0],0)+scipy.stats.iqr(front_matrix_outward[:,:,2,1,5,0],0,nan_policy='omit')/2, np.nanmedian(front_matrix_outward[:,:,2,1,5,0],0)-scipy.stats.iqr(front_matrix_outward[:,:,2,1,5,0],0,nan_policy='omit')/2,color='k',alpha=0.5)
    axs[0].legend(frameon=False)
    axs[1].set_xlabel('Relative gait fraction')
    axs[0].set_ylim([-0.05, 1.05])
    axs[1].set_xticks([10,12.5,15,17.5,20])
    axs[0].set_xlim([10,20])
    axs[1].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    # Compute the stats of the above figure
    list_00, list_01 = [], []
    pval00, pval01 = np.zeros((21,1)), np.zeros((21,1))
    for time in range(21):
        list_00.append(scipy.stats.ks_2samp(front_matrix_outward[:,:,2,1,5,2]**2, front_matrix_inward[:,:,2,1,5,2]**2, alternative='less'))
        list_01.append(scipy.stats.ks_2samp(front_matrix_outward[:,:,2,1,5,0], front_matrix_inward[:,:,2,1,5,0], alternative='less'))
        pval00[time], pval01[time] = list_00[time].pvalue, list_01[time].pvalue
    ptot = np.concatenate((pval00, pval01),1)
    np.savetxt(os.path.join(PATH_STATS,'stats_laterality_matthis.csv'), ptot, detlimiter=',')

def plot_laterality_mice(front_matrix_inward, front_matrix_outward, hind_matrix_inward, hind_matrix_outward, bool_show=False, bool_save=False, figname=None):
    """
    Plots the laterality for the mouse data
    """
    fig, axs = plt.subplots(2,2,figsize=(5,3),height_ratios=[4,0.5],sharey='row',sharex='row')
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right','bottom','left']].set_visible(False)
    axs[1,1].spines[['top','right','bottom','left']].set_visible(False)
    axs[1,0].set_xticks([]), axs[1,0].set_yticks([]), axs[1,1].set_xticks([]), axs[1,1].set_yticks([])
    axs[0,0].plot(np.nanmedian(front_matrix_inward[:,:,2,1,5,2]**2,0),color='k',lw=2,label='inward')
    axs[0,0].plot(np.nanmedian(front_matrix_outward[:,:,2,1,5,2]**2,0),color='r',lw=2,label='outward')
    axs[0,0].fill_between(np.arange(21), np.nanmedian(front_matrix_inward[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(front_matrix_inward[:,:,2,1,5,2]**2,0)/2, np.nanmedian(front_matrix_inward[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(front_matrix_inward[:,:,2,1,5,2]**2,0)/2,color='k',alpha=0.5)
    axs[0,0].fill_between(np.arange(21), np.nanmedian(front_matrix_outward[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(front_matrix_outward[:,:,2,1,5,2]**2,0)/2, np.nanmedian(front_matrix_outward[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(front_matrix_outward[:,:,2,1,5,2]**2,0)/2,color='r',alpha=0.5)
    axs[0,0].legend(frameon=False)

    axs[0,1].plot(np.nanmedian(hind_matrix_inward[:,:,2,1,5,2]**2,0),color='k',lw=2)
    axs[0,1].plot(np.nanmedian(hind_matrix_outward[:,:,2,1,2,2]**2,0),color='r',lw=2)
    axs[0,1].fill_between(np.arange(21), np.nanmedian(hind_matrix_inward[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(hind_matrix_inward[:,:,2,1,5,2]**2,0)/2, np.nanmedian(hind_matrix_inward[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(hind_matrix_inward[:,:,2,1,5,2]**2,0)/2,color='k',alpha=0.5)
    axs[0,1].fill_between(np.arange(21), np.nanmedian(hind_matrix_outward[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(hind_matrix_outward[:,:,2,1,5,2]**2,0)/2, np.nanmedian(hind_matrix_outward[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(hind_matrix_outward[:,:,2,1,5,2]**2,0)/2,color='r',alpha=0.5)
    
    axs[0,0].set_ylim([-0.1,0.5]), axs[0,1].set_ylim([-0.1,0.5])
    axs[0,0].set_xlim([10,21]), axs[0,1].set_xlim([10,21])
    axs[1,0].set_xlim([0,0.5]), axs[1,1].set_xlim([0,0.5])
    axs[1,0].add_patch(patches.Rectangle([0,0.5],height=0.5, width=0.25, color='k'))
    axs[1,0].add_patch(patches.Rectangle([0.5,0.5],height=0.5, width=0.25, color='k'))
    axs[1,1].add_patch(patches.Rectangle([0,0.5],height=0.5, width=0.25, color='k'))
    axs[1,1].add_patch(patches.Rectangle([0.5,0.5],height=0.5, width=0.25, color='k'))
    axs[0,0].set_xticks([10,12.5,15,17.5,20]), axs[0,0].set_xticklabels(['-1','-0.25','-0.5','-0.25','0'])
    axs[1,0].set_xlabel('Relative gait fraction'), axs[1,1].set_xlabel('Relative gait fraction')
    axs[0,0].set_ylabel('Coefficient of determination (R²)')    
    axs[0,0].set_title('Front limbs'), axs[0,1].set_title('Hind limbs')
    plt.tight_layout()
    if bool_show:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Compute the statistics associated with this figure
    list_00, list_01 = [], []
    pval00, pval01 = np.zeros((21,1)), np.zeros((21,1))
    for time in range(21):
        list_00.append(scipy.stats.ks_2samp(front_matrix_outward[:,time,2,1,5,2]**2, front_matrix_inward[:,time,2,1,5,2]**2,alternative='less'))
        list_01.append(scipy.stats.ks_2samp(hind_matrix_outward[:,time,2,1,5,2]**2, hind_matrix_inward[:,time,2,1,5,2]**2,alternative='less'))
        pval00[time], pval01[time] = list_00[time].pvalue, list_01[time].pvalue
    ptot = np.concatenate((pval00, pval01),1)
    np.savetxt(os.path.join(PATH_STATS,'stats_laterality.csv'),ptot,delimiter=',')


def plot_regressions_slopes(front_limbs, hind_limbs, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the feedback gains for the mouse dataset
    """
    idx_plots = [4,5,6,7]
    fig, axs = plt.subplots(2,2,figsize=(6,5),sharex=True,sharey='row')
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right']].set_visible(False)
    scale_change00 = copy.deepcopy(hind_limbs[:,17,2,0,idx_plots,0])
    scale_change01 = copy.deepcopy(front_limbs[:,17,2,0,idx_plots,0])
    scale_change10 = copy.deepcopy(front_limbs[:,17,2,1,idx_plots,0])
    scale_change11 = copy.deepcopy(hind_limbs[:,17,2,1,idx_plots,0])
    for vel_id in [2,3]:
        scale_change00[:,vel_id] = 10*scale_change00[:,vel_id]
        scale_change01[:,vel_id] = 10*scale_change01[:,vel_id]
        scale_change10[:,vel_id] = 10*scale_change10[:,vel_id]
        scale_change11[:,vel_id] = 10*scale_change11[:,vel_id]
    axs[0,0].boxplot(scale_change00, positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[0,1].boxplot(scale_change01, positions=np.arange(4), widths=0.5, boxprops=dict(color='k'), medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[1,0].boxplot(scale_change10, positions=np.arange(4), widths=0.5, boxprops=dict(color='k'),medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[1,1].boxplot(scale_change11, positions=np.arange(4), widths=0.5, boxprops=dict(color='k'), medianprops=dict(color='k'),showfliers=False,whis=(5,95), autorange=False)
    axs[0,0].axhline(0, color='k', ls=':')
    axs[0,1].axhline(0, color='k', ls=':')
    axs[1,0].axhline(0, color='k', ls=':')
    axs[1,1].axhline(0, color='k', ls=':') 
    axs[0,0].set_ylabel('Slope')
    axs[1,0].set_ylabel('Slope')
    axs[1,0].set_xlabel('Input variable')
    axs[1,1].set_xlabel('Input variable')

    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,'fb_gains_mice.svg'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,'fb_gains_mice.png'),bbox_inches='tight')


def plot_models_comparison(list_method1, list_method2, bool_plot=False, bool_save=False, figname=None):
    """
    Compare the two methods suggested by Nidhi
    """

    fig, axs = plt.subplots(2,2,figsize=(5,5),sharex=True, sharey=False)
    fig.suptitle('Mouse data',fontweight='bold')
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right']].set_visible(False)
    axs[0,0].plot(np.nanmean(list_method1[0],0),'k',lw=2,label='Deviations')
    axs[0,0].plot(np.nanmean(list_method2[0],0),'k:',lw=2,label='Steady state')
    axs[0,1].plot(np.nanmean(list_method1[1],0),'k',lw=2,label='Deviations')
    axs[0,1].plot(np.nanmean(list_method2[1],0),'k:',lw=2,label='Steady state')
    axs[1,0].plot(np.nanmean(list_method1[2],0),'k',lw=2,label='Deviations')
    axs[1,0].plot(np.nanmean(list_method2[2],0),'k:',lw=2,label='Steady state')
    axs[1,1].plot(np.nanmean(list_method1[3],0),'k',lw=2,label='Deviations')
    axs[1,1].plot(np.nanmean(list_method2[3],0),'k:',lw=2,label='Steady state')
    axs[0,0].legend(frameon=False)
    axs[1,0].set_xlabel('Relative gait fraction')
    axs[1,1].set_xlabel('Relative gait fraction')
    axs[0,0].set_xticks([10,12.5,15,17.5,20])
    axs[0,0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[0,0].set_xlim([10,20])
    axs[0,0].set_ylabel('Front limbs')
    axs[1,0].set_ylabel('Hind limbs')
    axs[0,0].set_title('Foreaft direction')
    axs[0,1].set_title('Lateral direction')
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    #Compute the stats of the above models 
    list_pval00, list_pval01, list_pval10, list_pval11 = [], [], [], []
    for time in range(list_method1[0].shape[1]):
        test_00 = scipy.stats.ks_2samp(list_method1[0][:,time], list_method2[0][:,time],alternative='less')
        list_pval00.append(test_00.pvalue)
        test_01 = scipy.stats.ks_2samp(list_method1[1][:,time], list_method2[1][:,time],alternative='less')
        list_pval01.append(test_01.pvalue)
        test_10 = scipy.stats.ks_2samp(list_method1[2][:,time], list_method2[2][:,time],alternative='less')
        list_pval10.append(test_10.pvalue)
        test_11 = scipy.stats.ks_2samp(list_method1[3][:,time], list_method2[3][:,time],alternative='less')
        list_pval11.append(test_11.pvalue)
    tot_pvalue = np.concatenate((np.expand_dims(np.array(list_pval00),-1), np.expand_dims(np.array(list_pval00),-1), np.expand_dims(np.array(list_pval10),-1), np.expand_dims(np.array(list_pval11),-1)),1)
    np.savetxt(os.path.join(PATH_STATS,'model_comparisons.csv'),tot_pvalue,delimiter=',')


def plot_figure_laterality_mouse(diag_front_large, diag_front_small, odiag_front_large, odiag_front_small,
                                 diag_hind_large, diag_hind_small, odiag_hind_large, odiag_hind_small, bool_save=False, bool_plot=True, figname=None):
    """
    Plots the laterality figure
    """
    front_in = np.concatenate((diag_front_large, odiag_front_small),0)
    front_out = np.concatenate((diag_front_small,odiag_front_large),0)
    hind_in = np.concatenate((diag_hind_large, odiag_hind_small),0)
    hind_out = np.concatenate((diag_hind_small, odiag_hind_large),0)
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(front_in[:,:,2,1,5,2]**2,0),color='k',lw=2)
    axs.plot(np.nanmean(front_out[:,:,2,1,5,2]**2,0),color=[0.7,0.7,0.7],lw=2)
    axs.fill_between(np.arange(21), np.nanmean(front_in[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(front_in[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,np.nanmean(front_in[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(front_in[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmean(front_out[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(front_out[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,np.nanmean(front_out[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(front_out[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,color=[0.7,0.7,0.7],alpha=0.5)
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.1,1.05])
    axs.set_xlim([10,20])
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.svg'))

    # Computing the stats for the above 
    list_pvalue1 = []
    for time in range(21):
        local_test = scipy.stats.ks_2samp(front_in[~np.isnan(front_in[:,time,2,1,5,2]),time,2,1,5,2]**2, front_out[~np.isnan(front_out[:,time,2,1,5,2]),time,2,1,5,2]**2)
        list_pvalue1.append(local_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(hind_in[:,:,2,1,5,2]**2,0),color='k',lw=2)
    axs.plot(np.nanmean(hind_out[:,:,2,1,5,2]**2,0),color=[0.7,0.7,0.7],lw=2)
    axs.fill_between(np.arange(21), np.nanmean(hind_in[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(hind_in[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,np.nanmean(hind_in[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(hind_in[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmean(hind_out[:,:,2,1,5,2]**2,0)+scipy.stats.iqr(hind_out[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,np.nanmean(hind_out[:,:,2,1,5,2]**2,0)-scipy.stats.iqr(hind_out[:,:,2,1,5,2]**2,0,nan_policy='omit')/2,color=[0.7,0.7,0.7],alpha=0.5)
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-0.1,1.05])
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlim([10,20])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.svg'))
    
    # Computing the stats for the above 
    list_pvalue2 = []
    for time in range(21):
        local_test = scipy.stats.ks_2samp(hind_in[~np.isnan(hind_in[:,time,2,1,5,2]),time,2,1,5,2]**2, hind_out[~np.isnan(hind_out[:,time,2,1,5,2]),time,2,1,5,2]**2)
        list_pvalue2.append(local_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(front_in[:,:,2,1,5,0],0),color='k',lw=2)
    axs.plot(np.nanmean(front_out[:,:,2,1,5,0],0),color=[0.7,0.7,0.7],lw=2)
    axs.fill_between(np.arange(21), np.nanmean(front_in[:,:,2,1,5,0],0)+scipy.stats.iqr(front_in[:,:,2,1,5,0],0,nan_policy='omit')/2,np.nanmean(front_in[:,:,2,1,5,0],0)-scipy.stats.iqr(front_in[:,:,2,1,5,0],0,nan_policy='omit')/2,color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmean(front_out[:,:,2,1,5,0],0)+scipy.stats.iqr(front_out[:,:,2,1,5,0],0,nan_policy='omit')/2,np.nanmean(front_out[:,:,2,1,5,0],0)-scipy.stats.iqr(front_out[:,:,2,1,5,0],0,nan_policy='omit')/2,color=[0.7,0.7,0.7],alpha=0.5)
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-1,1])
    axs.set_xlim([10,20])
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_3.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_3.svg'))

    # Computing the stats for the above 
    list_pvalue3 = []
    for time in range(21):
        local_test = scipy.stats.ks_2samp(front_in[~np.isnan(front_in[:,time,2,1,5,0]),time,2,1,5,2]**2, front_out[~np.isnan(front_out[:,time,2,1,5,0]),time,2,1,5,0])
        list_pvalue3.append(local_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(hind_in[:,:,2,1,5,0],0),color='k',lw=2)
    axs.plot(np.nanmean(hind_out[:,:,2,1,5,0],0),color=[0.7,0.7,0.7],lw=2)
    axs.fill_between(np.arange(21), np.nanmean(hind_in[:,:,2,1,5,0],0)+scipy.stats.iqr(hind_in[:,:,2,1,5,0],0,nan_policy='omit')/2,np.nanmean(hind_in[:,:,2,1,5,0],0)-scipy.stats.iqr(hind_in[:,:,2,1,5,0],0,nan_policy='omit')/2,color='k',alpha=0.5)
    axs.fill_between(np.arange(21), np.nanmean(hind_out[:,:,2,1,5,0],0)+scipy.stats.iqr(hind_out[:,:,2,1,5,0],0,nan_policy='omit')/2,np.nanmean(hind_out[:,:,2,1,5,0],0)-scipy.stats.iqr(hind_out[:,:,2,1,5,0],0,nan_policy='omit')/2,color=[0.7,0.7,0.7],alpha=0.5)
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.set_ylim([-1,1])
    axs.set_xlim([10,20])
    axs.set_xticks([10,12.5,15,17.5,20])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_4.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_4.svg'))

    # Computing the stats for the above 
    list_pvalue4 = []
    for time in range(21):
        local_test = scipy.stats.ks_2samp(hind_in[~np.isnan(hind_in[:,time,2,1,5,0]),time,2,1,5,0], hind_out[~np.isnan(hind_out[:,time,2,1,5,0]),time,2,1,5,0])
        list_pvalue4.append(local_test.pvalue)

    # Put the stats together ... 
    tot_array_stats = np.hstack((np.expand_dims(np.array(list_pvalue1),-1), np.expand_dims(np.array(list_pvalue2),-1), np.expand_dims(np.array(list_pvalue3),-1), np.expand_dims(np.array(list_pvalue4),-1)))
    np.savetxt(os.path.join(PATH_STATS,'stats_laterality_mice.csv'),tot_array_stats,delimiter=',')

    if bool_plot:
        plt.show()
