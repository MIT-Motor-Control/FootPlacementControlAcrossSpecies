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