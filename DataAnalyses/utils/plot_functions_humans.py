import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches 
from matplotlib import patches
import os
from sklearn.metrics import r2_score
from utils.utils_humans import *
from scipy.optimize import curve_fit


PATH_FIGURES = os.path.join(os.getcwd(), 'humans_results','figures')
PATH_STATS = os.path.join(os.getcwd(), 'humans_results','stats')
tmp_path = os.path.join(os.getcwd(), 'round_2_analysis')


def plot_laterality_humans(matrix_ref_fc_leftyang, matrix_reg_fc_rightyang, matrix_ref_fc_left_yang, matrix_reg_fc_right_yang, 
        matrix_ref_fc_leftmathis, matrix_reg_fc_rightmathis, matrix_ref_fc_left_mathis, matrix_reg_fc_right_mathis, bool_plot=False, bool_save=False, figname=None):
    fig, axs = plt.subplots(2,2,figsize=(6,3),height_ratios=[4,0.5],sharey='row',sharex='row')
    axs[0,0].spines[['top','right']].set_visible(False)
    axs[1,0].spines[['top','right','bottom','left']].set_visible(False)
    axs[0,1].spines[['top','right']].set_visible(False)
    axs[1,1].spines[['top','right','bottom','left']].set_visible(False)
    cat_outward_yang = np.nanmedian(np.concatenate((np.expand_dims(matrix_ref_fc_leftyang[:,:,0,2]**2,-1),np.expand_dims(matrix_reg_fc_right_yang[:,:,0,2]**2,-1)),2),-1)
    cat_inward_yang = np.nanmedian(np.concatenate((np.expand_dims(matrix_reg_fc_rightyang[:,:,0,2]**2,-1),np.expand_dims(matrix_ref_fc_left_yang[:,:,0,2]**2,-1)),2),-1)
    cat_outward_mathis = np.nanmedian(np.concatenate((np.expand_dims(matrix_ref_fc_leftmathis[:,:,1,2]**2,-1),np.expand_dims(matrix_reg_fc_right_mathis[:,:,1,2]**2,-1)),2),-1)
    cat_inward_mathis = np.nanmedian(np.concatenate((np.expand_dims(matrix_reg_fc_rightmathis[:,:,1,2]**2,-1),np.expand_dims(matrix_ref_fc_left_mathis[:,:,1,2]**2,-1)),2),-1)
    axs[0,0].plot(np.nanmean(cat_inward_yang,0),color=[0.5,0.5,0.5],lw=2,label='inward')
    axs[0,0].plot(np.nanmean(cat_outward_yang,0),color=[0,0,0],lw=2,label='outward')
    axs[0,1].fill_between(np.arange(101),np.nanmean(cat_inward_mathis,0)+np.nanstd(cat_inward_mathis,0),np.nanmean(cat_inward_mathis,0)-np.nanstd(cat_inward_mathis,0),alpha=0.5,color=[0.5,0.5,0.5])
    axs[0,1].fill_between(np.arange(101),np.nanmean(cat_outward_mathis,0)+np.nanstd(cat_outward_mathis,0),np.nanmean(cat_outward_mathis,0)-np.nanstd(cat_outward_mathis,0),alpha=0.5,color=[0,0,0])
    axs[0,0].fill_between(np.arange(101),np.nanmean(cat_inward_yang,0)+np.nanstd(cat_inward_yang,0),np.nanmean(cat_inward_yang,0)-np.nanstd(cat_inward_yang,0),alpha=0.5,color=[0.5,0.5,0.5])
    axs[0,0].fill_between(np.arange(101),np.nanmean(cat_outward_yang,0)+np.nanstd(cat_outward_yang,0),np.nanmean(cat_outward_yang,0)-np.nanstd(cat_outward_yang,0),alpha=0.5,color=[0,0,0])
    axs[0,1].plot(np.nanmean(cat_inward_mathis,0),color=[0.5,0.5,0.5],lw=2,label='inward')
    axs[0,1].plot(np.nanmean(cat_outward_mathis,0),color=[0,0,0],lw=2,label='outward')
    axs[0,0].set_xticks([0,25,50,75,100])
    axs[0,0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[0,0].set_ylabel('Coefficient of determination')
    axs[1,0].set_xlabel('Relative gait fraction')
    axs[0,0].legend(frameon=False,loc='upper left')
    axs[1,0].set_xticks([]), axs[1,0].set_yticks([])
    axs[1,1].set_xticks([]), axs[1,1].set_yticks([])
    axs[1,0].add_patch(patches.Rectangle([0,0.5],height=0.5, width=0.45,color='k'))
    axs[1,1].add_patch(patches.Rectangle([0,0.5],height=0.5, width=0.45,color='k'))
    axs[1,0].set_xlim([0,0.9])
    axs[1,1].set_xlabel('Relative gait fraction')
    axs[0,0].set_title('Treadmill walking',fontweight='bold'), axs[0,1].set_title('Overground walking',fontweight='bold')
    axs[0,1].set_xticks([0,25,50,75,100])
    axs[0,1].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[0,0].set_xlim([0,101]), axs[0,0].set_ylim([0,1])
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(os.getcwd(), 'humans_results','figures',f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(os.getcwd(), 'humans_results','figures',f'{figname}.svg'),bbox_inches='tight')

    

def exp_func(x,A,B,C,D):
    return A*np.exp(-B*(x-C))+D

def plot_feedforward_output_humans_time_paper(tot_input_leg1, tot_output_leg1, tot_subject1, bool_plot=False, bool_save=False, bool_fb=False, figname=None):
    """
    Plots the relationship between velocity and step length/width/duration for the camargo dataset
    """

    # Identify the subject with the more data (for exemplar representation)
    n_subjects = int(np.max(tot_subject1))+1
    len_sub = np.zeros((n_subjects,))
    for sub in range(n_subjects):
        len_sub[sub] = len(np.where(tot_subject1==sub)[0])
    local_sub = np.argmax(len_sub)

    tot_input = tot_input_leg1[tot_subject1==local_sub,:,:]
    tot_output = tot_output_leg1[tot_subject1==local_sub,:]
    tot_velocity = np.nanmean(tot_input_leg1[tot_subject1==local_sub,:,2],1)
    xinput = np.linspace(800,1800)


    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,2], s=20, color='b', alpha=0.5)
    axs.set_xlim([500,2000]), axs.set_xlabel('Velocity'), axs.set_ylabel('Step duration')
    idx_nans = np.where((~np.isnan(np.squeeze(tot_output[:,2]))) & (~np.isnan(tot_velocity)))[0]
    print(len(idx_nans), len(tot_velocity))
    popt, _ = curve_fit(exp_func, tot_velocity[idx_nans], tot_output[idx_nans,2], bounds=([100,0,0,50],[200,1,500,150]), p0=[120,0.00249,411,92])
    axs.set_xticks([500,1000,1500,2000]), axs.set_xticklabels(['0.5','1','1.5','2'])
    y_plot = exp_func(xinput, *popt)
    y_pred = exp_func(tot_velocity[idx_nans],*popt)
    axs.plot(xinput, y_plot,'k',lw=3)
    r2_exp = r2_score(tot_output[idx_nans,2], y_pred)
    print(popt, r2_exp)
    axs.set_ylim([150,325])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_duration.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_duration.svg'))


    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,3], s=20, color='b', alpha=0.5)
    axs.scatter(tot_velocity, tot_output[:,2], s=20, color='r', alpha=0.5)
    axs.set_xlim([500,2000]), axs.set_xlabel('Velocity'), axs.set_ylabel('Stance duration')
    idx_nans = np.where((~np.isnan(np.squeeze(tot_output[:,3]))) & (~np.isnan(tot_velocity)))[0]
    print(len(idx_nans), len(tot_velocity))
    popt, _ = curve_fit(exp_func, tot_velocity[idx_nans], tot_output[idx_nans,3], bounds=([100,0,0,50],[200,1,500,150]), p0=[120,0.00249,411,92])
    print(popt)
    axs.set_xticks([500,1000,1500,2000]), axs.set_xticklabels(['0.5','1','1.5','2'])
    y_plot = exp_func(xinput, *popt)
    y_pred = exp_func(tot_velocity[idx_nans],*popt)
    axs.plot(xinput, y_plot,'k',lw=3)
    r2_exp = r2_score(tot_output[idx_nans,2], y_pred)
    print(popt, r2_exp)
    axs.set_ylim([0,325])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_stance_duration.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_stance_duration.svg'))

    if bool_plot:
        plt.show()




def plot_feedforward_output_humans_paper(tot_input_leg1, tot_output_leg1, tot_subject1, bool_plot=False, bool_save=False, bool_fb=False, figname=None):
    """
    Plots the relationship between velocity and step length/width for the camargo dataset
    """
    # Identify the subject with the more data (for exemplar representation)
    n_subjects = int(np.max(tot_subject1))+1
    len_sub = np.zeros((n_subjects,))
    for sub in range(n_subjects):
        len_sub[sub] = len(np.where(tot_subject1==sub)[0])
    local_sub = np.argmax(len_sub)

    tot_input = tot_input_leg1[tot_subject1==local_sub,:,:]
    tot_output = tot_output_leg1[tot_subject1==local_sub,:]
    tot_velocity = np.nanmean(tot_input_leg1[tot_subject1==local_sub,:,2],1)

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,0], s=20, color='b', alpha=0.5)
    local_lin = scipy.stats.linregress(tot_velocity[tot_output[:,0]>400], tot_output[tot_output[:,0]>400,0])
    xinput = np.linspace(800,1800)
    print(local_lin.rvalue**2)
    axs.plot(xinput, local_lin.intercept + xinput*local_lin.slope,color='k',lw=3)
    axs.set_xlim([500,2000]), axs.set_ylim([400,900]), axs.set_xlabel('Velocity'), axs.set_ylabel('Step length')
    axs.set_yticks([400,500,600,700,800,900]), axs.set_yticklabels(['0.4','0.5','0.6','0.7','0.8','0.9'])
    axs.set_xticks([500,1000,1500,2000]), axs.set_xticklabels(['0.5','1','1.5','2'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_length.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_length.svg'))

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,1], s=20, color='b', alpha=0.5)
    local_lin = scipy.stats.linregress(tot_velocity[tot_output[:,0]>400], tot_output[tot_output[:,0]>400,1])
    print(local_lin.rvalue**2)
    xinput = np.linspace(800,1800)
    axs.plot(xinput, local_lin.intercept + xinput*local_lin.slope,color='k',lw=3)
    axs.set_xlim([500,2000]), axs.set_xlabel('Velocity'), axs.set_ylabel('Step width')
    axs.set_ylim([-450, 50]), axs.set_yticks([-400,-300,-200,-100,0]), axs.set_yticklabels(['-0.4','-0.3','-0.2','-0.1','0'])
    axs.set_xticks([500,1000,1500,2000]), axs.set_xticklabels(['0.5','1','1.5','2'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_width.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_data_width.svg'))

    # Compute the individual dependencies and represent them in a single figure ... 
    list_slopes_len, list_slopes_wid = [], []
    list_pvalues_len, list_pvalues_wid = [], []
    for sub in range(n_subjects):
        tmp_vel = np.nanmean(tot_input_leg1[tot_subject1==sub,:,2],1)
        tmp_len = tot_output_leg1[tot_subject1==sub,0]
        tmp_wid = tot_output_leg1[tot_subject1==sub,1]
        reg_len = scipy.stats.linregress(tmp_vel, tmp_len)
        reg_wid = scipy.stats.linregress(tmp_vel, tmp_wid)
        list_slopes_len.append(reg_len.slope), list_slopes_wid.append(reg_wid.slope)
        list_pvalues_len.append(reg_len.pvalue), list_pvalues_wid.append(reg_wid.pvalue)

    mat_pvalues_len = np.array(list_pvalues_len)
    mat_pvalues_wid = np.array(list_pvalues_wid) 
    str_len = f'{len(np.where(mat_pvalues_len<0.05)[0])}/{len(mat_pvalues_len)}'
    str_wid = f'{len(np.where(mat_pvalues_wid<0.05)[0])}/{len(mat_pvalues_wid)}'

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[0,1], height=[np.nanmean(np.array(list_slopes_len)), np.nanmean(np.array(list_slopes_wid))], width=0.2, color='b', yerr=[np.nanstd(np.array(list_slopes_len)), np.nanstd(np.array(list_slopes_wid))])
    axs.set_ylabel('Slope')
    axs.text(0.2, 0.3, str_len)
    axs.text(0.6, 0.3, str_wid)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slopes.png'))
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slopes.svg'))
    
    if bool_plot:
        plt.show()




def plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the laterality information for the human data
    """

    # Plots the feedback gains at contralateral midstance
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.set_xlabel('Direction'), axs.set_ylabel('Gain')
    axs.scatter(1, np.nanmean(regression_small[:,75,1,1,0]),color='r',s=20)
    axs.scatter(2, np.nanmean(regression_large[:,75,1,1,0]),color='b',s=20)
    axs.plot([1,1], [np.nanmean(regression_small[:,75,1,1,0])+scipy.stats.iqr(regression_small[:,75,1,1,0],nan_policy='omit')/2,np.nanmean(regression_small[:,75,1,1,0])-scipy.stats.iqr(regression_small[:,75,1,1,0],nan_policy='omit')/2],color='r',lw=2)
    axs.plot([2,2], [np.nanmean(regression_large[:,75,1,1,0])+scipy.stats.iqr(regression_large[:,75,1,1,0],nan_policy='omit')/2,np.nanmean(regression_large[:,75,1,1,0])-scipy.stats.iqr(regression_large[:,75,1,1,0],nan_policy='omit')/2],color='b',lw=2)
    x_vec1 = np.linspace(0.8,1.2,regression_large.shape[0])
    x_vec2 = np.linspace(1.8,2.2,regression_large.shape[0])
    for subject in range(regression_large.shape[0]):
        axs.scatter(x_vec1[subject], regression_small[subject,75,1,1,0], color='r', alpha=0.2, s=5)
        axs.scatter(x_vec2[subject], regression_large[subject,75,1,1,0], color='b', alpha=0.2, s=5)
    axs.set_ylim([0,5])
    axs.set_xlim([0,3])
    test1 = scipy.stats.wilcoxon(regression_large[:,75,1,1,0], regression_small[:,75,1,1,0],nan_policy='omit')
    print(test1)
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_comparison_midstance.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_comparison_midstance.svg'),bbox_inches='tight')
    plt.tight_layout()

    if bool_plot:
        plt.show()

def plot_correlation_horizon_final(list_input_horizon, list_subjects, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the correlation between successive foot contacts
    """

    matrix_correlation, matrix_slope = np.zeros((21,4)), np.zeros((21,4))
    for horizon in range(4):
        local_input = list_input_horizon[horizon]
        local_subject = list_subjects[horizon]
        local_sub_set = set(list(local_subject))
        for sub in local_sub_set:
            idx_subject = np.where(local_subject==sub)[0]
            local_input_h = local_input[idx_subject,:,:]
            idx_nans = np.where(~np.isnan(local_input_h[:,0,0]))[0]
            reg = scipy.stats.linregress(local_input_h[idx_nans,0,1], local_input_h[idx_nans,-1,1])
            matrix_correlation[int(sub),horizon] = reg.rvalue**2
            matrix_slope[int(sub),horizon] = reg.slope

    np.save(os.path.join(tmp_path,'matrix_correlation.npy'), matrix_slope)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,5), np.nanmean(matrix_slope,0),color='k',s=10)
    for horizon in range(4):
        axs.plot([horizon+1, horizon+1], [np.nanmean(matrix_slope[:,horizon],0)+scipy.stats.iqr(matrix_slope[:,horizon],0,nan_policy='omit')/2, np.nanmean(matrix_slope[:,horizon],0)-scipy.stats.iqr(matrix_slope[:,horizon],0,nan_policy='omit')/2],color='k',lw=2)
    axs.set_ylim([-1.05,1.05])
    axs.axhline(0,color='k',ls=':',lw=2)
    axs.set_xlabel('Number of gait cycle'), axs.set_ylabel('Pearson coefficient')
    plt.tight_layout()

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slope.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slope.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()



def plot_rsquares_final(foreaft_body, lateral_body, foreaft_self, lateral_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final rsquares figures for the human data 
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21),np.nanmedian(foreaft_body,0)[0::5],'b',lw=2)
    axs.plot(np.linspace(0,100,21),np.nanmedian(foreaft_self,0)[0::5],'r',lw=2)
    axs.fill_between(np.linspace(0,100,21), np.nanmedian(foreaft_body,0)[0::5]+scipy.stats.iqr(foreaft_body,0,nan_policy='omit')[0::5]/2,np.nanmedian(foreaft_body,0)[0::5]-scipy.stats.iqr(foreaft_body,0,nan_policy='omit')[0::5]/2,color='b',alpha=0.5)
    axs.fill_between(np.linspace(0,100,21), np.nanmedian(foreaft_self,0)[0::5]+scipy.stats.iqr(foreaft_self,0,nan_policy='omit')[0::5]/2,np.nanmedian(foreaft_self,0)[0::5]-scipy.stats.iqr(foreaft_self,0,nan_policy='omit')[0::5]/2,color='r',alpha=0.5)
    axs.set_ylim([-0.05, 1.05])
    axs.set_yticks([0,0.25,0.5,0.75,1])
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_foreaft.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_foreaft.svg'),bbox_inches='tight')
    
    list_pvalue_top = []
    time_vector = np.linspace(0,100,21)
    for ii in range(len(time_vector)):
        stat_test = scipy.stats.wilcoxon(foreaft_body[:,int(time_vector[ii])], foreaft_self[:,int(time_vector[ii])],alternative='greater')
        list_pvalue_top.append(stat_test.pvalue)
    
        
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21),np.nanmedian(lateral_body,0)[0::5],'b',lw=2)
    axs.plot(np.linspace(0,100,21),np.nanmedian(lateral_self,0)[0::5],'r',lw=2)
    axs.fill_between(np.linspace(0,100,21), np.nanmedian(lateral_body,0)[0::5]+scipy.stats.iqr(lateral_body,0,nan_policy='omit')[0::5]/2,np.nanmedian(lateral_body,0)[0::5]-scipy.stats.iqr(lateral_body,0,nan_policy='omit')[0::5]/2,color='b',alpha=0.5)
    axs.fill_between(np.linspace(0,100,21), np.nanmedian(lateral_self,0)[0::5]+scipy.stats.iqr(lateral_self,0,nan_policy='omit')[0::5]/2,np.nanmedian(lateral_self,0)[0::5]-scipy.stats.iqr(lateral_self,0,nan_policy='omit')[0::5]/2,color='r',alpha=0.5)
    axs.set_ylim([-0.05, 1.05])
    axs.set_yticks([0,0.25,0.5,0.75,1])
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_lateral.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_lateral.svg'),bbox_inches='tight')
    
    list_pvalue_bottom = []
    time_vector = np.linspace(0,100,21)
    for ii in range(len(time_vector)):
        stat_test = scipy.stats.wilcoxon(lateral_body[:,int(time_vector[ii])], lateral_self[:,int(time_vector[ii])],alternative='greater')
        list_pvalue_bottom.append(stat_test.pvalue)
    print(np.array(list_pvalue_top)<0.05)
    print(np.array(list_pvalue_bottom)<0.05)
    
    if bool_plot:
        plt.show()


