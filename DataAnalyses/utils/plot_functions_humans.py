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


def plot_laterality_matthis(matrix_ref_fc_leftmathis, matrix_reg_fc_rightmathis, matrix_ref_fc_left_mathis, matrix_reg_fc_right_mathis, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the laterality figure for matthis dataset
    """
    fig, axs = plt.subplots(2,1,figsize=(3,5),sharex=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible
    cat_outward_mathis = np.nanmedian(np.concatenate((np.expand_dims(matrix_ref_fc_leftmathis[:,:,0,2]**2,-1),np.expand_dims(matrix_reg_fc_right_mathis[:,:,0,2]**2,-1)),2),-1)
    cat_inward_mathis = np.nanmedian(np.concatenate((np.expand_dims(matrix_reg_fc_rightmathis[:,:,0,2]**2,-1),np.expand_dims(matrix_ref_fc_left_mathis[:,:,0,2]**2,-1)),2),-1)
    cat_outward_mathis_slope = np.nanmedian(np.concatenate((np.expand_dims(matrix_ref_fc_leftmathis[:,:,0,0],-1),np.expand_dims(matrix_reg_fc_right_mathis[:,:,0,0],-1)),2),-1)
    cat_inward_mathis_slope = np.nanmedian(np.concatenate((np.expand_dims(matrix_reg_fc_rightmathis[:,:,0,0],-1),np.expand_dims(matrix_ref_fc_left_mathis[:,:,0,0],-1)),2),-1)
    axs[0].fill_between(np.arange(101),np.nanmean(cat_inward_mathis,0)+np.nanstd(cat_inward_mathis,0),np.nanmean(cat_inward_mathis,0)-np.nanstd(cat_inward_mathis,0),alpha=0.5,color=[0.5,0.5,0.5])
    axs[0].fill_between(np.arange(101),np.nanmean(cat_outward_mathis,0)+np.nanstd(cat_outward_mathis,0),np.nanmean(cat_outward_mathis,0)-np.nanstd(cat_outward_mathis,0),alpha=0.5,color=[0,0,0])
    axs[0].plot(np.nanmean(cat_inward_mathis,0),color=[0.5,0.5,0.5],lw=2,label='inward')
    axs[0].plot(np.nanmean(cat_outward_mathis,0),color=[0,0,0],lw=2,label='outward')
    axs[1].fill_between(np.arange(101),np.nanmean(cat_inward_mathis_slope,0)+np.nanstd(cat_inward_mathis_slope,0),np.nanmean(cat_inward_mathis_slope,0)-np.nanstd(cat_inward_mathis_slope,0),alpha=0.5,color=[0.5,0.5,0.5])
    axs[1].fill_between(np.arange(101),np.nanmean(cat_outward_mathis_slope,0)+np.nanstd(cat_outward_mathis_slope,0),np.nanmean(cat_outward_mathis_slope,0)-np.nanstd(cat_outward_mathis_slope,0),alpha=0.5,color=[0,0,0])
    axs[1].plot(np.nanmean(cat_inward_mathis_slope,0),color=[0.5,0.5,0.5],lw=2,label='inward')
    axs[1].plot(np.nanmean(cat_outward_mathis_slope,0),color=[0,0,0],lw=2,label='outward')
    axs[0].set_xticks([0,25,50,75,100])
    axs[0].set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs[0].set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    # Compute the stats for the figure above 
    list_00, list_01 = [], []
    pval00, pval01 = np.zeros((101,1)), np.zeros((101,1))
    for time in range(101):
        list_00.append(scipy.stats.ks_2samp(cat_outward_mathis[:,time],cat_inward_mathis[:,time],alternative='less'))
        list_01.append(scipy.stats.ks_2samp(cat_outward_mathis_slope[:,time], cat_inward_mathis_slope[:,time],alternative='less'))
        pval00[time], pval01[time] = list_00[time].pvalue, list_01[time].pvalue
    ptot = np.concatenate((pval00, pval01),1)
    np.savetxt(os.path.join(PATH_STATS,'stats_laterality.csv'), ptot, delimiter=',')

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

def plot_contribution_variance_humans(bool_show=False, bool_save=False, figname=None):
    """
    Plots the contribution of individual models to the total variance
    """
    path_data = PATH_STATS
    ffwd_rsquare = np.loadtxt(os.path.join(path_data,'rsquares_ffwd.csv'),delimiter=',')
    fb_rsquare_foreaft = np.loadtxt(os.path.join(path_data,'rsquares_fb_foraft.csv'),delimiter=',')
    fb_rsquare_lateral = np.loadtxt(os.path.join(path_data,'rsquares_fb_lateral.csv'),delimiter=',')

    ffwd_means = np.nanmean(ffwd_rsquare,0)
    fb_means = np.array([np.nanmean(fb_rsquare_foreaft[:,75],0)*(1-ffwd_means[0]), np.nanmean(fb_rsquare_lateral[:,75],0)*(1-ffwd_means[1])])
    positions = [0,0.5]
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    width = 0.2
    axs.bar(positions, ffwd_means, width=width,color='k',edgecolor=None)
    axs.bar(positions, fb_means, width=width, bottom=ffwd_means,color=[0.7,0.7,0.7],edgecolor=None)
    axs.set_ylabel('Explained variance')
    axs.set_xticks([0,0.5])
    axs.set_xlim([-0.5,1])
    axs.set_ylim([-0.05,1.05])
    axs.plot([positions[0]-width/2, positions[1]+width/2], [1,1], color='k', lw=2, ls=':')
    axs.set_yticks([0,0.25,0.5,0.75,1])
    axs.set_yticklabels(['0','25','50','75','100'])
    axs.set_xticklabels(['Foreaft','Lateral'])
    plt.tight_layout()

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    if bool_show:
        plt.show()
    

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

def plot_feedforward_output_humans(tot_input_leg1, tot_output_leg1, tot_subject1, bool_plot=False, bool_save=False, bool_fb=False, figname=None):
    """
    Plots the output of the feedforward model for the camargo dataset
    """
    # Identify the subject with the more data 
    n_subjects = int(np.max(tot_subject1))+1
    len_sub = np.zeros((n_subjects,))
    for sub in range(n_subjects):
        len_sub[sub] = len(np.where(tot_subject1==sub)[0])
    local_sub = np.argmax(len_sub)

    tot_input = tot_input_leg1[tot_subject1==local_sub,:,:]
    tot_output = tot_output_leg1[tot_subject1==local_sub,:]

    tot_velocity = np.nanmean(tot_input_leg1[tot_subject1==local_sub,:,2],1)
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_velocity, tot_output[:,0],s=10,color='b',alpha=0.5)
    local_lin = scipy.stats.linregress(tot_velocity[tot_output[:,0]>400],tot_output[tot_output[:,0]>400,0])
    xinput = np.linspace(800,1800)
    axs.plot(xinput, local_lin.intercept + xinput*local_lin.slope,color='k',lw=3)
    axs.set_xlim([500,2000])
    axs.set_ylim([400,900])
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Step length')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig1.svg'),bbox_inches='tight')

    list_slopes1 = []
    list_slopes2 = []
    for sub in range(n_subjects):
        tmp_vel = np.nanmean(tot_input_leg1[tot_subject1==sub,:,2],1)
        tmp_len_1 = tot_output_leg1[tot_subject1==sub,0]
        tmp_len_2 = tot_output_leg1[tot_subject1==sub,1]
        reg1 = scipy.stats.linregress(tmp_vel, tmp_len_1)
        reg2 = scipy.stats.linregress(tmp_vel, tmp_len_2)
        list_slopes1.append(reg1.slope)
        list_slopes2.append(reg2.slope)
        print(f'Subject number {sub}')
        print(reg1.pvalue, reg2.pvalue)
    np.savetxt(os.path.join(PATH_STATS,'slopes_front.csv'),np.array(list_slopes1),delimiter=',')
    np.savetxt(os.path.join(PATH_STATS,'slopes_hind.csv'),np.array(list_slopes2),delimiter=',')
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    local_lin = scipy.stats.linregress(tot_velocity, tot_output[:,1])
    axs.scatter(tot_velocity, tot_output[:,1],color='b',alpha=0.5,s=10)
    axs.plot(xinput, local_lin.intercept + xinput*local_lin.slope,color='k',lw=3)
    axs.set_xlim([500,2000])
    axs.set_ylim([-500,100])
    axs.set_xlabel('Velocity')
    axs.set_ylabel('Step width')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig2.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def plot_feedforward_body_humans_timing(tot_input_leg1, tot_output_leg1, tot_subject1, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the impact of the velocity on the contact timing
    """

def plot_feedforward_body_humans_final(tot_input_leg1, tot_subject1, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the impact of the velocity on the pelvis position for the human data
    """

    n_humans = int(np.max(tot_subject1)+1)
    list_slopes, list_pvalues = [], []
    for subject in range(n_humans):
        tot_input = tot_input_leg1[tot_subject1==subject,:,:]
        tot_velocity = np.nanmean(tot_input[:,:,2],1)

        amplitude_x = tot_input[:,-1,0] - tot_input[:,0,0]
        amplitude_y = np.nanmax(tot_input[:,:,1],1) - np.nanmin(tot_input[:,:,1],1)
        idx_nans = np.where(~np.isnan(amplitude_y))[0]
        reg_ampl_x = scipy.stats.linregress(tot_velocity[idx_nans], amplitude_x[idx_nans])
        reg_ampl_y = scipy.stats.linregress(tot_velocity[idx_nans], amplitude_y[idx_nans])

        list_slopes.append(np.array([reg_ampl_x.slope, reg_ampl_y.slope]))
        list_pvalues.append(np.array([reg_ampl_x.pvalue, reg_ampl_y.pvalue]))

    mat_slopes = np.array(list_slopes)
    mat_pvalues = np.array(list_pvalues)
    print(f'{len(np.where(mat_pvalues[:,0]<0.05)[0])}/{mat_pvalues.shape[0]}')
    print(f'{len(np.where(mat_pvalues[:,1]<0.05)[0])}/{mat_pvalues.shape[0]}')

    n_subjects = int(np.max(tot_subject1))+1
    len_sub = np.zeros((n_subjects,))
    for sub in range(n_subjects):
        len_sub[sub] = len(np.where(tot_subject1==sub)[0])
    local_sub = np.argmax(len_sub)

    local_input = tot_input_leg1[tot_subject1==local_sub,:,:]
    local_amplitude_x = local_input[:,-1,0] - local_input[:,0,0]
    local_amplitude_y = np.nanmax(local_input[:,:,1],1) - np.nanmin(local_input[:,:,1],1)
    local_velocity = np.nanmean(tot_input_leg1[tot_subject1==local_sub,:,2],1)

    fig, axs = plt.subplots(1,1,figsize=(2,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity, local_amplitude_x,s=10,color='k',alpha=0.5,label='Foreaft')
    axs.scatter(local_velocity, local_amplitude_y,s=10,color='r',alpha=0.5,label='Lateral')
    axs.legend(frameon=False)
    axs.set_yticks([0,500,1000,1500]), axs.set_yticklabels(['0','0.5','1','1.5'])
    axs.set_xticks([1000,2000]), axs.set_xticklabels([1,2]), axs.set_xlim([500,2000])
    axs.set_ylabel('Amplitude')
    axs.set_xlabel('Velocity')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_amplitude.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_exemplar_amplitude.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[0,1], width=0.2, height=np.nanmean(mat_slopes,0), yerr=np.nanstd(mat_slopes,0),color='b')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()



def plot_feedforward_input_humans(tot_input_leg1, tot_self_input_leg1, tot_subject1, bool_plot=False, bool_save=False, bool_fb=False, figname=None):
    """
    Plots the feedforward behavior for the camargo dataset 
    """
    
    # Identify the subject with the more data 
    n_subjects = int(np.max(tot_subject1))+1
    len_sub = np.zeros((n_subjects,))
    for sub in range(n_subjects):
        len_sub[sub] = len(np.where(tot_subject1==sub)[0])
    local_sub = np.argmax(len_sub)

    tot_input = tot_input_leg1[tot_subject1==local_sub,:,:]
    tot_input_self = tot_self_input_leg1[tot_subject1==local_sub,:,:]

    tot_velocity = np.nanmean(tot_input_leg1[tot_subject1==local_sub,:,2],1)
    pctl_velocity = np.nanpercentile(tot_velocity,[33,66,100])
    tot_vel_bin = np.zeros((len(tot_velocity),))
    for line in range(len(tot_velocity)):
        if tot_velocity[line] < pctl_velocity[0]:
            tot_vel_bin[line] = 0
        elif tot_velocity[line] < pctl_velocity[1]:
            tot_vel_bin[line] = 1
        elif tot_velocity[line] < pctl_velocity[2]:
            tot_vel_bin[line] = 2 
        else:
            tot_vel_bin[line] = np.nan

    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(tot_input[tot_vel_bin==0,:,0],0) - np.nanmean(tot_input[tot_vel_bin==0,-1,0]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(tot_input[tot_vel_bin==1,:,0],0) - np.nanmean(tot_input[tot_vel_bin==1,-1,0]),'k',lw=1.2,label='intermediate')
    axs.plot(np.nanmean(tot_input[tot_vel_bin==2,:,0],0) - np.nanmean(tot_input[tot_vel_bin==2,-1,0]),'k',lw=1.7,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(tot_input[tot_vel_bin==0,:,1],0) - np.nanmean(tot_input[tot_vel_bin==0,:,1]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(tot_input[tot_vel_bin==1,:,1],0) - np.nanmean(tot_input[tot_vel_bin==1,:,1]),'k',lw=1.2,label='intermediate')
    axs.plot(np.nanmean(tot_input[tot_vel_bin==2,:,1],0) - np.nanmean(tot_input[tot_vel_bin==2,:,1]),'k',lw=1.7,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(tot_input_self[tot_vel_bin==0,:,0],0) - np.nanmean(tot_input_self[tot_vel_bin==0,-1,0]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(tot_input_self[tot_vel_bin==1,:,0],0) - np.nanmean(tot_input_self[tot_vel_bin==1,-1,0]),'k',lw=1.2,label='intermediate')
    axs.plot(np.nanmean(tot_input_self[tot_vel_bin==2,:,0],0) - np.nanmean(tot_input_self[tot_vel_bin==2,-1,0]),'k',lw=1.7,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Foreaft position')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig3.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(5,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(tot_input_self[tot_vel_bin==0,:,1],0) - np.nanmean(tot_input_self[tot_vel_bin==0,:,1]),'k',lw=0.5,label='slow')
    axs.plot(np.nanmean(tot_input_self[tot_vel_bin==1,:,1],0) - np.nanmean(tot_input_self[tot_vel_bin==1,:,1]),'k',lw=1.2,label='intermediate')
    axs.plot(np.nanmean(tot_input_self[tot_vel_bin==2,:,1],0) - np.nanmean(tot_input_self[tot_vel_bin==2,:,1]),'k',lw=1.7,label='fast')
    axs.set_xlabel('Relative gait fraction')
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.legend(frameon=False)
    axs.set_ylabel('Lateral position')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig4.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_fig4.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()
 
def plot_regression_matrix_humans(regression_matrix, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the feedback gains for the yang dataset
    """
    fig, axs = plt.subplots(1,2,figsize=(10,3),sharex=True,sharey=True)
    axs[0].boxplot(regression_matrix[:,75,:,0,0], positions=np.arange(6), widths=0.25, boxprops=dict(color='k'), medianprops=dict(color='k'), showfliers=False, whis=(5,95), autorange=False)
    axs[1].boxplot(regression_matrix[:,75,:,1,0], positions=np.arange(6), widths=0.25, boxprops=dict(color='k'), medianprops=dict(color='k'), showfliers=False, whis=(5,95), autorange=False)
    axs[0].spines[['top','right']].set_visible(False), axs[1].spines[['top','right']].set_visible(False)
    axs[0].axhline(0, color='k', lw=2, ls=':'), axs[1].axhline(0, color='k', lw=2, ls=':')

    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')


def plot_regression_matrix_matthis(regression_matrix, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the feedback gains for the matthis dataset - modified for the preprint
    """
    idx_plots = [0,1,3,4]
    fig, axs = plt.subplots(2,1,figsize=(3,5), sharex=True)
    axs[0].boxplot(regression_matrix[:,75,idx_plots,0,0], positions=np.arange(4), widths=0.5, boxprops=dict(color='k'), medianprops=dict(color='k'), showfliers=False, whis=(5,95), autorange=False)
    axs[1].boxplot(regression_matrix[:,75,idx_plots,1,0], positions=np.arange(4), widths=0.5, boxprops=dict(color='k'), medianprops=dict(color='k'), showfliers=False, whis=(5,95), autorange=False)
    axs[0].axhline(0, color='k', lw=2, ls=':'), axs[1].axhline(0, color='k', lw=2, ls=':')

    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].set_xticks(np.arange(4))
    axs[0].set_xticklabels([r'$p_x$',r'$p_y$',r'$v_x$',r'$v_y$'])
    axs[1].set_xlabel('Input variable')
    axs[0].set_ylabel('Slope')

    plt.tight_layout()
    if bool_plot:
        plt.show()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

def plot_normalized_feedback_gains(regression_matrix, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the final normalized feedabck gains for the lateral deviation
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter([1],[np.nanmean(regression_matrix[:,75,1,1,0],0)], color='k', s=20)
    axs.plot([1,1], [np.nanmean(regression_matrix[:,75,1,1,0],0)+np.nanstd(regression_matrix[:,75,1,1,0],0),np.nanmean(regression_matrix[:,75,1,1,0],0)-np.nanstd(regression_matrix[:,75,1,1,0],0)], color='k',lw=2)
    axs.set_xlim([0,4])
    axs.set_ylim([0,3])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()


def plot_scatter_laterality(tot_input_leg1, tot_output_leg1, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the scatter plot for the laterality
    """
    idx_nan = np.where(~np.isnan(tot_input_leg1[:,0,0]))[0]
    local_input = tot_input_leg1[idx_nan,:,:]
    local_output = tot_output_leg1[idx_nan,:]

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_input[:,75,1] - np.nanmean(local_input[:,75,1]), 
                local_output[:,1] - np.nanmean(local_output[:,1]), color='k', alpha=0.05)
    xneg, xpos = np.linspace(-100,0), np.linspace(0,100)
    tmp_inp = local_input[:,75,1]-np.nanmean(local_input[:,75,1])
    tmp_out = local_output[:,1]-np.nanmean(local_output[:,1])
    idx_pos = np.where(tmp_inp>0)[0]
    idx_neg = np.where(tmp_inp<0)[0]
    reg_pos = scipy.stats.linregress(tmp_inp[idx_pos], tmp_out[idx_pos])
    reg_neg = scipy.stats.linregress(tmp_inp[idx_neg], tmp_out[idx_neg])
    axs.plot(xneg, xneg*reg_neg.slope + reg_neg.intercept,color='b',lw=2)
    axs.plot(xpos, xpos*reg_pos.slope + reg_pos.intercept,color='r',lw=2)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()

def plot_laterality_regression(regression_small, regression_large, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the laterality information for the human data
    """

    diff_gains = regression_small[:,:,1,1,0] - regression_large[:,:,1,1,0]

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(diff_gains,0),color='k',lw=2)
    axs.fill_between(np.arange(101), np.nanmean(diff_gains,0)+np.nanstd(diff_gains,0),np.nanmean(diff_gains,0)-np.nanstd(diff_gains,0),color='k',alpha=0.3)
    axs.plot(np.nanmean(regression_small[:,:,1,1,0],0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,:,1,1,0],0),'b',lw=2)
    axs.fill_between(np.arange(101),np.nanmean(regression_small[:,:,1,1,0],0)-np.nanstd(regression_small[:,:,1,1,0],0),np.nanmean(regression_small[:,:,1,1,0],0)+np.nanstd(regression_small[:,:,1,1,0],0), color='r', alpha=0.5)
    axs.fill_between(np.arange(101),np.nanmean(regression_large[:,:,1,1,0],0)-np.nanstd(regression_large[:,:,1,1,0],0),np.nanmean(regression_large[:,:,1,1,0],0)+np.nanstd(regression_large[:,:,1,1,0],0), color='b', alpha=0.5)
    axs.axhline(0,color='k',lw=2,ls=':')
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_gains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_gains.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(diff_gains,0),color='k',lw=2)
    axs.axhline(0,color='k',lw=2,ls=':')
    axs.fill_between(np.arange(101), np.nanmean(diff_gains,0)+np.nanstd(diff_gains,0),np.nanmean(diff_gains,0)-np.nanstd(diff_gains,0),color='k',alpha=0.5)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_diffgains.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_diffgains.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(regression_small[:,:,1,1,2]**2,0),'r',lw=2)
    axs.plot(np.nanmean(regression_large[:,:,1,1,2]**2,0),'b',lw=2)
    axs.fill_between(np.arange(101),np.nanmean(regression_small[:,:,1,1,2]**2,0)-np.nanstd(regression_small[:,:,1,1,2]**2,0),np.nanmean(regression_small[:,:,1,1,2]**2,0)+np.nanstd(regression_small[:,:,1,1,2]**2,0), color='r', alpha=0.5)
    axs.fill_between(np.arange(101),np.nanmean(regression_large[:,:,1,1,2]**2,0)-np.nanstd(regression_large[:,:,1,1,2]**2,0),np.nanmean(regression_large[:,:,1,1,2]**2,0)+np.nanstd(regression_large[:,:,1,1,2]**2,0), color='b', alpha=0.5)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_rsquares.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES, f'{figname}_rsquares.svg'),bbox_inches='tight')

    # Plots the feedback gains at contralateral midstance
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)

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
    plt.tight_layout()

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slope.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_slope.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()
def plot_histograms_total_variance(input_variance, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the histograms for the total variance captured
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar(x=[1,1.3],height=np.nanmean(input_variance[:,0,:],0),width=0.2,color='b')
    axs.bar(x=[1,1.3],height=np.nanmean(input_variance[:,1,:],0),bottom=np.nanmean(input_variance[:,0,:],0),width=0.2,color='r')
    axs.set_xlim([0.9, 3.9])
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')
    
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
    axs.scatter([0.95],[np.nanmean(sum_variance[:,0])],color='k',s=20)
    axs.scatter([1.05],[np.nanmean(sum_variance[:,1])],color='r',s=20)
    axs.plot([0.95,0.95],[np.nanmean(sum_variance[:,0])-np.nanstd(sum_variance[:,0]),np.nanmean(sum_variance[:,0])+np.nanstd(sum_variance[:,0])],color='k',lw=2)
    axs.plot([1.05,1.05],[np.nanmean(sum_variance[:,1])-np.nanstd(sum_variance[:,1]),np.nanmean(sum_variance[:,1])+np.nanstd(sum_variance[:,1])],color='r',lw=2)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_total_var.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_total_var.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter([0.95],[np.nanmean(prop_variance[:,0])],color='k',s=20)
    axs.scatter([1.05],[np.nanmean(prop_variance[:,1])],color='r',s=20)
    axs.plot([0.95,0.95],[np.nanmean(prop_variance[:,0])-np.nanstd(prop_variance[:,0]),np.nanmean(prop_variance[:,0])+np.nanstd(prop_variance[:,0])],color='k',lw=2)
    axs.plot([1.05,1.05],[np.nanmean(prop_variance[:,1])-np.nanstd(prop_variance[:,1]),np.nanmean(prop_variance[:,1])+np.nanstd(prop_variance[:,1])],color='r',lw=2)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlim([0,4])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_proportion_var.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_proportion_var.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()


def plot_rsquares_time(time_body, time_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the time rsquares figures
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21), np.nanmedian(time_body,0)[0::5],'b',lw=2)
    axs.plot(np.linspace(0,100,21), np.nanmedian(time_self,0)[0::5],'r',lw=2)
    axs.fill_between(np.linspace(0,100,21), np.nanmedian(time_body,0)[0::5]+scipy.stats.iqr(time_body,0,nan_policy='omit')[0::5]/2,np.nanmedian(time_body,0)[0::5]-scipy.stats.iqr(time_body,0,nan_policy='omit')[0::5]/2,color='b',alpha=0.5)
    axs.fill_between(np.linspace(0,100,21), np.nanmedian(time_self,0)[0::5]+scipy.stats.iqr(time_self,0,nan_policy='omit')[0::5]/2,np.nanmedian(time_self,0)[0::5]-scipy.stats.iqr(time_self,0,nan_policy='omit')[0::5]/2,color='r',alpha=0.5)
    axs.set_ylim([-0.05, 1.05])
    axs.set_yticks([0,0.25,0.5,0.75,1])
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_time.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_time.svg'),bbox_inches='tight')

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

def plot_rsquares_camargo(list_rsquares_body, list_rsquares_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the figure for the rsquares of the camargo datases"""
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21),np.nanmean(list_rsquares_body[0],0)[0::5],'k',lw=2,label='Body')
    axs.plot(np.linspace(0,100,21),np.nanmean(list_rsquares_self[0],0)[0::5],'r',lw=2,label='Self')
    axs.fill_between(np.linspace(0,100,21), np.nanmean(list_rsquares_body[0],0)[0::5]+scipy.stats.iqr(list_rsquares_body[0],0,nan_policy='omit')[0::5]/2, np.nanmean(list_rsquares_body[0],0)[0::5]-scipy.stats.iqr(list_rsquares_body[0],0,nan_policy='omit')[0::5]/2, color='k', alpha=0.5)
    axs.fill_between(np.linspace(0,100,21), np.nanmean(list_rsquares_self[0],0)[0::5]+scipy.stats.iqr(list_rsquares_self[0],0,nan_policy='omit')[0::5]/2, np.nanmean(list_rsquares_self[0],0)[0::5]-scipy.stats.iqr(list_rsquares_self[0],0,nan_policy='omit')[0::5]/2, color='r', alpha=0.5)
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.legend(frameon=False)
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.svg'),bbox_inches='tight')

    # Stats for the figure above
    list_pvalue_1 = []
    for ii in range(list_rsquares_body[0].shape[1]):
        tmp_test = scipy.stats.ks_2samp(list_rsquares_body[0][~np.isnan(list_rsquares_body[0][:,ii]),ii], list_rsquares_self[0][~np.isnan(list_rsquares_self[0][:,ii]),ii], alternative='less')
        list_pvalue_1.append(tmp_test.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21),np.nanmean(list_rsquares_body[1],0)[0::5],'k',lw=2,label='Body')
    axs.plot(np.linspace(0,100,21),np.nanmean(list_rsquares_self[1],0)[0::5],'r',lw=2,label='Self')
    axs.fill_between(np.linspace(0,100,21), np.nanmean(list_rsquares_body[1],0)[0::5]+scipy.stats.iqr(list_rsquares_body[1],0,nan_policy='omit')[0::5]/2, np.nanmean(list_rsquares_body[1],0)[0::5]-scipy.stats.iqr(list_rsquares_body[1],0,nan_policy='omit')[0::5]/2, color='k', alpha=0.5)
    axs.fill_between(np.linspace(0,100,21), np.nanmean(list_rsquares_self[1],0)[0::5]+scipy.stats.iqr(list_rsquares_self[1],0,nan_policy='omit')[0::5]/2, np.nanmean(list_rsquares_self[1],0)[0::5]-scipy.stats.iqr(list_rsquares_self[1],0,nan_policy='omit')[0::5]/2, color='r', alpha=0.5)
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.legend(frameon=False)
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.svg'),bbox_inches='tight')
    if bool_plot:
        plt.show()
    list_pvalue_2 = []
    for ii in range(list_rsquares_body[0].shape[1]):
        tmp_test = scipy.stats.ks_2samp(list_rsquares_body[1][~np.isnan(list_rsquares_body[1][:,ii]),ii], list_rsquares_self[1][~np.isnan(list_rsquares_self[1][:,ii]),ii], alternative='less')
        list_pvalue_2.append(tmp_test.pvalue)

    # Saving the stats 
    tot_pvalue_array = np.hstack((np.expand_dims(np.array(list_pvalue_1),-1), np.expand_dims(np.array(list_pvalue_2),-1)))
    np.savetxt(os.path.join(PATH_STATS,'stats_rsquares_camargo.csv'),tot_pvalue_array,delimiter=',')

def plot_diff_rsquares_camargo(list_rsquares_body, list_rsquares_self, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the figure for the rsquares of the camargo datases"""
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21),np.nanmean(list_rsquares_body[0],0)[0::5]-np.nanmean(list_rsquares_self[0],0)[0::5],'k',lw=2,label='Body')
    axs.fill_between(np.linspace(0,100,21), np.nanmean(list_rsquares_body[0],0)[0::5]-np.nanmean(list_rsquares_self[0],0)[0::5]-scipy.stats.iqr(list_rsquares_body[0][:,0::5]-list_rsquares_self[0][:,0::5],0,nan_policy='omit')/2,
                     np.nanmean(list_rsquares_body[0],0)[0::5]-np.nanmean(list_rsquares_self[0],0)[0::5]+scipy.stats.iqr(list_rsquares_body[0][:,0::5]-list_rsquares_self[0][:,0::5],0,nan_policy='omit')/2, color='k', alpha=0.2)
    axs.set_xticks([0,25,50,75,100])
    axs.axhline(0,color='k',lw=2,ls=':')
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.legend(frameon=False)
    axs.set_ylim([-1.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,'rsquares_diff_humans1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,'rsquares_diff_humans1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.linspace(0,100,21),np.nanmean(list_rsquares_body[1],0)[0::5]-np.nanmean(list_rsquares_self[1],0)[0::5],'k',lw=2,label='Body')
    axs.fill_between(np.linspace(0,100,21), np.nanmean(list_rsquares_body[1],0)[0::5]-np.nanmean(list_rsquares_self[1],0)[0::5]-scipy.stats.iqr(list_rsquares_body[0][:,0::5]-list_rsquares_self[1][:,0::5],0,nan_policy='omit')/2,
                     np.nanmean(list_rsquares_body[1],0)[0::5]-np.nanmean(list_rsquares_self[1],0)[0::5]+scipy.stats.iqr(list_rsquares_body[1][:,0::5]-list_rsquares_self[0][:,0::5],0,nan_policy='omit')/2, color='k', alpha=0.2)
    axs.set_xticks([0,25,50,75,100])
    axs.axhline(0,color='k',lw=2,ls=':')
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    axs.legend(frameon=False)
    axs.set_ylim([-1.05,1.05])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,'rsquares_diff_humans2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,'rsquares_diff_humans2.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

def plot_horizon_humans(list_input_time, list_subjects, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the horizon study on human data
    """
    vec_slope = np.zeros((4,1))
    vec_rsquare = np.zeros((4,1))
    for hor in range(len(list_input_time)):
        local_input = list_input_time[hor]
        plt.plot(np.nanmean(local_input[:,:,1],0),'r')
        idx_nans = np.where(~np.isnan(local_input[:,0,0]))[0]
        reg = scipy.stats.linregress(local_input[idx_nans,0,1], local_input[idx_nans,-1,1])
        vec_slope[hor] = reg.slope
        vec_rsquare[hor] = reg.rvalue
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,5), vec_rsquare**2, color='k',s=20)
    axs.set_xlabel('Horizon')
    axs.set_ylabel('Rsquare')
    plt.tight_layout()

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.arange(1,5), vec_rsquare, color='k', s=20)
    axs.set_xlabel('Horizon')
    axs.set_ylabel('Slope')
    plt.tight_layout()

    if bool_plot:
        plt.show()


def plot_feedforward_humans(tot_input_matrix, tot_output_matrix, tot_trial, tot_leg, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the information for the human feedforward behavior
    """
    n_trial = np.max(tot_trial) + 1
    # FIGURE 0 - Plot the variability of an exemplar subject
    trial = 11
    idx_leg_1 = np.where((tot_trial==trial) & (tot_leg==0))[0]
    idx_leg_2 = np.where((tot_trial==trial) & (tot_leg==1))[0]
    fig, axs = plt.subplots(1,1,figsize=(3,5))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_output_matrix[idx_leg_1,0], tot_output_matrix[idx_leg_1,1],color='r',s=10,alpha=0.5)
    axs.scatter(tot_output_matrix[idx_leg_2,0], tot_output_matrix[idx_leg_2,1],color='b',s=10,alpha=0.5)
    plt.tight_layout()


    # FIGURE 1 - Plot the variability of contact locations
    list_var, list_var_od = [], []
    for trial in range(n_trial.astype(int)):
        idx_trial_leg_1 = np.where((tot_trial==trial) & (tot_leg==0))[0]
        idx_trial_leg_2 = np.where((tot_trial==trial) & (tot_leg==1))[0]
        tmp_len_1 = tot_output_matrix[(idx_trial_leg_1),1]
        tmp_wid_1 = tot_output_matrix[(idx_trial_leg_1),0]
        tmp_len_1_od = tot_output_matrix[(idx_trial_leg_2),1]
        tmp_wid_1_od = tot_output_matrix[(idx_trial_leg_2),0]
        list_var.append(np.array([scipy.stats.iqr(tmp_len_1, nan_policy='omit'),
                                  scipy.stats.iqr(tmp_wid_1, nan_policy='omit')]))
        list_var_od.append(np.array([scipy.stats.iqr(tmp_len_1_od, nan_policy='omit'),
                                     scipy.stats.iqr(tmp_wid_1_od, nan_policy='omit')]))
    array_var, array_var_od = np.array(list_var), np.array(list_var_od)

    fig, axs = plt.subplots(1, 1, figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.bar([0,0.5], np.nanmedian(array_var,0), width=0.2, yerr=scipy.stats.iqr(array_var,0,nan_policy='omit'), color=['r','r'], ecolor='k')
    plt.tight_layout()
    plt.show()

def plot_laterality_camargo(large_diag, small_diag, large_odiag, small_odiag, bool_plot=False, bool_save=False, figname=None):
    """
    Plot the laterality metrics for the Camargo
    """
    input_in = np.concatenate((large_diag, small_odiag),0)
    input_out = np.concatenate((small_diag, large_odiag),0)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_in[:,:,1,1,2]**2,0), color='k',lw=2,label='inward')
    axs.plot(np.nanmean(input_out[:,:,1,1,2]**2,0), color=[0.7,0.7,0.7], lw=2, label='outward')
    axs.fill_between(np.arange(101), np.nanmean(input_in[:,:,1,1,2]**2,0)+scipy.stats.iqr(input_in[:,:,1,1,2]**2,0,nan_policy='omit')/2,np.nanmean(input_in[:,:,1,1,2]**2,0)-scipy.stats.iqr(input_in[:,:,1,1,2]**2,0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(101), np.nanmean(input_out[:,:,1,1,2]**2,0)+scipy.stats.iqr(input_out[:,:,1,1,2]**2,0,nan_policy='omit')/2,np.nanmean(input_out[:,:,1,1,2]**2,0)-scipy.stats.iqr(input_out[:,:,1,1,2]**2,0,nan_policy='omit')/2, color=[0.7,0.7,0.7], alpha=0.5)
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-0.05,1.05])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_1.svg'),bbox_inches='tight')

    # Statistics for the figure above
    list_pvalue1 = []
    for time in range(input_in.shape[1]):
        local_stats = scipy.stats.ks_2samp(input_in[:,time,1,1,2]**2, input_out[:,time,1,1,2]**2)
        list_pvalue1.append(local_stats.pvalue)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmean(input_in[:,:,1,1,0],0), color='k',lw=2,label='inward')
    axs.plot(np.nanmean(input_out[:,:,1,1,0],0), color=[0.7,0.7,0.7], lw=2, label='outward')
    axs.fill_between(np.arange(101), np.nanmean(input_in[:,:,1,1,0],0)+scipy.stats.iqr(input_in[:,:,1,1,0],0,nan_policy='omit')/2,np.nanmean(input_in[:,:,1,1,0],0)-scipy.stats.iqr(input_in[:,:,1,1,0],0,nan_policy='omit')/2, color='k', alpha=0.5)
    axs.fill_between(np.arange(101), np.nanmean(input_out[:,:,1,1,0],0)+scipy.stats.iqr(input_out[:,:,1,1,0],0,nan_policy='omit')/2,np.nanmean(input_out[:,:,1,1,0],0)-scipy.stats.iqr(input_out[:,:,1,1,0],0,nan_policy='omit')/2, color=[0.7,0.7,0.7], alpha=0.5)
    axs.set_xticks([0,25,50,75,100])
    axs.set_xticklabels(['-1','-0.75','-0.5','-0.25','0'])
    axs.set_ylim([-1.5,3])
    axs.set_xlabel('Relative gait fraction')
    axs.set_ylabel('Slope')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}_2.svg'),bbox_inches='tight')

    list_pvalue2 = []
    for time in range(input_in.shape[1]):
        local_stats = scipy.stats.ks_2samp(input_in[:,time,1,1,0], input_out[:,time,1,1,0])
        list_pvalue2.append(local_stats.pvalue)

    tot_array_stats = np.hstack((np.expand_dims(np.array(list_pvalue1),-1), np.expand_dims(np.array(list_pvalue2),-1)))
    np.savetxt(os.path.join(PATH_STATS,'stats_laterality_camargo.csv'),tot_array_stats,delimiter=',')

    if bool_plot:
        plt.show()

def plot_sample_data_laterality_camargo(tot_input_leg1, tot_output_leg1, time, bool_plot=False, bool_save=False, figname=None):
    """
    Plotting an exemplar sample data from the camargo dataset
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(tot_input_leg1[:,time, 1] - np.nanmean(tot_input_leg1[:,time,1],0), tot_output_leg1[:,1] - np.nanmean(tot_output_leg1[:,1]),color='k', alpha=0.5, s=5)
    axs.set_xlabel('Pelvis lateral position')
    axs.set_ylabel('Foot lateral position')
    axs.axhline(0, color='k', ls=':', lw=2)
    axs.axvline(0, color='k', ls=':', lw=2)
    axs.set_xlim([-150,150])
    axs.set_ylim([-300,300])
    plt.tight_layout()

    if bool_save:
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.png'),bbox_inches='tight')
        fig.savefig(os.path.join(PATH_FIGURES,f'{figname}.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()

    
