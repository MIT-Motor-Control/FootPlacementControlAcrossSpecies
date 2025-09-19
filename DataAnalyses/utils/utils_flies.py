import numpy as np 
import pickle
import os, sys
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import _pca
import scipy.signal
import scipy.stats
from scipy.signal import find_peaks
from tqdm import tqdm

def compute_variance_contributions(matrix_rsquares_ffwd, leg1_foreaft, leg1_lateral, leg2_foreaft, leg2_lateral, leg3_foreaft, leg3_lateral):
    """
    Computes the individual contributions of feedforward and feedback to the total variance explained for each individual
    """
    output_matrix = np.zeros((matrix_rsquares_ffwd.shape[0],2,6))
    for animal in range(output_matrix.shape[0]):
        output_matrix[animal,0,0] = matrix_rsquares_ffwd[animal,0]
        output_matrix[animal,0,1] = matrix_rsquares_ffwd[animal,1]
        output_matrix[animal,0,2] = matrix_rsquares_ffwd[animal,2]
        output_matrix[animal,0,3] = matrix_rsquares_ffwd[animal,3]
        output_matrix[animal,0,4] = matrix_rsquares_ffwd[animal,4]
        output_matrix[animal,0,5] = matrix_rsquares_ffwd[animal,5]
        output_matrix[animal,1,0] = leg1_foreaft[animal,15]*(1-matrix_rsquares_ffwd[animal,0])
        output_matrix[animal,1,1] = leg1_lateral[animal,15]*(1-matrix_rsquares_ffwd[animal,1])
        output_matrix[animal,1,2] = leg2_foreaft[animal,15]*(1-matrix_rsquares_ffwd[animal,2])
        output_matrix[animal,1,3] = leg2_lateral[animal,15]*(1-matrix_rsquares_ffwd[animal,3])
        output_matrix[animal,1,4] = leg3_foreaft[animal,15]*(1-matrix_rsquares_ffwd[animal,4])
        output_matrix[animal,1,5] = leg3_lateral[animal,15]*(1-matrix_rsquares_ffwd[animal,5])
    
    return output_matrix

def compute_variance_contributions_time(matrix_rsquares_ffwd, leg1_foreaft, leg1_lateral, leg2_foreaft, leg2_lateral, leg3_foreaft, leg3_lateral):
    """
    Computes the individual contributions of feedforward and feedback to the total variance explained for each individual
    """
    output_matrix = np.zeros((matrix_rsquares_ffwd.shape[0],2,6,21))
    for animal in range(output_matrix.shape[0]):
        for time in range(21):
            output_matrix[animal,0,0,time] = matrix_rsquares_ffwd[animal,0]
            output_matrix[animal,0,1,time] = matrix_rsquares_ffwd[animal,1]
            output_matrix[animal,0,2,time] = matrix_rsquares_ffwd[animal,2]
            output_matrix[animal,0,3,time] = matrix_rsquares_ffwd[animal,3]
            output_matrix[animal,0,4,time] = matrix_rsquares_ffwd[animal,4]
            output_matrix[animal,0,5,time] = matrix_rsquares_ffwd[animal,5]
            output_matrix[animal,1,0,time] = leg1_foreaft[animal,time]*(1-matrix_rsquares_ffwd[animal,0])
            output_matrix[animal,1,1,time] = leg1_lateral[animal,time]*(1-matrix_rsquares_ffwd[animal,1])
            output_matrix[animal,1,2,time] = leg2_foreaft[animal,time]*(1-matrix_rsquares_ffwd[animal,2])
            output_matrix[animal,1,3,time] = leg2_lateral[animal,time]*(1-matrix_rsquares_ffwd[animal,3])
            output_matrix[animal,1,4,time] = leg3_foreaft[animal,time]*(1-matrix_rsquares_ffwd[animal,4])
            output_matrix[animal,1,5,time] = leg3_lateral[animal,time]*(1-matrix_rsquares_ffwd[animal,5])
    
    return output_matrix


def compute_ffwd_inputs(tot_input, tot_video):
    """
    Computes the body fixed marker information for the feedforward controller
    """
    n_video = 8
    mat_slope, mat_pvalue = np.zeros((n_video,2)), np.zeros((n_video,2))
    for video in range(1,n_video+1):
        local_input = tot_input[tot_video[:,0]==video,:,:]
        local_velocity = np.nanmean(local_input[:,:,2],1)
        local_x_amplitude = np.zeros((local_velocity.shape[0],))
        local_y_amplitude = np.zeros((local_velocity.shape[0],))
        for line in range(local_x_amplitude.shape[0]):
            local_x_amplitude[line] = local_input[line,-1,0] - local_input[line,0,0]
            local_y_amplitude[line] = np.max(local_input[line,:,1]) - np.min(local_input[line,:,1])

        # Performing the regressions between these factors 
        idx_nans = np.where(~np.isnan(local_velocity))[0]
        reg_x_amplitude = scipy.stats.linregress(local_velocity[idx_nans], local_x_amplitude[idx_nans])
        reg_y_amplitude = scipy.stats.linregress(local_velocity[idx_nans], local_y_amplitude[idx_nans])
        mat_slope[video-1,0], mat_slope[video-1,1] = reg_x_amplitude.slope, reg_y_amplitude.slope
        mat_pvalue[video-1,0], mat_pvalue[video-1,1] = reg_x_amplitude.pvalue, reg_y_amplitude.pvalue
    
    return mat_slope, mat_pvalue

def extract_metrics_threshold(foot_output, chunk, raw_data):
    """Extracts the contact metrics based on the threshold method"""
    idx_nans = np.where(np.isnan(foot_output[:,0,0]))[0]
    output_metrics = np.zeros((1,8))
    local_begin = idx_nans[chunk]
    local_end = idx_nans[chunk+1]
    #local_matrix_foot = foot_output[local_begin:local_end,:,:]
    local_matrix_raw = reshape_raw_data(raw_data[local_begin:local_end,:])
    local_matrix_foot = reshape_foot_data(foot_output[local_begin:local_end,:,:])
    time_vector = np.linspace(0,1/150*local_matrix_foot.shape[0],local_matrix_foot.shape[0])
    local_velocity = compute_velocity_com(np.squeeze(local_matrix_raw[:,0,6]))
    for time in range(1,len(time_vector)-2):
        # First leg
        if (local_matrix_foot[time-1,0,0]==0 and local_matrix_foot[time,0,0]==0 and local_matrix_foot[time+1,0,0]!=0 and local_matrix_foot[time+2,0,0]!=0):
            init_pos_x = local_matrix_foot[time+1,0,0]
            init_pos_y = local_matrix_foot[time+1,1,0]
            idx_leg0 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,0]==1)[0]
            idx_leg1 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,1]==1)[0]
            idx_leg2 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,2]==1)[0]
            idx_leg3 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,3]==1)[0]
            idx_leg4 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,4]==1)[0]
            idx_leg5 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,5]==1)[0]
            if (len(idx_leg0)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg0[0],0,0]
                final_pos_y = local_matrix_foot[time+2+idx_leg0[0],1,0]
                output_metrics = np.vstack((output_metrics, np.array([chunk,1,1,1/150*(idx_leg0[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg0[0]]),local_begin+time+1])))
            if (len(idx_leg1)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg1[0],0,1]
                final_pos_y = local_matrix_foot[time+2+idx_leg1[0],1,1]
                output_metrics = np.vstack((output_metrics, np.array([chunk,1,2,1/150*(idx_leg1[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg1[0]]),local_begin+time+1])))
            if (len(idx_leg2)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg2[0],0,2]
                final_pos_y = local_matrix_foot[time+2+idx_leg2[0],1,2]
                output_metrics = np.vstack((output_metrics, np.array([chunk,1,3,1/150*(idx_leg2[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg2[0]]),local_begin+time+1])))
            if (len(idx_leg3)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg3[0],0,3]
                final_pos_y = local_matrix_foot[time+2+idx_leg3[0],1,3]
                output_metrics = np.vstack((output_metrics, np.array([chunk,1,4,1/150*(idx_leg3[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg3[0]]),local_begin+time+1])))
            if (len(idx_leg4)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg4[0],0,4]
                final_pos_y = local_matrix_foot[time+2+idx_leg4[0],1,4]
                output_metrics = np.vstack((output_metrics, np.array([chunk,1,5,1/150*(idx_leg4[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg4[0]]),local_begin+time+1])))
            if (len(idx_leg5)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg5[0],0,5]
                final_pos_y = local_matrix_foot[time+2+idx_leg5[0],1,5]
                output_metrics = np.vstack((output_metrics, np.array([chunk,1,6,1/150*(idx_leg5[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg5[0]]),local_begin+time+1])))
        # Second leg
        elif (local_matrix_foot[time-1,0,1]==0 and local_matrix_foot[time,0,1]==0 and local_matrix_foot[time+1,0,1]!=0 and local_matrix_foot[time+2,0,1]!=0):
            init_pos_x = local_matrix_foot[time+1,0,1]
            init_pos_y = local_matrix_foot[time+1,1,1]
            idx_leg0 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,0]==1)[0]
            idx_leg1 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,1]==1)[0]
            idx_leg2 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,2]==1)[0]
            idx_leg3 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,3]==1)[0]
            idx_leg4 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,4]==1)[0]
            idx_leg5 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,5]==1)[0]
            if (len(idx_leg0)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg0[0],0,0]
                final_pos_y = local_matrix_foot[time+2+idx_leg0[0],1,0]
                output_metrics = np.vstack((output_metrics, np.array([chunk,2,1,1/150*(idx_leg0[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg0[0]]),local_begin+time+1])))
            if (len(idx_leg1)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg1[0],0,1]
                final_pos_y = local_matrix_foot[time+2+idx_leg1[0],1,1]
                output_metrics = np.vstack((output_metrics, np.array([chunk,2,2,1/150*(idx_leg1[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg1[0]]),local_begin+time+1])))
            if (len(idx_leg2)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg2[0],0,2]
                final_pos_y = local_matrix_foot[time+2+idx_leg2[0],1,2]
                output_metrics = np.vstack((output_metrics, np.array([chunk,2,3,1/150*(idx_leg2[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg2[0]]),local_begin+time+1])))
            if (len(idx_leg3)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg3[0],0,3]
                final_pos_y = local_matrix_foot[time+2+idx_leg3[0],1,3]
                output_metrics = np.vstack((output_metrics, np.array([chunk,2,4,1/150*(idx_leg3[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg3[0]]),local_begin+time+1])))
            if (len(idx_leg4)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg4[0],0,4]
                final_pos_y = local_matrix_foot[time+2+idx_leg4[0],1,4]
                output_metrics = np.vstack((output_metrics, np.array([chunk,2,5,1/150*(idx_leg4[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg4[0]]),local_begin+time+1])))
            if (len(idx_leg5)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg5[0],0,5]
                final_pos_y = local_matrix_foot[time+2+idx_leg5[0],1,5]
                output_metrics = np.vstack((output_metrics, np.array([chunk,2,6,1/150*(idx_leg5[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmean(local_velocity[time:time+idx_leg5[0]]),local_begin+time+1])))
        # Third leg
        elif (local_matrix_foot[time-1,0,2]==0 and local_matrix_foot[time,0,2]==0 and local_matrix_foot[time+1,0,2]!=0 and local_matrix_foot[time+2,0,2]!=0):
            init_pos_x = local_matrix_foot[time+1,0,2]
            init_pos_y = local_matrix_foot[time+1,1,2]
            idx_leg0 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,0]==1)[0]
            idx_leg1 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,1]==1)[0]
            idx_leg2 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,2]==1)[0]
            idx_leg3 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,3]==1)[0]
            idx_leg4 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,4]==1)[0]
            idx_leg5 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,5]==1)[0]
            if (len(idx_leg0)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg0[0],0,0]
                final_pos_y = local_matrix_foot[time+2+idx_leg0[0],1,0]
                output_metrics = np.vstack((output_metrics, np.array([chunk,3,1,1/150*(idx_leg0[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg0[0]]),local_begin+time+1])))
            if (len(idx_leg1)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg1[0],0,1]
                final_pos_y = local_matrix_foot[time+2+idx_leg1[0],1,1]
                output_metrics = np.vstack((output_metrics, np.array([chunk,3,2,1/150*(idx_leg1[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg1[0]]),local_begin+time+1])))
            if (len(idx_leg2)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg2[0],0,2]
                final_pos_y = local_matrix_foot[time+2+idx_leg2[0],1,2]
                output_metrics = np.vstack((output_metrics, np.array([chunk,3,3,1/150*(idx_leg2[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg2[0]]),local_begin+time+1])))
            if (len(idx_leg3)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg3[0],0,3]
                final_pos_y = local_matrix_foot[time+2+idx_leg3[0],1,3]
                output_metrics = np.vstack((output_metrics, np.array([chunk,3,4,1/150*(idx_leg3[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg3[0]]),local_begin+time+1])))
            if (len(idx_leg4)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg4[0],0,4]
                final_pos_y = local_matrix_foot[time+2+idx_leg4[0],1,4]
                output_metrics = np.vstack((output_metrics, np.array([chunk,3,5,1/150*(idx_leg4[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg4[0]]),local_begin+time+1])))
            if (len(idx_leg5)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg5[0],0,5]
                final_pos_y = local_matrix_foot[time+2+idx_leg5[0],1,5]
                output_metrics = np.vstack((output_metrics, np.array([chunk,3,6,1/150*(idx_leg5[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg5[0]]),local_begin+time+1])))
        # Fourth leg
        elif (local_matrix_foot[time-1,0,3]==0 and local_matrix_foot[time,0,3]==0 and local_matrix_foot[time+1,0,3]!=0 and local_matrix_foot[time+2,0,3]!=0):
            init_pos_x = local_matrix_foot[time+1,0,3]
            init_pos_y = local_matrix_foot[time+1,1,3]
            idx_leg0 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,0]==1)[0]
            idx_leg1 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,1]==1)[0]
            idx_leg2 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,2]==1)[0]
            idx_leg3 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,3]==1)[0]
            idx_leg4 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,4]==1)[0]
            idx_leg5 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,5]==1)[0]
            if (len(idx_leg0)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg0[0],0,0]
                final_pos_y = local_matrix_foot[time+2+idx_leg0[0],1,0]
                output_metrics = np.vstack((output_metrics, np.array([chunk,4,1,1/150*(idx_leg0[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg0[0]]),local_begin+time+1])))
            if (len(idx_leg1)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg1[0],0,1]
                final_pos_y = local_matrix_foot[time+2+idx_leg1[0],1,1]
                output_metrics = np.vstack((output_metrics, np.array([chunk,4,2,1/150*(idx_leg1[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg1[0]]),local_begin+time+1])))
            if (len(idx_leg2)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg2[0],0,2]
                final_pos_y = local_matrix_foot[time+2+idx_leg2[0],1,2]
                output_metrics = np.vstack((output_metrics, np.array([chunk,4,3,1/150*(idx_leg2[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg2[0]]),local_begin+time+1])))
            if (len(idx_leg3)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg3[0],0,3]
                final_pos_y = local_matrix_foot[time+2+idx_leg3[0],1,3]
                output_metrics = np.vstack((output_metrics, np.array([chunk,4,4,1/150*(idx_leg3[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg3[0]]),local_begin+time+1])))
            if (len(idx_leg4)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg4[0],0,4]
                final_pos_y = local_matrix_foot[time+2+idx_leg4[0],1,4]
                output_metrics = np.vstack((output_metrics, np.array([chunk,4,5,1/150*(idx_leg4[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg4[0]]),local_begin+time+1])))
            if (len(idx_leg5)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg5[0],0,5]
                final_pos_y = local_matrix_foot[time+2+idx_leg5[0],1,5]
                output_metrics = np.vstack((output_metrics, np.array([chunk,4,6,1/150*(idx_leg5[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg5[0]]),local_begin+time+1])))
        # Fifth leg
        elif (local_matrix_foot[time-1,0,4]==0 and local_matrix_foot[time,0,4]==0 and local_matrix_foot[time+1,0,4]!=0 and local_matrix_foot[time+2,0,4]!=0):
            init_pos_x = local_matrix_foot[time+1,0,4]
            init_pos_y = local_matrix_foot[time+1,1,4]
            idx_leg0 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,0]==1)[0]
            idx_leg1 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,1]==1)[0]
            idx_leg2 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,2]==1)[0]
            idx_leg3 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,3]==1)[0]
            idx_leg4 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,4]==1)[0]
            idx_leg5 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,5]==1)[0]
            if (len(idx_leg0)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg0[0],0,0]
                final_pos_y = local_matrix_foot[time+2+idx_leg0[0],1,0]
                output_metrics = np.vstack((output_metrics, np.array([chunk,5,1,1/150*(idx_leg0[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg0[0]]),local_begin+time+1])))
            if (len(idx_leg1)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg1[0],0,1]
                final_pos_y = local_matrix_foot[time+2+idx_leg1[0],1,1]
                output_metrics = np.vstack((output_metrics, np.array([chunk,5,2,1/150*(idx_leg1[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg1[0]]),local_begin+time+1])))
            if (len(idx_leg2)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg2[0],0,2]
                final_pos_y = local_matrix_foot[time+2+idx_leg2[0],1,2]
                output_metrics = np.vstack((output_metrics, np.array([chunk,5,3,1/150*(idx_leg2[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg2[0]]),local_begin+time+1])))
            if (len(idx_leg3)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg3[0],0,3]
                final_pos_y = local_matrix_foot[time+2+idx_leg3[0],1,3]
                output_metrics = np.vstack((output_metrics, np.array([chunk,5,4,1/150*(idx_leg3[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg3[0]]),local_begin+time+1])))
            if (len(idx_leg4)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg4[0],0,4]
                final_pos_y = local_matrix_foot[time+2+idx_leg4[0],1,4]
                output_metrics = np.vstack((output_metrics, np.array([chunk,5,5,1/150*(idx_leg4[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg4[0]]),local_begin+time+1])))
            if (len(idx_leg5)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg5[0],0,5]
                final_pos_y = local_matrix_foot[time+2+idx_leg5[0],1,5]
                output_metrics = np.vstack((output_metrics, np.array([chunk,5,6,1/150*(idx_leg5[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg5[0]]),local_begin+time+1])))
        # Sixth leg
        elif (local_matrix_foot[time-1,0,5]==0 and local_matrix_foot[time,0,5]==0 and local_matrix_foot[time+1,0,5]!=0 and local_matrix_foot[time+2,0,5]!=0):
            init_pos_x = local_matrix_foot[time+1,0,5]
            init_pos_y = local_matrix_foot[time+1,1,5]
            idx_leg0 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,0]==1)[0]
            idx_leg1 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,1]==1)[0]
            idx_leg2 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,2]==1)[0]
            idx_leg3 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,3]==1)[0]
            idx_leg4 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,4]==1)[0]
            idx_leg5 = np.where(foot_contact_detection_threshold(local_matrix_foot[time+2:,:,:6])[:,5]==1)[0]
            if (len(idx_leg0)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg0[0],0,0]
                final_pos_y = local_matrix_foot[time+2+idx_leg0[0],1,0]
                output_metrics = np.vstack((output_metrics, np.array([chunk,6,1,1/150*(idx_leg0[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg0[0]]),local_begin+time+1])))
            if (len(idx_leg1)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg1[0],0,1]
                final_pos_y = local_matrix_foot[time+2+idx_leg1[0],1,1]
                output_metrics = np.vstack((output_metrics, np.array([chunk,6,2,1/150*(idx_leg1[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg1[0]]),local_begin+time+1])))
            if (len(idx_leg2)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg2[0],0,2]
                final_pos_y = local_matrix_foot[time+2+idx_leg2[0],1,2]
                output_metrics = np.vstack((output_metrics, np.array([chunk,6,3,1/150*(idx_leg2[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg2[0]]),local_begin+time+1])))
            if (len(idx_leg3)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg3[0],0,3]
                final_pos_y = local_matrix_foot[time+2+idx_leg3[0],1,3]
                output_metrics = np.vstack((output_metrics, np.array([chunk,6,4,1/150*(idx_leg3[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg3[0]]),local_begin+time+1])))
            if (len(idx_leg4)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg4[0],0,4]
                final_pos_y = local_matrix_foot[time+2+idx_leg4[0],1,4]
                output_metrics = np.vstack((output_metrics, np.array([chunk,6,5,1/150*(idx_leg4[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg4[0]]),local_begin+time+1])))
            if (len(idx_leg5)!=0):
                final_pos_x = local_matrix_foot[time+2+idx_leg5[0],0,5]
                final_pos_y = local_matrix_foot[time+2+idx_leg5[0],1,5]
                output_metrics = np.vstack((output_metrics, np.array([chunk,6,6,1/150*(idx_leg5[0]+1),np.abs(final_pos_x-init_pos_x),np.abs(final_pos_y-init_pos_y),np.nanmedian(local_velocity[time:time+idx_leg5[0]]),local_begin+time+1])))
    return output_metrics[1:,:]

def compute_velocity_com(com_pos_x):
    """Computes the velocity of the com based on the 4th order centered finite difference scheme"""
    dt = 1/150
    output_velocity = np.zeros((com_pos_x.shape[0],))
    output_velocity[2:-2] = (-com_pos_x[4:]+8*com_pos_x[3:-1]-8*com_pos_x[1:-3]+com_pos_x[:-4])/(12*dt)
    return output_velocity

def reshape_raw_data_or(local_raw_data):
    """Reshapes the raw data to work with the foot contact information + orientation info"""
    n_time = local_raw_data.shape[0]
    output_matrix = np.zeros((n_time,2,8))
    for axis in range(2):
        output_matrix[:,axis,0] = local_raw_data[:,8+axis]
        output_matrix[:,axis,1] = local_raw_data[:,10+axis]
        output_matrix[:,axis,2] = local_raw_data[:,12+axis]
        output_matrix[:,axis,3] = local_raw_data[:,14+axis]
        output_matrix[:,axis,4] = local_raw_data[:,16+axis]
        output_matrix[:,axis,5] = local_raw_data[:,18+axis]
        output_matrix[:,axis,6] = local_raw_data[:,6+axis]

    output_matrix[:,0,7] = local_raw_data[:,3]
    output_matrix[:,1,7] = local_raw_data[:,3]
    return output_matrix

def reshape_raw_data(local_raw_data):
    """Reshapes the raw data to work with the foot contact information"""
    n_time = local_raw_data.shape[0]
    output_matrix = np.zeros((n_time,2,7))
    for axis in range(2):
        output_matrix[:,axis,0] = local_raw_data[:,8+axis]
        output_matrix[:,axis,1] = local_raw_data[:,10+axis]
        output_matrix[:,axis,2] = local_raw_data[:,12+axis]
        output_matrix[:,axis,3] = local_raw_data[:,14+axis]
        output_matrix[:,axis,4] = local_raw_data[:,16+axis]
        output_matrix[:,axis,5] = local_raw_data[:,18+axis]
        output_matrix[:,axis,6] = local_raw_data[:,6+axis]
    return output_matrix

def reshape_foot_data(input_foot_data):
    """Reshapes the foot data (inverse the last two dimensions)"""
    output_foot_data = np.zeros((input_foot_data.shape[0],input_foot_data.shape[2], input_foot_data.shape[1]))
    for dim2 in range(output_foot_data.shape[1]):
        output_foot_data[:,dim2,0] = input_foot_data[:,0,dim2]
        output_foot_data[:,dim2,1] = input_foot_data[:,1,dim2]
        output_foot_data[:,dim2,2] = input_foot_data[:,2,dim2]
        output_foot_data[:,dim2,3] = input_foot_data[:,3,dim2]
        output_foot_data[:,dim2,4] = input_foot_data[:,4,dim2]
        output_foot_data[:,dim2,5] = input_foot_data[:,5,dim2]
    return output_foot_data

def foot_contact_detection_threshold(input_data):
    """Extracts the foot contact based on a velocity threshold"""
    foot_contact_matrix = np.zeros((input_data.shape[0], input_data.shape[2]))
    for ii in range(input_data.shape[2]):
        for jj in range(1,input_data.shape[0]-2):
            if ((int(input_data[jj-1,0,ii]==0)) and (int(input_data[jj,0,ii]==0)) and (int(input_data[jj+1,0,ii]!=0)) and (int(input_data[jj+2,0,ii]!=0))):
                foot_contact_matrix[jj+1,ii] = 1 
            elif ((int(input_data[jj-1,0,ii]!=0)) and (int(input_data[jj,0,ii]!=0)) and (int(input_data[jj+1,0,ii]==0)) and (int(input_data[jj+2,0,ii]==0))):
                foot_contact_matrix[jj+1,ii] = -1
    return foot_contact_matrix

def load_fly_data():
    """
    Load the fly related to input data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    fly_matrix_diag = np.load(os.path.join(path_loading,'fly_id_diag.npy'))
    fly_matrix_odiag = np.load(os.path.join(path_loading,'fly_id_odiag.npy'))
    return fly_matrix_diag, fly_matrix_odiag

def load_video_data():
    """
    Load the video related input data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    video_matrix_diag = np.load(os.path.join(path_loading,'video_id_diag.npy'))
    video_matrix_odiag = np.load(os.path.join(path_loading,'video_id_odiag.npy'))
    return video_matrix_diag, video_matrix_odiag

def load_video_data_mm():
    """
    Load the video related input data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    video_matrix_diag = np.load(os.path.join(path_loading,'video_id_diag_mm_fixed.npy'))
    video_matrix_odiag = np.load(os.path.join(path_loading,'video_id_odiag_mm_fixed.npy'))
    return video_matrix_diag, video_matrix_odiag

def load_video_data_mm_abs():
    """
    Load the video related input data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io_abs')
    video_matrix_diag = np.load(os.path.join(path_loading, 'video_id_diag_mm_norm_fixed.npy'))
    video_matrix_odiag = np.load(os.path.join(path_loading, 'video_id_odiag_mm_norm_fixed.npy'))
    return video_matrix_diag, video_matrix_odiag

def load_video_data_mm_touchdown():
    """
    Load the video related input data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io_touchdown')
    video_matrix_diag = np.load(os.path.join(path_loading, 'video_id_diag_mm_norm_fixed.npy'))
    video_matrix_odiag = np.load(os.path.join(path_loading, 'video_id_odiag_mm_norm_fixed.npy'))
    return video_matrix_diag, video_matrix_odiag

def load_video_data_mm_():
    """
    Load the video related input data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    video_matrix_diag = np.load(os.path.join(path_loading,'video_id_diag_mm_norm_fixed_sub.npy'))
    return video_matrix_diag

def load_video_data_mm_norm():
    """
    Load the video related to the input (normalized) data
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    video_matrix_diag = np.load(os.path.join(path_loading,'video_id_diag_mm_norm.npy'))
    video_matrix_odiag = np.load(os.path.join(path_loading,'video_id_odiag_mm_norm.npy'))
    return video_matrix_diag, video_matrix_odiag


def load_video_data_horizon(horizon):
    """
    Load the video related input data for a specific horizon
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    video_matrix_diag = np.load(os.path.join(path_loading,f'list_video_id_{horizon}.npy'))
    return video_matrix_diag

def load_all_data_horizon(horizon):
    """
    Load the input data for the fly controller inference (with horizon)
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,f'list_input_fr_{horizon}.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,f'list_output_fr_{horizon}.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,f'list_input_fl_{horizon}.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,f'list_output_fl_{horizon}.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)

    with open(os.path.join(path_loading,f'list_input_self_fr_{horizon}.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,f'list_output_self_fr_{horizon}.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,f'list_input_self_fl_{horizon}.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,f'list_output_self_fl_{horizon}.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two
    

def load_all_data_or():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,'list_input_fr_or.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr_mm_fixed.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl_or.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl_mm_fixed.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr_mm_fixed.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr_mm_fixed.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl_mm_fixed.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl_mm_fixed.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two


def load_all_data():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,'list_input_fr.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two

def load_all_data_mm_():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,'list_input_fr_mm_fixed_sub.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr_mm_fixed_sub.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl_mm_fixed_sub.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl_mm_fixed_sub.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr_mm_fixed_sub.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr_mm_fixed_sub.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl_mm_fixed_sub.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl_mm_fixed_sub.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two


def load_avg_velocity():
    """
    Load the average velocity file 
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,'list_avg_velocity_mm_fixed.pkl'),'rb') as f1:
        avg_velocity = pickle.load(f1)


    return avg_velocity

def load_all_data_mm():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,'list_input_fr_mm_fixed.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr_mm_fixed.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl_mm_fixed.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl_mm_fixed.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr_mm_fixed.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr_mm_fixed.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl_mm_fixed.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl_mm_fixed.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two


def load_all_data_mm_touchdown():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io_touchdown_v2')
    with open(os.path.join(path_loading,'list_input_fr_mm_fixed.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr_mm_fixed.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl_mm_fixed.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl_mm_fixed.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr_mm_fixed.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr_mm_fixed.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl_mm_fixed.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl_mm_fixed.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two

def load_all_data_mm_abs():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io_abs')
    with open(os.path.join(path_loading,'list_input_fr_mm_fixed.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr_mm_fixed.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl_mm_fixed.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl_mm_fixed.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr_mm_fixed.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr_mm_fixed.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl_mm_fixed.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl_mm_fixed.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two

def load_all_data_mm_norm():
    """
    Load the input data for the fly controller inference (normalized data)
    
    Returns two lists which respectively contain the data for body and self prediction
    """
    path_loading = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io')
    with open(os.path.join(path_loading,'list_input_fr_mm_norm.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'list_output_fr_mm_norm.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'list_input_fl_mm_norm.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'list_output_fl_mm_norm.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    
    with open(os.path.join(path_loading,'list_input_self_fr_mm_norm.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'list_output_self_fr_mm_norm.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'list_input_self_fl_mm_norm.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'list_output_self_fl_mm_norm.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)
    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]

    return list_one, list_two

def concatenate_data_vel(input_list, output_list, input_self, output_self, avg_velocity):
    """
    Concatenate the (off) diagonal data + avg bout velocity
    """
    input_list_matrix, output_list_matrix = input_list[0], output_list[0]
    input_self_matrix, output_self_matrix = input_self[0], output_self[0]
    avg_velocity_matrix = avg_velocity[0] * np.ones((input_list[0].shape[0],1))
    for ii in tqdm(range(1, len(input_list))):
        input_list_matrix = np.vstack((input_list_matrix, input_list[ii]))
        output_list_matrix = np.vstack((output_list_matrix, output_list[ii]))
        input_self_matrix = np.vstack((input_self_matrix, input_self[ii]))
        output_self_matrix = np.vstack((output_self_matrix, output_self[ii]))
        avg_velocity_matrix = np.vstack((avg_velocity_matrix, avg_velocity[ii]*np.ones((input_list[ii].shape[0],1))))

    return input_list_matrix, input_self_matrix, output_list_matrix, output_self_matrix, avg_velocity_matrix

def concatenate_data(input_list, output_list, input_list_self, output_list_self):
    """
    Concatenate the (off) diagonal data
    """
    input_list_matrix, output_list_matrix = input_list[0], output_list[0]
    input_self_matrix, output_self_matrix = input_list_self[0], output_list_self[0]
    for ii in tqdm(range(1, len(input_list))):
        input_list_matrix = np.vstack((input_list_matrix, input_list[ii]))
        output_list_matrix = np.vstack((output_list_matrix, output_list[ii]))
        input_self_matrix = np.vstack((input_self_matrix, input_list_self[ii]))
        output_self_matrix = np.vstack((output_self_matrix, output_list_self[ii]))
    return input_list_matrix, input_self_matrix, output_list_matrix, output_self_matrix

def flip_data(input_list, input_self, output_list, output_self):
    """
    Flip the data that goes in the other diretion (negative head velocity along the x-axis)
    """
    idx_flip = np.where(np.nanmean(input_list[:,:,2],1)<0)[0]
    input_list_flip, output_list_flip = copy.deepcopy(input_list), copy.deepcopy(output_list)
    input_self_flip, output_self_flip = copy.deepcopy(input_self), copy.deepcopy(output_self)
    input_list_flip[idx_flip,:,:] = - input_list_flip[idx_flip,:,:]
    input_self_flip[idx_flip,:,:] = - input_self_flip[idx_flip,:,:]
    col_flip = [0,1,3,4,6,7]
    for col in col_flip:
        output_list_flip[idx_flip,col] = - output_list_flip[idx_flip,col]
        output_self_flip[idx_flip,col] = - output_self_flip[idx_flip,col]
    return input_list_flip, input_self_flip, output_list_flip, output_self_flip


def extract_metrics_distance(foot_output, animal, raw_data):
    """Extracts the contact metrics based on the distance method"""
    return

def foot_contact_detection_distance(input_data):
    """Extracts the foot contact based on the distance between the foot and the head marker"""
    return

def compute_local_velocity(raw_data):
    """Computes the local raw velocity of the data"""
    dt = 1/150
    output_matrix = np.zeros((raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]))
    output_matrix[2:-2,0,:] = (-raw_data[4:,0,:]+8*raw_data[3:-1,0,:]-8*raw_data[1:-3,0,:]+raw_data[:-4,0,:])/(12*dt)
    output_matrix[2:-2,1,:] = (-raw_data[4:,1,:]+8*raw_data[3:-1,1,:]-8*raw_data[1:-3,1,:]+raw_data[:-4,1,:])/(12*dt)
    return output_matrix


def get_io_time_model_self_fl(time_vector, line, foot_data, raw_data, marker, input_data, padding_time):
    """Extracts the gait cycle preceding the contact (self prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,0]) - np.abs(foot_data[idx_test:-1,0,0]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,2]) - np.abs(foot_data[idx_test:-1,0,2]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,4]) - np.abs(foot_data[idx_test:-1,0,4]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,0])) - np.flip(np.abs(foot_data[1:idx_test,0,0])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,2])) - np.flip(np.abs(foot_data[1:idx_test,0,2])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,4])) - np.flip(np.abs(foot_data[1:idx_test,0,4])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)==0) | (len(idx_prev_fc2)==0) | (len(idx_prev_fc3)==0)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[0], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[0], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[0], idx_test+idx_next_fc3[0]+1)
    leg_position_x_1, leg_position_y_1 = raw_data[time_input_1,0,0] - ref_position_x, raw_data[time_input_1,1,0] - ref_position_y
    leg_position_x_2, leg_position_y_2 = raw_data[time_input_2,0,2] - ref_position_x, raw_data[time_input_2,1,2] - ref_position_y
    leg_position_x_3, leg_position_y_3 = raw_data[time_input_3,0,4] - ref_position_x, raw_data[time_input_3,1,4] - ref_position_y
    leg_velocity_x_1, leg_velocity_y_1 = raw_data_velocity[time_input_1,0,0], raw_data_velocity[time_input_1,1,0]
    leg_velocity_x_2, leg_velocity_y_2 = raw_data_velocity[time_input_2,0,2], raw_data_velocity[time_input_2,1,2]
    leg_velocity_x_3, leg_velocity_y_3 = raw_data_velocity[time_input_3,0,4], raw_data_velocity[time_input_3,1,4]

    # Interpolation of the inputs 
    output_time = np.linspace(0,1,21)
    leg_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_1)),leg_position_x_1),-1)
    leg_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_2)),leg_position_x_2),-1)
    leg_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_3)),leg_position_x_3),-1)
    leg_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_1)),leg_position_y_1),-1)
    leg_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_2)),leg_position_y_2),-1)
    leg_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_3)),leg_position_y_3),-1)
    leg_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_1)),leg_velocity_x_1),-1)
    leg_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_2)),leg_velocity_x_2),-1)
    leg_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_3)),leg_velocity_x_3),-1)
    leg_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_1)),leg_velocity_y_1),-1)
    leg_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_2)),leg_velocity_y_2),-1)
    leg_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_3)),leg_velocity_y_3),-1)
    input_array = np.hstack((leg_position_x_1_, leg_position_y_1_, leg_velocity_x_1_, leg_velocity_y_1_,
                             leg_position_x_2_, leg_position_y_2_, leg_velocity_x_2_, leg_velocity_y_2_,
                             leg_position_x_3_, leg_position_y_3_, leg_velocity_x_3_, leg_velocity_y_3_))
    time_cycle = line[3]

    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==1))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,0] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,0] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==3))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,2] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,2] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==5))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,4] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,4] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_self_fr(time_vector, line, foot_data, raw_data, marker, input_data, padding_time):
    """Extracts the gait cycle preceding the contact (self prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,1]) - np.abs(foot_data[idx_test:-1,0,1]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,3]) - np.abs(foot_data[idx_test:-1,0,3]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,5]) - np.abs(foot_data[idx_test:-1,0,5]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,1])) - np.flip(np.abs(foot_data[1:idx_test,0,1])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,3])) - np.flip(np.abs(foot_data[1:idx_test,0,3])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,5])) - np.flip(np.abs(foot_data[1:idx_test,0,5])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)==0) | (len(idx_prev_fc2)==0) | (len(idx_prev_fc3)==0)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[0], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[0], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[0], idx_test+idx_next_fc3[0]+1)
    leg_position_x_1, leg_position_y_1 = raw_data[time_input_1,0,1] - ref_position_x, raw_data[time_input_1,1,1] - ref_position_y
    leg_position_x_2, leg_position_y_2 = raw_data[time_input_2,0,3] - ref_position_x, raw_data[time_input_2,1,3] - ref_position_y
    leg_position_x_3, leg_position_y_3 = raw_data[time_input_3,0,5] - ref_position_x, raw_data[time_input_3,1,5] - ref_position_y
    leg_velocity_x_1, leg_velocity_y_1 = raw_data_velocity[time_input_1,0,1], raw_data_velocity[time_input_1,1,1]
    leg_velocity_x_2, leg_velocity_y_2 = raw_data_velocity[time_input_2,0,3], raw_data_velocity[time_input_2,1,3]
    leg_velocity_x_3, leg_velocity_y_3 = raw_data_velocity[time_input_3,0,5], raw_data_velocity[time_input_3,1,5]

    # Interpolation of the inputs 
    output_time = np.linspace(0,1,21)
    leg_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_1)),leg_position_x_1),-1)
    leg_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_2)),leg_position_x_2),-1)
    leg_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_3)),leg_position_x_3),-1)
    leg_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_1)),leg_position_y_1),-1)
    leg_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_2)),leg_position_y_2),-1)
    leg_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_3)),leg_position_y_3),-1)
    leg_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_1)),leg_velocity_x_1),-1)
    leg_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_2)),leg_velocity_x_2),-1)
    leg_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_3)),leg_velocity_x_3),-1)
    leg_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_1)),leg_velocity_y_1),-1)
    leg_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_2)),leg_velocity_y_2),-1)
    leg_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_3)),leg_velocity_y_3),-1)
    input_array = np.hstack((leg_position_x_1_, leg_position_y_1_, leg_velocity_x_1_, leg_velocity_y_1_,
                             leg_position_x_2_, leg_position_y_2_, leg_velocity_x_2_, leg_velocity_y_2_,
                             leg_position_x_3_, leg_position_y_3_, leg_velocity_x_3_, leg_velocity_y_3_))
    time_cycle = line[3]

    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==2))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,1] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,1] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==4))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,3] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,3] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==6))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,5] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,5] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array


def get_io_time_model_self_fr_long(time_vector, line, foot_data, raw_data, marker, input_data, padding_time, horizon):
    """Extracts the gait cycle preceding the contact (self prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,1]) - np.abs(foot_data[idx_test:-1,0,1]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,3]) - np.abs(foot_data[idx_test:-1,0,3]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,5]) - np.abs(foot_data[idx_test:-1,0,5]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,1])) - np.flip(np.abs(foot_data[1:idx_test,0,1])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,3])) - np.flip(np.abs(foot_data[1:idx_test,0,3])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,5])) - np.flip(np.abs(foot_data[1:idx_test,0,5])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)<horizon) | (len(idx_prev_fc2)<horizon) | (len(idx_prev_fc3)<horizon)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[horizon-1], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[horizon-1], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[horizon-1], idx_test+idx_next_fc3[0]+1)
    leg_position_x_1, leg_position_y_1 = raw_data[time_input_1,0,1] - ref_position_x, raw_data[time_input_1,1,1] - ref_position_y
    leg_position_x_2, leg_position_y_2 = raw_data[time_input_2,0,3] - ref_position_x, raw_data[time_input_2,1,3] - ref_position_y
    leg_position_x_3, leg_position_y_3 = raw_data[time_input_3,0,5] - ref_position_x, raw_data[time_input_3,1,5] - ref_position_y
    leg_velocity_x_1, leg_velocity_y_1 = raw_data_velocity[time_input_1,0,1], raw_data_velocity[time_input_1,1,1]
    leg_velocity_x_2, leg_velocity_y_2 = raw_data_velocity[time_input_2,0,3], raw_data_velocity[time_input_2,1,3]
    leg_velocity_x_3, leg_velocity_y_3 = raw_data_velocity[time_input_3,0,5], raw_data_velocity[time_input_3,1,5]

    # Interpolation of the inputs 
    output_time = np.linspace(0,1,1+20*horizon)
    leg_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_1)),leg_position_x_1),-1)
    leg_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_2)),leg_position_x_2),-1)
    leg_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_3)),leg_position_x_3),-1)
    leg_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_1)),leg_position_y_1),-1)
    leg_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_2)),leg_position_y_2),-1)
    leg_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_3)),leg_position_y_3),-1)
    leg_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_1)),leg_velocity_x_1),-1)
    leg_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_2)),leg_velocity_x_2),-1)
    leg_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_3)),leg_velocity_x_3),-1)
    leg_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_1)),leg_velocity_y_1),-1)
    leg_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_2)),leg_velocity_y_2),-1)
    leg_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_3)),leg_velocity_y_3),-1)
    input_array = np.hstack((leg_position_x_1_, leg_position_y_1_, leg_velocity_x_1_, leg_velocity_y_1_,
                             leg_position_x_2_, leg_position_y_2_, leg_velocity_x_2_, leg_velocity_y_2_,
                             leg_position_x_3_, leg_position_y_3_, leg_velocity_x_3_, leg_velocity_y_3_))
    time_cycle = line[3]

    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==2))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,1] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,1] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==4))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,3] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,3] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==6))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,5] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,5] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_fr_long(time_vector, line, foot_data, raw_data, marker, input_data, padding_time, horizon):
    """Extracts the gait cycle(s) preceding the contact (body prediction)"""
    # Compute the velocity forthe raw data
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,1]) - np.abs(foot_data[idx_test:-1,0,1]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,3]) - np.abs(foot_data[idx_test:-1,0,3]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,5]) - np.abs(foot_data[idx_test:-1,0,5]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,1])) - np.flip(np.abs(foot_data[1:idx_test,0,1])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,3])) - np.flip(np.abs(foot_data[1:idx_test,0,3])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,5])) - np.flip(np.abs(foot_data[1:idx_test,0,5])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)<horizon) | (len(idx_prev_fc2)<horizon) | (len(idx_prev_fc3)<horizon)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[horizon-1], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[horizon-1], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[horizon-1], idx_test+idx_next_fc3[0]+1)
    head_position_x_1, head_position_y_1 = raw_data[time_input_1,0,6] - ref_position_x, raw_data[time_input_1,1,6] - ref_position_y
    head_position_x_2, head_position_y_2 = raw_data[time_input_2,0,6] - ref_position_x, raw_data[time_input_2,1,6] - ref_position_y
    head_position_x_3, head_position_y_3 = raw_data[time_input_3,0,6] - ref_position_x, raw_data[time_input_3,1,6] - ref_position_y
    head_velocity_x_1, head_velocity_y_1 = raw_data_velocity[time_input_1,0,6], raw_data_velocity[time_input_1,1,6]
    head_velocity_x_2, head_velocity_y_2 = raw_data_velocity[time_input_2,0,6], raw_data_velocity[time_input_2,1,6]
    head_velocity_x_3, head_velocity_y_3 = raw_data_velocity[time_input_3,0,6], raw_data_velocity[time_input_3,1,6]
    # Interpolation of the inputs
    output_time = np.linspace(0,1,1+horizon*20)
    head_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_1)), head_position_x_1),-1)
    head_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_2)), head_position_x_2),-1)
    head_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_3)), head_position_x_3),-1)
    head_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_1)), head_velocity_x_1),-1)
    head_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_2)), head_velocity_x_2),-1)
    head_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_3)), head_velocity_x_3),-1)
    head_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_1)), head_position_y_1),-1)
    head_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_2)), head_position_y_2),-1)
    head_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_3)), head_position_y_3),-1)
    head_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_1)), head_velocity_y_1),-1)
    head_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_2)), head_velocity_y_2),-1)
    head_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_3)), head_velocity_y_3),-1)
    input_array = np.hstack((head_position_x_1_, head_position_y_1_, head_velocity_x_1_, head_velocity_y_1_, head_position_x_2_, head_position_y_2_,
                             head_velocity_x_2_, head_velocity_y_2_, head_position_x_3_, head_position_y_3_, head_velocity_x_3_, head_velocity_y_3_))
    time_cycle = line[3]
    
    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==2))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,1] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,1] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==4))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,3] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,3] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==6))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,5] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,5] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_self_fl_long(time_vector, line, foot_data, raw_data, marker, input_data, padding_time, horizon):
    """Extracts the gait cycle preceding the contact (self prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,0]) - np.abs(foot_data[idx_test:-1,0,0]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,2]) - np.abs(foot_data[idx_test:-1,0,2]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,4]) - np.abs(foot_data[idx_test:-1,0,4]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,0])) - np.flip(np.abs(foot_data[1:idx_test,0,0])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,2])) - np.flip(np.abs(foot_data[1:idx_test,0,2])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,4])) - np.flip(np.abs(foot_data[1:idx_test,0,4])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)<horizon) | (len(idx_prev_fc2)<horizon) | (len(idx_prev_fc3)<horizon)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[horizon-1], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[horizon-1], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[horizon-1], idx_test+idx_next_fc3[0]+1)
    leg_position_x_1, leg_position_y_1 = raw_data[time_input_1,0,0] - ref_position_x, raw_data[time_input_1,1,0] - ref_position_y
    leg_position_x_2, leg_position_y_2 = raw_data[time_input_2,0,2] - ref_position_x, raw_data[time_input_2,1,2] - ref_position_y
    leg_position_x_3, leg_position_y_3 = raw_data[time_input_3,0,4] - ref_position_x, raw_data[time_input_3,1,4] - ref_position_y
    leg_velocity_x_1, leg_velocity_y_1 = raw_data_velocity[time_input_1,0,0], raw_data_velocity[time_input_1,1,0]
    leg_velocity_x_2, leg_velocity_y_2 = raw_data_velocity[time_input_2,0,2], raw_data_velocity[time_input_2,1,2]
    leg_velocity_x_3, leg_velocity_y_3 = raw_data_velocity[time_input_3,0,4], raw_data_velocity[time_input_3,1,4]

    # Interpolation of the inputs 
    output_time = np.linspace(0,1,1+20*horizon)
    leg_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_1)),leg_position_x_1),-1)
    leg_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_2)),leg_position_x_2),-1)
    leg_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_x_3)),leg_position_x_3),-1)
    leg_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_1)),leg_position_y_1),-1)
    leg_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_2)),leg_position_y_2),-1)
    leg_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_position_y_3)),leg_position_y_3),-1)
    leg_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_1)),leg_velocity_x_1),-1)
    leg_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_2)),leg_velocity_x_2),-1)
    leg_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_x_3)),leg_velocity_x_3),-1)
    leg_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_1)),leg_velocity_y_1),-1)
    leg_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_2)),leg_velocity_y_2),-1)
    leg_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(leg_velocity_y_3)),leg_velocity_y_3),-1)
    input_array = np.hstack((leg_position_x_1_, leg_position_y_1_, leg_velocity_x_1_, leg_velocity_y_1_,
                             leg_position_x_2_, leg_position_y_2_, leg_velocity_x_2_, leg_velocity_y_2_,
                             leg_position_x_3_, leg_position_y_3_, leg_velocity_x_3_, leg_velocity_y_3_))
    time_cycle = line[3]

    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==1))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,0] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,0] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==3))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,2] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,2] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==5))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,4] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,4] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_fl_long(time_vector, line, foot_data, raw_data, marker, input_data, padding_time, horizon):
    """Extracts the gait cycle(s) preceding the contact (body prediction)"""
    # Compute the velocity forthe raw data
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,0]) - np.abs(foot_data[idx_test:-1,0,0]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,2]) - np.abs(foot_data[idx_test:-1,0,2]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,4]) - np.abs(foot_data[idx_test:-1,0,4]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,0])) - np.flip(np.abs(foot_data[1:idx_test,0,0])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,2])) - np.flip(np.abs(foot_data[1:idx_test,0,2])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,4])) - np.flip(np.abs(foot_data[1:idx_test,0,4])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)<horizon) | (len(idx_prev_fc2)<horizon) | (len(idx_prev_fc3)<horizon)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[horizon-1], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[horizon-1], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[horizon-1], idx_test+idx_next_fc3[0]+1)
    head_position_x_1, head_position_y_1 = raw_data[time_input_1,0,6] - ref_position_x, raw_data[time_input_1,1,6] - ref_position_y
    head_position_x_2, head_position_y_2 = raw_data[time_input_2,0,6] - ref_position_x, raw_data[time_input_2,1,6] - ref_position_y
    head_position_x_3, head_position_y_3 = raw_data[time_input_3,0,6] - ref_position_x, raw_data[time_input_3,1,6] - ref_position_y
    head_velocity_x_1, head_velocity_y_1 = raw_data_velocity[time_input_1,0,6], raw_data_velocity[time_input_1,1,6]
    head_velocity_x_2, head_velocity_y_2 = raw_data_velocity[time_input_2,0,6], raw_data_velocity[time_input_2,1,6]
    head_velocity_x_3, head_velocity_y_3 = raw_data_velocity[time_input_3,0,6], raw_data_velocity[time_input_3,1,6]
    # Interpolation of the inputs
    output_time = np.linspace(0,1,1+horizon*20)
    head_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_1)), head_position_x_1),-1)
    head_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_2)), head_position_x_2),-1)
    head_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_3)), head_position_x_3),-1)
    head_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_1)), head_velocity_x_1),-1)
    head_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_2)), head_velocity_x_2),-1)
    head_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_3)), head_velocity_x_3),-1)
    head_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_1)), head_position_y_1),-1)
    head_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_2)), head_position_y_2),-1)
    head_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_3)), head_position_y_3),-1)
    head_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_1)), head_velocity_y_1),-1)
    head_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_2)), head_velocity_y_2),-1)
    head_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_3)), head_velocity_y_3),-1)
    input_array = np.hstack((head_position_x_1_, head_position_y_1_, head_velocity_x_1_, head_velocity_y_1_, head_position_x_2_, head_position_y_2_,
                             head_velocity_x_2_, head_velocity_y_2_, head_position_x_3_, head_position_y_3_, head_velocity_x_3_, head_velocity_y_3_))
    time_cycle = line[3]
    
    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==1))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,0] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,0] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==3))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,2] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,2] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==5))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,4] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,4] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_fr(time_vector, line, foot_data, raw_data, marker, input_data, padding_time):
    """Extracts the gait cycle preceding the contact (body prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,1]) - np.abs(foot_data[idx_test:-1,0,1]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,3]) - np.abs(foot_data[idx_test:-1,0,3]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,5]) - np.abs(foot_data[idx_test:-1,0,5]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,1])) - np.flip(np.abs(foot_data[1:idx_test,0,1])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,3])) - np.flip(np.abs(foot_data[1:idx_test,0,3])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,5])) - np.flip(np.abs(foot_data[1:idx_test,0,5])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)==0) | (len(idx_prev_fc2)==0) | (len(idx_prev_fc3)==0)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[0], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[0], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[0], idx_test+idx_next_fc3[0]+1)
    head_position_x_1, head_position_y_1 = raw_data[time_input_1,0,6] - ref_position_x, raw_data[time_input_1,1,6] - ref_position_y
    head_position_x_2, head_position_y_2 = raw_data[time_input_2,0,6] - ref_position_x, raw_data[time_input_2,1,6] - ref_position_y
    head_position_x_3, head_position_y_3 = raw_data[time_input_3,0,6] - ref_position_x, raw_data[time_input_3,1,6] - ref_position_y
    head_velocity_x_1, head_velocity_y_1 = raw_data_velocity[time_input_1,0,6], raw_data_velocity[time_input_1,1,6]
    head_velocity_x_2, head_velocity_y_2 = raw_data_velocity[time_input_2,0,6], raw_data_velocity[time_input_2,1,6]
    head_velocity_x_3, head_velocity_y_3 = raw_data_velocity[time_input_3,0,6], raw_data_velocity[time_input_3,1,6]

    # Interpolation of the inputs
    output_time = np.linspace(0,1,21)
    head_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_1)), head_position_x_1),-1)
    head_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_2)), head_position_x_2),-1)
    head_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_3)), head_position_x_3),-1)
    head_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_1)), head_velocity_x_1),-1)
    head_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_2)), head_velocity_x_2),-1)
    head_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_3)), head_velocity_x_3),-1)
    head_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_1)), head_position_y_1),-1)
    head_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_2)), head_position_y_2),-1)
    head_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_3)), head_position_y_3),-1)
    head_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_1)), head_velocity_y_1),-1)
    head_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_2)), head_velocity_y_2),-1)
    head_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_3)), head_velocity_y_3),-1)
    input_array = np.hstack((head_position_x_1_, head_position_y_1_, head_velocity_x_1_, head_velocity_y_1_, head_position_x_2_, head_position_y_2_,
                             head_velocity_x_2_, head_velocity_y_2_, head_position_x_3_, head_position_y_3_, head_velocity_x_3_, head_velocity_y_3_))
    time_cycle = line[3]
    
    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==2))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,1] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,1] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==4))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,3] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,3] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==6))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,5] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,5] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array


def get_io_time_model_fr_or(time_vector, line, foot_data, raw_data, marker, input_data, padding_time):
    """Extracts the gait cycle preceding the contact (body prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,1]) - np.abs(foot_data[idx_test:-1,0,1]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,3]) - np.abs(foot_data[idx_test:-1,0,3]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,5]) - np.abs(foot_data[idx_test:-1,0,5]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,1])) - np.flip(np.abs(foot_data[1:idx_test,0,1])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,3])) - np.flip(np.abs(foot_data[1:idx_test,0,3])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,5])) - np.flip(np.abs(foot_data[1:idx_test,0,5])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)==0) | (len(idx_prev_fc2)==0) | (len(idx_prev_fc3)==0)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[0], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[0], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[0], idx_test+idx_next_fc3[0]+1)
    head_position_x_1, head_position_y_1 = raw_data[time_input_1,0,6] - ref_position_x, raw_data[time_input_1,1,6] - ref_position_y
    head_position_x_2, head_position_y_2 = raw_data[time_input_2,0,6] - ref_position_x, raw_data[time_input_2,1,6] - ref_position_y
    head_position_x_3, head_position_y_3 = raw_data[time_input_3,0,6] - ref_position_x, raw_data[time_input_3,1,6] - ref_position_y
    head_velocity_x_1, head_velocity_y_1 = raw_data_velocity[time_input_1,0,6], raw_data_velocity[time_input_1,1,6]
    head_velocity_x_2, head_velocity_y_2 = raw_data_velocity[time_input_2,0,6], raw_data_velocity[time_input_2,1,6]
    head_velocity_x_3, head_velocity_y_3 = raw_data_velocity[time_input_3,0,6], raw_data_velocity[time_input_3,1,6]
    orientation_1 = np.arctan2(head_velocity_y_1, head_velocity_x_1)*360/(2*np.pi)
    orientation_2 = np.arctan2(head_velocity_y_1, head_velocity_x_1)*360/(2*np.pi)
    orientation_3 = np.arctan2(head_velocity_y_1, head_velocity_x_1)*360/(2*np.pi)
    # Forcing the angles between -179 & 180 ..
    orientation_1 = orientation_1 % 360
    orientation_1 = (orientation_1+360) % 360
    idx_flip1 = np.where(orientation_1> 180)[0]
    orientation_1[idx_flip1] = orientation_1[idx_flip1] - 360

    orientation_2 = orientation_2 % 360
    orientation_2 = (orientation_2+360) % 360
    idx_flip2 = np.where(orientation_2> 180)[0]
    orientation_2[idx_flip2] = orientation_2[idx_flip2] - 360

    orientation_3 = orientation_3 % 360
    orientation_3 = (orientation_3+360) % 360
    idx_flip3 = np.where(orientation_3> 180)[0]
    orientation_3[idx_flip3] = orientation_3[idx_flip3] - 360
    
    

    # Interpolation of the inputs
    output_time = np.linspace(0,1,21)
    head_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_1)), head_position_x_1),-1)
    head_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_2)), head_position_x_2),-1)
    head_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_3)), head_position_x_3),-1)
    head_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_1)), head_velocity_x_1),-1)
    head_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_2)), head_velocity_x_2),-1)
    head_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_3)), head_velocity_x_3),-1)
    head_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_1)), head_position_y_1),-1)
    head_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_2)), head_position_y_2),-1)
    head_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_3)), head_position_y_3),-1)
    head_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_1)), head_velocity_y_1),-1)
    head_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_2)), head_velocity_y_2),-1)
    head_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_3)), head_velocity_y_3),-1)
    orientation_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(orientation_1)), orientation_1),-1)
    orientation_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(orientation_2)), orientation_2),-1)
    orientation_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(orientation_3)), orientation_3),-1)
    input_array = np.hstack((head_position_x_1_, head_position_y_1_, head_velocity_x_1_, head_velocity_y_1_, orientation_1_, head_position_x_2_, head_position_y_2_,
                             head_velocity_x_2_, head_velocity_y_2_, orientation_2_, head_position_x_3_, head_position_y_3_, head_velocity_x_3_, head_velocity_y_3_, orientation_3_))
    time_cycle = line[3]
    
    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==2))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,1] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,1] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==4))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,3] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,3] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==6))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,5] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,5] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_fl_or(time_vector, line, foot_data, raw_data, marker, input_data, padding_time):
    """Extracts the gait cycle preceding the contact (body prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,0]) - np.abs(foot_data[idx_test:-1,0,0]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,2]) - np.abs(foot_data[idx_test:-1,0,2]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,4]) - np.abs(foot_data[idx_test:-1,0,4]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,0])) - np.flip(np.abs(foot_data[1:idx_test,0,0])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,2])) - np.flip(np.abs(foot_data[1:idx_test,0,2])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,4])) - np.flip(np.abs(foot_data[1:idx_test,0,4])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)==0) | (len(idx_prev_fc2)==0) | (len(idx_prev_fc3)==0)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[0], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[0], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[0], idx_test+idx_next_fc3[0]+1)
    head_position_x_1, head_position_y_1 = raw_data[time_input_1,0,6] - ref_position_x, raw_data[time_input_1,1,6] - ref_position_y
    head_position_x_2, head_position_y_2 = raw_data[time_input_2,0,6] - ref_position_x, raw_data[time_input_2,1,6] - ref_position_y
    head_position_x_3, head_position_y_3 = raw_data[time_input_3,0,6] - ref_position_x, raw_data[time_input_3,1,6] - ref_position_y
    head_velocity_x_1, head_velocity_y_1 = raw_data_velocity[time_input_1,0,6], raw_data_velocity[time_input_1,1,6]
    head_velocity_x_2, head_velocity_y_2 = raw_data_velocity[time_input_2,0,6], raw_data_velocity[time_input_2,1,6]
    head_velocity_x_3, head_velocity_y_3 = raw_data_velocity[time_input_3,0,6], raw_data_velocity[time_input_3,1,6]
    orientation_1 = np.arctan2(head_velocity_y_1, head_velocity_x_1)*360/(2*np.pi)
    orientation_2 = np.arctan2(head_velocity_y_1, head_velocity_x_1)*360/(2*np.pi)
    orientation_3 = np.arctan2(head_velocity_y_1, head_velocity_x_1)*360/(2*np.pi)
    # Forcing the angles between -179 & 180 ...
    orientation_1 = orientation_1 % 360
    orientation_1 = (orientation_1+360) % 360
    idx_flip1 = np.where(orientation_1> 180)[0]
    orientation_1[idx_flip1] = orientation_1[idx_flip1] - 360

    orientation_2 = orientation_2 % 360
    orientation_2 = (orientation_2+360) % 360
    idx_flip2 = np.where(orientation_2> 180)[0]
    orientation_2[idx_flip2] = orientation_2[idx_flip2] - 360

    orientation_3 = orientation_3 % 360
    orientation_3 = (orientation_3+360) % 360
    idx_flip3 = np.where(orientation_3> 180)[0]
    orientation_3[idx_flip3] = orientation_3[idx_flip3] - 360

    # Interpolation of the inputs
    output_time = np.linspace(0,1,21)
    head_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_1)), head_position_x_1),-1)
    head_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_2)), head_position_x_2),-1)
    head_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_3)), head_position_x_3),-1)
    head_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_1)), head_velocity_x_1),-1)
    head_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_2)), head_velocity_x_2),-1)
    head_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_3)), head_velocity_x_3),-1)
    head_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_1)), head_position_y_1),-1)
    head_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_2)), head_position_y_2),-1)
    head_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_3)), head_position_y_3),-1)
    head_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_1)), head_velocity_y_1),-1)
    head_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_2)), head_velocity_y_2),-1)
    head_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_3)), head_velocity_y_3),-1)
    orientation_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(orientation_1)), orientation_1),-1)
    orientation_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(orientation_2)), orientation_2),-1)
    orientation_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(orientation_3)), orientation_3),-1)
    input_array = np.hstack((head_position_x_1_, head_position_y_1_, head_velocity_x_1_, head_velocity_y_1_, orientation_1_, head_position_x_2_, head_position_y_2_,
                             head_velocity_x_2_, head_velocity_y_2_, orientation_2_, head_position_x_3_, head_position_y_3_, head_velocity_x_3_, head_velocity_y_3_, orientation_3_))
    time_cycle = line[3]
    
    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==1))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,0] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,0] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==3))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,2] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,2] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==5))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,4] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,4] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_io_time_model_fl(time_vector, line, foot_data, raw_data, marker, input_data, padding_time):
    """Extracts the gait cycle preceding the contact (body prediction)"""
    # Compute the velocity for the raw data 
    raw_data_velocity = compute_local_velocity(raw_data)
    idx_meta = np.where(time_vector==line[-1])[0][0]
    idx_test = idx_meta - padding_time
    idx_next_fc1 = np.where((np.abs(foot_data[idx_test+1:,0,0]) - np.abs(foot_data[idx_test:-1,0,0]))>0)[0]
    idx_next_fc2 = np.where((np.abs(foot_data[idx_test+1:,0,2]) - np.abs(foot_data[idx_test:-1,0,2]))>0)[0]
    idx_next_fc3 = np.where((np.abs(foot_data[idx_test+1:,0,4]) - np.abs(foot_data[idx_test:-1,0,4]))>0)[0]
    idx_prev_fc1 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,0])) - np.flip(np.abs(foot_data[1:idx_test,0,0])))<0)[0]
    idx_prev_fc2 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,2])) - np.flip(np.abs(foot_data[1:idx_test,0,2])))<0)[0]
    idx_prev_fc3 = np.where((np.flip(np.abs(foot_data[:idx_test-1,0,4])) - np.flip(np.abs(foot_data[1:idx_test,0,4])))<0)[0]
    if ((len(idx_next_fc1)==0) | (len(idx_next_fc2)==0) | (len(idx_next_fc3)==0) | (len(idx_prev_fc1)==0) | (len(idx_prev_fc2)==0) | (len(idx_prev_fc3)==0)):
        return None, None
    ref_position_x, ref_position_y = foot_data[idx_test,0,marker-1], foot_data[idx_test,1,marker-1]
    time_input_1 = np.arange(idx_test-idx_prev_fc1[0], idx_test+idx_next_fc1[0]+1)
    time_input_2 = np.arange(idx_test-idx_prev_fc2[0], idx_test+idx_next_fc2[0]+1)
    time_input_3 = np.arange(idx_test-idx_prev_fc3[0], idx_test+idx_next_fc3[0]+1)
    head_position_x_1, head_position_y_1 = raw_data[time_input_1,0,6] - ref_position_x, raw_data[time_input_1,1,6] - ref_position_y
    head_position_x_2, head_position_y_2 = raw_data[time_input_2,0,6] - ref_position_x, raw_data[time_input_2,1,6] - ref_position_y
    head_position_x_3, head_position_y_3 = raw_data[time_input_3,0,6] - ref_position_x, raw_data[time_input_3,1,6] - ref_position_y
    head_velocity_x_1, head_velocity_y_1 = raw_data_velocity[time_input_1,0,6], raw_data_velocity[time_input_1,1,6]
    head_velocity_x_2, head_velocity_y_2 = raw_data_velocity[time_input_2,0,6], raw_data_velocity[time_input_2,1,6]
    head_velocity_x_3, head_velocity_y_3 = raw_data_velocity[time_input_3,0,6], raw_data_velocity[time_input_3,1,6]

    # Interpolation of the inputs
    output_time = np.linspace(0,1,21)
    head_position_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_1)), head_position_x_1),-1)
    head_position_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_2)), head_position_x_2),-1)
    head_position_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x_3)), head_position_x_3),-1)
    head_velocity_x_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_1)), head_velocity_x_1),-1)
    head_velocity_x_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_2)), head_velocity_x_2),-1)
    head_velocity_x_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x_3)), head_velocity_x_3),-1)
    head_position_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_1)), head_position_y_1),-1)
    head_position_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_2)), head_position_y_2),-1)
    head_position_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y_3)), head_position_y_3),-1)
    head_velocity_y_1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_1)), head_velocity_y_1),-1)
    head_velocity_y_2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_2)), head_velocity_y_2),-1)
    head_velocity_y_3_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y_3)), head_velocity_y_3),-1)
    input_array = np.hstack((head_position_x_1_, head_position_y_1_, head_velocity_x_1_, head_velocity_y_1_, head_position_x_2_, head_position_y_2_,
                             head_velocity_x_2_, head_velocity_y_2_, head_position_x_3_, head_position_y_3_, head_velocity_x_3_, head_velocity_y_3_))
    time_cycle = line[3]
    
    idx_same2 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==1))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2 = raw_data[idx_test+idx_next_fc1[0]+2,0,0] - ref_position_x
    final_position_y_2 = raw_data[idx_test+idx_next_fc1[0]+2,1,0] - ref_position_y
    time_contact_2 = (input_data[idx_same2,3]/time_cycle)[0]

    idx_same3 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==3))[0]
    if len(idx_same3)==0:
        return input_array, None
    final_position_x_3 = raw_data[idx_test+idx_next_fc2[0]+2,0,2] - ref_position_x
    final_position_y_3 = raw_data[idx_test+idx_next_fc2[0]+2,1,2] - ref_position_y
    time_contact_3 = (input_data[idx_same3,3]/time_cycle)[0]

    idx_same4 = np.where((input_data[:,0]==line[0]) & (input_data[:,-1]==line[-1]) & (input_data[:,2]==5))[0]
    if len(idx_same4)==0:
        return input_array, None
    final_position_x_4 = raw_data[idx_test+idx_next_fc3[0]+2,0,4] - ref_position_x
    final_position_y_4 = raw_data[idx_test+idx_next_fc3[0]+2,1,4] - ref_position_y
    time_contact_4 = (input_data[idx_same4,3]/time_cycle)[0]

    output_array = np.expand_dims(np.array([final_position_x_2, final_position_y_2, time_contact_2, final_position_x_3, final_position_y_3, time_contact_3, final_position_x_4, final_position_y_4, time_contact_4]),-1)
    return input_array, output_array

def get_fly_id_chunk(padded_video, validity):
    """Gets the video id of each valid chunks"""
    idx_nans = np.where(np.isnan(padded_video))[0]
    n_chunks = len(idx_nans) - 1
    output_list = []
    count_chunk = 0
    for chunk in range(n_chunks):
        if validity[chunk] != 4:
            continue
        else:
            output_list.append(padded_video[idx_nans[chunk]+1])
        count_chunk += 1
    return output_list


def get_video_id_chunk(padded_video, validity):
    """Gets the video id of each valid chunks"""
    idx_nans = np.where(np.isnan(padded_video))[0]
    n_chunks = len(idx_nans) - 1
    output_list = []
    count_chunk = 0
    for chunk in range(n_chunks):
        if validity[chunk] != 4:
            continue
        else:
            output_list.append(padded_video[idx_nans[chunk]+1])
        count_chunk += 1
    return output_list


def get_input_output_fr_long(padded_raw, padded_foot, tot_list_data, validity, horizon):
    """Gets the inputs and outputs for the controller inference for the first tripod gait - body prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==1) & (input_data[:,2]==1) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),horizon*20+1,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_fr_long(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 1, input_data, idx_nans[chunk]+1, horizon)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output

def get_input_output_fl_long(padded_raw, padded_foot, tot_list_data, validity, horizon):
    """Gets the inputs and outputs for the controller inference for the first tripod gait - body prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==4) & (input_data[:,2]==4) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),horizon*20+1,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_fl_long(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 4, input_data, idx_nans[chunk]+1, horizon)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output


def get_input_output_fr_or(padded_raw, padded_foot, tot_list_data, validity):
    """Gets the inputs and outputs (with orientation) for the controller inference for the first tripod gait - body prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==1) & (input_data[:,2]==1) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),21,15))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_fr_or(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 1, input_data, idx_nans[chunk]+1)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output


def get_input_output_fr(padded_raw, padded_foot, tot_list_data, validity):
    """Gets the input and outputs for the controller inference for the first tripod gait - body prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0,padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==1) & (input_data[:,2]==1) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),21,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_fr(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 1, input_data, idx_nans[chunk]+1)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output


def get_input_output_fl(padded_raw, padded_foot, tot_list_data, validity):
    """Gets the input and outputs for the controller inference for the first tripod gait"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0,padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==4) & (input_data[:,2]==4) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),21,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_fl(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 4, input_data, idx_nans[chunk]+1)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output

def get_input_output_fl_or(padded_raw, padded_foot, tot_list_data, validity):
    """Gets the input and outputs for the controller inference for the first tripod gait"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0,padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==4) & (input_data[:,2]==4) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),21,15))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_fl_or(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 4, input_data, idx_nans[chunk]+1)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output

def get_input_output_self_fl_long(padded_raw, padded_foot, tot_list_data, validity, horizon):
    """Gets the input and output for the controller inference for the first tripod gait - self prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==4) & (input_data[:,2]==4) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),1+20*horizon,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_self_fl_long(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 4, input_data, idx_nans[chunk]+1, horizon)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output



def get_input_output_self_fr_long(padded_raw, padded_foot, tot_list_data, validity, horizon):
    """Gets the input and output for the controller inference for the first tripod gait - self prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==1) & (input_data[:,2]==1) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),1+20*horizon,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_self_fr_long(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 1, input_data, idx_nans[chunk]+1, horizon)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output

def get_input_output_self_fr(padded_raw, padded_foot, tot_list_data, validity):
    """Gets the input and outpus for the controller inference for the first tripod gait - self prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==1) & (input_data[:,2]==1) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),21,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_self_fr(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 1, input_data, idx_nans[chunk]+1)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output

def get_input_output_self_fl(padded_raw, padded_foot, tot_list_data, validity):
    """Gets the input and outpus for the controller inference for the second tripod gait - self prediction"""
    idx_nans = np.where(np.isnan(padded_raw[:,0,0]))[0]
    n_chunk = len(idx_nans)-1
    new_input, new_output = [], []
    time_vector = np.arange(0, padded_raw.shape[0])
    count_chunk = 0
    for chunk in tqdm(range(n_chunk)):
        if validity[chunk]!=4:
            continue
        else:
            input_data = tot_list_data[count_chunk]
            local_idx = np.arange(idx_nans[chunk]+1,idx_nans[chunk+1])
            idx_to_keep = np.where((input_data[:,1]==4) & (input_data[:,2]==4) & (input_data[:,3]>0.075))[0]
            total_input = np.zeros((len(idx_to_keep),21,12))
            total_output = np.zeros((len(idx_to_keep),9))
            for line in range(len(idx_to_keep)):
                tmp_input, tmp_output = get_io_time_model_self_fl(time_vector, input_data[idx_to_keep[line],:], padded_foot[local_idx,:,:], padded_raw[local_idx,:,:], 4, input_data, idx_nans[chunk]+1)
                if tmp_output is None:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
                elif tmp_output.shape[1]!=0:
                    total_input[line,:] = tmp_input
                    total_output[line,0] = tmp_output[0][0]
                    total_output[line,1] = tmp_output[1][0]
                    total_output[line,2] = tmp_output[2][0]
                    total_output[line,3] = tmp_output[3][0]
                    total_output[line,4] = tmp_output[4][0]
                    total_output[line,5] = tmp_output[5][0]
                    total_output[line,6] = tmp_output[6][0]
                    total_output[line,7] = tmp_output[7][0]
                    total_output[line,8] = tmp_output[8][0]
                else:
                    total_input[line,:], total_output[line,:] = np.nan, np.nan
            count_chunk += 1
            new_input.append(total_input)
            new_output.append(total_output)
    return new_input, new_output


def add_velocity_tier(input_list):
    """Generates the velocity tier list for the input data"""
    velocity_tier = np.zeros((input_list.shape[0],))
    fr_speed = np.zeros((input_list.shape[0],))
    for line in range(input_list.shape[0]):
        fr_speed[line] = np.nanmean(input_list[line,:,2])
    fr_speed_ = fr_speed[~np.isnan(fr_speed)]
    pctl_fr = np.percentile(fr_speed_,[5,20,40,60,80,95])

    for line in range(fr_speed.shape[0]):
        if np.isnan(fr_speed[line]):
            velocity_tier[line] = np.nan
        elif fr_speed[line] > pctl_fr[4]:
            velocity_tier[line] = 5
        elif fr_speed[line] > pctl_fr[3]:
            velocity_tier[line] = 4
        elif fr_speed[line] > pctl_fr[2]:
            velocity_tier[line] = 3
        elif fr_speed[line] > pctl_fr[1]:
            velocity_tier[line] = 2
        elif fr_speed[line] > pctl_fr[0]:
            velocity_tier[line] = 1
        else:
            velocity_tier[line] = np.nan
    return velocity_tier

def regression_fr_baseline_laterality(tot_input, tot_output, tot_video, id_leg):
    """Baseline for the figure below"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix = np.zeros((n_videos,21,3,4,5))
    for video in tqdm(range(n_videos)):
        local_idx = np.where((tot_video[:,0]==video+1))[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4*id_leg+4]
        local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
        for output_var in range(local_output.shape[-1]):
            for input_var in range(local_input.shape[-1]):
                # Normalize the inputs and outputs 
                input_tmp = local_input[:,:,input_var]
                output_tmp = local_output[:,output_var]
                if output_var==0:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                    output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                else:
                    output_tmp = output_tmp - np.nanmean(output_tmp)

                if input_var==0:
                    xaxis = np.arange(21)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                for time in range(21):
                    if input_tmp.shape[0]<5:
                        regression_matrix[video,time,output_var,input_var,:] = np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
                    else:
                        reg_local = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                        regression_matrix[video,time,output_var,input_var,:] = np.array([reg_local.slope, reg_local.pvalue, reg_local.rvalue, reg_local.intercept, reg_local.intercept_stderr])

    return regression_matrix


def regression_fr_laterality_velidp(tot_input, tot_output, tot_video, id_leg):
    """Computes the lateral regression coefficient for the fly data - velocity independant"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix_small = np.zeros((n_videos,21,3,4,5))
    regression_matrix_large = np.zeros((n_videos,21,3,4,5))
    for video in tqdm(range(n_videos)):
        local_idx = np.where((tot_video[:,0]==video+1))[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4*id_leg+4]
        local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
        for output_var in range(local_output.shape[-1]):
            for input_var in range(local_input.shape[-1]):
                #Normalize the inputs and outputs 
                input_tmp = local_input[:,:,input_var]
                output_tmp = local_output[:,output_var]
                if output_var==0:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                    output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                else:
                    output_tmp = output_tmp - np.nanmean(output_tmp,0)

                if input_var==0:
                    xaxis = np.arange(21)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                idx_large = np.where(input_tmp>0)[0]
                idx_small = np.where(input_tmp<0)[0]
                for time in range(21):
                    if input_tmp.shape[0] < 5:
                        regression_matrix_large[video,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        regression_matrix_small[video,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                        reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                        regression_matrix_large[video,time,output_var,input_var,:] = np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        regression_matrix_small[video,time,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large

def regression_fr_laterality_velidp_multilinear(tot_input, tot_output, tot_video, id_leg):
    """Computes the lateral regression coefficient for the fly data - velocity independant"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix_small = np.zeros((n_videos,21,3,4,2))
    regression_matrix_large = np.zeros((n_videos,21,3,4,2))
    for video in tqdm(range(n_videos)):
        local_idx = np.where((tot_video[:,0]==video+1))[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4*id_leg+4]
        local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
        for output_var in range(local_output.shape[-1]):
            # Normalize the outputs
            output_tmp = local_output[:,output_var]
            input_tmp = copy.deepcopy(local_input)
            if output_var==0:
                tmp_vel = np.nanmean(local_input[:,:,2],1)
                subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
            else:
                output_tmp = output_tmp - np.nanmean(output_tmp,0)
            # Normalize the inputs 
            xaxis = np.arange(21)
            for line in range(input_tmp.shape[0]):
                subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:,0])
                input_tmp[line,:,0] = input_tmp[line,:,0] - (xaxis*subjectlin.slope+subjectlin.intercept)
            tmp_vel = np.nanmean(input_tmp[:,:,2],1)
            input_tmp[:,:,2] = input_tmp[:,:,2] - np.expand_dims(tmp_vel,-1)
            print(np.nanmean(input_tmp[:,:,1],0))
            input_tmp[:,:,1] = input_tmp[:,:,1] - np.nanmean(input_tmp[:,:,1],0)
            input_tmp[:,:,3] = input_tmp[:,:,3] - np.nanmean(input_tmp[:,:,3],0)
            for time in range(21):
                idx_large = np.where(input_tmp[:,time,1]>0)[0]
                idx_small = np.where(input_tmp[:,time,1]<0)[0]
                if input_tmp.shape[0] < 5:
                    regression_matrix_large[video,time,output_var,:,:] = np.nan
                    regression_matrix_small[video,time,output_var,:,:] = np.nan
                else:
                    design_mat_large = np.hstack((np.ones((input_tmp[idx_large,time].shape[0],1)),input_tmp[idx_large,time]))
                    design_mat_small = np.hstack((np.ones((input_tmp[idx_small,time].shape[0],1)),input_tmp[idx_small,time]))
                    rsquare_large, gains_large = multilinear_ols_rsquare_gains(design_mat_large,output_tmp[idx_large])
                    rsquare_small, gains_small = multilinear_ols_rsquare_gains(design_mat_small, output_tmp[idx_small])
                    #reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                    #reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                    for ii in range(4):
                        regression_matrix_large[video,time,output_var,ii,:] = np.array([rsquare_large,gains_large[ii+1]])#np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        regression_matrix_small[video,time,output_var,ii,:] = np.array([rsquare_small,gains_small[ii+1]])#np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large


def regression_fr_laterality(velocity_tier, tot_input, tot_output, tot_video, id_leg):
    """Computes the regression coefficient for the fly data"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix_small = np.zeros((n_videos,21,5,3,4,5))
    regression_matrix_large = np.zeros((n_videos,21,5,3,4,5))
    for video in tqdm(range(n_videos)):
        for vel_tier in range(5):
            local_idx = np.where((tot_video[:,0]==video+1) & (velocity_tier==vel_tier+1))[0]
            idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
            local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4*id_leg+4]
            local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
            for output_var in range(local_output.shape[-1]):
                for input_var in range(local_input.shape[-1]):
                    # Normalize the outputs
                    input_tmp = local_input[:,:,input_var]
                    output_tmp = local_output[:,output_var]
                    if output_var==0:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                        output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                    else:
                        output_tmp = output_tmp - np.nanmean(output_tmp,0)
                    # Normalize the inputs 
                    if input_var==0:
                        xaxis = np.arange(21)
                        for line in range(input_tmp.shape[0]):
                            subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                            input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                    elif input_var==2:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                    else:
                        input_tmp = input_tmp - np.nanmean(input_tmp,0)
                    idx_large = np.where(input_tmp>0)[0]
                    idx_small = np.where(input_tmp<0)[0]
                    for time in range(21):
                        if input_tmp.shape[0] < 5:
                            regression_matrix_large[video,time,vel_tier,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                            regression_matrix_small[video,time,vel_tier,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                            reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                            regression_matrix_large[video,time,vel_tier,output_var,input_var,:] = np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                            regression_matrix_small[video,time,vel_tier,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large



def regression_fr_body_norm(velocity_tier, tot_input, tot_output, tot_video, id_leg):
    """Computes the regression coefficient for the fly data"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix = np.zeros((n_videos,21,5,3,4,5))
    for video in tqdm(range(n_videos)):
        for vel_tier in range(5):
            local_idx = np.where((tot_video[:,0]==video+1) & (velocity_tier==vel_tier+1))[0]
            idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
            local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4*id_leg+4]
            local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
            for output_var in range(local_output.shape[-1]):
                for input_var in range(local_input.shape[-1]):
                    # Normalize the outputs
                    input_tmp = local_input[:,:,input_var]
                    output_tmp = local_output[:,0]
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    tmp_len = local_output[:,0]
                    if output_var==0:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                        output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                    else:
                        output_tmp = output_tmp - np.nanmean(output_tmp,0)
                    # Normalize the inputs 
                    if input_var==0:
                        xaxis = np.arange(21)
                        for line in range(input_tmp.shape[0]):
                            subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                            input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                    elif input_var==2:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                    else:
                        input_tmp = input_tmp - np.nanmean(input_tmp,0)
                    if input_var==3:
                        for line in range(input_tmp.shape[0]):
                            input_tmp[line,:] = input_tmp[line,:] / tmp_vel[line]
                    if output_var==1:
                        for line in range(output_tmp.shape[0]):
                            output_tmp[line] = output_tmp[line] / tmp_len[line]
                    for time in range(21):
                        if input_tmp.shape[0] < 5:
                            regression_matrix[video,time,vel_tier,output_var-3*id_leg,input_var-4*id_leg,:] = np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
                        else:
                            reg = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                            regression_matrix[video,time,vel_tier,output_var,input_var,:] =np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])
                        
    return regression_matrix


def regression_fr_body_or(tot_input, tot_output, tot_video, id_leg):
    """Computes the regression coefficient for the fly data"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix = np.zeros((n_videos,21,3,5,5))
    for video in tqdm(range(n_videos)):
        local_idx = np.where((tot_video[:,0]==video+1))[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
        local_input = tot_input[local_idx[idx_nan],:,5*id_leg:5*id_leg+5]
        local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
        for output_var in range(local_output.shape[-1]):
            for input_var in range(local_input.shape[-1]):
                # Normalize the outputs
                input_tmp = local_input[:,:,input_var]
                output_tmp = local_output[:,output_var]
                if output_var==0:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                    output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                else:
                    output_tmp = output_tmp - np.nanmean(output_tmp,0)
                # Normalize the inputs 
                if input_var==0:
                    xaxis = np.arange(21)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                for time in range(21):
                    if input_tmp.shape[0] < 10:
                        regression_matrix[video,time,output_var-3*id_leg,input_var-4*id_leg,:] = np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
                    else:
                        reg = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                        regression_matrix[video,time,output_var,input_var,:] =np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])
                        
    return regression_matrix

def regression_fr_body(velocity_tier, tot_input, tot_output, tot_video, id_leg):
    """Computes the regression coefficient for the fly data"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    regression_matrix = np.zeros((n_videos,21,5,3,4,5))
    for video in tqdm(range(n_videos)):
        for vel_tier in range(5):
            local_idx = np.where((tot_video[:,0]==video+1) & (velocity_tier==vel_tier+1))[0]
            idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
            local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4*id_leg+4]
            local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
            for output_var in range(local_output.shape[-1]):
                for input_var in range(local_input.shape[-1]):
                    # Normalize the outputs
                    input_tmp = local_input[:,:,input_var]
                    output_tmp = local_output[:,output_var]
                    if output_var==0:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                        output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                    else:
                        output_tmp = output_tmp - np.nanmean(output_tmp,0)
                    # Normalize the inputs 
                    if input_var==0:
                        xaxis = np.arange(21)
                        for line in range(input_tmp.shape[0]):
                            subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                            input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                    elif input_var==2:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                    else:
                        input_tmp = input_tmp - np.nanmean(input_tmp,0)
                    for time in range(21):
                        if input_tmp.shape[0] < 5:
                            regression_matrix[video,time,vel_tier,output_var-3*id_leg,input_var-4*id_leg,:] = np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
                        else:
                            reg = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                            regression_matrix[video,time,vel_tier,output_var,input_var,:] =np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])
                        
    return regression_matrix


def get_rsquare_matrix_normalization_dir(tot_input, tot_output, tot_video, id_leg, bool_lat, dir):
    """
    Computes the rsquare by integrating the feedforward model
    """
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos, 21))
    for video in range(1,n_videos+1):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (tot_output[local_idx,3*id_leg+2]<0.6))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4+4*id_leg]
        local_output = tot_output[local_idx[idx_nan],3*id_leg+bool_lat]
        # Normalization of the inputs
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(21)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs
        if bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(local_input.shape[1]):
            if dir:
                idx_plot = np.where(local_input[:,time,1]>0)[0]
            else:
                idx_plot = np.where(local_input[:,time,1]<0)[0]
            design_mat = np.hstack((np.ones((local_input[idx_plot].shape[0],1)),np.squeeze(local_input[idx_plot,time,:])))
            rsquare_diagonal[video-1, time] = multilinear_ols_rsquare(design_mat, local_output[idx_plot])

    return rsquare_diagonal

def get_rsquare_matrix_normalization_or(tot_input, tot_output, tot_video, id_leg, bool_lat):
    """
    Computes the rsquare by integrating the feedforward model
    """
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos, tot_input.shape[1]))
    for video in range(1,n_videos+1):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (tot_output[local_idx,3*id_leg+2]<0.6))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:5+5*id_leg]
        local_output = tot_output[local_idx[idx_nan],3*id_leg+bool_lat]
        # Normalization of the inputs
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        local_input[:,:,4] = local_input[:,:,4] - np.nanmean(local_input[:,:,4],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(tot_input.shape[1])
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs
        if bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(local_input.shape[1]):
            design_mat = np.hstack((np.ones((local_input.shape[0],1)),np.squeeze(local_input[:,time,:])))
            print(time)
            print(design_mat.shape)
            if design_mat.shape[0]<5:
                rsquare_diagonal[video-1, time] = np.nan
            else:
                rsquare_diagonal[video-1, time] = multilinear_ols_rsquare(design_mat, local_output)

    return rsquare_diagonal

def get_rsquare_matrix_normalization_vel(tot_input, tot_output, tot_video, tot_velocity, id_leg, bool_lat):
    """
    Computes the rsquare by integrating the feedforward model
    """
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos, tot_input.shape[1]))
    gains_diagonal = np.zeros((n_videos, tot_input.shape[1], 5))
    for video in range(1,n_videos+1):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (tot_output[local_idx,3*id_leg+2]<0.6))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4+4*id_leg]
        local_output = tot_output[local_idx[idx_nan],3*id_leg+bool_lat]
        # Inputs normalization
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        print(tot_velocity.shape)
        tmp_vel = np.abs(tot_velocity[local_idx[idx_nan]]) #np.nanmean(local_input[:,:,2],1)
        print(tmp_vel.shape)
        local_input[:,:,2] = local_input[:,:,2] - tmp_vel
        for line in range(local_input.shape[0]):
            xinput = np.arange(tot_input.shape[1])
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Outputs normalization
        if not bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        # Compute the individual linear regressions
        for time in range(local_input.shape[1]):
            design_mat = np.hstack((np.ones((local_input.shape[0],1)),np.squeeze(local_input[:,time,:])))
            if design_mat.shape[0]<5:
                rsquare_diagonal[video-1, time] = np.nan
            else:
                a,b = multilinear_ols_rsquare_gains(design_mat, local_output)
                pred_output = b @ design_mat.T
                if time == 110:
                    fig, axs = plt.subplots(1,1,figsize=(3,3))
                    axs.spines[['top','right']].set_visible(False)
                    axs.scatter(local_output, pred_output, color='b', alpha=0.3, s=20)
                    axs.plot([-0.06,0.06],[-0.06,0.06],'k:',lw=2)
                    axs.set_xlabel('True deviations')
                    axs.set_ylabel('Predicted deviations')
                    axs.set_xlim([-0.06,0.06]), axs.set_ylim([-0.06,0.06])
                    plt.tight_layout()
                    fig.savefig(os.path.join(os.getcwd(), 'fly_results','figures','scatterplot_deviations.png'),bbox_inches='tight')
                    fig.savefig(os.path.join(os.getcwd(), 'fly_results','figures','scatterplot_deviations.svg'),bbox_inches='tight')
                    plt.show()
                rsquare_diagonal[video-1, time] = a# multilinear_ols_rsquare(design_mat, local_output)
                gains_diagonal[video-1, time, :] = b
    return rsquare_diagonal, gains_diagonal

def get_rsquare_time_body(tot_input, difference_matrix, tot_video, id_leg):
    """
    Computes the rsquare for the timing prediction
    """
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos, tot_input.shape[1]))
    for video in range(1,n_videos+1):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (difference_matrix[local_idx,3*id_leg+2]<0.6))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4+4*id_leg]
        local_output = difference_matrix[local_idx[idx_nan],3*id_leg+2] - np.nanmean(difference_matrix[local_idx[idx_nan],3*id_leg+2])
        # Inputs normalization
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(tot_input.shape[1])
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Compute the individual linear regressions
        for time in range(local_input.shape[1]):
            design_mat = np.hstack((np.ones((local_input.shape[0],1)),np.squeeze(local_input[:,time,:])))
            if design_mat.shape[0]<5:
                rsquare_diagonal[video-1, time] = np.nan
            else:
                a,b = multilinear_ols_rsquare_gains(design_mat, local_output)
                # pred_output = b @ design_mat.T
                # if time == 110:
                #     fig, axs = plt.subplots(1,1,figsize=(3,3))
                #     axs.spines[['top','right']].set_visible(False)
                #     axs.scatter(local_output, pred_output, color='b', alpha=0.3, s=20)
                #     axs.plot([-0.06,0.06],[-0.06,0.06],'k:',lw=2)
                #     axs.set_xlabel('True deviations')
                #     axs.set_ylabel('Predicted deviations')
                #     axs.set_xlim([-0.06,0.06]), axs.set_ylim([-0.06,0.06])
                #     plt.tight_layout()
                #     fig.savefig(os.path.join(os.getcwd(), 'fly_results','figures','scatterplot_deviations.png'),bbox_inches='tight')
                #     fig.savefig(os.path.join(os.getcwd(), 'fly_results','figures','scatterplot_deviations.svg'),bbox_inches='tight')
                    # plt.show()
                rsquare_diagonal[video-1, time] = a# multilinear_ols_rsquare(design_mat, local_output)
    return rsquare_diagonal

def get_rsquare_matrix_normalization(tot_input, tot_output, tot_video, id_leg, bool_lat):
    """
    Computes the rsquare by integrating the feedforward model
    """
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos, tot_input.shape[1]))
    gains_diagonal = np.zeros((n_videos, tot_input.shape[1], 5))
    for video in range(1,n_videos+1):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (tot_output[local_idx,3*id_leg+2]<0.6))[0]
        local_input = tot_input[local_idx[idx_nan],:,4*id_leg:4+4*id_leg]
        local_output = tot_output[local_idx[idx_nan],3*id_leg+bool_lat]
        # Inputs normalization
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(tot_input.shape[1])
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Outputs normalization
        if not bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        # Compute the individual linear regressions
        for time in range(local_input.shape[1]):
            design_mat = np.hstack((np.ones((local_input.shape[0],1)),np.squeeze(local_input[:,time,:])))
            if design_mat.shape[0]<5:
                rsquare_diagonal[video-1, time] = np.nan
            else:
                a,b = multilinear_ols_rsquare_gains(design_mat, local_output)
                pred_output = b @ design_mat.T
                if time == 110:
                    fig, axs = plt.subplots(1,1,figsize=(3,3))
                    axs.spines[['top','right']].set_visible(False)
                    axs.scatter(local_output, pred_output, color='b', alpha=0.3, s=20)
                    axs.plot([-0.06,0.06],[-0.06,0.06],'k:',lw=2)
                    axs.set_xlabel('True deviations')
                    axs.set_ylabel('Predicted deviations')
                    axs.set_xlim([-0.06,0.06]), axs.set_ylim([-0.06,0.06])
                    plt.tight_layout()
                    fig.savefig(os.path.join(os.getcwd(), 'fly_results','figures','scatterplot_deviations.png'),bbox_inches='tight')
                    fig.savefig(os.path.join(os.getcwd(), 'fly_results','figures','scatterplot_deviations.svg'),bbox_inches='tight')
                    plt.show()
                rsquare_diagonal[video-1, time] = a# multilinear_ols_rsquare(design_mat, local_output)
                gains_diagonal[video-1, time, :] = b
    return rsquare_diagonal, gains_diagonal

def compute_feedforward_rsquare(tot_input, tot_output, tot_animal):
    """
    Computes the rsquares of the linear fit of the velocity dependent controller
    """
    n_animals = int(np.max(tot_animal))+1
    matrix_rsquares = np.zeros((n_animals-1,6))
    for subject in range(1,n_animals):
        idx_local = np.where(tot_animal[:,0]==subject)[0]
        idx_nans = np.where(~np.isnan(tot_input[idx_local,0,0]))[0]
        idx_nans_ = np.random.choice(np.arange(len(idx_nans)),int(np.floor(len(idx_nans)/10)))
        if len(idx_nans_)<5:
            matrix_rsquares[subject-1,:] = np.array([np.nan, np.nan, np.nan, np.nan,np.nan,np.nan])
        else:
            for leg in range(3):
                input_vel = np.nanmean(tot_input[idx_local[idx_nans[idx_nans_]],:,2],1)
                for output_var in range(2):
                    local_output = tot_output[idx_local[idx_nans[idx_nans_]],output_var+3*leg]
                    local_reg = scipy.stats.linregress(input_vel, local_output)
                    matrix_rsquares[subject-1,output_var+2*leg] = local_reg.rvalue**2
    
    return matrix_rsquares


def get_rsquare_matrix(tot_input, tot_output, tot_video, id_leg, bool_lat):
    """Computes the rsquare based on the multilinear regression"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos,tot_input.shape[1]))
    for video in range(1,9):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (tot_output[local_idx,3*id_leg+2]<0.6))[0]
        tmp_output = tot_output[local_idx[idx_nan],3*id_leg+bool_lat] - np.nanmean(tot_output[local_idx[idx_nan],3*id_leg+bool_lat],0)
        for time in range(tot_input.shape[1]):
            tmp_input = tot_input[local_idx[idx_nan],time,4*id_leg:4+4*id_leg]
            design_mat = np.hstack((np.ones((tmp_input.shape[0],1)),tmp_input))
            pred_value = tmp_output
            if design_mat.shape[0]<10:
                rsquare_diagonal[video-1, time] = np.nan
            else:
                a,b = multilinear_ols_rsquare_gains(design_mat, pred_value)
                pred_local = b @ design_mat.T
                # if time == 100:
                #     fig, axs = plt.subplots(1,1,figsize=(3,3))
                #     axs.spines[['top','right']].set_visible(False)
                #     axs.scatter(pred_value,pred_local,color='r',s=20,alpha=0.5)
                #     axs.set_xlim([-0.06,0.06]), axs.set_ylim([-0.06,0.06])
                #     axs.plot([-0.05,0.05],[-0.05,0.05],color='k',lw=2,ls=':')
                #     axs.set_xlabel('True deviation'), axs.set_ylabel('Predicted deviation')
                #     plt.tight_layout()
                #     fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','scatter_baseline.png'),bbox_inches='tight')
                #     fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','scatter_baseline.svg'),bbox_inches='tight')
                #     plt.show()
                rsquare_diagonal[video-1, time] = multilinear_ols_rsquare(design_mat, pred_value)

    return rsquare_diagonal


def get_rsquare_time_self(tot_input, difference_matrix, tot_video, id_leg):
    """Computes the rsquare based on the multilinear regression"""
    n_videos = len(set(tot_video[~np.isnan(tot_video)]))
    rsquare_diagonal = np.zeros((n_videos,tot_input.shape[1]))
    for video in range(1,9):
        local_idx = np.where(tot_video==video)[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,15,0]) & (difference_matrix[local_idx,3*id_leg+2]<0.6))[0]
        tmp_output = difference_matrix[local_idx[idx_nan],3*id_leg+2] - np.nanmean(difference_matrix[local_idx[idx_nan],3*id_leg+2])
        for time in range(tot_input.shape[1]):
            tmp_input = tot_input[local_idx[idx_nan],time,4*id_leg:4+4*id_leg]
            design_mat = np.hstack((np.ones((tmp_input.shape[0],1)),tmp_input))
            pred_value = tmp_output
            if design_mat.shape[0]<10:
                rsquare_diagonal[video-1, time] = np.nan
            else:
                a,b = multilinear_ols_rsquare_gains(design_mat, pred_value)
                # pred_local = b @ design_mat.T
                # if time == 100:
                #     fig, axs = plt.subplots(1,1,figsize=(3,3))
                #     axs.spines[['top','right']].set_visible(False)
                #     axs.scatter(pred_value,pred_local,color='r',s=20,alpha=0.5)
                #     axs.set_xlim([-0.06,0.06]), axs.set_ylim([-0.06,0.06])
                #     axs.plot([-0.05,0.05],[-0.05,0.05],color='k',lw=2,ls=':')
                #     axs.set_xlabel('True deviation'), axs.set_ylabel('Predicted deviation')
                #     plt.tight_layout()
                #     fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','scatter_baseline.png'),bbox_inches='tight')
                #     fig.savefig(os.path.join(os.getcwd(),'fly_results','figures','scatter_baseline.svg'),bbox_inches='tight')
                #     plt.show()
                rsquare_diagonal[video-1, time] = multilinear_ols_rsquare(design_mat, pred_value)

    return rsquare_diagonal

def multilinear_ols_thetahat(X,y):
    """Returns the optimal gains of the multilinear regression"""
    theta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta_hat

def multilinear_ols_rsquare(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare

def multilinear_ols_rsquare_gains(X,y):
    """Returns the optimal gains and te associated r-squares for the multilinear regression between X and y"""
    theta_hat = np.linalg.inv(X.T@X) @(X.T@y)
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare, theta_hat

def multilinear_ols_rsquare_absolute(X, y , y_mean):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat + y_mean
    ytrue = y + y_mean
    rsquare = 1 - np.sum(np.square(yhat-ytrue)) / np.sum(np.square(ytrue))
    return rsquare