import numpy as np
from tqdm import tqdm
import scipy.stats
import os, sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import pickle

def exp_func(x,A,B,C,D):
    return A*np.exp(-B*(x-C))+D

def compute_velocity(input_matrix):
    """Computes the velocity of the input matrix using 4th order finite differnece"""
    tmp_vel_matrix = np.zeros((input_matrix.shape[0], input_matrix.shape[1], 3))
    dt = 1/200
    for slab in range(3):
        tmp_vel_matrix[2:-2,:,slab] = ( - input_matrix[4:,:,slab] + 8*input_matrix[3:-1,:,slab] - 8*input_matrix[1:-3,:,slab] + input_matrix[:-4,:,slab]) /(12*dt)
    output_matrix = np.concatenate((input_matrix, tmp_vel_matrix),-1)
    return output_matrix

def load_camargo_horizon_io(horizon):
    """
    Load the processed io data from the Camargo dataset
    """
    input_path = os.path.join(os.getcwd(),'Datasets','Humans','Camargo','processed')
    with open(os.path.join(input_path,f'tot_input_fr_{horizon}.pkl'),'rb') as f1:
        tot_input = pickle.load(f1)
    with open(os.path.join(input_path,f'tot_output_fr_{horizon}.pkl'),'rb') as f2:
        tot_output = pickle.load(f2)
    with open(os.path.join(input_path,f'tot_subject1_{horizon}.pkl'),'rb') as f3:
        tot_subject = pickle.load(f3)
    return tot_input, tot_output, tot_subject

def load_camargo_horizon(horizon):
    """ Load the processed data from the Camargo dataset"""
    input_path = os.path.join(os.getcwd(), 'Datasets','Humans','Camargo','processed')
    with open(os.path.join(input_path,f'tot_input_self_fr_{horizon}.pkl'),'rb') as f1:
        tot_input_self = pickle.load(f1)
    with open(os.path.join(input_path,f'tot_subject1_{horizon}.pkl'),'rb') as f2:
        tot_subject = pickle.load(f2)
    return tot_input_self, tot_subject

def load_data_camargo():
    """ Load the processed data from the Camargo dataset"""
    input_path = os.path.join(os.getcwd(),'Datasets','Humans','Camargo','processed')
    with open(os.path.join(input_path,'tot_input_fr.pkl'),'rb') as f1:
        tot_input_leg1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_output_fr.pkl'),'rb') as f2:
        tot_output_leg1 = pickle.load(f2)
    with open(os.path.join(input_path,'tot_input_fl.pkl'),'rb') as f3:
        tot_input_leg2 = pickle.load(f3)
    with open(os.path.join(input_path,'tot_output_fl.pkl'),'rb') as f4:
        tot_output_leg2 = pickle.load(f4)
    with open(os.path.join(input_path,'tot_input_self_fr.pkl'),'rb') as f1:
        tot_input_self_leg1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_input_self_fl.pkl'),'rb') as f1:
        tot_input_self_leg2 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_output_self_fr.pkl'),'rb') as f1:
        tot_output_self_leg1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_output_self_fl.pkl'),'rb') as f1:
        tot_output_self_leg2 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_subject1.pkl'),'rb') as f1:
        tot_subject1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_subject2.pkl'),'rb') as f1:
        tot_subject2 = pickle.load(f1)
    list_io1 = [tot_input_leg1, tot_input_self_leg1, tot_output_leg1, tot_output_self_leg1]
    list_io2 = [tot_input_leg2, tot_input_self_leg2, tot_output_leg2, tot_output_self_leg2]
    list_sub = [tot_subject1, tot_subject2]
    return list_io1, list_io2, list_sub

def load_data_camargo_timing():
    """ Load the processed data from the Camargo dataset"""
    input_path = os.path.join(os.getcwd(),'Datasets','Humans','Camargo','processed_time_bis')
    with open(os.path.join(input_path,'tot_input_fr.pkl'),'rb') as f1:
        tot_input_leg1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_output_fr.pkl'),'rb') as f2:
        tot_output_leg1 = pickle.load(f2)
    with open(os.path.join(input_path,'tot_input_fl.pkl'),'rb') as f3:
        tot_input_leg2 = pickle.load(f3)
    with open(os.path.join(input_path,'tot_output_fl.pkl'),'rb') as f4:
        tot_output_leg2 = pickle.load(f4)
    with open(os.path.join(input_path,'tot_input_self_fr.pkl'),'rb') as f1:
        tot_input_self_leg1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_input_self_fl.pkl'),'rb') as f1:
        tot_input_self_leg2 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_output_self_fr.pkl'),'rb') as f1:
        tot_output_self_leg1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_output_self_fl.pkl'),'rb') as f1:
        tot_output_self_leg2 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_subject1.pkl'),'rb') as f1:
        tot_subject1 = pickle.load(f1)
    with open(os.path.join(input_path,'tot_subject2.pkl'),'rb') as f1:
        tot_subject2 = pickle.load(f1)
    list_io1 = [tot_input_leg1, tot_input_self_leg1, tot_output_leg1, tot_output_self_leg1]
    list_io2 = [tot_input_leg2, tot_input_self_leg2, tot_output_leg2, tot_output_self_leg2]
    list_sub = [tot_subject1, tot_subject2]
    return list_io1, list_io2, list_sub

def get_stats_feedback_controller(foreaft_body, foreaft_self, lateral_body, lateral_self):
    """
    Stats for the distribution level of the rsquares in humans 
    """
    
    output_matrix = np.zeros((foreaft_body.shape[1],2))
    for time in range(output_matrix.shape[0]):
        test_1 = scipy.stats.ks_2samp(foreaft_body[:,time], foreaft_self[:,time],alternative='less')
        test_2 = scipy.stats.ks_2samp(lateral_body[:,time], lateral_self[:,time],alternative='less')
        output_matrix[time,0] = test_1.pvalue
        output_matrix[time,1] = test_2.pvalue

    return output_matrix

def gait_segmentation_camargo_horizon(input_matrix, horizon):
    """Performs the gait segmentation for Camargo dataset"""
    idx_nans = np.where(np.isnan(input_matrix[:,0,0]))[0]
    tot_input_pelvis1 = np.zeros((1,1+100*horizon,3))
    tot_input_leg1 = np.zeros((1,1+100*horizon,3))
    tot_input_pelvis2 = np.zeros((1,1+100*horizon,3))
    tot_input_leg2 = np.zeros((1,1+100*horizon,3))
    # Compute the velocity of each marker here - this is important for later ...
    tot_input_pelvis1 = compute_velocity(tot_input_pelvis1)
    tot_input_pelvis2 = compute_velocity(tot_input_pelvis2)
    tot_input_leg1 = compute_velocity(tot_input_leg1)
    tot_input_leg2 = compute_velocity(tot_input_leg2)
    local_count, local_count_ = 0, 0
    for ii in range(len(idx_nans)-1):
        local_matrix = input_matrix[idx_nans[ii]+1:idx_nans[ii+1],:,:]
        local_matrix = compute_velocity(local_matrix)
        diff_leg1 = local_matrix[:,1,:] - local_matrix[:,0,:]
        diff_leg2 = local_matrix[:,2,:] - local_matrix[:,0,:]
        b,a = scipy.signal.butter(6, 20/100, btype='lowpass')
        diff_leg1_x = scipy.signal.filtfilt(b,a,diff_leg1[:,0])
        diff_leg2_x = scipy.signal.filtfilt(b,a,diff_leg2[:,0])
        # Find the heel strike (max) and toe off (min) of the first leg
        hs_leg1, _ = scipy.signal.find_peaks(diff_leg1_x, distance=50)
        to_leg1, _ = scipy.signal.find_peaks(-diff_leg1_x, distance=50)

        # Find the heel strike (max) and toe off (min) of the second leg
        hs_leg2, _ = scipy.signal.find_peaks(diff_leg2_x, distance=50)
        to_leg2, _ = scipy.signal.find_peaks(-diff_leg2_x, distance=50)
        local_count += len(hs_leg1)
        # Extract the gait cycles for the first leg
        for idx in range(len(hs_leg1)):
            if ((len(hs_leg1[idx:])>1) & (len(hs_leg1[:idx+1])>horizon)):
                idx_middle = np.where((hs_leg2>hs_leg1[idx]) & (hs_leg2<hs_leg1[idx+1]))[0]
                if len(idx_middle)!=0:
                    # if the middle point is to close to one of the edge, drop it...
                    dist = hs_leg1[idx+1] - hs_leg1[idx]
                    if ((hs_leg2[idx_middle[0]]-0.4*dist < hs_leg1[idx]) or (hs_leg2[idx_middle[0]]+0.4*dist > hs_leg1[idx+1])):
                        continue
                    local_ref = local_matrix[hs_leg2[idx_middle[0]],2,:]
                    input_idx = np.arange(hs_leg1[idx-(horizon-1)],hs_leg1[idx+1])
                    input_pelvis = local_matrix[input_idx,0,:]
                    input_leg = local_matrix[input_idx,1,:]
                    local_time = np.linspace(0,len(input_idx),1+100*horizon)
                    local_interpolation_pelvis = np.zeros((1,1+100*horizon,6))
                    local_interpolation_leg = np.zeros((1,1+100*horizon,6))
                    for dim in range(6):
                        if dim<3:
                            local_interpolation_pelvis[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_pelvis[:,dim]) - local_ref[dim]
                            local_interpolation_leg[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_leg[:,dim]) - local_ref[dim]
                        else:
                            local_interpolation_pelvis[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_pelvis[:,dim])
                            local_interpolation_leg[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_leg[:,dim])
                    tot_input_pelvis1 = np.concatenate((tot_input_pelvis1, local_interpolation_pelvis),0)
                    tot_input_leg1 = np.concatenate((tot_input_leg1, local_interpolation_leg),0)

        for idx in range(len(hs_leg2)):
            if ((len(hs_leg2[idx:])>1) & (len(hs_leg2[:idx+1])>horizon)):
                idx_middle = np.where((hs_leg1>hs_leg2[idx]) & (hs_leg1<hs_leg2[idx+1]))[0]
                if len(idx_middle)!=0:
                    # if the middle point is to close to one of the edge, drop it...
                    dist = hs_leg2[idx+1] - hs_leg2[idx]
                    if ((hs_leg1[idx_middle[0]]-0.4*dist < hs_leg2[idx]) or (hs_leg1[idx_middle[0]]+0.4*dist > hs_leg2[idx+1])):
                        continue
                    local_ref = local_matrix[hs_leg1[idx_middle[0]],1,:]
                    input_idx = np.arange(hs_leg2[idx-(horizon-1)],hs_leg2[idx+1])
                    input_pelvis = local_matrix[input_idx,0,:]
                    input_leg = local_matrix[input_idx,2,:]
                    local_time = np.linspace(0,len(input_idx),1+100*horizon)
                    local_interpolation_pelvis = np.zeros((1,1+100*horizon,6))
                    local_interpolation_leg = np.zeros((1,1+100*horizon,6))
                    for dim in range(6):
                        local_interpolation_pelvis[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_pelvis[:,dim]) - local_ref[dim]
                        local_interpolation_leg[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_leg[:,dim]) - local_ref[dim]
                    tot_input_pelvis2 = np.concatenate((tot_input_pelvis2, local_interpolation_pelvis),0)
                    tot_input_leg2 = np.concatenate((tot_input_leg2, local_interpolation_leg),0)
    return tot_input_pelvis1[1:,:,:], tot_input_leg1[1:,:,:], tot_input_pelvis2[1:,:,:], tot_input_leg2[1:,:,:]


def gait_segmentation_camargo(input_matrix):
    """Performs the gait segmentation for Camargo dataset"""
    idx_nans = np.where(np.isnan(input_matrix[:,0,0]))[0]
    tot_input_pelvis1 = np.zeros((1,101,3))
    tot_input_leg1 = np.zeros((1,101,3))
    tot_input_pelvis2 = np.zeros((1,101,3))
    tot_input_leg2 = np.zeros((1,101,3))
    # Compute the velocity of each marker here - this is important for later ... 
    tot_input_pelvis1 = compute_velocity(tot_input_pelvis1)
    tot_input_pelvis2 = compute_velocity(tot_input_pelvis2)
    tot_input_leg1 = compute_velocity(tot_input_leg1)
    tot_input_leg2 = compute_velocity(tot_input_leg2)
    local_count = 0
    local_count_ = 0
    for ii in range(len(idx_nans)-1):
        local_matrix = input_matrix[idx_nans[ii]+1:idx_nans[ii+1],:,:]
        local_matrix = compute_velocity(local_matrix)
        diff_leg1 = local_matrix[:,1,:] - local_matrix[:,0,:]
        diff_leg2 = local_matrix[:,2,:] - local_matrix[:,0,:]
        b,a = scipy.signal.butter(6,20/100,btype='lowpass')
        diff_leg1_x = scipy.signal.filtfilt(b,a,diff_leg1[:,0])
        diff_leg2_x = scipy.signal.filtfilt(b,a,diff_leg2[:,0])
        # Find heel strike (max) and toe off (min) of the first leg
        hs_leg1, _ = scipy.signal.find_peaks(diff_leg1_x, distance=50)
        to_leg1, _ = scipy.signal.find_peaks(-diff_leg1_x, distance=50)
        

        # Find heel strike (max) and toe off (min) of the second leg
        hs_leg2, _ = scipy.signal.find_peaks(diff_leg2_x, distance=50)
        to_leg2, _ = scipy.signal.find_peaks(-diff_leg2_x, distance=50)
        local_count += len(hs_leg1)
        # Extract the gait cycles for the first leg 
        for idx in range(len(hs_leg1)):
            if len(hs_leg1[idx:])>1:
                idx_middle = np.where((hs_leg2>hs_leg1[idx]) & (hs_leg2<hs_leg1[idx+1]))[0]
                if len(idx_middle)!=0:
                    # if the middle point is to close to one of the edge, drop it...
                    dist = hs_leg1[idx+1] - hs_leg1[idx]
                    if ((hs_leg2[idx_middle[0]]-0.4*dist < hs_leg1[idx]) or (hs_leg2[idx_middle[0]]+0.4*dist > hs_leg1[idx+1])):
                        continue
                    local_ref = local_matrix[hs_leg2[idx_middle[0]],2,:]
                    input_idx = np.arange(hs_leg1[idx],hs_leg1[idx+1])
                    input_pelvis = local_matrix[input_idx,0,:]
                    input_leg = local_matrix[input_idx,1,:]
                    local_time = np.linspace(0,len(input_idx),101)
                    local_interpolation_pelvis = np.zeros((1,101,6))
                    local_interpolation_leg = np.zeros((1,101,6))
                    for dim in range(6):
                        if dim<3:
                            local_interpolation_pelvis[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_pelvis[:,dim]) - local_ref[dim]
                            local_interpolation_leg[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_leg[:,dim]) - local_ref[dim]
                        else:
                            local_interpolation_pelvis[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_pelvis[:,dim])
                            local_interpolation_leg[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_leg[:,dim])
                    tot_input_pelvis1 = np.concatenate((tot_input_pelvis1, local_interpolation_pelvis),0)
                    tot_input_leg1 = np.concatenate((tot_input_leg1, local_interpolation_leg),0)

        for idx in range(len(hs_leg2)):
            if len(hs_leg2[idx:])>1:
                idx_middle = np.where((hs_leg1>hs_leg2[idx]) & (hs_leg1<hs_leg2[idx+1]))[0]
                if len(idx_middle)!=0:
                    # if the middle point is to close to one of the edge, drop it...
                    dist = hs_leg2[idx+1] - hs_leg2[idx]
                    if ((hs_leg1[idx_middle[0]]-0.4*dist < hs_leg2[idx]) or (hs_leg1[idx_middle[0]]+0.4*dist > hs_leg2[idx+1])):
                        continue
                    local_ref = local_matrix[hs_leg1[idx_middle[0]],1,:]
                    input_idx = np.arange(hs_leg2[idx],hs_leg2[idx+1])
                    input_pelvis = local_matrix[input_idx,0,:]
                    input_leg = local_matrix[input_idx,2,:]
                    local_time = np.linspace(0,len(input_idx),101)
                    local_interpolation_pelvis = np.zeros((1,101,6))
                    local_interpolation_leg = np.zeros((1,101,6))
                    for dim in range(6):
                        local_interpolation_pelvis[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_pelvis[:,dim]) - local_ref[dim]
                        local_interpolation_leg[:,:,dim] = np.interp(local_time, np.arange(0,len(input_idx)), input_leg[:,dim]) - local_ref[dim]
                    tot_input_pelvis2 = np.concatenate((tot_input_pelvis2, local_interpolation_pelvis),0)
                    tot_input_leg2 = np.concatenate((tot_input_leg2, local_interpolation_leg),0)
    return tot_input_pelvis1[1:,:,:], tot_input_leg1[1:,:,:], tot_input_pelvis2[1:,:,:], tot_input_leg2[1:,:,:]

def get_input_output_fr(matrix_body, matrix_leg):
    """Gets the input and output of the linear model in the right format"""
    tot_input_fr = np.zeros((matrix_body.shape[0], matrix_body.shape[1],4))
    tot_input_fr[:,:,0] = matrix_body[:,:,0]
    tot_input_fr[:,:,1] = matrix_body[:,:,2]
    tot_input_fr[:,:,2] = matrix_body[:,:,3]
    tot_input_fr[:,:,3] = matrix_body[:,:,5]

    tot_output_fr = matrix_leg[:,-1,[0,2]]

    return tot_input_fr, tot_output_fr


def get_timing_contact(input_time):
    """Get the relative contact timings"""
    left_vec, right_vec = input_time[::2], input_time[1::2]
    left_diff, right_diff = np.diff(left_vec), np.diff(right_vec)
    right_m_left_diff = right_vec - left_vec
    left_m_right_diff = left_vec - right_vec
    right_m_left_diff = right_vec - left_vec
    left_m_right_diff = left_vec[1:] - right_vec[:-1]
    right_contact_timing = right_m_left_diff[:-1] / left_diff
    left_contact_timing = left_m_right_diff/right_diff

    return right_contact_timing, left_contact_timing  

def get_input_lines(input_matrix,rt,lt):
    """Returns the lines usefull for the inputs of the model"""
    input_matrix_lt = input_matrix[::2,:,:]
    input_matrix_rt = input_matrix[1:-1:2,:,:]
    output_matrix_lt = lt
    output_matrix_rt = rt[1:]

    return input_matrix_lt, input_matrix_rt, output_matrix_lt, output_matrix_rt   

def get_regression_matrix_yang(input_data_yang, list_inputs_yang, bool_right):
    """
    Computes the feedback gains for the human data
    """
    matrix_regression = np.zeros((30,101,6,2,5))
    map_variable = [0,1,2,9,10,11]
    for subject in tqdm(range(30)):
        subject_timings = input_data_yang[subject]
        subject_inputs = list_inputs_yang[subject]

        rt, lt = get_timing_contact(subject_timings)
        ilt, irt, olt, ort = get_input_lines(subject_inputs, rt, lt)
        local_output_vec = np.array([3+3*int(bool_right), 4+3*int(bool_right)])
        for time in range(irt.shape[1]):
            for input_var in range(len(map_variable)):
                for output_var in range(2):
                    if bool_right:
                        local_reg = scipy.stats.linregress(irt[:,time,map_variable[input_var]], irt[:,-1,local_output_vec[output_var]])
                    else:
                        local_reg = scipy.stats.linregress(ilt[:,time,map_variable[input_var]], ilt[:,-1,local_output_vec[output_var]])
                    matrix_regression[subject,time,input_var,output_var,:] = np.array([local_reg.slope, local_reg.intercept, local_reg.rvalue, local_reg.pvalue, local_reg.stderr])

    return matrix_regression

def get_regression_matrix_mathis(list_events, list_procs, bool_right):
    n_subjects = len(list_events)
    map_variable = [0, 1, 2, 9, 10, 11]
    matrix_regression = np.zeros((n_subjects, 101, len(map_variable), 2, 5))
    for subject in tqdm(range(n_subjects)):
        subject_timings = list_events[subject]
        subject_inputs = list_procs[subject]

        rt, lt = get_timing_contact(subject_timings)
        ilt, irt, _, _ = get_input_lines(subject_inputs, rt, lt)
        local_output_vec = np.array([3+3*int(bool_right), 4+3*int(bool_right)])
        for time in range(irt.shape[1]):
            for input_var in range(len(map_variable)):
                for output_var in range(2):
                    if bool_right:
                        local_reg = scipy.stats.linregress(irt[:,time,map_variable[input_var]], irt[:,-1,local_output_vec[output_var]])
                    else:
                        local_reg = scipy.stats.linregress(ilt[:,time,map_variable[input_var]], ilt[:,-1,local_output_vec[output_var]])
                    matrix_regression[subject,time,input_var,output_var,:] = np.array([local_reg.slope, local_reg.pvalue, local_reg.rvalue, local_reg.intercept, local_reg.intercept_stderr])
    return matrix_regression

def multilinear_ols_rsquare(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare

def multilinear_ols_rsquare_gains(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(y-yhat)) / np.sum(np.square(y))
    return rsquare, theta_hat


def get_rsquare_self_matrix_baseline(tot_input_self_matrix, tot_output_matrix, tot_subject, direction):
    """
    Computes the rsquare based on the multilinear regression for the camargo dataset
    """
    n_subjects = int(np.max(tot_subject))+1
    r_square_matrix = np.zeros((n_subjects,201))
    for subject in tqdm(range(n_subjects)):
        local_input = tot_input_self_matrix[tot_subject==subject,:,:]
        local_output = tot_output_matrix[tot_subject==subject, direction]
        local_output = local_output - np.nanmean(local_output,0)
        for time in range(local_input.shape[1]):
            if local_input.shape[0]==1:
                r_square_matrix[subject,time] = np.nan
                continue
            design_matrix = np.hstack((np.ones((local_input.shape[0],1)), np.squeeze(local_input[:,time,:])))
            a ,b = multilinear_ols_rsquare_gains(design_matrix, local_output)
            pred_output = b @ design_matrix.T
            r_square_matrix[subject, time] = multilinear_ols_rsquare(design_matrix, local_output)
    
    return r_square_matrix


def compute_regression_fr_norm(tot_input_matrix, tot_output_matrix, tot_sub):
    """
    Computes the regression for the human dataset
    """
    n_sub = int(np.max(tot_sub))+1
    regression_matrix_small = np.zeros((n_sub,101,3,4,5))
    for subject in tqdm(range(n_sub)):
        local_idx = np.where((tot_sub==subject))[0]
        idx_nan = np.where(~np.isnan(tot_input_matrix[local_idx,0,0]))[0]
        local_input = tot_input_matrix[local_idx[idx_nan],:,:]
        local_output = tot_output_matrix[local_idx[idx_nan],:]
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
                    x_axis = np.arange(101)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(x_axis,input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (x_axis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                if input_var==3:
                    for line in range(input_tmp.shape[0]):
                        input_tmp[line,:] = input_tmp[line,:] / tmp_len[line]
                if input_var==1:
                    for line in range(input_tmp.shape[0]):
                        input_tmp[line,:] = input_tmp[line,:] / tmp_len[line]
                if output_var==1:
                    for line in range(output_tmp.shape[0]):
                        output_tmp[line] = output_tmp[line] / tmp_len[line]
                for time in range(101):
                    idx_small = np.where(input_tmp[:,time]!=0)[0]
                    if len(idx_small)<5:
                        regression_matrix_small[subject,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                        regression_matrix_small[subject,time,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small

def compute_ffwd_inputs(tot_input_leg1, tot_subject1):
    """
    Computes the body fixed marker information for the feedforward controller
    """

    n_subjects = int(np.max(tot_subject1))+1
    mat_slope, mat_pvalue = np.zeros((n_subjects,2)), np.zeros((n_subjects,2))
    for subject in range(n_subjects):
        local_input = tot_input_leg1[tot_subject1==subject,:,:]
        local_velocity = np.nanmean(local_input[:,:,2],1)
        local_x_amplitude = np.zeros((local_velocity.shape[0],))
        local_y_amplitude = np.zeros((local_velocity.shape[0],))
        for line in range(local_x_amplitude.shape[0]):
            local_x_amplitude[line] = local_input[line,-1,0] - local_input[line,0,0]
            local_y_amplitude[line] = np.max(local_input[line,:,1]) - np.min(local_input[line,:,1])

        # Performing regressions between these factors ...
        idx_nans = np.where(~np.isnan(local_velocity))[0]
        reg_x_amplitude = scipy.stats.linregress(local_velocity[idx_nans], local_x_amplitude[idx_nans])
        reg_y_amplitude = scipy.stats.linregress(local_velocity[idx_nans], local_y_amplitude[idx_nans])
        mat_slope[subject,0], mat_slope[subject,1] = reg_x_amplitude.slope, reg_y_amplitude.slope
        mat_pvalue[subject,0], mat_pvalue[subject,1] = reg_x_amplitude.pvalue, reg_y_amplitude.pvalue

    # Let's generate the figure for the last individual
    x_reg = np.linspace(1250,2000)
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(local_velocity, local_y_amplitude,color='k',s=10,alpha=0.2)
    axs.plot(x_reg, x_reg*reg_y_amplitude.slope+reg_y_amplitude.intercept,'k',lw=2)
    axs.set_xlim([1250,2000])
    plt.tight_layout()
    plt.show()
    return mat_slope, mat_pvalue



def compute_regression_fr(tot_input_matrix, tot_output_matrix, tot_sub):
    """
    Computes the regression for the human dataset
    """
    n_sub = int(np.max(tot_sub))+1
    regression_matrix_small = np.zeros((n_sub,101,3,4,5))
    for subject in tqdm(range(n_sub)):
        local_idx = np.where((tot_sub==subject))[0]
        idx_nan = np.where(~np.isnan(tot_input_matrix[local_idx,0,0]))[0]
        local_input = tot_input_matrix[local_idx[idx_nan],:,:]
        local_output = tot_output_matrix[local_idx[idx_nan],:]
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
                    x_axis = np.arange(101)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(x_axis,input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (x_axis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                for time in range(101):
                    idx_small = np.where(input_tmp[:,time]!=0)[0]
                    if len(idx_small)<5:
                        regression_matrix_small[subject,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                        regression_matrix_small[subject,time,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small


def regression_laterality_baseline(tot_input_matrix, tot_output_matrix, tot_sub):
    """
    Computes the laterality metrics for the human dataset
    """
    n_sub = int(np.max(tot_sub))+1
    regression_matrix = np.zeros((n_sub,101,3,4,5))
    for subject in tqdm(range(n_sub)):
        local_idx = np.where((tot_sub==subject))[0]
        idx_nan = np.where(~np.isnan(tot_input_matrix[local_idx,0,0]))[0]
        local_input = tot_input_matrix[local_idx[idx_nan],:,:]
        local_output = tot_output_matrix[local_idx[idx_nan],:]
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
                    x_axis = np.arange(101)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(x_axis,input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (x_axis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                for time in range(101):
                    if (input_tmp.shape[0]<5):
                        regression_matrix[subject,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_large = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                        regression_matrix[subject,time,output_var,input_var,:] = np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        
    return regression_matrix

def regression_laterality_multilinear(tot_input_matrix, tot_output_matrix, tot_sub):
    """
    Computes the laterality metrics for the human dataset
    """
    n_sub = int(np.max(tot_sub))+1
    regression_matrix_small = np.zeros((n_sub,101,3,4,2))
    regression_matrix_large = np.zeros((n_sub,101,3,4,2))
    for subject in tqdm(range(n_sub)):
        local_idx = np.where((tot_sub==subject))[0]
        idx_nan = np.where(~np.isnan(tot_input_matrix[local_idx,0,0]))[0]
        local_input = tot_input_matrix[local_idx[idx_nan],:,:]
        local_output = tot_output_matrix[local_idx[idx_nan],:]
        for output_var in range(local_output.shape[-1]):
            # Normalize the inputs 
            input_tmp = copy.deepcopy(local_input)
            output_tmp = local_output[:,output_var]
            if output_var==0:
                tmp_vel = np.nanmean(local_input[:,:,2],1)
                subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
            else:
                output_tmp = output_tmp - np.nanmean(output_tmp,0)
            x_axis = np.arange(101)
            for line in range(input_tmp.shape[0]):
                subjectlin = scipy.stats.linregress(x_axis, input_tmp[line,:,0])
                input_tmp[line,:,0] = input_tmp[line,:,0] - (x_axis*subjectlin.slope + x_axis*subjectlin.intercept)
            input_tmp[:,:,2] = input_tmp[:,:,2] - np.expand_dims(np.nanmean(input_tmp[:,:,2],1),-1)
            input_tmp[:,:,1] = input_tmp[:,:,1] - np.nanmean(input_tmp[:,:,1],0)
            input_tmp[:,:,3] = input_tmp[:,:,3] - np.nanmean(input_tmp[:,:,3],0)
            for time in range(101):
                idx_large = np.where(input_tmp[:,time,1]>0)[0]
                idx_small = np.where(input_tmp[:,time,1]<0)[0]
                print(len(idx_large))
                print(len(idx_small))
                if ((len(idx_large)<5) or (len(idx_small)<5)):
                    regression_matrix_large[subject,time,output_var,:,:] = np.nan
                    regression_matrix_small[subject,time,output_var,:,:] = np.nan
                else:
                    design_mat_large = np.hstack((np.ones((input_tmp[idx_large,time,:].shape[0],1)),input_tmp[idx_large,time,:]))
                    design_mat_small = np.hstack((np.ones((input_tmp[idx_small,time,:].shape[0],1)),input_tmp[idx_small,time,:]))
                    rsquare_large, gains_large = multilinear_ols_rsquare_gains(design_mat_large,output_tmp[idx_large])
                    rsquare_small, gains_small = multilinear_ols_rsquare_gains(design_mat_small, output_tmp[idx_small])
                    #reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                    #reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                    for ii in range(4):
                        regression_matrix_large[subject,time,output_var,ii,:] = np.array([rsquare_large,gains_large[ii+1]])#np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        regression_matrix_small[subject,time,output_var,ii,:] = np.array([rsquare_small,gains_small[ii+1]])#np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large

def regression_laterality(tot_input_matrix, tot_output_matrix, tot_sub):
    """
    Computes the laterality metrics for the human dataset
    """
    n_sub = int(np.max(tot_sub))+1
    regression_matrix_small = np.zeros((n_sub,101,3,4,5))
    regression_matrix_large = np.zeros((n_sub,101,3,4,5))
    for subject in tqdm(range(n_sub)):
        local_idx = np.where((tot_sub==subject))[0]
        idx_nan = np.where(~np.isnan(tot_input_matrix[local_idx,0,0]))[0]
        local_input = tot_input_matrix[local_idx[idx_nan],:,:]
        local_output = tot_output_matrix[local_idx[idx_nan],:]
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
                    x_axis = np.arange(101)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(x_axis,input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (x_axis*subjectlin.slope + subjectlin.intercept)
                elif input_var==2:
                    tmp_vel = np.nanmean(local_input[:,:,2],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                for time in range(101):
                    idx_large = np.where(input_tmp[:,time]>0)[0]
                    idx_small = np.where(input_tmp[:,time]<0)[0]
                    if ((len(idx_small)<5) or (len(idx_large)<5)):
                        regression_matrix_large[subject,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        regression_matrix_small[subject,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                        reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                        regression_matrix_large[subject,time,output_var,input_var,:] = np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        regression_matrix_small[subject,time,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large


def get_rsquare_time_matrix_camargo(tot_input_matrix, difference_matrix, tot_subject):
    """
    Computes the rsquares based on the multilinear regression for the camargo time dataset
    """
    # First find the popt parameters for the entire data
    tot_velocity = np.nanmean(tot_input_matrix[:,:,2],1)
    idx_nans = np.where(~np.isnan(tot_velocity))[0]

    n_subjects = int(np.max(tot_subject))+1
    r_square_matrix = np.zeros((n_subjects,101))
    for subject in range(n_subjects):
        local_input = tot_input_matrix[tot_subject==subject,:,:]
        local_output = difference_matrix[tot_subject==subject,3] - np.nanmean(difference_matrix[tot_subject==subject,3])
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        # Normalization of the inputs
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(101)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Output normalization
        idx_local_nans = np.where((~np.isnan(local_output)) & (~np.isnan(tmp_vel)))[0]
        for time in range(local_input.shape[1]):
            design_matrix = np.hstack((np.ones((local_input.shape[0],1)), np.squeeze(local_input[:,time,:])))
            a,b = multilinear_ols_rsquare_gains(design_matrix[idx_local_nans,:],local_output[idx_local_nans])
            r_square_matrix[subject, time] = a
    return r_square_matrix

def get_rsquare_matrix_camargo(tot_input_matrix, tot_output_matrix, tot_subject, direction):
    """
    Computes the rsquare based on the multilinear regression for the camargo dataset
    """
    n_subjects = int(np.max(tot_subject))+1
    r_square_matrix = np.zeros((n_subjects, 101))
    gains_matrix = np.zeros((n_subjects,101,5))
    for subject in range(n_subjects):
        local_input = tot_input_matrix[tot_subject==subject,:,:]
        local_output = tot_output_matrix[tot_subject==subject,direction]
        # Normalization of the inputs (most important one is the foreaft position)
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(101)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs 
        if direction==0:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel*subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(local_input.shape[1]):
            if local_input.shape[0]==1:
                r_square_matrix[subject,time] = np.nan
                gains_matrix[subject,time] = np.nan
                continue
            design_matrix = np.hstack((np.ones((local_input.shape[0],1)),np.squeeze(local_input[:,time,:])))
            a, b = multilinear_ols_rsquare_gains(design_matrix, local_output)
            r_square_matrix[subject,time] = a
            gains_matrix[subject,time,:] = b
            test_pred_output = b @ design_matrix.T
    return r_square_matrix, gains_matrix


def concatenate_data(tot_input_list, tot_input_list_self, tot_output_list, list_trial_id):
    """
    Concatenate the human data so that it can be used for further data analyses
    """
    tot_input_matrix = tot_input_list[0]
    tot_input_self_matrix = tot_input_list_self[0]
    tot_output_matrix = tot_output_list[0]
    tot_trial_matrix = list_trial_id[0] * np.ones((tot_input_matrix.shape[0],1))
    tot_leg_matrix = np.expand_dims(np.mod(np.arange(tot_input_list[0].shape[0]),2),1)
    for chunk in range(1,len(tot_input_list)):
        tot_input_matrix = np.concatenate((tot_input_matrix, tot_input_list[chunk]),0)
        tot_input_self_matrix = np.concatenate((tot_input_self_matrix, tot_input_list_self[chunk]),0)
        tot_output_matrix = np.concatenate((tot_output_matrix, tot_output_list[chunk]),0)
        tot_trial_matrix = np.concatenate((tot_trial_matrix, list_trial_id[chunk] * np.ones((tot_input_list[chunk].shape[0],1))),0)
        tot_leg_matrix = np.concatenate((tot_leg_matrix, np.expand_dims(np.mod(np.arange(tot_input_list[chunk].shape[0]),2),-1)),0)

    return tot_input_matrix, tot_input_self_matrix, tot_output_matrix, tot_output_matrix, tot_trial_matrix, tot_leg_matrix


def compute_feedforward_rsquare(tot_input, tot_output, tot_sub):
    """
    Computes the rsquares of the linear fit of the velocity dependent controller
    """
    n_subjects = int(np.max(tot_sub))+1
    matrix_rsquares = np.zeros((n_subjects,2))
    for subject in range(n_subjects):
        idx_local = np.where(tot_sub==subject)[0]
        input_vel = np.nanmean(tot_input[idx_local,:,2],1)
        for output_var in range(2):
            local_output = tot_output[idx_local,output_var]
            local_reg = scipy.stats.linregress(input_vel, local_output)
            matrix_rsquares[subject,output_var] = local_reg.rvalue**2
    
    return matrix_rsquares
