import numpy as np
import pickle
import os
from tqdm import tqdm
import copy
import scipy
import matplotlib.pyplot as plt


def concatenate_data_horizon(input_list_self, output_list_self):
    """
    Concatenate the data for the horizon investigation
    """
    tot_input_self, tot_output_self = np.zeros((1,input_list_self[0].shape[1],8)), np.zeros((1,6))
    tot_animal = np.zeros((1,1))
    for animal in tqdm(range(80)):
        local_input_self, local_output_self = np.array(input_list_self[animal]), np.array(output_list_self[animal])
        idx_flip = np.where(np.nanmean(local_input_self[:,:,2],1)<0)[0]
        local_input_self_flip, local_output_self_flip = copy.deepcopy(local_input_self), copy.deepcopy(local_output_self)
        local_input_self_flip[idx_flip,:,:] = - local_input_self_flip[idx_flip,:,:]
        col_flip = [0,1,3,4]
        for col in col_flip:
            local_output_self_flip[idx_flip, col] = - local_output_self_flip[idx_flip,col]
        local_animal_list = animal * np.ones((local_input_self_flip.shape[0],1))
        tot_input_self = np.concatenate((tot_input_self, local_input_self_flip),0)
        tot_output_self = np.concatenate((tot_output_self, local_output_self_flip),0)
        tot_animal = np.concatenate((tot_animal, local_animal_list),0)

    tot_input_self = tot_input_self[1:,:,:]
    tot_output_self = tot_output_self[1:,:]
    return tot_input_self, tot_output_self



def concatenate_data(input_list, output_list, input_list_self, output_list_self):
    """
    Concatenate the data for the (off) diagonal data
    """
    tot_input_list, tot_output_list = np.zeros((1,21,18)), np.zeros((1,6))
    tot_input_self, tot_output_self = np.zeros((1,21,8)), np.zeros((1,6))

    tot_animal = np.zeros((1,1))
    for animal in tqdm(range(80)):
        local_input, local_output = np.array(input_list[animal]), np.array(output_list[animal])
        local_input_self, local_output_self = np.array(input_list_self[animal]), np.array(output_list_self[animal])
        idx_flip = np.where(np.nanmean(local_input[:,:,2],1)<0)[0]
        local_input_flip, local_output_flip = copy.deepcopy(local_input), copy.deepcopy(local_output)
        local_input_self_flip, local_output_self_flip = copy.deepcopy(local_input_self), copy.deepcopy(local_output_self)
        local_input_flip[idx_flip,:,:] = - local_input_flip[idx_flip,:,:]
        local_input_self_flip[idx_flip,:,:] = -local_input_self_flip[idx_flip,:,:]
        col_flip = [0,1,3,4]
        for col in col_flip:
            local_output_flip[idx_flip, col] = - local_output_flip[idx_flip, col]
            local_output_self_flip[idx_flip, col] = - local_output_self_flip[idx_flip, col]
        local_animal_list = animal * np.ones((local_input_flip.shape[0],1))
        tot_input_list = np.concatenate((tot_input_list, local_input_flip),0)
        tot_output_list = np.concatenate((tot_output_list, local_output_flip),0)
        tot_input_self = np.concatenate((tot_input_self, local_input_self_flip),0)
        tot_output_self = np.concatenate((tot_output_self, local_output_self_flip),0)
        tot_animal = np.concatenate((tot_animal,local_animal_list),0)
    
    tot_input_list, tot_input_self = tot_input_list[1:,:,:], tot_input_self[1:,:,:]
    tot_output_list, tot_output_self = tot_output_list[1:,:], tot_output_self[1:,:]
    tot_animal = tot_animal[1:]
    return tot_input_list, tot_input_self, tot_output_list, tot_output_self, tot_animal


def get_velocity_vector(input_list, input_list_self):
    """
    Get the velocity vector for individual inputs
    """
    assert (input_list.shape[0] == input_list_self.shape[0])

    velocity_vector = np.zeros((input_list.shape[0],))
    for line in range(input_list.shape[0]):
        velocity_vector[line] = np.nanmean(input_list[line,:,2])
    return velocity_vector


def load_data_horizon(horizon):
    """
    Load the input data from the current directory for a selected horizon value
    """
    cwd = os.getcwd()
    path_loading = os.path.join(cwd,'Datasets','Mouse','Klibaite','Control')
    with open(os.path.join(path_loading,f'input_2cycles_controls_self_{horizon}.pkl'),'rb') as f1:
        input_self_list = pickle.load(f1)
    with open(os.path.join(path_loading,f'output_2cycles_self_{horizon}.pkl'),'rb') as f2:
        output_self_list = pickle.load(f2)

    return input_self_list, output_self_list


def load_all_data():
    """
    Load the input data from the current directory

    Returns three lists which respectively contains the data for body, self and timing prediction
    """
    cwd = os.getcwd()
    path_loading = os.path.join(cwd,'Datasets','Mouse','Klibaite','Control')
    with open(os.path.join(path_loading,'input_2cycles_control.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'output_2cycles_control.pkl'),'rb') as f2:
        output_list = pickle.load(f2)
    with open(os.path.join(path_loading,'input_2cycles_control_fl.pkl'),'rb') as f3:
        input_list_od = pickle.load(f3)
    with open(os.path.join(path_loading,'output_2cycles_control_fl.pkl'),'rb') as f4:
        output_list_od = pickle.load(f4)
    with open(os.path.join(path_loading,'input_2cycles_control_self.pkl'),'rb') as f5:
        input_list_self = pickle.load(f5)
    with open(os.path.join(path_loading,'output_2cycles_control_self.pkl'),'rb') as f6:
        output_list_self = pickle.load(f6)
    with open(os.path.join(path_loading,'input_2cycles_control_self_fl.pkl'),'rb') as f7:
        input_list_self_od = pickle.load(f7)
    with open(os.path.join(path_loading,'output_2cycles_control_self_fl.pkl'),'rb') as f8:
        output_list_self_od = pickle.load(f8)

    with open(os.path.join(path_loading,'input_2cycles_controls_selftiming.pkl'),'rb') as f_in_timing:
        input_list_selftiming = pickle.load(f_in_timing)
    with open(os.path.join(path_loading,'output_2cycles_controls_selftiming.pkl'),'rb') as f_out_timing:
        output_list_selftiming = pickle.load(f_out_timing)
    with open(os.path.join(path_loading,'input_2cycles_controls_selftiming_od.pkl'),'rb') as f_in_timing_od:
        input_list_selftiming_od = pickle.load(f_in_timing_od)
    with open(os.path.join(path_loading,'output_2cycles_controls_selftiming_od.pkl'),'rb') as f_out_timing_od:
        output_list_selftiming_od = pickle.load(f_out_timing_od)

    list_one = [input_list, output_list, input_list_od, output_list_od]
    list_two = [input_list_self, output_list_self, input_list_self_od, output_list_self_od]
    list_three = [input_list_selftiming, output_list_selftiming, input_list_selftiming_od, output_list_selftiming_od]


    return list_one, list_two, list_three

def multilinear_ols_rsquare_gains(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare, theta_hat

def multilinear_ols_rsquare(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare


def get_rsquare_matrix_feedback(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location (around the feedforward controller)
    """
    n_animal = np.max(tot_animal).astype(int) + 1 
    rsquare_diagonal = np.zeros((n_animal, 21))
    gains_diagonal = np.zeros((n_animal,21,5))
    for animal in tqdm(range(n_animal)):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        local_input = tot_input_list[idx_animal[idx_nan],:,8*bool_hind:8+8*bool_hind]
        local_output = tot_output_list[idx_animal[idx_nan],3*bool_hind+bool_lat]
        # Normalization of the inputs 
        local_input[:,:,5] = local_input[:,:,5] - np.nanmean(local_input[:,:,5],0)
        local_input[:,:,7] = local_input[:,:,7] - np.nanmean(local_input[:,:,7],0)
        tmp_vel = np.nanmean(local_input[:,:,6],1)
        local_input[:,:,6] = local_input[:,:,6] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(21)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,4])
            local_input[line,:,4] = local_input[line,:,4] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs
        if bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(21):
            if bool_lat:
                idx_plot = np.where(local_input[:,time,5]!=0)[0]
            else:
                idx_plot = np.where(local_input[:,time,5]!=0)[0]
            design_mat = np.hstack((np.ones((local_input[idx_plot].shape[0],1)),local_input[idx_plot,time,:]))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,5,6,7,8]]
            else:
                design_mat_y = design_mat[:,[0,5,6,7,8]]
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                a, b = multilinear_ols_rsquare_gains(design_mat_y, local_output[idx_plot])
                pred_output = b @ design_mat_y.T
                gains_diagonal[animal,time,:] = b
                rsquare_diagonal[animal,time] = a 
    
    return rsquare_diagonal, gains_diagonal


def regression_fr_laterality_velidp(tot_animal, tot_input, tot_output, id_leg):
    """Computes the regression coefficient for the mouse data - laterality without speed influence"""
    n_animals = 80
    regression_matrix_small = np.zeros((n_animals,21,3,8,5))
    regression_matrix_large = np.zeros((n_animals,21,3,8,5))
    for animal in tqdm(range(n_animals)):
        local_idx = np.where((tot_animal[:,0]==animal))[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
        local_input = tot_input[local_idx[idx_nan],:,8*id_leg:8*id_leg+8]
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
                if input_var==4:
                    xaxis = np.arange(21)
                    for line in range(input_tmp.shape[0]):
                        subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                        input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                elif input_var==6:
                    tmp_vel = np.nanmean(local_input[:,:,6],1)
                    input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                else:
                    input_tmp = input_tmp - np.nanmean(input_tmp,0)
                for time in range(21):
                    idx_large = np.where(input_tmp[:,time]>0)[0]
                    idx_small = np.where(input_tmp[:,time]<0)[0]
                    if ((len(idx_small)<5) or (len(idx_large)<5)):
                        regression_matrix_large[animal,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        regression_matrix_small[animal,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                        reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                        regression_matrix_large[animal,time,output_var,input_var,:] = np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        regression_matrix_small[animal,time,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large


def load_data_absolute():
    cwd = os.getcwd()
    path_loading = os.path.join(cwd, 'Datasets','Mouse','Klibaite','Control')
    with open(os.path.join(path_loading,'input_2cycles_control_fr_abs.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading, 'output_2cycles_control_fr_abs.pkl'),'rb') as f2:
        output_list = pickle.load(f2)


    return input_list, output_list


def get_rsquare_matrix_self(tot_animal, tot_animal_od, tot_input_list, tot_input_self, tot_input_list_od, tot_input_self_od, tot_output_self, tot_output_self_od, bool_hind, bool_lat):
    """
    Computes the rsquares matrix for the self prediction
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal, rsquare_offdiagonal = np.zeros((n_animal,21)), np.zeros((n_animal,21))
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_self[idx_animal,15,0]))[0]
        tmp_output = tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind] - np.nanmean(tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind])
        for time in range(21):
            tmp_input = tot_input_list[idx_animal[idx_nan],time,5] - np.nanmean(tot_input_list[idx_animal[idx_nan],time,5],0)
            idx_plot = np.where(tmp_input!=0)[0]
            tmp_input_ = tot_input_self[idx_animal[idx_nan][idx_plot],time,4*bool_hind:4+4*bool_hind]
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            design_mat_y = design_mat
            pred_value = tmp_output[idx_plot]
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                a, b = multilinear_ols_rsquare_gains(design_mat_y, pred_value)
                local_pred_output = b @ design_mat_y.T
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, pred_value)

    for animal in range(n_animal):
        idx_animal = np.where(tot_animal_od==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_self_od[idx_animal,15,0]))[0]
        tmp_output = tot_output_self_od[idx_animal[idx_nan],bool_lat+3*bool_hind] - np.nanmean(tot_output_self_od[idx_animal[idx_nan],bool_lat+3*bool_hind])
        for time in range(21):
            tmp_input = tot_input_list_od[idx_animal[idx_nan],time,5] - np.nanmean(tot_input_list_od[idx_animal[idx_nan],time,5],0)
            idx_plot = np.where(tmp_input!=0)[0]
            tmp_input_ = tot_input_self_od[idx_animal[idx_nan][idx_plot],time,4*bool_hind:4+4*bool_hind] - np.nanmean(tot_input_self_od[idx_animal[idx_nan][idx_plot],time,4*bool_hind:4+4*bool_hind],0)
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            design_mat_y = design_mat
            pred_value = tmp_output[idx_plot]
            if design_mat_y.shape[0]<10:
                rsquare_offdiagonal[animal,time] = np.nan 
            else:
                rsquare_offdiagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, pred_value)

    rsquare = rsquare_diagonal #np.concatenate((rsquare_diagonal, rsquare_diagonal),0)
    return rsquare
