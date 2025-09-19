import numpy as np
import pickle
import os
from tqdm import tqdm
import copy
import scipy
import matplotlib.pyplot as plt

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image.plt.imread(image)
    except TypeError:
        pass
    im = OffsetImage(image,zoom=zoom)
    x, y = np.atleast_1d(x,y)
    artists = []
    for x0, y0 in zip(x,y):
        ab = AnnotationBbox(im,(x0,y0),xycoords='data',frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x,y]))
    ax.autoscale()
    return artists

def compute_variance_contributions(matrix_rsquares_ffwd, matrix_foreaft_front, matrix_lateral_front, matrix_foreaft_hind, matrix_lateral_hind):
    """
    Computes the individual contributions of feedfroward and feedback to the total variance explained for each individual
    """
    output_matrix = np.zeros((matrix_rsquares_ffwd.shape[0],2,4))
    for animal in range(output_matrix.shape[0]):
        output_matrix[animal,0,0] = matrix_rsquares_ffwd[animal,0]
        output_matrix[animal,0,1] = matrix_rsquares_ffwd[animal,1]
        output_matrix[animal,0,2] = matrix_rsquares_ffwd[animal,2]
        output_matrix[animal,0,3] = matrix_rsquares_ffwd[animal,3]
        output_matrix[animal,1,0] = matrix_foreaft_front[animal,17]*(1-matrix_rsquares_ffwd[animal,0])
        output_matrix[animal,1,1] = matrix_lateral_front[animal,17]*(1-matrix_rsquares_ffwd[animal,1])
        output_matrix[animal,1,2] = matrix_foreaft_hind[animal,17]*(1-matrix_rsquares_ffwd[animal,2])
        output_matrix[animal,1,3] = matrix_lateral_hind[animal,17]*(1-matrix_rsquares_ffwd[animal,3])
    
    return output_matrix

def compute_variance_contributions_time(matrix_rsquares_ffwd, matrix_foreaft_front, matrix_lateral_front, matrix_foreaft_hind, matrix_lateral_hind):
    """
    Computes the individual contributions of feedfroward and feedback to the total variance explained for each individual
    """
    output_matrix = np.zeros((matrix_rsquares_ffwd.shape[0],2,4,21))
    for animal in range(output_matrix.shape[0]):
        for time in range(21):
            output_matrix[animal,0,0,time] = matrix_rsquares_ffwd[animal,0]
            output_matrix[animal,0,1,time] = matrix_rsquares_ffwd[animal,1]
            output_matrix[animal,0,2,time] = matrix_rsquares_ffwd[animal,2]
            output_matrix[animal,0,3,time] = matrix_rsquares_ffwd[animal,3]
            output_matrix[animal,1,0,time] = matrix_foreaft_front[animal,time]*(1-matrix_rsquares_ffwd[animal,0])
            output_matrix[animal,1,1,time] = matrix_lateral_front[animal,time]*(1-matrix_rsquares_ffwd[animal,1])
            output_matrix[animal,1,2,time] = matrix_foreaft_hind[animal,time]*(1-matrix_rsquares_ffwd[animal,2])
            output_matrix[animal,1,3,time] = matrix_lateral_hind[animal,time]*(1-matrix_rsquares_ffwd[animal,3])
    
    return output_matrix

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

def concatenate_data_hind(input_list, output_list):
    """
    Concatenate the data for the hind reference data 
    """
    tot_input_list, tot_output_list = np.zeros((1,21,18)), np.zeros((1,6))

    tot_animal = np.zeros((1,1))
    for animal in tqdm(range(80)):
        local_input, local_output = np.array(input_list[animal]), np.array(output_list[animal])
        idx_flip = np.where(np.nanmean(local_input[:,:,2],1)<0)[0]
        local_input_flip, local_output_flip = copy.deepcopy(local_input), copy.deepcopy(local_output)
        local_input_flip[idx_flip,:,:] = - local_input_flip[idx_flip,:,:]
        col_flip = [0,1,3,4]
        for col in col_flip:
            local_output_flip[idx_flip, col] = - local_output_flip[idx_flip, col]
        local_animal_list = animal * np.ones((local_input_flip.shape[0],1))
        tot_input_list = np.concatenate((tot_input_list, local_input_flip),0)
        tot_output_list = np.concatenate((tot_output_list, local_output_flip),0)
        tot_animal = np.concatenate((tot_animal, local_animal_list),0)

    tot_input_list, tot_output_list = tot_input_list[1:,:,:], tot_output_list[1:,:]
    tot_animal = tot_animal[1:]
    return tot_input_list, tot_output_list, tot_animal


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

def regression_pooled_limbs(tot_animal, tot_animal_od, vel_tier, vel_tier_od,
                            tot_input_list, tot_output_list, tot_input_list_od, tot_output_list_od, matrix_flip,bool_hind=0):
    n_animal = 80
    regression_front_limbs = np.zeros((n_animal, 21, 5, 3, 9, 5))
    for animal in tqdm(range(n_animal)):
        for time in range(21):
            for velocity_tier in range(5):
                ind_list = []
                idx_local_tier = np.where((tot_animal.flatten()==animal) & (vel_tier==velocity_tier+1))[0]
                idx_local_tier_od = np.where((tot_animal_od.flatten()==animal) & (vel_tier_od==velocity_tier+1))[0]
                for output_var in range(bool_hind*3,3+bool_hind*3):
                    for input_var in range(bool_hind*9,9+bool_hind*9):
                        tmp_input_ = tot_input_list[idx_local_tier, time, input_var] - np.nanmean(tot_input_list[idx_local_tier, time, input_var])
                        tmp_input_od = tot_input_list_od[idx_local_tier_od, time, input_var] - np.nanmean(tot_input_list_od[idx_local_tier_od, time, input_var])
                        idx_nan = np.where(~np.isnan(tmp_input_))[0]
                        idx_nan_od = np.where(~np.isnan(tmp_input_od))[0]
                        if (len(idx_nan) + len(idx_nan_od)) < 5 :
                            regression_front_limbs[animal, time, velocity_tier, output_var-3*bool_hind, input_var-9*bool_hind] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            if matrix_flip[output_var-3*bool_hind, input_var-9*bool_hind]==1:
                                tot_tmp_input = np.concatenate((tmp_input_[idx_nan], -tmp_input_od[idx_nan_od]),0)
                            else: 
                                tot_tmp_input = np.concatenate((tmp_input_[idx_nan], tmp_input_od[idx_nan_od]),0)
                            tot_tmp_output = np.concatenate((tot_output_list[idx_local_tier, output_var][idx_nan], tot_output_list_od[idx_local_tier_od, output_var][idx_nan_od]),0) 
                            reg = scipy.stats.linregress(tot_tmp_input, tot_tmp_output)
                            regression_front_limbs[animal, time, velocity_tier, output_var-3*bool_hind, input_var-9*bool_hind] = np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])
    return regression_front_limbs

def get_velocity_vector(input_list, input_list_self):
    """
    Get the velocity vector for individual inputs
    """
    assert (input_list.shape[0] == input_list_self.shape[0])

    velocity_vector = np.zeros((input_list.shape[0],))
    for line in range(input_list.shape[0]):
        velocity_vector[line] = np.nanmean(input_list[line,:,2])
    return velocity_vector

def add_velocity_tier(input_list,input_list_self):
    """
    Generates the velocity tier list for the input data given in input_list, etc
    !!! the first dimension of all the inputs should be similar
    """
    assert (input_list.shape[0] == input_list_self.shape[0])

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

def load_data_hind_ref():
    """
    Load the mouse data with the hind legs as reference
    """
    cwd = os.getcwd()
    path_loading = os.path.join(cwd,'Datasets','Mouse','Klibaite','Control')
    with open(os.path.join(path_loading,'input_2cycles_control_hl.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading,'output_2cycles_control_hl.pkl'),'rb') as f2:
        output_list = pickle.load(f2)

    list_output =[input_list, output_list]
    return list_output

def load_data_absolute():
    cwd = os.getcwd()
    path_loading = os.path.join(cwd, 'Datasets','Mouse','Klibaite','Control')
    with open(os.path.join(path_loading,'input_2cycles_control_fr_abs.pkl'),'rb') as f1:
        input_list = pickle.load(f1)
    with open(os.path.join(path_loading, 'output_2cycles_control_fr_abs.pkl'),'rb') as f2:
        output_list = pickle.load(f2)

    # Loading the files for the self predictions as well

    return input_list, output_list

def load_all_data_time():
    """
    Load the data for the time subfigures
    """
    cwd = os.getcwd()
    path_loading = os.path.join(cwd, 'Datasets','Mouse','Klibaite','Control')
    with open(os.path.join(path_loading,'input_2cycles_control_fr_abs.pkl'),'rb') as f1:
        input_data = pickle.load(f1)
    with open(os.path.join(path_loading,'output_2cycles_control_fr_abs.pkl'),'rb') as f2:
        output_data = pickle.load(f2)
    return input_data, output_data

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

def multilinear_ols(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_hat

def multilinear_ols_pred(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    return yhat

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

def multilinear_ols_rsquare_absolute(X,y,y_mean):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y 
    yhat = X @ theta_hat + y_mean
    ytrue = y + y_mean
    rsquare = 1 - np.sum(np.square(yhat - ytrue)) / np.sum(np.square(ytrue))
    return rsquare

def multilinear_ols_rsquare_atlernative(X,y,X_tot,y_tot):
    """
    Evaluating the model trained on average and tested on varations
    """
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X_tot @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat - y_tot)) / np.sum(np.square(y_tot))
    return rsquare

def add_velocity_tier(input_list,input_list_self):
    """
    Generates the velocity tier list for the input data given in input_list, etc
    !!! the first dimension of all the inputs should be similar
    """
    assert (input_list.shape[0] == input_list_self.shape[0])

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


def get_rsquare_matrix_method2(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind, bool_lat):
    """
    1. Fit the model on the entire input output and observe how well it does
    """
    n_animal = np.max(tot_animal).astype(int) + 1 
    r_square_diagonal, r_square_offdiagonal = np.zeros((n_animal,21)), np.zeros((n_animal,21))
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        tmp_output = tot_output_list[idx_animal[idx_nan], bool_hind*3+bool_lat]
        for time in range(21):
            tmp_input = tot_input_list[idx_animal[idx_nan],time,0+9*bool_hind:9+9*bool_hind]
            design_mat = np.hstack((np.ones((tmp_input.shape[0],1)),tmp_input))
            if design_mat.shape[0]<10:
                r_square_diagonal[animal,time] = np.nan
            else:
                r_square_diagonal[animal,time] = multilinear_ols_rsquare(design_mat, tmp_output)

    for animal in range(n_animal):
        idx_animal = np.where(tot_animal_od==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list_od[idx_animal,15,0]))[0]
        tmp_output = tot_output_list_od[idx_animal[idx_nan], bool_hind*3+bool_lat]
        for time in range(21):
            tmp_input = tot_input_list_od[idx_animal[idx_nan],time,0+9*bool_hind:9+9*bool_hind]
            design_mat = np.hstack((np.ones((tmp_input.shape[0],1)),tmp_input))
            if design_mat.shape[0]<10:
                r_square_offdiagonal[animal,time] = np.nan
            else:
                r_square_offdiagonal[animal,time] = multilinear_ols_rsquare(design_mat, tmp_output)
    rsquare_matrix = np.concatenate((r_square_diagonal, r_square_offdiagonal),0)
    return rsquare_matrix

def get_rsquare_matrix_method1(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location by proceding as follows
    1. Subtract for each individual the average behavior in that velocity bin
    2. Fit the controller inference model
    3. Add the average that was subtracted 
    4. Compute the Rsquare
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal, rsquare_offdiagonal = np.zeros((n_animal,21)),np.zeros((n_animal,21)) 
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        local_output_mean = np.nanmean(tot_output_list[idx_animal[idx_nan],bool_hind*3+bool_lat],0)
        tmp_output = tot_output_list[idx_animal[idx_nan],bool_hind*3+bool_lat] - np.nanmean(tot_output_list[idx_animal[idx_nan],bool_hind*3+bool_lat])
        for time in range(21):
            tmp_input = tot_input_list[idx_animal[idx_nan],time,5] - np.nanmean(tot_input_list[idx_animal[idx_nan],time,5],0)
            if bool_lat:
                idx_plot = np.where(tmp_input>0)[0]
            else:
                idx_plot = np.where(tmp_input!=0)[0]
            tmp_input_ = tot_input_list[idx_animal[idx_nan][idx_plot],time,0+9*bool_hind:9+9*bool_hind] - np.nanmean(tot_input_list[idx_animal[idx_nan][idx_plot],time,9*bool_hind:9+9*bool_hind],0)
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,1,3,5,7]]
            else:
                design_mat_y = design_mat[:,[0,2,4,6,8]]
            pred_value = tmp_output[idx_plot] + local_output_mean
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare_absolute(design_mat_y, pred_value,local_output_mean)

    for animal in range(n_animal):
        idx_animal = np.where(tot_animal_od==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list_od[idx_animal,15,0]))[0]
        local_output_mean = np.nanmean(tot_output_list_od[idx_animal[idx_nan],bool_hind*3+bool_lat],0)
        tmp_output = tot_output_list_od[idx_animal[idx_nan],bool_hind*3+bool_lat] - np.nanmean(tot_output_list_od[idx_animal[idx_nan],bool_hind*3+bool_lat])
        for time in range(21):
            tmp_input = tot_input_list_od[idx_animal[idx_nan],time,5] - np.nanmean(tot_input_list_od[idx_animal[idx_nan],time,5],0)
            if bool_lat:
                idx_plot = np.where(tmp_input>0)[0]
            else:
                idx_plot = np.where(tmp_input!=0)[0]
            tmp_input_ = tot_input_list_od[idx_animal[idx_nan][idx_plot],time,0+9*bool_hind:9+9*bool_hind] - np.nanmean(tot_input_list_od[idx_animal[idx_nan][idx_plot],time,9*bool_hind:9+9*bool_hind],0)
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,1,3,5,7]]
            else:
                design_mat_y = design_mat[:,[0,2,4,6,8]]
            pred_value = tmp_output[idx_plot]
            if design_mat_y.shape[0]<10:
                rsquare_offdiagonal[animal,time] = np.nan 
            else:
                rsquare_offdiagonal[animal,time] = multilinear_ols_rsquare_absolute(design_mat_y, pred_value, local_output_mean)

    rsquare_matrix = np.concatenate((rsquare_diagonal, rsquare_offdiagonal),0)
    return rsquare_matrix

def normalize_output_feedback(tot_input_list, tot_output_list, idx_animal, idx_nan, bool_hind, bool_lat):
    """
    Normalize the local output around the feedforward controller 
    """
    tmp_output_raw = tot_output_list[idx_animal[idx_nan], bool_hind*3+bool_lat]
    if bool_lat:
        tmp_output_norm = tmp_output_raw - np.nanmean(tmp_output_raw, 0)
    else:
        # Fit a linear model between the inputs and outputs 
        tmp_input_lin_nose = np.nanmean(tot_input_list[idx_animal[idx_nan],:,6+bool_hind*9],1)
        tmp_output_lin = tmp_output_raw
        lin_nose = scipy.stats.linregress(tmp_input_lin_nose, tmp_output_lin)
        tmp_output_norm = tmp_output_raw - (tmp_input_lin_nose*lin_nose.slope + lin_nose.intercept)

    return tmp_output_norm

def normalize_input_feedback(tot_input_list, idx_animal, idx_nan, idx_plot, time, bool_hind, bool_lat):
    """
    Normalize the local output around the feedforward controller
    """
    tmp_input_raw = tot_input_list[idx_animal[idx_nan][idx_plot], :, 0+9*bool_hind:9+9*bool_hind]
    if False:
        tmp_input_norm = tmp_input_raw - np.nanmean(tmp_input_raw, 0)
    else:
        speed_vector = np.nanmean(tmp_input_raw[:,:,6],1)
        tmp_input_norm = tmp_input_raw - np.nanmean(tmp_input_raw,0)
        # update the foraft velocity with velocity tracking 
        tmp_input_norm[:,:,2] = tmp_input_raw[:,:,2] - np.repeat(np.expand_dims(speed_vector,-1),21,1)
        tmp_input_norm[:,:,6] = tmp_input_raw[:,:,6] - np.repeat(np.expand_dims(speed_vector,-1),21,1)
        # update the foreaft position with velocity tracking
        for line in range(tmp_input_norm.shape[0]):
            local_lin_reg = scipy.stats.linregress(tmp_input_raw[line,:,4], np.arange(21))
            x_axis = np.arange(21)
            tmp_input_norm[line,:,4] = tmp_input_raw[line,:,4] - (x_axis * local_lin_reg.slope + local_lin_reg.intercept)
            tmp_input_norm[line,:,0] = tmp_input_raw[line,:,0] - (x_axis * local_lin_reg.slope + tmp_input_raw[line,0,0])
    
    return tmp_input_norm[:,time,:]

def normalize_input_feedback_self(tot_input_self, idx_animal, idx_nan, idx_plot, time, bool_hind, bool_lat):
    """
    Normalize the local output around the feedforward controller
    """
    tmp_input_raw = tot_input_self[idx_animal[idx_nan][idx_plot], :, 0+4*bool_hind:4+4*bool_hind]
    if False:
        tmp_input_norm = tmp_input_raw - np.nanmean(tmp_input_raw, 0)
    else:
        tmp_input_norm = np.zeros((tmp_input_raw.shape[0],tmp_input_raw.shape[1],4))
        local_mean_f = np.nanmean(tmp_input_raw[:,:,0],0)
        local_mean_vf = np.nanmean(tmp_input_raw[:,:,2],0)
        dtot = local_mean_f[-1] - np.nanmean(local_mean_f[4:8])
        dind = local_mean_vf[-1] - np.nanmean(tmp_input_raw[:,4:8,0],1)
        scaling_factor = dind / dtot
        for line in range(tmp_input_raw.shape[0]):
            local_norm = scaling_factor[line] * (local_mean_f - np.nanmean(local_mean_f[4:8])) + np.nanmean(tmp_input_raw[line,4:8,0])
            local_norm_v = scaling_factor[line] * (local_mean_vf - np.nanmean(local_mean_vf[4:8])) + np.nanmean(tmp_input_raw[line,4:8,2])
            tmp_input_norm[line,:,0] = tmp_input_raw[line,:,0] - local_norm
            tmp_input_norm[line,:,2] = tmp_input_raw[line,:,2] - local_norm_v
        tmp_input_norm[:,:,1] = tmp_input_raw[:,:,1] - np.nanmean(tmp_input_raw[:,:,1],0)
        tmp_input_norm[:,:,3] = tmp_input_raw[:,:,3] - np.nanmean(tmp_input_raw[:,:,3],0)
    return tmp_input_norm[:,time,:]


def get_rsquare_matrix_feedback_bis(tot_animal, tot_input_list, tot_output_list, direction, leg):
    """
    Computes the rsquare matrix for the deviations after having removed the feedforward component of the model
    """
    n_animal = np.max(tot_animal).astype(int)+1
    rsquare_diagonal = np.zeros((n_animal, 21))
    for animal in tqdm(range(n_animal)):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        local_input = tot_input_list[idx_animal[idx_nan],:,:]
        local_output = tot_output_list[idx_animal[idx_nan],3*leg+direction]
        # Normalization of the inputs (most important is the foreaft direction)
        tmp_vel = np.nanmean(local_input[:,:,6],1)
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        local_input[:,:,5] = local_input[:,:,5] - np.nanmean(local_input[:,:,5],0)
        local_input[:,:,7] = local_input[:,:,7] - np.nanmean(local_input[:,:,7],0)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        local_input[:,:,6] = local_input[:,:,6] - np.expand_dims(tmp_vel,-1)
        xinput = np.arange(21)
        for line in range(local_input.shape[0]):
            local_input[line,:,0] = local_input[line,:,0] - (local_input[line,-1,0] + xinput*tmp_vel[line] - xinput[-1]*tmp_vel[line])
        # Normalization of the outputs
        if direction==0:
            subjectlin = scipy.stats.linregress(tmp_vel,local_output)
            local_output = local_output - (tmp_vel*subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(local_input.shape[1]):
            design_matrix = np.hstack((np.ones((local_input.shape[0],1)),np.squeeze(local_input[:,time,:])))
            rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_matrix, local_output)
    
    return rsquare_diagonal

def compute_ffwd_inputs(tot_input, tot_animals):
    """
    Computes the body fixed marker information for the feedforward controller
    """
    n_animals = 80
    mat_slope, mat_pvalue = np.zeros((n_animals,2)), np.zeros((n_animals,2))
    for animal in range(n_animals):
        local_input = tot_input[tot_animals[:,0]==animal,:,:]
        local_velocity = np.nanmean(local_input[:,:,6],1)
        local_x_amplitude = np.zeros((local_velocity.shape[0],))
        local_y_amplitude = np.zeros((local_velocity.shape[0],))
        for line in range(local_x_amplitude.shape[0]):
            local_x_amplitude[line] = local_input[line,-1,4] - local_input[line,0,4]
            local_y_amplitude[line] = np.max(local_input[line,:,5]) - np.min(local_input[line,:,5])

        # Performing regressions between these factors ...
        idx_nans = np.where(~np.isnan(local_velocity))[0]
        reg_x_amplitude = scipy.stats.linregress(local_velocity[idx_nans], local_x_amplitude[idx_nans])
        reg_y_amplitude = scipy.stats.linregress(local_velocity[idx_nans], local_y_amplitude[idx_nans])
        mat_slope[animal,0], mat_slope[animal,1] = reg_x_amplitude.slope, reg_y_amplitude.slope
        mat_pvalue[animal,0], mat_pvalue[animal,1] = reg_x_amplitude.pvalue, reg_y_amplitude.pvalue

    return mat_slope, mat_pvalue
   
def compute_regression_fr(tot_animal, tot_input, tot_output, velocity_tier, id_leg):
    """
    Computes the parameters of the linear regression
    """
    n_animals = 80
    regression_matrix = np.zeros((n_animals,21,5,3,8,5))
    for animal in tqdm(range(n_animals)):
        for vel_tier in range(5):
            local_idx = np.where((tot_animal[:,0]==animal) & (velocity_tier==vel_tier+1))[0]
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
                        if (input_tmp.shape[0]<5):
                            regression_matrix[animal,time,vel_tier,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            reg = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                            regression_matrix[animal,time,vel_tier,output_var,input_var,:] = np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])
                        
    return regression_matrix


def compute_regression_fr_norm(tot_animal, tot_input, tot_output, velocity_tier, id_leg):
    """
    Computes the parameters of the linear regression
    """
    n_animals = 80
    regression_matrix = np.zeros((n_animals,21,5,3,8,5))
    for animal in tqdm(range(n_animals)):
        for vel_tier in range(5):
            local_idx = np.where((tot_animal[:,0]==animal) & (velocity_tier==vel_tier+1))[0]
            idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
            local_input = tot_input[local_idx[idx_nan],:,8*id_leg:8*id_leg+8]
            local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
            for output_var in range(local_output.shape[-1]):
                for input_var in range(local_input.shape[-1]):
                    # Normalize the outputs
                    tmp_len = local_output[:,0]
                    input_tmp = local_input[:,:,input_var]
                    output_tmp = local_output[:,0]
                    if output_var==0:
                        tmp_vel = np.nanmean(local_input[:,:,2],1)
                        subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                        output_tmp = output_tmp - (tmp_vel*subjectlin.slope + subjectlin.intercept)
                    else:
                        output_tmp = output_tmp - np.nanmean(output_tmp,0)
                    # Normalize the inputs 
                    tmp_vel = np.nanmean(local_input[:,:,6],1)
                    if input_var==4:
                        xaxis = np.arange(21)
                        for line in range(input_tmp.shape[0]):
                            subjectlin = scipy.stats.linregress(xaxis, input_tmp[line,:])
                            input_tmp[line,:] = input_tmp[line,:] - (xaxis*subjectlin.slope + subjectlin.intercept)
                    elif input_var==6:
                        input_tmp = input_tmp - np.expand_dims(tmp_vel,-1)
                    else:
                        input_tmp = input_tmp - np.nanmean(input_tmp,0)
                
                    if input_var == 7:
                        for line in range(input_tmp.shape[0]):
                            input_tmp[line,:] = input_tmp[line,:] / tmp_len[line]
                    if output_var == 1:
                        for line in range(output_tmp.shape[0]):
                            output_tmp[line] = output_tmp[line] / tmp_len[line]
                    for time in range(21):
                        if (input_tmp.shape[0]<5):
                            regression_matrix[animal,time,vel_tier,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            reg = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                            regression_matrix[animal,time,vel_tier,output_var,input_var,:] = np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])
                        
    return regression_matrix

def compute_feedforward_rsquare(tot_input, tot_output, tot_animal):
    """
    Computes the rsquares of the linear fit of the velocity dependent controller
    """
    n_animals = 80
    matrix_rsquares = np.zeros((n_animals,4))
    for subject in range(n_animals):
        idx_local = np.where(tot_animal[:,0]==subject)[0]
        idx_nans = np.where(~np.isnan(tot_input[idx_local,0,0]))[0]
        for leg in range(2):
            input_vel = np.nanmean(tot_input[idx_local[idx_nans],:,6],1)
            for output_var in range(2):
                local_output = tot_output[idx_local[idx_nans],output_var+3*leg]
                local_reg = scipy.stats.linregress(input_vel, local_output)
                matrix_rsquares[subject,output_var+2*leg] = local_reg.rvalue**2
    
    return matrix_rsquares


def get_rsquare_matrix_feedback_time(tot_animal, tot_input_list, difference_matrix, bool_hind):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact timing 
    """
    n_animal = np.max(tot_animal).astype(int) + 1 
    rsquare_diagonal = np.zeros((n_animal, 21))
    for animal in tqdm(range(n_animal)):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        local_input = tot_input_list[idx_animal[idx_nan],:,8*bool_hind:8+8*bool_hind]
        local_output = difference_matrix[idx_animal[idx_nan],3*bool_hind+2] - np.nanmean(difference_matrix[idx_animal[idx_nan],3*bool_hind+2])
        # Normalization of the inputs 
        local_input[:,:,5] = local_input[:,:,5] - np.nanmean(local_input[:,:,5],0)
        local_input[:,:,7] = local_input[:,:,7] - np.nanmean(local_input[:,:,7],0)
        tmp_vel = np.nanmean(local_input[:,:,6],1)
        local_input[:,:,6] = local_input[:,:,6] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(21)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,4])
            local_input[line,:,4] = local_input[line,:,4] - (xinput*subjectlin.slope + subjectlin.intercept)
        
        for time in range(21):
            idx_plot = np.where(local_input[:,time,5]!=0)[0]
            design_mat = np.hstack((np.ones((local_input[idx_plot].shape[0],1)),local_input[idx_plot,time,:]))
            if design_mat.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                a, b = multilinear_ols_rsquare_gains(design_mat, local_output[idx_plot])
                pred_output = b @ design_mat.T
                # if time == 115:
                #     fig, axs = plt.subplots(1,1,figsize=(3,3))
                #     axs.spines[['top','right']].set_visible(False)
                #     axs.scatter(local_output[idx_plot], pred_output, color='b', alpha=0.5, s=20)
                #     axs.plot([-25,25],[-25,25],'k:',lw=2)
                #     axs.set_xlabel('True deviations')
                #     axs.set_ylabel('Predicted deviations')
                #     axs.set_xlim([-25,25])
                #     axs.set_ylim([-25,25])
                #     plt.tight_layout()
                #     fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatterplot_predictions.png'),bbox_inches='tight')
                #     fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatterplot_predictions.svg'),bbox_inches='tight')
                #     plt.show()
                rsquare_diagonal[animal,time] = a #multilinear_ols_rsquare(design_mat_y, local_output[idx_plot])
    
    return rsquare_diagonal

def get_rsquare_matrix_feedback_verpeut(tot_animal, tot_input_list, tot_output_list, bool_hind, bool_lat):
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
                idx_plot = np.where(local_input[:,time,5]>0)[0]
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
                if time == 115:
                    fig, axs = plt.subplots(1,1,figsize=(3,3))
                    axs.spines[['top','right']].set_visible(False)
                    axs.scatter(local_output[idx_plot], pred_output, color='b', alpha=0.5, s=20)
                    axs.plot([-25,25],[-25,25],'k:',lw=2)
                    axs.set_xlabel('True deviations')
                    axs.set_ylabel('Predicted deviations')
                    axs.set_xlim([-25,25])
                    axs.set_ylim([-25,25])
                    plt.tight_layout()
                    fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatterplot_predictions.png'),bbox_inches='tight')
                    fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatterplot_predictions.svg'),bbox_inches='tight')
                    plt.show()
                gains_diagonal[animal,time,:] = b
                rsquare_diagonal[animal,time] = a #multilinear_ols_rsquare(design_mat_y, local_output[idx_plot])
    
    return rsquare_diagonal, gains_diagonal

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
                idx_plot = np.where(local_input[:,time,5]>0)[0]
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
                if time == 115:
                    fig, axs = plt.subplots(1,1,figsize=(3,3))
                    axs.spines[['top','right']].set_visible(False)
                    axs.scatter(local_output[idx_plot], pred_output, color='b', alpha=0.5, s=20)
                    axs.plot([-25,25],[-25,25],'k:',lw=2)
                    axs.set_xlabel('True deviations')
                    axs.set_ylabel('Predicted deviations')
                    axs.set_xlim([-25,25])
                    axs.set_ylim([-25,25])
                    plt.tight_layout()
                    fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatterplot_predictions.png'),bbox_inches='tight')
                    fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatterplot_predictions.svg'),bbox_inches='tight')
                    plt.show()
                gains_diagonal[animal,time,:] = b
                rsquare_diagonal[animal,time] = a #multilinear_ols_rsquare(design_mat_y, local_output[idx_plot])
    
    return rsquare_diagonal, gains_diagonal

def regression_fr_laterality_velidp_multilinear(tot_animal, tot_input, tot_output, id_leg):
    """Computes the regression coefficient for the mouse data - laterality without speed influence"""
    n_animals = 80
    regression_matrix_small = np.zeros((n_animals,21,3,8,2))
    regression_matrix_large = np.zeros((n_animals,21,3,8,2))
    for animal in tqdm(range(n_animals)):
        local_idx = np.where((tot_animal[:,0]==animal))[0]
        idx_nan = np.where(~np.isnan(tot_input[local_idx,0,0]))[0]
        local_input = tot_input[local_idx[idx_nan],:,8*id_leg:8*id_leg+8]
        local_output = tot_output[local_idx[idx_nan],3*id_leg:3*id_leg+3]
        for output_var in range(local_output.shape[-1]):
            # Normalize the outputs 
            output_tmp = local_output[:,output_var]
            input_tmp = copy.deepcopy(local_input[:,:,4:])
            if output_var==0:
                tmp_vel = np.nanmean(local_input[:,:,2],1)
                subjectlin = scipy.stats.linregress(tmp_vel, output_tmp)
                output_tmp = output_tmp - (tmp_vel*subjectlin.slope+subjectlin.intercept)
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
                    regression_matrix_large[animal,time,output_var,:,:] = np.nan
                    regression_matrix_small[animal,time,output_var,:,:] = np.nan
                else:
                    design_mat_large = np.hstack((np.ones((input_tmp[idx_large,time].shape[0],1)),input_tmp[idx_large,time]))
                    design_mat_small = np.hstack((np.ones((input_tmp[idx_small,time].shape[0],1)),input_tmp[idx_small,time]))
                    rsquare_large, gains_large = multilinear_ols_rsquare_gains(design_mat_large,output_tmp[idx_large])
                    rsquare_small, gains_small = multilinear_ols_rsquare_gains(design_mat_small, output_tmp[idx_small])
                    #reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                    #reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                    for ii in range(4):
                        regression_matrix_large[animal,time,output_var,ii,:] = np.array([rsquare_large,gains_large[ii+1]])#np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                        regression_matrix_small[animal,time,output_var,ii,:] = np.array([rsquare_small,gains_small[ii+1]])#np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large




def regression_fr_baseline_laterality(tot_animal, tot_input, tot_output, id_leg):
    """
    Regression baseline for the laterality
    """
    n_animals = 80
    regression_matrix = np.zeros((n_animals,21,3,8,5))
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
                    if (input_tmp.shape[0]<5):
                        regression_matrix[animal,time,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    else:
                        reg_local = scipy.stats.linregress(input_tmp[:,time], output_tmp)
                        regression_matrix[animal,time,output_var,input_var,:] = np.array([reg_local.slope, reg_local.pvalue, reg_local.rvalue, reg_local.intercept, reg_local.intercept_stderr])
                        
    return regression_matrix

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

def regression_fr_laterality(tot_animal, tot_input, tot_output, velocity_tier, id_leg):
    """Computes the regression coefficient for the mouse data"""
    n_animals = 80
    regression_matrix_small = np.zeros((n_animals,21,5,3,8,5))
    regression_matrix_large = np.zeros((n_animals,21,5,3,8,5))
    for animal in tqdm(range(n_animals)):
        for vel_tier in range(5):
            local_idx = np.where((tot_animal[:,0]==animal) & (velocity_tier==vel_tier+1))[0]
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
                            regression_matrix_large[animal,time,vel_tier,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                            regression_matrix_small[animal,time,vel_tier,output_var,input_var,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            reg_large = scipy.stats.linregress(input_tmp[idx_large,time], output_tmp[idx_large])
                            reg_small = scipy.stats.linregress(input_tmp[idx_small,time], output_tmp[idx_small])
                            regression_matrix_large[animal,time,vel_tier,output_var,input_var,:] = np.array([reg_large.slope, reg_large.pvalue, reg_large.rvalue, reg_large.intercept, reg_large.intercept_stderr])
                            regression_matrix_small[animal,time,vel_tier,output_var,input_var,:] = np.array([reg_small.slope, reg_small.pvalue, reg_small.rvalue, reg_small.intercept, reg_small.intercept_stderr])
                        
    return regression_matrix_small, regression_matrix_large



def get_rsquare_matrix(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal, rsquare_offdiagonal = np.zeros((n_animal,21)),np.zeros((n_animal,21)) 
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        tmp_output = tot_output_list[idx_animal[idx_nan],bool_hind*3+bool_lat] - np.nanmean(tot_output_list[idx_animal[idx_nan],bool_hind*3+bool_lat])
        for time in range(21):
            tmp_input = tot_input_list[idx_animal[idx_nan],time,5] - np.nanmean(tot_input_list[idx_animal[idx_nan],time,5],0)
            if bool_lat:
                idx_plot = np.where(tmp_input>0)[0]
            else:
                idx_plot = np.where(tmp_input!=0)[0]
            tmp_input_ = tot_input_list[idx_animal[idx_nan][idx_plot],time,0+9*bool_hind:9+9*bool_hind] - np.nanmean(tot_input_list[idx_animal[idx_nan][idx_plot],time,9*bool_hind:9+9*bool_hind],0)
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,1,3,5,7]]
            else:
                design_mat_y = design_mat[:,[0,2,4,6,8]]
            pred_value = tmp_output[idx_plot]
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, pred_value)

    for animal in range(n_animal):
        idx_animal = np.where(tot_animal_od==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list_od[idx_animal,15,0]))[0]
        tmp_output = tot_output_list_od[idx_animal[idx_nan],bool_hind*3+bool_lat] - np.nanmean(tot_output_list_od[idx_animal[idx_nan],bool_hind*3+bool_lat])
        for time in range(21):
            tmp_input = tot_input_list_od[idx_animal[idx_nan],time,5] - np.nanmean(tot_input_list_od[idx_animal[idx_nan],time,5],0)
            if bool_lat:
                idx_plot = np.where(tmp_input>0)[0]
            else:
                idx_plot = np.where(tmp_input!=0)[0]
            tmp_input_ = tot_input_list_od[idx_animal[idx_nan][idx_plot],time,0+9*bool_hind:9+9*bool_hind] - np.nanmean(tot_input_list_od[idx_animal[idx_nan][idx_plot],time,9*bool_hind:9+9*bool_hind],0)
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,1,3,5,7]]
            else:
                design_mat_y = design_mat[:,[0,2,4,6,8]]
            pred_value = tmp_output[idx_plot]
            if design_mat_y.shape[0]<10:
                rsquare_offdiagonal[animal,time] = np.nan 
            else:
                rsquare_offdiagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, pred_value)

    rsquare_matrix = np.concatenate((rsquare_diagonal, rsquare_offdiagonal),0)
    return rsquare_matrix


def get_rsquare_matrix_self_time(tot_animal, tot_input_list, tot_input_self, difference_matrix, bool_hind):
    """
    Computes the rsquares matrix for the self prediction for the contact timing
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal, rsquare_offdiagonal = np.zeros((n_animal,21)), np.zeros((n_animal,21))
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_self[idx_animal,15,0]))[0]
        tmp_output = difference_matrix[idx_animal[idx_nan],2+3*bool_hind] - np.nanmean(difference_matrix[idx_animal[idx_nan],2+3*bool_hind])
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
                # local_pred_output = b @ design_mat_y.T
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, pred_value)

    return rsquare_diagonal


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
                if time == 115:
                    fig, axs = plt.subplots(1,1,figsize=(3,3))
                    axs.spines[['top','right']].set_visible(False)
                    axs.scatter(pred_value, local_pred_output,color='r',s=20,alpha=0.5)
                    axs.set_xlim([-25,25]), axs.set_ylim([-25,25])
                    axs.plot([-25,25],[-25,25],color='k',lw=2,ls=':')
                    axs.set_xlabel('True deviation'), axs.set_ylabel('Predicted deviation')
                    plt.tight_layout()
                    fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatter_baseline.png'),bbox_inches='tight')
                    fig.savefig(os.path.join(os.getcwd(),'mice_results','figures','scatter_baseline.svg'),bbox_inches='tight')
                    plt.show()
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

def get_matrix_laterality(tot_animal, tot_animal_od, tot_input_list, tot_input_list_od, tot_output_list, tot_output_list_od, vel_tier, vel_tier_od, bool_hind, bool_in):
    """
    Computes the direction-dependent matrices
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    regression_matrix_left, regression_matrix_right = np.zeros((n_animal,21,5,3,9,5)), np.zeros((n_animal,21,5,3,9,5))
    for animal in tqdm(range(n_animal)):
        for time in range(21):
            for velocity_tier in range(5):
                idx_local_tier = np.where((tot_animal.flatten()==animal) & (vel_tier==velocity_tier+1))[0]
                for output_var in range(3):
                    tmp_input = tot_input_list[idx_local_tier, time,5+5*bool_hind] - np.nanmean(tot_input_list[idx_local_tier,time,5+5*bool_hind])
                    if bool_in:
                        idx_plot = np.where(tmp_input<0)[0]
                    else:
                        idx_plot = np.where(tmp_input>0)[0]
                    for input_var in range(9):
                        tmp_input_ = tot_input_list[idx_local_tier[idx_plot],time,input_var+9*bool_hind] - np.nanmean(tot_input_list[idx_local_tier[idx_plot],time,input_var+9*bool_hind])
                        idx_nan = np.where(~np.isnan(tmp_input_))[0]
                        if len(idx_nan)==0:
                            regression_matrix_left[animal,time,velocity_tier,output_var,input_var] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            reg = scipy.stats.linregress(tmp_input_[idx_nan], tot_output_list[idx_local_tier[idx_plot],output_var+3*bool_hind][idx_nan])
                            regression_matrix_left[animal,time,velocity_tier,output_var,input_var] = np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])

    for animal in tqdm(range(n_animal)):
        for time in range(21):
            for velocity_tier in range(5):
                idx_local_tier = np.where((tot_animal_od.flatten()==animal) & (vel_tier_od==velocity_tier+1))[0]
                for output_var in range(3):
                    tmp_input = tot_input_list_od[idx_local_tier, time,5+5*bool_hind] - np.nanmean(tot_input_list_od[idx_local_tier,time,5+5*bool_hind])
                    if not bool_in:
                        idx_plot = np.where(tmp_input<0)[0]
                    else:
                        idx_plot = np.where(tmp_input>0)[0]
                    for input_var in range(9):
                        tmp_input_ = tot_input_list_od[idx_local_tier[idx_plot],time,input_var] - np.nanmean(tot_input_list_od[idx_local_tier[idx_plot],time,input_var+9*bool_hind])
                        idx_nan = np.where(~np.isnan(tmp_input_))[0]
                        if len(idx_nan)==0:
                            regression_matrix_right[animal,time,velocity_tier,output_var,input_var] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                        else:
                            reg = scipy.stats.linregress(tmp_input_[idx_nan], tot_output_list_od[idx_local_tier[idx_plot],output_var+3*bool_hind][idx_nan])
                            regression_matrix_right[animal,time,velocity_tier,output_var,input_var] = np.array([reg.slope, reg.pvalue, reg.rvalue, reg.intercept, reg.intercept_stderr])

    regression_matrix = np.concatenate((regression_matrix_left, regression_matrix_right),0)
    return regression_matrix

def model_alternative_hypothesis(tot_animal, vel_tier, tot_input_list, tot_output_list):
    """
    Designed a regression model based on the average behavior
    version 1.0 - March 17th
    """
    n_animal = 80
    avg_inputs_matrix = np.zeros((n_animal, 5, 21, 9))
    avg_outputs_matrix = np.zeros((n_animal, 5, 3))
    for animal in range(n_animal):
        for velocity_tier in range(5):
            idx_local_tier = np.where((tot_animal.flatten()==animal) & (vel_tier==velocity_tier+1))[0]
            for output_var in range(3):
                avg_outputs_matrix[animal, velocity_tier,output_var] = np.nanmean(tot_output_list[idx_local_tier,output_var])
            for time in range(21):
                for input_var in range(9):
                    avg_inputs_matrix[animal, velocity_tier, time, input_var] = np.nanmean(tot_input_list[idx_local_tier,time,input_var])
    
    return avg_inputs_matrix, avg_outputs_matrix