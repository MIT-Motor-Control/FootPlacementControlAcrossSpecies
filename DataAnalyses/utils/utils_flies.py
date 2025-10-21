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


def compute_velocity_com(com_pos_x):
    """
    Computes the velocity of the com based on the 4th order centered finite difference scheme

    INPUT
        - com_pos_x : position of the center of mass along the foreaft axis
    OUTPUT
        - output_velocity : speed of the center of mass along the foreaft axis
    """
    dt = 1/150 # Replace that quantity based on your framerate (here 150 Hz)
    output_velocity = np.zeros((com_pos_x.shape[0],))
    output_velocity[2:-2] = (-com_pos_x[4:]+8*com_pos_x[3:-1]-8*com_pos_x[1:-3]+com_pos_x[:-4])/(12*dt)
    return output_velocity


def load_video_data_mm():
    """
    Load the data to identify individual video

    OUTPUT
        - video_matrix_diag : identifies video for front-right based gait cycles
        - video_matrix_odiag : identifies video for front-left based gait cycles
    """
    path_loading = os.path.join(os.getcwd(), 'Datasets','Fly','controllers_io')
    video_matrix_diag = np.load(os.path.join(path_loading,'video_id_diag_mm_fixed.npy'))
    video_matrix_odiag = np.load(os.path.join(path_loading,'video_id_odiag_mm_fixed.npy'))
    return video_matrix_diag, video_matrix_odiag


def load_all_data_horizon(horizon):
    """
    Load the input data for the fly controller inference for a specific horizon
    
    Returns two lists which respectively contain the data for body and self prediction

    INPUTS 
        - horizon is the number of gait cycle for that specific data loading
    OUTPUTS
        - list_one : contains the body based data
        - list_two : contains the foot based data
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
    
def load_all_data_mm():
    """
    Load the input data for the fly controller inference 
    
    Returns two lists which respectively contain the data for body and self prediction

    OUTPUTS
        - list_one : contains the body based data
        - list_two : contains the foot based data
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


def concatenate_data(input_list, output_list, input_list_self, output_list_self):
    """
    Concatenate the data of individual individual bouts
    
    INPUTS
        - input_list : contains the body based input data
        - output_list : contains the body based output data
        - input_self : contains the foot based input data
        - output_self : contains the foot based output data

    OUTPUTS
        Same as inputs but concatenated in a numpy array
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

    INPUTS
        - input_list : contains the body based input data
        - output_list : contains the body based output data
        - input_self : contains the foot based input data
        - output_self : contains the foot based output data

    OUTPUTS
        Same as inputs but all direction are aligned
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




def regression_fr_laterality_velidp(tot_input, tot_output, tot_video, id_leg):
    """
    Computes the lateral regression coefficient for the fly data (ie medial and lateral directions)

    INPUTS
        - tot_input : contains the body based input to foot placement controller
        - tot_output : contains the body based output to foot placement controller
        - tot_video : contains the video identifier
        - id_leg : leg identifier whether you want to analyze the first (0), second (1), or third (2) leg of the group

    OUTPUTS
        - regression_matrix_small : contains the model information for medial direction
        - regression_matrix_large : contains the model information for lateral direction

    """
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



def get_rsquare_matrix_normalization(tot_input, tot_output, tot_video, id_leg, bool_lat):
    """
    Computes the rsquare matrix for the body based model (as deviations from the nominal behavior)

    INPUTS
     - tot_input : contains the body based input to foot placement controller
     - tot_output : contains the body based output to foot placement controller
     - tot_video : contains the video identifier
     - id_leg : leg identifier whether you want to analyze the first (0), second (1), or third (2) leg of the group
     - bool_lat : (0) for foreaft, (1) for lateral

     OUTPUTS
     - rsquare_diagonal : n_animal  x n_timestep matrix containing the individual rsquares 
     - gains_diagonal : n_animal x n_timestep x n_input x n_output contiaining the individual gains of the multilinear regression
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


def get_rsquare_matrix(tot_input, tot_output, tot_video, id_leg, bool_lat):
    """
    Computes the rsquare matrix for the foot based model 

    INPUTS
     - tot_input : contains the foot based input to foot placement controller
     - tot_output : contains the foot based output to foot placement controller
     - tot_video : contains the video identifier
     - id_leg : leg identifier whether you want to analyze the first (0), second (1), or third (2) leg of the group
     - bool_lat : (0) for foreaft, (1) for lateral

     OUTPUTS
     - rsquare_diagonal : n_animal  x n_timestep matrix containing the individual rsquares 
    """
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
                rsquare_diagonal[video-1, time] = multilinear_ols_rsquare(design_mat, pred_value)

    return rsquare_diagonal


def multilinear_ols_rsquare(X,y):
    """
    Computes the rsquare matrix for a given input/output relationship
    
    INPUTS
     - X : input matrix of the predictors 
     - y : output matrix of the predicted variable
     
    OUTPUT
     - rsquare : rsquare matrix for the input/output couple
     """
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare

def multilinear_ols_rsquare_gains(X,y):
    """
    Computes the rsquare matrix for a given input/output relationship
    
    INPUTS
     - X : input matrix of the predictors 
     - y : output matrix of the predicted variable
     
    OUTPUT
     - rsquare : rsquare matrix for the input/output couple
     - theta_hat : coefficients of the multilinear relationship
     """
    theta_hat = np.linalg.inv(X.T@X) @(X.T@y)
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare, theta_hat