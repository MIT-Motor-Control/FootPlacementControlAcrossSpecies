import os, sys 
import copy
import numpy as np 
import pandas as pd


def extractStraightLineSegments(angle_vector):
    """
    Extract the straight line locomotion bouts from the Camargo dataset
    """
    # Transform the angles in degrees 
    angle_deg = np.rad2deg(angle_vector)
    angle_deg = np.abs(angle_deg)

    # Identifying the straight line subsections
    bool_angle = ((angle_deg<110) & (angle_deg>70))
    begins, lengths = np.zeros((1,1)), np.zeros((1,1))
    for time in range(1, len(bool_angle)):
        if (bool_angle[time]==True) and (bool_angle[time-1]==False):
            next_zero = np.where(bool_angle[time:]==False)[0][0] if len(np.where(bool_angle[time:]==False)[0])!=0 else np.nan
            begins = np.vstack((begins, time))
            lengths = np.vstack((lengths, next_zero))
    idx_long = np.argmax(lengths)
    tmp_lengths = copy.deepcopy(lengths)
    tmp_lengths[idx_long] = 0
    idx_2ndmax = np.argmax(tmp_lengths)    

    bool_vector = np.zeros((len(bool_angle),1))
    bool_vector[begins[idx_long][0].astype(int):begins[idx_long][0].astype(int)+lengths[idx_long][0].astype(int)] = 1
    bool_vector[begins[idx_2ndmax][0].astype(int):begins[idx_2ndmax][0].astype(int)+lengths[idx_2ndmax][0].astype(int)] = 1
    return bool_vector

def rotate_data(input_markers, average_angle, time_vector):
    """
    Rotate the data to align the movement axis with the first direction of the cartesian space
    """
    tmp_data = input_markers[:,time_vector,:]
    local_data = copy.deepcopy(tmp_data)
    data_output = np.zeros((local_data.shape))
    data_output[0,:,:] =  np.cos(average_angle) * local_data[0,:,:] + np.sin(average_angle) * local_data[2,:,:]
    data_output[2,:,:] = -np.sin(average_angle) * local_data[0,:,:] + np.cos(average_angle) * local_data[2,:,:]
    return data_output

def compute_velocity(input_matrix):
    """
    Computes the velocity of the input matrix using a centered 4th order finite difference method
    """
    dt = 1/200
    tmp_vel_matrix = np.zeros((input_matrix.shape[0], input_matrix.shape[1], 3))
    tmp_vel_matrix[:,2:-2,:] = (-input_matrix[:,4:,:] + 8*input_matrix[:,3:-1,:] - 8 * input_matrix[:,1:-3,:] + input_matrix[:,:-4,:]) / (12*dt)

    output_matrix = np.concatenate((input_matrix, tmp_vel_matrix),0)
    return output_matrix