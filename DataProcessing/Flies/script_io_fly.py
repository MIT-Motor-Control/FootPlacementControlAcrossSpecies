# This script transforms the preprocessed fly dataset into a format that is useable for the foot placement controller inference 

import os, sys
import copy
import numpy as np 
import pickle 
from tqdm import tqdm
import scipy.io as spio
from utils_fly import *
import matplotlib.pyplot as plt

INPUTS_FOLDER =  os.path.join(os.getcwd(),'Datasets','Fly','processed_data')
OUTPUTS_FOLDER = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io')
bool_save_list = True
bool_save_io = True

# Load the input data
foot_data = spio.loadmat(os.path.join(INPUTS_FOLDER, 'contact_data_mm_fixed.mat'))['padded_contacts_data']
raw_data = spio.loadmat(os.path.join(INPUTS_FOLDER, 'raw_pos_mm.mat'))['reconstructed_padded_matrix']
validity = spio.loadmat(os.path.join(INPUTS_FOLDER, 'validity_mm.mat'))['vec_message']
n_good_chunks = len(np.where(validity==4)[0])
n_good_chunks = len(np.where(validity==4)[0])
tot_animal_vector = raw_data[:,2]-1
id_animals = list(set(tot_animal_vector))
idx_nan = np.where(np.isnan(foot_data[:,0,0]))[0]

# We only investigated those locomotion bouts for which the animal was neither turning nor stopping =
list_output_chunks = []

for chunks in tqdm(range(len(idx_nan)-1)):
    id_animal = id_animals[chunks]
    if validity[chunks]!=4:
        continue
    else:
        tmp_mat = extract_metrics_threshold(foot_data, chunks, raw_data)
        list_output_chunks.append(tmp_mat)


with open(os.path.join(OUTPUTS_FOLDER,'list_outputs_animals_threshold_mm.pkl'),'wb') as f1:
    pickle.dump(list_output_chunks, f1)

print(len(list_output_chunks))

# Generate the input/output for the controller inference
video_id_list = get_video_id_chunk(raw_data[:,2], validity)
fly_id_list = get_fly_id_chunk(raw_data[:,1], validity)
raw_data = reshape_raw_data(raw_data)
foot_data = reshape_foot_data(foot_data)


tot_input_fr, tot_output_fr, avg_velocity = get_input_output_fr_vel(raw_data, foot_data, list_output_chunks, validity)
tot_input_fl, tot_output_fl = get_input_output_fl(raw_data, foot_data, list_output_chunks, validity)
tot_input_self_fr, tot_output_self_fr = get_input_output_self_fr(raw_data, foot_data, list_output_chunks, validity)
tot_input_self_fl, tot_output_self_fl = get_input_output_self_fl(raw_data, foot_data, list_output_chunks, validity)

with open(os.path.join(OUTPUTS_FOLDER,'list_avg_velocity_mm_fixed.pkl'),'wb') as f0:
    pickle.dump(avg_velocity, f0)
with open(os.path.join(OUTPUTS_FOLDER,'list_input_fr_mm_fixed.pkl'),'wb') as f1:
    pickle.dump(tot_input_fr, f1)
with open(os.path.join(OUTPUTS_FOLDER,'list_output_fr_mm_fixed.pkl'),'wb') as f2:
    pickle.dump(tot_output_fr, f2)
with open(os.path.join(OUTPUTS_FOLDER,'list_input_fl_mm_fixed.pkl'),'wb') as f3:
    pickle.dump(tot_input_fl, f3)
with open(os.path.join(OUTPUTS_FOLDER,'list_output_fl_mm_fixed.pkl'),'wb') as f4:
    pickle.dump(tot_output_fl, f4)
with open(os.path.join(OUTPUTS_FOLDER,'list_input_self_fr_mm_fixed.pkl'),'wb') as f5:
    pickle.dump(tot_input_self_fr, f5)
with open(os.path.join(OUTPUTS_FOLDER,'list_output_self_fr_mm_fixed.pkl'),'wb') as f6:
    pickle.dump(tot_output_self_fr, f6)
with open(os.path.join(OUTPUTS_FOLDER,'list_input_self_fl_mm_fixed.pkl'),'wb') as f7:
    pickle.dump(tot_input_self_fl, f7)
with open(os.path.join(OUTPUTS_FOLDER,'list_output_self_fl_mm_fixed.pkl'),'wb') as f8:
    pickle.dump(tot_output_self_fl, f8)

with open(os.path.join(OUTPUTS_FOLDER,'list_video_id_mm_fixed.pkl'),'wb') as f9:
    pickle.dump(video_id_list, f9)
with open(os.path.join(OUTPUTS_FOLDER,'list_fly_id_mm_fixed.pkl'),'wb') as f10:
    pickle.dump(fly_id_list, f10)

print(f'Data saved in the {OUTPUTS_FOLDER} folder')

# Also saving these quantities as matrices 
input_fr_matrix, output_fr_matrix = tot_input_fr[0], tot_output_fr[0]
input_fl_matrix, output_fl_matrix = tot_input_fl[0], tot_output_fl[0]
input_self_fr_matrix, output_self_fr_matrix = tot_input_self_fr[0], tot_output_self_fr[0]
input_self_fl_matrix, output_self_fl_matrix = tot_input_self_fl[0], tot_output_self_fl[0]
matrix_video_id = video_id_list[0] * np.ones((tot_input_fr[0].shape[0],1))
matrix_video_id_od = video_id_list[0] * np.ones((tot_input_fl[0].shape[0],1))
matrix_fly_id = fly_id_list[0] * np.ones((tot_input_fr[0].shape[0],1))
matrix_fly_id_od = fly_id_list[0] * np.ones((tot_input_fl[0].shape[0],1))
for ii in range(1, len(tot_input_fr)):
    input_fr_matrix = np.vstack((input_fr_matrix, tot_input_fr[ii]))
    output_fr_matrix = np.vstack((output_fr_matrix, tot_output_fr[ii]))
    input_fl_matrix = np.vstack((input_fl_matrix, tot_input_fl[ii]))
    output_fl_matrix = np.vstack((output_fl_matrix, tot_output_fl[ii]))
    input_self_fr_matrix = np.vstack((input_self_fr_matrix, tot_input_self_fr[ii]))
    output_self_fr_matrix = np.vstack((output_self_fr_matrix, tot_output_self_fr[ii]))
    input_self_fl_matrix = np.vstack((input_self_fl_matrix, tot_input_self_fl[ii]))
    output_self_fl_matrix = np.vstack((output_self_fl_matrix, tot_output_self_fl[ii]))
    matrix_video_id = np.vstack((matrix_video_id, video_id_list[ii] * np.ones((tot_input_fr[ii].shape[0],1))))
    matrix_video_id_od = np.vstack((matrix_video_id_od, video_id_list[ii] * np.ones((tot_input_fl[ii].shape[0],1))))
    matrix_fly_id = np.vstack((matrix_fly_id, fly_id_list[ii] * np.ones((tot_input_fr[ii].shape[0],1))))
    matrix_fly_id_od = np.vstack((matrix_fly_id_od, fly_id_list[ii] * np.ones((tot_input_fl[ii].shape[0],1))))

np.save(os.path.join(OUTPUTS_FOLDER,'video_id_diag_mm_norm_fixed.npy'),matrix_video_id)
np.save(os.path.join(OUTPUTS_FOLDER,'video_id_odiag_mm_norm_fixed.npy'),matrix_video_id_od)


np.save(os.path.join(OUTPUTS_FOLDER,'input_fr_matrix.npy'),input_fr_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'output_fr_matrix.npy'),output_fr_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'input_fl_matrix.npy'),input_fl_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'output_fl_matrix.npy'),output_fl_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'input_self_fr_matrix.npy'),input_self_fr_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'output_self_fr_matrix.npy'),output_self_fr_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'input_self_fl_matrix.npy'),input_self_fl_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'output_self_fl_matrix.npy'),output_self_fl_matrix)
np.save(os.path.join(OUTPUTS_FOLDER,'video_id_diag.npy'),matrix_video_id)
np.save(os.path.join(OUTPUTS_FOLDER,'video_id_odiag.npy'),matrix_video_id_od)
np.save(os.path.join(OUTPUTS_FOLDER,'fly_id_diag.npy'),matrix_fly_id)
np.save(os.path.join(OUTPUTS_FOLDER,'fly_id_odiag.npy'),matrix_fly_id_od)


