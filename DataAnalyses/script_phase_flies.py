import os, sys 
import numpy as np 
import pickle 
import scipy.io as spio
from scipy.signal import hilbert, savgol_filter 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 

output_figures_path = os.path.join(os.getcwd(), 'fly_results','figures')
os.makedirs(output_figures_path, exist_ok=True)
# Loading the data 
path_to_data = os.path.join(os.getcwd(),'Datasets','Fly','controllers_io_abs')
input_fr_matrix = np.load(os.path.join(path_to_data,'input_fr_matrix.npy'))
output_fr_matrix = np.load(os.path.join(path_to_data,'output_fr_matrix.npy'))
phase_fr_matrix = np.load(os.path.join(path_to_data,'phase_fr_matrix_bis.npy'))
modulus_fr_matrix = np.load(os.path.join(path_to_data,'modulus_fr_matrix_bis.npy'))

print(input_fr_matrix.shape, output_fr_matrix.shape, phase_fr_matrix.shape, modulus_fr_matrix.shape)
nans_inputs = np.where(~np.isnan(input_fr_matrix[:,0,0]))[0]
nans_phases = np.where(np.isnan(phase_fr_matrix[:,0,0,0]))[0]
fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.unwrap(phase_fr_matrix[0,:,0,1]))
axs.plot(np.unwrap(phase_fr_matrix[1,:,0,1]))
axs.plot(np.unwrap(phase_fr_matrix[2,:,0,1]))
axs.plot(np.unwrap(phase_fr_matrix[3,:,0,1]))
axs.plot(np.unwrap(phase_fr_matrix[4,:,0,1]))
axs.plot(np.unwrap(phase_fr_matrix[5,:,0,1]))
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmean(np.unwrap(phase_fr_matrix[:,:,0,1]),0))
axs.plot(np.nanmean(np.unwrap(phase_fr_matrix[:,:,1,1]),0))
axs.plot(np.nanmean(np.unwrap(phase_fr_matrix[:,:,2,1]),0))
axs.plot(np.nanmean(np.unwrap(phase_fr_matrix[:,:,3,1]),0))
axs.plot(np.nanmean(np.unwrap(phase_fr_matrix[:,:,4,1]),0))
axs.plot(np.nanmean(np.unwrap(phase_fr_matrix[:,:,5,1]),0))
plt.tight_layout()

dataset_pca_1 = np.zeros((len(nans_inputs)*21, 12))
for i,idx in enumerate(nans_inputs):
    for leg in range(6):
        local_phase = phase_fr_matrix[idx,:,leg,1] #- phase_fr_matrix[idx,0,leg,1]
        dataset_pca_1[i*21:(i+1)*21,2*leg] = np.sin(local_phase) #modulus_fr_matrix[idx,leg,0] * np.sin(local_phase)
        dataset_pca_1[i*21:(i+1)*21,2*leg+1] = np.cos(local_phase) #modulus_fr_matrix[idx,leg,0] * np.cos(local_phase)

dataset_pca_2 = np.zeros((len(nans_inputs)*21, 12))
for i,idx in enumerate(nans_inputs):
    for leg in range(6):
        local_phase = phase_fr_matrix[idx,:,leg,0] #- phase_fr_matrix[idx,0,leg,1]
        dataset_pca_2[i*21:(i+1)*21,2*leg] = np.sin(local_phase) #modulus_fr_matrix[idx,leg,0] * np.sin(local_phase)
        dataset_pca_2[i*21:(i+1)*21,2*leg+1] = np.cos(local_phase) #modulus_fr_matrix[idx,leg,0] * np.cos(local_phase)


dataset_pca_3 = np.zeros((len(nans_inputs)*21, 12))
for i,idx in enumerate(nans_inputs):
    for leg in range(6):
        local_phase = phase_fr_matrix[idx,:,leg,2] #- phase_fr_matrix[idx,0,leg,1]
        dataset_pca_3[i*21:(i+1)*21,2*leg] = np.sin(local_phase) #modulus_fr_matrix[idx,leg,0] * np.sin(local_phase)
        dataset_pca_3[i*21:(i+1)*21,2*leg+1] = np.cos(local_phase) #modulus_fr_matrix[idx,leg,0] * np.cos(local_phase)



pca_object = PCA(n_components=2)
transformed_dataset = pca_object.fit_transform(dataset_pca_1)

pca_object_2 = PCA(n_components=2)
transformed_dataset_2 = pca_object_2.fit_transform(dataset_pca_2)

pca_object_3 = PCA(n_components=2)
transformed_dataset_3 = pca_object_3.fit_transform(dataset_pca_3)

reconstructed_phase = np.zeros((len(nans_inputs),21))
stacked_components = np.expand_dims((pca_object.components_[0,:]+1j*pca_object.components_[1,:]),-1).T
for line in range(reconstructed_phase.shape[0]):
    local_product = np.angle(stacked_components @ dataset_pca_1[21*line:21*(line+1),:].T)
    reconstructed_phase[line,:] = np.unwrap(local_product[0,:])
    reconstructed_phase[line,:] = reconstructed_phase[line,:] - reconstructed_phase[line,0]
    reconstructed_phase[line,:] = reconstructed_phase[line,:] / (2*np.pi)

reconstructed_phase_2 = np.zeros((len(nans_inputs),21))
stacked_components = np.expand_dims((pca_object_2.components_[0,:]+1j*pca_object_2.components_[1,:]),-1).T
for line in range(reconstructed_phase_2.shape[0]):
    local_product = np.angle(stacked_components @ dataset_pca_2[21*line:21*(line+1),:].T)
    reconstructed_phase_2[line,:] = np.unwrap(local_product[0,:])
    reconstructed_phase_2[line,:] = reconstructed_phase_2[line,:] - reconstructed_phase_2[line,0]
    reconstructed_phase_2[line,:] = reconstructed_phase_2[line,:] / (2*np.pi)


reconstructed_phase_3 = np.zeros((len(nans_inputs),21))
stacked_components = np.expand_dims((pca_object_3.components_[0,:]+1j*pca_object_3.components_[1,:]),-1).T
for line in range(reconstructed_phase_3.shape[0]):
    local_product = np.angle(stacked_components @ dataset_pca_3[21*line:21*(line+1),:].T)
    reconstructed_phase_3[line,:] = np.unwrap(local_product[0,:])
    reconstructed_phase_3[line,:] = reconstructed_phase_3[line,:] - reconstructed_phase_3[line,0]
    reconstructed_phase_3[line,:] = reconstructed_phase_3[line,:] / (2*np.pi)

fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Flies',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
idx_line = np.random.choice(reconstructed_phase_2.shape[0],150,replace=True)
for idx in idx_line:
    idx_col = np.random.choice(21,1)
    axs.scatter(idx_col, reconstructed_phase_2[idx,idx_col]/0.85,s=5,color='k',alpha=0.2)
axs.set_xlabel('Time-based phase'), axs.set_ylabel('Oscillators-based phase')
axs.set_xticks([0,5,10,15,20]), axs.set_xticklabels(['0','0.25','0.50','0.75','1'])
axs.set_yticks([0,0.25,0.50,0.75,1])
axs.set_ylim([-0.05,1.05]), axs.set_xlim([-1,21])
plt.tight_layout()
fig.savefig(os.path.join(output_figures_path,'phase_flies_dots.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figures_path,'phase_flies_dots.svg'),bbox_inches='tight')




fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmedian(reconstructed_phase,0)/0.85,'b',lw=2)
axs.plot(np.nanmedian(reconstructed_phase_2,0)/0.85,'r',lw=2)
axs.plot(np.nanmedian(reconstructed_phase_3,0)/0.85,'k',lw=2)
axs.plot([0,20],[0,1],'k:',lw=2)
axs.set_xlabel('Time-based phase'), axs.set_ylabel('Oscillators-based phase')
axs.set_xticks([0,5,10,15,20]), axs.set_xticklabels(['0','0.25','0.50','0.75','1'])
axs.set_yticks([0,0.25,0.50,0.75,1])
plt.tight_layout()
fig.savefig(os.path.join(output_figures_path,'phase_flies.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figures_path,'phase_flies.svg'),bbox_inches='tight')
plt.show()