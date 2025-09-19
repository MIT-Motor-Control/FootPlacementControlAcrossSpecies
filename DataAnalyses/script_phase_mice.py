import os, sys 
import numpy as np 
import pickle 
import scipy.io as spio
from scipy.signal import hilbert, savgol_filter
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 

output_figure_path = os.path.join(os.getcwd(),'mice_results','figures')
os.makedirs(output_figure_path, exist_ok=True)
# Loading the data 
path_to_data = os.path.join(os.getcwd(),'Datasets','Mouse','Klibaite','Control')
with open(os.path.join(path_to_data,'aaa_phase_2cycles_control_fr_abs.pkl'),'rb') as f1:
    total_phase_fr = pickle.load(f1)
with open(os.path.join(path_to_data,'aaa_input_2cycles_control_fr_abs.pkl'),'rb') as f2:
    total_input_fr = pickle.load(f2)
with open(os.path.join(path_to_data,'aaa_output_2cycles_control_fr_abs.pkl'),'rb') as f3:
    total_output_fr = pickle.load(f3)
with open(os.path.join(path_to_data,'aaa_modulus_2cycles_control_fr_abs.pkl'),'rb') as f4:
    total_modulus_fr = pickle.load(f4)




# Put these data in a matricial form

matrix_input = total_input_fr[0]
matrix_output = total_output_fr[0]
matrix_phase = total_phase_fr[0]
matrix_modulus = total_modulus_fr[0]
for ii in range(1,len(total_input_fr)):
    matrix_input = np.concatenate((matrix_input, total_input_fr[ii]),0)
    matrix_output = np.concatenate((matrix_output, total_output_fr[ii]),0)
    matrix_phase = np.concatenate((matrix_phase, total_phase_fr[ii]),0)
    matrix_modulus = np.concatenate((matrix_modulus, total_modulus_fr[ii]),0)

print(matrix_input.shape, matrix_output.shape, matrix_phase.shape, matrix_modulus.shape)

nan_inputs = np.where(~np.isnan(matrix_input[:,0,0]))[0]

dataset_pca_1 = np.zeros((len(nan_inputs)*21,8))
for i, idx in enumerate(nan_inputs):
    for leg in range(4):
        local_phase = matrix_phase[idx,:,leg,0]
        dataset_pca_1[i*21:(i+1)*21,2*leg] = np.sin(local_phase)
        dataset_pca_1[i*21:(i+1)*21,2*leg+1] = np.cos(local_phase)  

dataset_pca_2 = np.zeros((len(nan_inputs)*21,8))
for i, idx in enumerate(nan_inputs):
    for leg in range(4):
        local_phase = matrix_phase[idx,:,leg,1]
        dataset_pca_2[i*21:(i+1)*21,2*leg] = np.sin(local_phase)
        dataset_pca_2[i*21:(i+1)*21,2*leg+1] = np.cos(local_phase)  

pca_object = PCA(n_components=2)
transformed_dataset = pca_object.fit_transform(dataset_pca_1)

pca_object_2 = PCA(n_components=2)
transformed_dataset_2 = pca_object_2.fit_transform(dataset_pca_2)

reconstructed_phase = np.zeros((len(nan_inputs),21))
stacked_components = np.expand_dims((pca_object.components_[0,:]+1j*pca_object.components_[1,:]),-1).T
for line in range(reconstructed_phase.shape[0]):
    local_product = np.angle(stacked_components @ dataset_pca_1[21*line:21*(line+1),:].T)
    reconstructed_phase[line,:] = np.unwrap(local_product[0,:])
    reconstructed_phase[line,:] = reconstructed_phase[line,:] - reconstructed_phase[line,9]
    reconstructed_phase[line,:] = reconstructed_phase[line,:] / (2*np.pi)

reconstructed_phase_2 = np.zeros((len(nan_inputs),21))
stacked_components_2 = np.expand_dims((pca_object_2.components_[0,:]+1j*pca_object_2.components_[1,:]),-1).T
for line in range(reconstructed_phase_2.shape[0]):
    local_product = np.angle(stacked_components_2 @ dataset_pca_2[21*line:21*(line+1),:].T)
    reconstructed_phase_2[line,:] = np.unwrap(local_product[0,:])
    reconstructed_phase_2[line,:] = reconstructed_phase_2[line,:] - reconstructed_phase_2[line,9]
    reconstructed_phase_2[line,:] = reconstructed_phase_2[line,:] / (2*np.pi)


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Mice',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
idx_line = np.random.choice(reconstructed_phase_2.shape[0],500,replace=True)
for idx in idx_line:
    idx_col = np.random.choice(11,1)
    axs.scatter(idx_col, reconstructed_phase_2[idx,idx_col+9],s=5,color='k',alpha=0.2)
axs.set_ylabel('Oscillators-based phase'), axs.set_xlabel('Time-based phase')
axs.set_xticks([0,11/4,11/2,3*11/4,11])
axs.set_xticklabels([0,0.25,0.5,0.75,1])
axs.set_yticks([0,0.25,0.5,0.75,1])
axs.set_ylim([-0.05,1.05])
plt.tight_layout()
fig.savefig(os.path.join(output_figure_path,'mice_phase_dots.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figure_path,'mice_phase_dots.svg'),bbox_inches='tight')


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmedian(reconstructed_phase[:,9:],0),'r',lw=2)
axs.plot(np.nanmedian(reconstructed_phase_2[:,9:],0),'b',lw=2)
axs.plot([0,11],[0,1],'b:',lw=2), axs.plot([0,11],[0,1],'r:',lw=2)
axs.set_ylabel('Oscillators-based phase'), axs.set_xlabel('Time-based phase')
axs.set_xticks([0,11/4,11/2,3*11/4,11])
axs.set_xticklabels([0,0.25,0.5,0.75,1])
axs.set_yticks([0,0.25,0.5,0.75,1])
plt.tight_layout()
# fig.savefig(os.path.join(output_figure_path,'mice_phase.png'),bbox_inches='tight')
# fig.savefig(os.path.join(output_figure_path,'mice_phase.svg'),bbox_inches='tight')
plt.show()