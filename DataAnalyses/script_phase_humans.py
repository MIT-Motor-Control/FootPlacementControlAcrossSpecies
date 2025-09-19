import os, sys 
import numpy as np 
import pickle
import scipy.io as spio
from scipy.signal import hilbert, savgol_filter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

output_figure_path = os.path.join(os.getcwd(),'humans_results','figures')
os.makedirs(output_figure_path,exist_ok=True)

# Loading the data 
path_to_data = os.path.join(os.getcwd(),'Datasets','Humans','Camargo','processed_time')
with open(os.path.join(path_to_data, 'tot_phase_leg1.pkl'),'rb') as f1:
    phase_list1 = pickle.load(f1)
with open(os.path.join(path_to_data, 'tot_phase_leg2.pkl'),'rb') as f2:
    phase_list2 = pickle.load(f2)
with open(os.path.join(path_to_data, 'tot_modulus_leg1.pkl'),'rb') as f3:
    modulus_list1 = pickle.load(f3)
with open(os.path.join(path_to_data, 'tot_modulus_leg2.pkl'),'rb') as f4:
    modulus_list2 = pickle.load(f4)

cleaned_phase_1 = np.zeros((phase_list1.shape[0],phase_list1.shape[1]))
for line in range(phase_list1.shape[0]):
    local_phase1 = savgol_filter(np.unwrap(phase_list1[line,:,0])-phase_list1[line,0,0],15,1)
    cleaned_phase_1[line,:] = local_phase1



# Building the datasets for PCA
dataset_pca_1 = np.zeros((phase_list1.shape[0]*101,4))
for line in range(phase_list1.shape[0]):
    local_phase1 = savgol_filter(np.unwrap(phase_list1[line,:,0])-phase_list1[line,0,0],15,1)
    local_phase2 = savgol_filter(np.unwrap(phase_list1[line,:,1])-phase_list1[line,0,1],15,1)
    dataset_pca_1[line*101:(line+1)*101,0] = modulus_list1[line,0] * np.sin(local_phase1)
    dataset_pca_1[line*101:(line+1)*101,1] = modulus_list1[line,0] * np.cos(local_phase1)
    dataset_pca_1[line*101:(line+1)*101,2] = modulus_list1[line,1] * np.sin(local_phase2)
    dataset_pca_1[line*101:(line+1)*101,3] = modulus_list1[line,1] * np.cos(local_phase2)

pca_object_final = PCA(n_components=2)
transformed_dataset = pca_object_final.fit_transform(dataset_pca_1)

reconstructed_phase = np.zeros((phase_list1.shape[0],101))
stacked_components = np.expand_dims((pca_object_final.components_[1,:]+1j*pca_object_final.components_[0,:]),-1).T
for line in range(reconstructed_phase.shape[0]):
    local_product = np.angle(stacked_components @ dataset_pca_1[101*line:101*(line+1),:].T)
    reconstructed_phase[line,:] = np.unwrap(local_product[0,:])
    reconstructed_phase[line,:] = reconstructed_phase[line,:] - reconstructed_phase[line,0]
    reconstructed_phase[line,:] = reconstructed_phase[line,:] / (2*np.pi)


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(modulus_list1[:,0], modulus_list1[:,1], color='k',alpha=0.5,s=5)
axs.plot([200,450],[200,450],'r:',lw=2)
axs.set_xlabel('Modulus leg 1')
axs.set_xlabel('Modulus leg 2')
plt.tight_layout()
    
tmp_plot1 = np.unwrap(phase_list1[0,:,0]) - phase_list1[0,0,0]
tmp_plot2 = np.unwrap(phase_list1[0,:,1]) - phase_list1[0,0,1]
tmp_plot1_filtered = savgol_filter(tmp_plot1, 15, 1)
tmp_plot2_filtered = savgol_filter(tmp_plot2, 15, 1)

idx_random_plot = np.random.choice(np.arange(phase_list1.shape[0]), 9, replace=False)


fig, axs = plt.subplots(1,1,figsize=(3,3))
fig.suptitle('Humans',fontweight='bold')
axs.spines[['top','right']].set_visible(False)
idx_line = np.random.choice(reconstructed_phase.shape[0],300,replace=True)
for idx in idx_line:
    idx_col = np.random.choice(101,1)
    axs.scatter(idx_col, reconstructed_phase[idx,idx_col],s=5,color='k')

axs.set_ylabel('Oscillators-based phase')
axs.set_xlabel('Time-based phase')
axs.set_xticks([0,25,50,75,100]), axs.set_xticklabels(['0','0.25','0.5','0.75','1'])
axs.set_yticks([0,0.25,0.50,0.75,1])
axs.set_ylim([-0.05,1.05])
plt.tight_layout()
fig.savefig(os.path.join(output_figure_path,'phase_humans_dots.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figure_path,'phase_humans_dots.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.nanmean(reconstructed_phase,0),'k',lw=2)
axs.plot([0,101],[0,1],'k:',lw=2)
axs.fill_between(np.arange(101), np.nanmean(reconstructed_phase,0)+np.nanstd(reconstructed_phase,0),np.nanmean(reconstructed_phase,0)-np.nanstd(reconstructed_phase,0),color='k',alpha=0.5)
axs.set_ylabel('Oscillators-based phase')
axs.set_xlabel('Time-based phase')
axs.set_xticks([0,25,50,75,100]), axs.set_xticklabels(['0','0.25','0.5','0.75','1'])
axs.set_yticks([0,0.25,0.50,0.75,1])
plt.tight_layout()
fig.savefig(os.path.join(output_figure_path,'phase_humans.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_figure_path,'phase_humans.svg'),bbox_inches='tight')



plt.show()

