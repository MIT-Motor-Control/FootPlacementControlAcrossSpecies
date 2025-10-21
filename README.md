# FootPlacementControlAcrossSpecies
Foot placement control underlies stable locomotion across species, Antoine De Comite and Nidhi Seethapathi, PNAS, 122 (43), 2025 , (https://doi.org/10.1073/pnas.2413958122)

This repository contains the code associated with the above manuscript.

---
## Data processing 

### Data requirements

The foot placement control pipeline used in the manuscript are useable for any legged agent (artificial or biological) independently of the agent's dimension and experimental procedure used for the data collection. The only requirements on the data are the following 

- The horizontal positions (i.e., parallel to the agent's transverse plane) of the extremity of each limb should be availalbe throughoug the trial.  
- The horizontal position of at least one point of the body (not located on one of the legs used above) should be available throughout the trial.
- The kinematic data has been collected at a high enough framerate (ideally a temporal resolution of at least 10 frames per gait cycles)

### Gait segmentation

If your data meet the aformentionned requirements, it can be preprocessed to run the foot placement control analysis. This preprocessing consists of (1) detecting the straight line locomotion bouts, (2) detecting the individual limb contacts, and (3) segmenting and normalizing (both spatially and temporally) each gait cycle.

1. Detecting straight line locomotion bouts.

Different methods are available to detect locomotion bouts. If your data comes from an animal, it is likely that it contains a decent amount of non walking behaviors that would need to be filtered out. To do this, you can either use existing approaches of behavioral clustering, which allows you to discriminate *locomotion* bouts from other behaviors or use a threshold of the amplitude of the velocity of the body marker.  The non straight line segments can be filtered out by computing for each locomotion bout the body sway from the foreaft body axis if available or from the orientation of the velocity vector. Each straight line locomotion bouts subsequently has to be rotated around the vertical axis (i.e., yaw rotation) to align the main movement direction with the first cartesian axis. 



2. Detecting individual limb contacts

The kinematics of the limb and body markers can be used to detect individual limb contacts, assuming that they are not directly available in the data. For each individual contact, one has to identify the time of contact initiation (*heel strike*), the time of contact termination (*toe off*), and the contact location (*foot placement*). The contact initiation and termination of each limb can respectively be approximated by the maxima and minima of the distance between that limb marker and the body markers. The time interval between the contact initiation and termination corresponds to the stance phase associated with that contact. The contact location is defined as the average location of each limb marker during its own stance phase. To store these contact locations and timings, we suggest to create intermediate matrix of dimension $n_{limb} * n_{time} * 2$, where n_time is the number of time samples for that locomotion bout. The entries of these matrices would be the value of the contact location during its stance phase and zero otherwise.

Once all the contacts location and timing have been identified, it is possible to extract the information related to the velocity-dependent foot placement. For this purpose, it is recommended to identify all the pairs of subsequent contacts of any two legs and compute the associated foreaft and lateral distance, time interval between the contacts and average velocity during that time interval (exemplar functions for bipeds, quadrupeds, and hexapeds are available in this repository). These caracteristics can therefore be investigated at a stride (two successive contacts of a single limb) or step level to represent for instance the results of Figure 2. 



3. Segmenting and normalizing gait cycles


This section requires the more fine-tuning to your specific data and species as the gait segmentation assumes a specific stereotypical gait pattern. The analyses presented in the papers were focussed on biphasic gaits (or quasi-biphasic gaits) but our approach is in theory generalizable to any gait patterns.


To gait segmentation can be performed based on the contact timing of the front right limb (in the agent's referenc frame). One gait cycle being defined as the time interval between two successive contacts. The inputs of the foot placement controller is the kinematics of the body marker during that gait cycle. The output of the foot placement controller are the contact location and timing of the legs belonging to the other group of limbs (i.e., the ones not in phase with the front right limb), hence the species- and gait pattern specificity of that step. This repository contains code example for bipedal, diagonal quadrupedal, and hexapedal tripod gaits.

For each gait cycle, the spatial normalization is obtained by subtracting the contact location (during that gait cycle) of the front left limb from the position variables (both inputs and outputs). The temporal normalization, applied only to the input data, is obtained by linearly interpolating the body kinematics on a fixed number of points, which allows to handle gait cycles of different durations.

### Data format

At the end of the data processing procedure, the input and output data of the foot placement analyses should be two matrices. The input data has dimension $n_{sample} * n_{time} * n_{input_var}$ and the output data has dimension $n_{sample} * n_{output_var}$, where $n_{sample}$ is the number of individual gait cycles, $n_{time}$ is the number of points on which the input kinematics was interpolated. The functions `get_rsquares_matrix_*` in the humans, mice, and flies utils should generate the associated $R^2$ matrices. They can also be adapted to output the feedback gains alongside those $R^2$.

---

## Data analysis

The data analysis is less agent-specific than the processing part. The code contained in the `DataAnalysis` section contains the code that were used to generate the results from the paper and can be minimally adapt to be working with your own data. The rsquare matrices spitted out by the different functions contains one raw per subject / animal and one column by interpolated time points. The options of the function call have to be adjusted to investigated specific outputs (see function documentation).

---


## Citation
If you use this code pipeline in your work, please cite the following 

```
@misc{DeComite2025Code,
  author = {De Comite, A. and Seethapathi, N.},
  title = {{Foot placement control underlies stable locomotion across species (Code)}},
  howpublished = {\url{https://github.com/MIT-Motor-Control/FootPlacementControlAcrossSpecies}},
  year = {2025},
  note = {GitHub repository},
  urldate = {2025-09-23},
}
```