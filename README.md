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

Different methods are available to detect locomotion bouts. If your data comes from an animal, it is likely that it contains a decent amount of non walking behaviors that would need to be filtered out. To do this, you can either use existing approaches of behavioral clustering, which allows you to discriminate *locomotion* bouts from other behaviors or use a threshold of the amplitude of the velocity of the body marker.  The non straight line segments can be filtered out by computing for each locomotion bout the body sway from the foreaft body axis if available or from the orientation of the velocity vector.



2. Detecting individual limb contacts




3. Segmenting and normalizing gait cycles

### Data format


---

## Data analysis

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