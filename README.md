# FootPlacementControlAcrossSpecies
Foot placement control underlies stable locomotion across species, Antoine De Comite and Nidhi Seethapathi, PNAS, 122 (43), 2025 , (https://doi.org/10.1073/pnas.2413958122)

This repository contains the code associated with the above manuscript.

---
## Data processing 

The foot placement control pipeline used in the manuscript are useable for any legged agent (artificial or biological) independently of the agent's dimension and experimental procedure used for the data collection. The only conditions on the data are the following 

- The horizontal positions (i.e., parallel to the agent's transverse plane) of the extremity of each limb should be availalbe throughoug the trial.  
- The horizontal position of at least one point of the body (not located on one of the legs used above) should be available throughout the trial.
- The kinematic data has been collected at a high enough framerate (ideally a temporal resolution of at least 10 frames per gait cycles)


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