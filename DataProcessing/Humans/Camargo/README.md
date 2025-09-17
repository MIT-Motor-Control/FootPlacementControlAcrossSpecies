This folder contains the code used to extract the data from the Camargo dataset. This data was published alongside the following paper :

"A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions"
Camargo, J and Ramanathan, A and Flanagan, W and Young, A 2021, Journal of Biomechanics 119, 110320
# How to run the code 
These three matlab files have to be located in the directory containing the marker data from the individual trial from the level ground locomotion. As it is now, they have to be copied and pasted for each subject. The path for these files look something like


`Camargo2023/[SubjectID]/[SubjectDate]/levelground/markers`

The code can be run by executing the file load_data.m in the MatlabTerminal and it will create a data file named `data_subject_[SubjectID]`.

