This folder contains the code used to extract the data from the Klibaite dataset. This data was published alongside the following paper: 

"Deep phenotyping reveals movement phenotypes in mouse neurodevelopmental models", Klibaite, U and Kislin, M and Verpeut, J L and Bergeler, S and Sun, X and Shaevitz, J W and Wang, S S H, 2022, Molecular Autism, 13 (12)

# How to run the code 
Once you have downloaded the original data (available at [this link](https://datacommons.princeton.edu/discovery/catalog/doi-10-34770-bzkz-j672)) you first need to split the whole dataset into different subsets as it can be too large to process in a single pass with a classical laptop RAM memory. For this purpose run the `split_data.m` code which will generates different matlab datafiles for each group.

Then you can run the file ScriptMetricsStraight.m which is the main processing code. It will (1) extract the straight line locomotion bouts, (2) extracts the foot contacts and compute their location, and (3) save the processed data. It will generates four types of file : 
- output_time{group} : contains the time stamps corresponding to the beginning and end of each straight line locomotion bout
- raw_group{group} : contains the raw data for each straight line locomotion bouts, interleaved with slabs of NaN values
- mat_group{group} : contains the foot placement data for each straight line locomotiob bouts, interleaved with slabs of NaN values
- output_metrics_straight : contains the straight line locomotion metrics (step length, width, and duration as well as the step-wise average velocity)

