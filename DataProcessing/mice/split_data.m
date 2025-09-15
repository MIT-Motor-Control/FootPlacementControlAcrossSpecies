% This script splits the dataset into cohort specific ones - It is necessary when trying to load the whole data on a computer that has small amount of RAM memory.
data_input = load('jointsCON.mat').jointsCON;
disp("Data loaded");

% There are two functionning versions for this script. The first one evaluates the non mutants group (it corresponds to the uncommented current version of the file),
% the second one evaluates the mutant groups (it corresponds to the commented version of this file). To go from one to the other, all you have to do is to change which parts of the code are commmented.

n_group =  2;%size(data_input,2);


names = cell(n_group);
% names{1} = 'group1_joints.mat';
% names{2} = 'group2_joints.mat';
% names{3} = 'group3_joints.mat';
% names{4} = 'group4_joints.mat';
% names{5} = 'group5_joints.mat';
% names{6} = 'group6_joints.mat';
names{1} = 'group7_joints.mat';
names{2} = 'group8_joints.mat';
for ii = 7 : 8
    disp("Saving a dataset");
    tmp_struct = data_input{:,ii};
    save(names{ii-6}, 'tmp_struct');
end
