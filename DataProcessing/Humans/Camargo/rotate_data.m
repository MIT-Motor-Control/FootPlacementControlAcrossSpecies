function data_output = rotate_data(data_input, angle, time_vector)
% DATA_OUTPUT : Rotates the data to align the data points along the main movement direction
%
% INPUTS:
%  data_input : Data to be rotated
%  angle : Rotation angle to be applied
%  time_vector : Mask to extract the relevant data 
%
% OUTPUTS:
%  data_output : Rotated data
%
% @Antoine De Comite - MIT 2025
tmp_data =  data_input(time_vector,:,:);
data_output = tmp_data;
data_output(:,:,1) = cos(angle) * tmp_data(:,:,1) + sin(angle) * tmp_data(:,:,3);
data_output(:,:,3) = -sin(angle) * tmp_data(:,:,1) + cos(angle) * tmp_data(:,:,3);

data_output(:,:,1) = data_output(:,:,1) - data_output(1,1,1);
data_output(:,:,3) = data_output(:,:,3) - data_output(1,1,3);


end
