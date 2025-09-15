function data_output = rotate_data(data_input, angle, time_vector)


tmp_data =  data_input(time_vector,:,:);
data_output = tmp_data;
data_output(:,:,1) = cos(angle) * tmp_data(:,:,1) + sin(angle) * tmp_data(:,:,3);
data_output(:,:,3) = -sin(angle) * tmp_data(:,:,1) + cos(angle) * tmp_data(:,:,3);

data_output(:,:,1) = data_output(:,:,1) - data_output(1,1,1);
data_output(:,:,3) = data_output(:,:,3) - data_output(1,1,3);


end
