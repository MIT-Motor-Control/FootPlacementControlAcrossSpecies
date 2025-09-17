function [output_data] = rescale_data_to_mm(input_data)
%RESCALE_DATA_TO_MM transforms the data from pixels to mm
% 
% INPUTS 
% - input_data: contains the input unnormalized data
%
% OUTPUTS
% - output_data: contains the output normalized data
%
% @Antoine De Comite - MIT 2025

scaling_factor_x = 2.75/584.7894;
scaling_factor_y = 2.2/458.1300;
output_data = zeros(size(input_data));
x_indices = [5 7 9 11 13 15 17 19];
y_indices = [6 8 10 12 14 16 18 20];

for variable = 1 : size(input_data,2)
    if any(x_indices==variable)
        output_data(:,variable) = input_data(:,variable) * scaling_factor_x;
    elseif any(y_indices==variable)
        output_data(:,variable) = input_data(:,variable) * scaling_factor_y;
    else
        output_data(:,variable) = input_data(:,variable);
    end
end
end
