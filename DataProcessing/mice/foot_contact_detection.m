function [foot_contact_matrix] = foot_contact_detection(input_data)
%foot_contact_detection extracts the timings of contact for each individual foot
% 
% INPUTS
% ======
% input_data is a matrix containing the timeseries (row) of the positions
% of each foot (column) - the value is 0 when the foot isn't in contact
%
% OUTPUTS
% =======
% foot_contact_matrix is -1 for toe off and 1 for foot strike
% Antoine De Comite - 11.19.2023


foot_contact_matrix = zeros(size(input_data,2),size(input_data,3));
for ii = 1 : size(input_data,3)
    for jj = 1 : size(input_data,2)-1
        if (input_data(1,jj,ii)==0 && input_data(1,jj+1,ii)~=0)
            foot_contact_matrix(jj+1,ii) = 1 ;
        elseif (input_data(1,jj,ii)~=0 && input_data(1,jj+1,ii)==0)
            foot_contact_matrix(jj,ii) = -1;
        end
    end
end
end