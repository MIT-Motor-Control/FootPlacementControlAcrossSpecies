function [foot_contact_matrix] = foot_contact_detection(input_data)
% FOOT_CONTACT_DETECTION extracts the timings of contact for each individual foot
% 
% INPUTS
%     - input_data : contains the timeseries (in rows) of each foot positions in the foreaft direction - 0 when the foot is in its swing phase
%
% OUTPUTS
% - foot_contact_matrix : contains -1 for toe off, 1 for foot strike and 0 for any other event
% @Antoine De Comite - MIT 2025


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
