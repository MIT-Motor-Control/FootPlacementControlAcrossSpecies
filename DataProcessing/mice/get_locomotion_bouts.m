function [output_matrix] = get_locomotion_bouts(behaviors)
%GETLOCOMOTIONBOUTS extracts the locomotion bouts from the Klibaite paper 
%
% INPUTS 
% ======
% behaviors is the timeseries of the classified behaviors
%
% OUTPUTS
% =======
% output_matrix contains the timestamp of the beginning and end of each
% locomotion bout (1st & 2nd column for the beginning and the end
% respectively).
% @Antoine De Comite - MIT 2025
% v1.1. updated documentation


output_matrix = zeros(1,2);
for idx = 1 : length(behaviors)-1
    if (behaviors(idx)~=8) && (behaviors(idx+1)==8)
        next_non_loco = find(behaviors(idx+1:end)~=8,1);
        if isempty(next_non_loco)
            output_matrix(idx,:) = [idx+1, length(behaviors(idx+1:end))-1];
        else
            output_matrix(idx,:) = [idx+1,next_non_loco-1];
        end
    end
end
end
