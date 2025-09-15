function bool_vector = extractStraightLineSegments(angle_vector)
% EXTRACTSTRAIGHTLINESEGMENTS : Identifies the straight line locomotion segments and returns the corresponding time masks
% 
% INPUTS:
%    angle_vector : Movement direction vector
%
% OUTPUTS:
%    bool_vector : Masking vector identifiying the straight line locomotion bouts
% 
% @Antoine De Comite - MIT 2025

% Transforming the original angles (rad) in degrees 
angle_deg = 360*angle_vector / (2*pi);
angle_deg = abs(angle_deg);

% Indentifying the time points where the orientation is around the straight line
bool_angle = ((angle_deg<110) & (angle_deg>70));
begins = zeros(1,1); lengths = zeros(1,1);
for ii = 2:length(bool_angle)
    if((bool_angle(ii)==1) && (bool_angle(ii-1)==0))
        next_zero = find(bool_angle(ii:end)==0,1);
        if isempty(next_zero)
            break
        else 
            begins = [begins, ii]; 
            lengths = [lengths, next_zero];
        end
    end
end

[len_max,idx_max] = maxk(lengths,2);

bool_vector = zeros(length(bool_angle),1);
bool_vector(begins(idx_max(1)):begins(idx_max(1))+len_max(1)) = 1; 
bool_vector(begins(idx_max(2)):begins(idx_max(2))+len_max(2)) = 1; 

end
