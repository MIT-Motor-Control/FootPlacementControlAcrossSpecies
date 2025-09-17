function [position_matrix] = extract_foot_placement(time_vector, x_position, y_position, body_xposition, head_marker)
% EXTRACT_FOOT_PLACEMENT constructs the foot placement mask for the raw data based on the distance detection method
%
% INPUTS
% - time_vector : time mask corresponding to the input data
% - x_position : x-position (along the movement direction) of the foot marker
% - y_position : y-position (orthogonal to movement direction) of the foot marker
% - body_xposition : x-position of the head marker
% - head_marker : velocity of the head marker
%
% OUTPUTS
% - position_matrix : Matrix containing the foot placement location in the entries corresponding to the time in stance phase.
%
% @Antoine De Comite - MIT 2025

Ts = 1/80; % The data was collected at 80 Hz
% Identifying whether this is a leftward or rightward locomotion bout (wrt to the first dimension)
avg_vel = mean(head_marker);
if avg_vel>0
    bool_up=1;
else
    bool_up=0;
end
% Computing the distance (along the first dimension) between the head and the foot markers
diff_leg_x  = squeeze(x_position - body_xposition);
foot_placement = 0;
head_velocity = 0;

% Identifying heel strike and toe off
[~, max_leg] = findpeaks(diff_leg_x, 'MinPeakDistance', 10);
[~, min_leg] = findpeaks(-diff_leg_x, 'MinPeakDistance', 10);
position_matrix = zeros(2,length(time_vector));
bool_contact = zeros(length(time_vector),1);
bool_contact(max_leg) = 1;
bool_contact(min_leg) = -1;

% For each detected contact, compute the foot contact location as the average location of the marker during the stance phase and populate the output files. 
for time = 1 : size(x_position,3)
    if bool_up
        cdt1 = 1; cdt2 = -1;
    else
        cdt1 = -1; cdt2 = 1;
    end
    len_before_time = length(bool_contact(1:time-1));
    len_after_time = length(bool_contact(time:end));
    if bool_contact(time)==cdt1
        idx_next_toeoff = find(bool_contact(time:end)==cdt2,1);
        if ~isempty(idx_next_toeoff)
            idx_endc = idx_next_toeoff;
        else
            idx_endc = length(bool_contact(time:end));
        end
        x_value = mean(x_position(1,1,time:time+idx_endc-1));
        y_value = mean(y_position(1,1,time:time+idx_endc-1));
        position_matrix(1,time:time+idx_endc-1) = x_value;
        position_matrix(2,time:time+idx_endc-1) = y_value;
    end
end



end
