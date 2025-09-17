function output_message = get_message(padded_list, com_vel, idx_input)
idx_nan = find(isnan(padded_list(:,1)));
idx_begin = idx_nan(idx_input); idx_end = idx_nan(idx_input+1);
angle_vec = padded_list(idx_begin:idx_end,4);
idx_nonnan = find(~isnan(angle_vec),1);
idx_spin = find(abs(angle_vec-angle_vec(idx_nonnan))>30);
scalingmm = 0.0047;
idx_slow = find(com_vel(idx_begin:idx_end)<0.1*scalingmm); 
if (~isempty(idx_slow)) && (~isempty(idx_spin))
    output_message = 1;
elseif (~isempty(idx_slow))
    output_message = 2;
elseif (~isempty(idx_spin))
    output_message = 3;
else
    output_message = 4;
end