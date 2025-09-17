function [output_metrics_] = extract_metrics(input_data, idx_begin, idx_end,animal,tmp_vel)
% EXTRACT_METRICS : extracts all the step-to-step contacts and their associated metrics
%
% INPUTS 
% - input_data : input foot placement data
% - idx_begin : initial time stamp of the current input data
% - idx_end : final time stamp of the current input data
% - animal : animal identity
% - tmp_vel : average velocity of the current locomotion bout
%
% OUTPUTS
% - output_metrics_ : Nx8 matrix containing all the contact information as follows (per column)
%                    - animal identity
%                    - first contact leg
%                    - second contact leg
%                    - step duration
%                    - step length
%                    - step width
%                    - average velocity during step
%                    - time stamp of the first contact
%
% @Antoine DE Comite - MIT 2025

output_metrics = zeros(1,8);
len_bout = idx_begin:idx_end;
for ii = 2 : length(len_bout)-2
    % Checking transitions for the front right foot
    if ((input_data(1,ii-1,1)==0) && (input_data(1,ii,1)==0) && (input_data(1,ii+1,1)~=0) && (input_data(1,ii+2,1)~=0))
        init_pos_x = input_data(1,ii+1,1);
        init_pos_y = input_data(2,ii+1,1);
        tmp = foot_contact_detection(input_data(:,ii:end,:));
        tmp_same = foot_contact_detection(input_data(:,ii+1:end,:));
        idx_leg1 = find(tmp_same(:,1)==1,1);
        idx_leg2 = find(tmp(:,2)==1,1);
        idx_leg3 = find(tmp(:,3)==1,1);
        idx_leg4 = find(tmp(:,4)==1,1);
        if ~isempty(idx_leg1)
            final_pos_x = input_data(1,ii+idx_leg1,1);
            final_pos_y = input_data(2,ii+idx_leg1,1);
            output_metrics = [output_metrics; [animal,1,1,1/80*(idx_leg1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg2)
            final_pos_x = input_data(1,ii+idx_leg2-1,2);
            final_pos_y = input_data(2,ii+idx_leg2-1,2);
            output_metrics = [output_metrics; [animal,1,2,1/80*(idx_leg2-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg3)
            final_pos_x = input_data(1,ii+idx_leg3-1,3);
            final_pos_y = input_data(2,ii+idx_leg3-1,3);
            output_metrics = [output_metrics; [animal,1,3,1/80*(idx_leg3-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg4)
            final_pos_x = input_data(1,ii+idx_leg4-1,4);
            final_pos_y = input_data(2,ii+idx_leg4-1,4);
            output_metrics = [output_metrics; [animal,1,4,1/80*(idx_leg4-2),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
    % Checking the transitions for the front left foot
    elseif ((input_data(1,ii-1,2)==0) && (input_data(1,ii,2)==0) && (input_data(1,ii+1,2)~=0) && (input_data(1,ii+2,2)~=2))
        init_pos_x = input_data(1,ii+1,2);
        init_pos_y = input_data(2,ii+1,2);
        tmp = foot_contact_detection(input_data(:,ii:end,:));
        tmp_same = foot_contact_detection(input_data(:,ii+1:end,:));
        idx_leg1 = find(tmp(:,1)==1,1);
        idx_leg2 = find(tmp_same(:,2)==1,1);
        idx_leg3 = find(tmp(:,3)==1,1);
        idx_leg4 = find(tmp(:,4)==1,1);
        if ~isempty(idx_leg1)
            final_pos_x = input_data(1,ii+idx_leg1-1,1);
            final_pos_y = input_data(2,ii+idx_leg1-1,1);
            output_metrics = [output_metrics; [animal,2,1,1/80*(idx_leg1-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg2)
            final_pos_x = input_data(1,ii+idx_leg2,2);
            final_pos_y = input_data(2,ii+idx_leg2,2);
            output_metrics = [output_metrics; [animal,2,2,1/80*(idx_leg2),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg3)
            final_pos_x = input_data(1,ii+idx_leg3-1,3);
            final_pos_y = input_data(2,ii+idx_leg3-1,3);
            output_metrics = [output_metrics; [animal,2,3,1/80*(idx_leg3-2),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg4)
            final_pos_x = input_data(1,ii+idx_leg4-1,4);
            final_pos_y = input_data(2,ii+idx_leg4-1,4);
            output_metrics = [output_metrics; [animal,2,4,1/80*(idx_leg4-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
    % Checking the transitions for the hind right foot
    elseif ((input_data(1,ii-1,3)==0) && (input_data(1,ii,3)==0) && (input_data(1,ii+1,3)~=0) && (input_data(1,ii+2,3)~=0))
        init_pos_x = input_data(1,ii+1,3);
        init_pos_y = input_data(2,ii+1,3);
        tmp = foot_contact_detection(input_data(:,ii:end,:));
        tmp_same = foot_contact_detection(input_data(:,ii+1:end,:));
        idx_leg1 = find(tmp(:,1)==1,1);
        idx_leg2 = find(tmp(:,2)==1,1);
        idx_leg3 = find(tmp_same(:,3)==1,1);
        idx_leg4 = find(tmp(:,4)==1,1);
        if ~isempty(idx_leg1)
            final_pos_x = input_data(1,ii+idx_leg1-1,1);
            final_pos_y = input_data(2,ii+idx_leg1-1,1);
            output_metrics = [output_metrics; [animal,3,1,1/80*(idx_leg1-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg2)
            final_pos_x = input_data(1,ii+idx_leg2-1,2);
            final_pos_y = input_data(2,ii+idx_leg2-1,2);
            output_metrics = [output_metrics; [animal,3,2,1/80*(idx_leg2-2),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg3)
            final_pos_x = input_data(1,ii+idx_leg3,3);
            final_pos_y = input_data(2,ii+idx_leg3,3);
            output_metrics = [output_metrics; [animal,3,3,1/80*(idx_leg3),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg4)
            final_pos_x = input_data(1,ii+idx_leg4-1,4);
            final_pos_y = input_data(2,ii+idx_leg4-1,4);
            output_metrics = [output_metrics; [animal,3,4,1/80*(idx_leg4-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
    % Checking the transitions for the hind left foot
    elseif ((input_data(1,ii-1,4)==0) && (input_data(1,ii,4)==0) && (input_data(1,ii+1,4)~=0) && (input_data(1,ii+2,4)~=0))
        init_pos_x = input_data(1,ii+1,4);
        init_pos_y = input_data(2,ii+1,4);
        tmp = foot_contact_detection(input_data(:,ii:end,:));
        tmp_same = foot_contact_detection(input_data(:,ii+1:end,:));
        idx_leg1 = find(tmp(:,1)==1,1);
        idx_leg2 = find(tmp(:,2)==1,1);
        idx_leg3 = find(tmp(:,3)==1,1);
        idx_leg4 = find(tmp_same(:,4)==1,1);
        if ~isempty(idx_leg1)
            final_pos_x = input_data(1,ii+idx_leg1-1,1);
            final_pos_y = input_data(2,ii+idx_leg1-1,1);
            output_metrics = [output_metrics; [animal,4,1,1/80*(idx_leg1-2),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg2)
            final_pos_x = input_data(1,ii+idx_leg2-1,2);
            final_pos_y = input_data(2,ii+idx_leg2-1,2);
            output_metrics = [output_metrics; [animal,4,2,1/80*(idx_leg2-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg3)
            final_pos_x = input_data(1,ii+idx_leg3-1,3);
            final_pos_y = input_data(2,ii+idx_leg3-1,3);
            output_metrics = [output_metrics; [animal,4,3,1/80*(idx_leg3-1),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
        if ~isempty(idx_leg4)
            final_pos_x = input_data(1,ii+idx_leg4,4);
            final_pos_y = input_data(2,ii+idx_leg4,4);
            output_metrics = [output_metrics; [animal,4,4,1/80*(idx_leg4),abs(final_pos_x-init_pos_x),abs(final_pos_y-init_pos_y),tmp_vel,ii+idx_begin+1]];
        end
    end
% Removing the zero padding from the output matrix.
output_metrics_ = output_metrics(2:end,:);


end
