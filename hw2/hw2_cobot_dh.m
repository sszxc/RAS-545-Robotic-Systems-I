function dobot_kinematics()
    % define D-H parameters [theta_offset, d, a, alpha]
    dh_params = [    0, 219.34,    0,   pi/2;  % joint 1-2
                -pi/2,      0, -250,     0;  % joint 2-3
                   0,      0, -250,     0;  % joint 3-4
                 pi/2,   -108,    0, -pi/2;  % joint 4-5
                   0, 109.10,    0,  pi/2;  % joint 5-6
                   0, -75.86,    0,    pi]; % joint 6-end

    test_angles_deg = [0, 30, 30, 30, -45, 0];
    joint_angles = deg2rad(test_angles_deg);
    
    % compute forward kinematics
    T = forward_kinematics(dh_params, joint_angles);
    
    % extract position and orientation
    position = T(1:3, 4)';
    
    % display results
    fprintf('test joint angles (deg):\n'); disp(test_angles_deg);
    fprintf('end effector pose:\n'); disp(T);
    fprintf('end effector position (x, y, z):\n'); disp(position);
end

function T_total = forward_kinematics(dh_params, joint_angles)
    % compute forward kinematics
    T_total = eye(4);
    for i = 1:size(dh_params, 1)
        % get current joint parameters
        theta_offset = dh_params(i, 1);
        d = dh_params(i, 2);
        a = dh_params(i, 3);
        alpha = dh_params(i, 4);
        
        % compute actual joint angle (with offset)
        theta = joint_angles(i) + theta_offset;
        
        % compute current transformation matrix
        Ti = dh_transform(theta, d, a, alpha);
        fprintf('joint %d transformation matrix:\n', i); disp(Ti);
        
        % accumulate transformation matrix
        T_total = T_total * Ti;
    end
end

function Ti = dh_transform(theta, d, a, alpha)
    % compute single D-H transformation matrix
    ct = cos(theta);
    st = sin(theta);
    ca = cos(alpha);
    sa = sin(alpha);
    
    Ti = [ct   -st*ca   st*sa   a*ct;
          st    ct*ca  -ct*sa   a*st;
          0      sa      ca      d;
          0       0       0      1];
end

% run main function
dobot_kinematics();
