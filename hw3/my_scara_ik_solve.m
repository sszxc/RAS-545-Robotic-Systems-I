% Target position
x_target = 0.2;
y_target = 0.5;



% Define symbol and fk equations
syms theta1 theta2 theta3
x = 0.4 * cos(theta1) + 0.3 * cos(theta1 + theta2) + 0.15 * cos(theta1 + theta2 + theta3);
y = 0.4 * sin(theta1) + 0.3 * sin(theta1 + theta2) + 0.15 * sin(theta1 + theta2 + theta3);

% Solve
eq1 = x == x_target;
eq2 = y == y_target;
[theta1_sol, theta2_sol, theta3_sol] = solve([eq1, eq2], [theta1, theta2, theta3], 'Real', true);

% Convert solutions
theta1_num = double(theta1_sol);
theta2_num = double(theta2_sol);
theta3_num = double(theta3_sol);

theta1_deg = rad2deg(theta1_num);
theta2_deg = rad2deg(theta2_num);
theta3_deg = rad2deg(theta3_num);

disp('Solutions for theta1, theta2, theta3 (in degrees):');
disp([theta1_deg, theta2_deg, theta3_deg]);

% Validate the first solution
if ~isempty(theta1_num)
    x_check = double(subs(x, [theta1, theta2, theta3], [theta1_num(1), theta2_num(1), theta3_num(1)]));
    y_check = double(subs(y, [theta1, theta2, theta3], [theta1_num(1), theta2_num(1), theta3_num(1)]));
    disp('Validation (x, y):');
    disp([x_check, y_check]);
end