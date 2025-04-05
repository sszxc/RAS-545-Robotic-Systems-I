function [M, dM] = massFunction(theta, m1, m2, m3, l1, l2)
    % 几何关系
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin2_theta = sin_theta.^2;
    cos2_theta = cos_theta.^2;

    k = l1 / l2;
    sin_phi = k * sin_theta;
    cos_phi = sqrt(1 - sin_phi.^2);
    tan_phi = sin_phi ./ cos_phi;

    % M(θ)
    M1 = (1/3)*m1*l1^2;
    M2 = m2*l1^2*(1/8 + (13/48)*(cos2_theta ./ (cos_phi.^2)));
    M3 = m3*l1^2*sin2_theta ./ (1 - (l1^2/l2^2)*sin2_theta);
    M = M1 + M2 + M3;

    % 数值导数
    h = 1e-6;
    theta_plus = theta + h;
    theta_minus = theta - h;

    %[M_p, ~] = massFunction_noDeriv(theta_plus, m1, m2, m3, l1, l2);
    %[M_m, ~] = massFunction_noDeriv(theta_minus, m1, m2, m3, l1, l2);
    M_p = massFunction_noDeriv(theta_plus, m1, m2, m3, l1, l2);
    M_m = massFunction_noDeriv(theta_minus, m1, m2, m3, l1, l2);
    dM = (M_p - M_m)/(2*h);
end

function [M] = massFunction_noDeriv(theta, m1, m2, m3, l1, l2)
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    sin2_theta = sin_theta.^2;
    cos2_theta = cos_theta.^2;

    k = l1 / l2;
    sin_phi = k * sin_theta;
    cos_phi = sqrt(1 - sin_phi.^2);

    M1 = (1/3)*m1*l1^2;
    M2 = m2*l1^2*(1/8 + (13/48)*(cos2_theta ./ (cos_phi.^2)));
    M3 = m3*l1^2*sin2_theta ./ (1 - (l1^2/l2^2)*sin2_theta);
    M = M1 + M2 + M3;
end