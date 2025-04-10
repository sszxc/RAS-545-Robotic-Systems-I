%% =============================================================
%  FILE: crank_slider_sim.m
%  滑块-曲柄机构的拉格朗日方程数值求解 (示例)
%  初始条件: theta(0) = pi/3, dtheta(0) = 10 rad/s
%% =============================================================
clc; clear; close all;

%% 1. 定义系统参数
params.m1 = 1.0;    % 曲柄质量 [kg]
params.m2 = 2.0;    % 连杆质量 [kg]
params.m3 = 0.5;    % 滑块质量 [kg]
params.l1 = 0.3;    % 曲柄长度 [m]
params.l2 = 0.5;    % 连杆长度 [m]
params.g  = 9.81;   % 重力加速度 [m/s^2]

% 初始条件
theta0  = pi/3;     % [rad]
omega0  = 5;       % 初始角速度 [rad/s]
y0 = [theta0; omega0];

% 仿真时间区间
tspan = [0, 60.0];   % 视情况可延长, 例如 [0, 2] 或更长

%% 2. 调用 ODE 求解器
%   状态量 y = [theta; dtheta]
%   y' = [ dtheta; ddtheta ]
[tSol, ySol] = ode45(@(t, y) crankSliderODE(t, y, params), tspan, y0);

% 提取解
thetaSol = ySol(:,1);   % 角度 theta(t)
omegaSol = ySol(:,2);   % 角速度 dtheta(t)

% 根据几何关系, 计算滑块位移 x(t)
%   闭链方程: x - l1*cos(theta) = ± sqrt(l2^2 - l1^2*sin^2(theta))
%   取正号(通常滑块在曲柄右侧), 则:
%       x(t) = l1*cos(theta) + sqrt(l2^2 - l1^2*sin^2(theta))
xSol = arrayfun(@(th) ...
    params.l1*cos(th) + sqrt(params.l2^2 - params.l1^2*sin(th)^2), ...
    thetaSol);

%% 3. 绘图
% === Plot ===
figure;
subplot(3,1,1);
plot(tSol, thetaSol, 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('\theta [rad]');
title('Crank Angle \theta');

subplot(3,1,2);
plot(tSol, omegaSol, 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('\omega = d\theta/dt [rad/s]');
title('Crank Angle Velocity \omega(t)');

subplot(3,1,3);
plot(tSol, xSol, 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('x [m]');
title('Slider Position x');

% 结束
save('sim_data.mat', 'tSol', 'thetaSol', 'xSol');
disp('仿真完成。');

%% ================ 局部函数: crankSliderODE =========================
function dydt = crankSliderODE(~, y, p)
% crankSliderODE
%   依据拉格朗日方程: M(theta)*ddtheta + 0.5*M'(theta)*dtheta^2 + ...
%                       0.5*(m1+m2)*g*l1*cos(theta) = 0
%   返回状态导数: dydt = [ dtheta; ddtheta ]

theta = y(1);
omega = y(2);  % dtheta/dt

% 计算 M(theta) 和 M'(theta)
[M_val, M_deriv] = inertiaFunction(theta, p);

% 计算 ddtheta
%   M(theta)*ddtheta = - 0.5*M'(theta)*omega^2 - 0.5*(m1+m2)*g*l1*cos(theta)
ddtheta = - ( 0.5*M_deriv*omega^2 + 0.5*(p.m1 + p.m2)*p.g*p.l1*cos(theta) ) ...
           / M_val;

% ODE 返回
dydt = [omega; ddtheta];
end

%% ================ 局部函数: inertiaFunction ========================
function [M_val, M_deriv] = inertiaFunction(theta, p)
% inertiaFunction
%   计算 M(theta) 和它对 theta 的导数 dM/dtheta
%   M(theta) = 等效惯性项(曲柄 + 连杆 + 滑块)

% 提取参数
m1 = p.m1; m2 = p.m2; m3 = p.m3;
l1 = p.l1; l2 = p.l2;

% 先确定连杆角度 varphi(θ), 满足 sin(varphi) = (l1 / l2)*sin(theta).
% 这里假设机构几何有效(l2 >= l1), varphi 在 [0, pi] 范围。
varphi = asin( (l1/l2)*sin(theta) );

% cos(varphi) 带正号, 若连杆与曲柄同侧(常规滑块曲柄机构)
cosVarphi = sqrt(1 - (l1/l2)^2 * sin(theta)^2);

% -- 1) 曲柄转动惯性 --
I_crank = (1/3)*m1*l1^2;

% -- 2) 连杆的等效惯性(包含平动+绕质心转动) --
%   常见推导给出: T2 ~ [A1 + A2*cos^2(theta)/cos^2(varphi)] * dot(theta)^2
%   这里我们仅需把系数提炼, 形成等效 M2(θ).
%   [简略写法: 省去中间展开, 直接给出常见简化结构]
%   示例(与之前推导对应): M2(θ) = (m2*l1^2)/8 + (13*m2*l1^2)/(48) * (cos^2θ / cos^2varphi)
%   可做更精确或项目定制的推导.
M2_term = (m2*l1^2)/8 + (13*m2*l1^2/48)*(cos(theta)^2 / cosVarphi^2);

% -- 3) 滑块的等效惯性 --
%   x(θ) = ...;  dx/dt = (dx/dθ)*dot(θ).   =>   T3 = (1/2)*m3*(dx/dθ)^2 * dot(θ)^2
%   其中 dx/dθ = d/dθ of [ l1*cosθ + sqrt(l2^2 - l1^2*sin^2θ ] 
%   不展开写了, 只给出等效结果, 亦可显式计算.
%   这里为演示，采用一种“直接”写法:
%   令   tan(varphi) = (l1*sinθ)/(x - l1*cosθ).
%   最终 T3 ~ (1/2)*m3*x^2*tan^2(varphi)*dot(θ)^2.
%   用 x=..., varphi=... 的函数也可形成 M3(θ).
%   这里可以直接数值计算, 也可做解析近似.

% 为简化演示，这里用更常见的近似形式(如 many references)：
%   M3_term = (m3 * l1^2 * sin^2(theta)) / [1 - (l1^2/l2^2)*sin^2(theta)] 
%   (此公式可由 x(θ)、tan(φ) 等关系代入并化简得到)
M3_term = (m3 * l1^2 * sin(theta)^2) / ...
          (1 - (l1^2 / l2^2)*sin(theta)^2);

% ==== 将以上三部分相加 ====
M_val = I_crank + M2_term + M3_term;

% 若需要 M'(theta), 可以做数值微分或解析微分
% 这里演示数值微分(小步近似):
h = 1e-6;
M_plus  = inertiaFunction_noDeriv(theta + h, p);
M_minus = inertiaFunction_noDeriv(theta - h, p);
M_deriv = (M_plus - M_minus) / (2*h);

end

function Mv = inertiaFunction_noDeriv(theta, p)
% 只返回 M(theta), 不做导数
m1 = p.m1; m2 = p.m2; m3 = p.m3;
l1 = p.l1; l2 = p.l2;

% 计算 varphi
varphi = asin( (l1/l2)*sin(theta) );
cosVarphi = sqrt(1 - (l1/l2)^2 * sin(theta)^2);

% 曲柄
I_crank = (1/3)*m1*l1^2;
% 连杆
M2_term = (m2*l1^2)/8 + (13*m2*l1^2/48)*(cos(theta)^2 / cosVarphi^2);
% 滑块
M3_term = (m3 * l1^2 * sin(theta)^2) / ...
          (1 - (l1^2 / l2^2)*sin(theta)^2);

Mv = I_crank + M2_term + M3_term;
end