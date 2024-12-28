clearvars; clc; 
close all;
%Loading the true data belonging to the target 
load("trueTarget.mat");

time_steps = trueTarget(1, :);
x_true = trueTarget(2, :);
y_true = trueTarget(3, :);

noisy_measurements = zeros(2,length(time_steps));

sigma_x = 20;  % Standard deviation for x measurement noise 
mu_x = 0;       % Mean for x measurement noise 
sigma_y = 20;  % Standard deviation for y measurement noise 
mu_y = 0;       % Mean for y measurement noise 


R = diag([sigma_x^2,sigma_y^2]);

measurement_noise_mu = [mu_x; mu_y];

for k = 1:length(time_steps)
    noisy_measurements(:,k) = generate_measurements(trueTarget(2:3,k),measurement_noise_mu,R);
end

figure;
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(noisy_measurements(1,:),noisy_measurements(2,:),'r.');
title("True Target Trajectory and Noisy Measurements");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Noisy Measurements");
grid on;


%sampling period
T=1;
vmax = 50;
kappa = 3;


% State transition matrix A
A = [eye(2),T*eye(2);
    zeros(2,2),eye(2)];

% Process Noise matrix B
B = [T^2/2*eye(2);
    T*eye(2)];

C = [eye(2),zeros(2)];
H = eye(2);

Q1 = 0.1^2 * eye(2);
Q2 = 1^2 * eye(2);
Q3 = 10^2 * eye(2);

[estimated_states1,estimated_covariance1,predicted_states1,predicted_innovation_covariance1] = ...
    KalmanFilter(A,B,C,Q1,H,R,time_steps,noisy_measurements,vmax,kappa,sigma_x);
[estimated_states2,estimated_covariance2,predicted_states2,predicted_innovation_covariance2] = ...
    KalmanFilter(A,B,C,Q2,H,R,time_steps,noisy_measurements,vmax,kappa,sigma_x);
[estimated_states3,estimated_covariance3,predicted_states3,predicted_innovation_covariance3] = ...
    KalmanFilter(A,B,C,Q3,H,R,time_steps,noisy_measurements,vmax,kappa,sigma_x);
figure;
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(noisy_measurements(1,:),noisy_measurements(2,:),'r.');
title("True Target Trajectory, Noisy Measurements, and Estimated States");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
grid on;
plot(estimated_states1(1,:),estimated_states1(2,:),LineWidth=1.5);
plot(estimated_states2(1,:),estimated_states2(2,:),LineWidth=1.5);
plot(estimated_states3(1,:),estimated_states3(2,:),LineWidth=1.5);
legend("True Target Trajectory","Noisy Measurements","\sigma = 0.1m/s^2","\sigma = 1m/s^2","\sigma = 10m/s^2");

estimation_error1 = sqrt((x_true-estimated_states1(1,:)).^2+(y_true-estimated_states1(2,:)).^2);
estimation_error2 = sqrt((x_true-estimated_states2(1,:)).^2+(y_true-estimated_states2(2,:)).^2);
estimation_error3 = sqrt((x_true-estimated_states3(1,:)).^2+(y_true-estimated_states3(2,:)).^2);

figure;
plot(time_steps,estimation_error1,LineWidth=1.5);
hold on;
plot(time_steps,estimation_error2,LineWidth=1.5);
plot(time_steps,estimation_error3,LineWidth=1.5);
title("Estimation Errors");
xlabel("Time (s)");
ylabel("Error (m)");
grid on;
legend("Q1 = 0.1^2*I","Q2 = 1^2*I","Q3 = 10^2*I");

rms_estimation1 = sqrt(1/length(time_steps)*(sum(estimation_error1.^2)));
rms_estimation2 = sqrt(1/length(time_steps)*(sum(estimation_error2.^2)));
rms_estimation3 = sqrt(1/length(time_steps)*(sum(estimation_error3.^2)));

fprintf("Q1: Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation1);
fprintf("Q2: Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation2);
fprintf("Q3: Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation3);

gate_threshold = chi2inv(0.99,2);

% with gates

figure;
subplot(1,3,1);
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(estimated_states1(1,:),estimated_states1(2,:),'.');
for k = 2:length(time_steps)
    U = cholcov(predicted_innovation_covariance1{k});
    theta = linspace(0,2*pi,100);
    y = zeros(2,100);
    for i = 1:100
        y(:,i) =  C*predicted_states1(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
    end
    plot(y(1,:),y(2,:),'r');
end
title("Gates Q1 = 0.1^2*I");
grid on;

subplot(1,3,2);
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(estimated_states2(1,:),estimated_states2(2,:),'.');
for k = 2:length(time_steps)
    U = cholcov(predicted_innovation_covariance2{k});
    theta = linspace(0,2*pi,100);
    y = zeros(2,100);
    for i = 1:100
        y(:,i) =  C*predicted_states2(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
    end
    plot(y(1,:),y(2,:),'r');
end
title("Gates Q2 = 1^2*I");
grid on;

subplot(1,3,3);
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(estimated_states3(1,:),estimated_states3(2,:),'.');
for k = 2:length(time_steps)
    U = cholcov(predicted_innovation_covariance3{k});
    theta = linspace(0,2*pi,100);
    y = zeros(2,100);
    for i = 1:100
        y(:,i) =  C*predicted_states3(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
    end
    plot(y(1,:),y(2,:),'r');
end

title("Gates Q3 = 10^2*I");
grid on;

% Gate volumes


gatevolumes1 = zeros(1,length(time_steps));
gatevolumes2 = zeros(1,length(time_steps));
gatevolumes3 = zeros(1,length(time_steps));
for k = 1:length(time_steps)
    gatevolumes1(k) = pi*gate_threshold*sqrt(det(predicted_innovation_covariance1{k}));
    gatevolumes2(k) = pi*gate_threshold*sqrt(det(predicted_innovation_covariance2{k}));
    gatevolumes3(k) = pi*gate_threshold*sqrt(det(predicted_innovation_covariance3{k}));
end

figure;
plot(time_steps,gatevolumes1);
hold on;
plot(time_steps,gatevolumes2);
plot(time_steps,gatevolumes3);
grid on;
title("Gate Volumes");
legend("Q1 = 0.1^2*I","Q2 = 1^2*I","Q3 = 10^2*I");


% IMM

TPM1 = [0.99,0.01;0.01,0.99];
TPM2 = [0.999,0.001;0.001,0.999]; % daha fazla hatırlayacak
TPM3 = [0.5,0.5;0.5,0.5];
TPM4 = [1,0;0,1];
initial_mode_probabilities = [0.5 0.5];
initial_mode_probabilities2 = [1,0];
initial_mode_probabilities3 = [0,1];
Nr = length(initial_mode_probabilities);

As = {A,A};
Bs = {B,B};
Cs = {C,C};
Qs = {Q1,Q3};
Hs = {H,H};
Rs = {R,R};


[output_estimated_states,output_estimated_covariance,output_predicted_measurements,output_predicted_innovation_covariance,mode_probabilities] = IMM(TPM1,initial_mode_probabilities,time_steps,Nr,As,Bs,Cs,Qs,Hs,Rs,noisy_measurements,vmax,kappa,sigma_x);


figure;
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(noisy_measurements(1,:),noisy_measurements(2,:),'r.');
title("True Target Trajectory, Noisy Measurements, and Estimated States");
plot(output_estimated_states(1,:),output_estimated_states(2,:),LineWidth=3.5);
plot(estimated_states1(1,:),estimated_states1(2,:),LineWidth=1.5);
plot(estimated_states2(1,:),estimated_states2(2,:),LineWidth=1.5);
plot(estimated_states3(1,:),estimated_states3(2,:),LineWidth=1.5);


ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Noisy Measurements","IMM Filter","Q1 = 0.1^2*I","Q2 = 1^2*I","Q3 = 10^2*I");
grid on;

output_estimation_error = sqrt((x_true-output_estimated_states(1,:)).^2+(y_true-output_estimated_states(2,:)).^2);

figure;
plot(time_steps,estimation_error1,LineWidth=1.5);
hold on;
plot(time_steps,estimation_error2,LineWidth=1.5);
plot(time_steps,estimation_error3,LineWidth=1.5);
plot(time_steps,output_estimation_error,LineWidth=1.5);
title("Estimation Errors");
xlabel("Time (s)");
ylabel("Error (m)");
grid on;
legend("Q1 = 0.1^2*I","Q2 = 1^2*I","Q3 = 10^2*I","IMM Filter");

rms_estimation1 = sqrt(1/length(time_steps)*(sum(estimation_error1.^2)));
rms_estimation2 = sqrt(1/length(time_steps)*(sum(estimation_error2.^2)));
rms_estimation3 = sqrt(1/length(time_steps)*(sum(estimation_error3.^2)));
rms_imm = sqrt(1/length(time_steps)*(sum(output_estimation_error.^2)));

fprintf("Q1: Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation1);
fprintf("Q2: Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation2);
fprintf("Q3: Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation3);
fprintf("IMM: Root Mean Square Error of Estimated Position: %0.5g \n",rms_imm);

figure;
% subplot(1,4,1);
% ylabel("y position");
% ylim([0,2500]);
% xlabel("x-position");
% xlim([500,3000]);
% plot(x_true,y_true,LineWidth=1.5);
% hold on;
% plot(estimated_states1(1,:),estimated_states1(2,:),'.');
% for k = 2:length(time_steps)
%     U = cholcov(predicted_innovation_covariance1{k});
%     theta = linspace(0,2*pi,100);
%     y = zeros(2,100);
%     for i = 1:100
%         y(:,i) =  C*predicted_states1(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
%     end
%     plot(y(1,:),y(2,:),'r');
% end
% title("Gates Q1 = 0.1^2*I");
% grid on;
% 
% subplot(1,4,2);
% ylabel("y position");
% ylim([0,2500]);
% xlabel("x-position");
% xlim([500,3000]);
% plot(x_true,y_true,LineWidth=1.5);
% hold on;
% plot(estimated_states2(1,:),estimated_states2(2,:),'.');
% for k = 2:length(time_steps)
%     U = cholcov(predicted_innovation_covariance2{k});
%     theta = linspace(0,2*pi,100);
%     y = zeros(2,100);
%     for i = 1:100
%         y(:,i) =  C*predicted_states2(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
%     end
%     plot(y(1,:),y(2,:),'r');
% end
% title("Gates Q2 = 1^2*I");
% grid on;
% 
% subplot(1,4,3);
% ylabel("y position");
% ylim([0,2500]);
% xlabel("x-position");
% xlim([500,3000]);
% plot(x_true,y_true,LineWidth=1.5);
% hold on;
% plot(estimated_states3(1,:),estimated_states3(2,:),'.');
% for k = 2:length(time_steps)
%     U = cholcov(predicted_innovation_covariance3{k});
%     theta = linspace(0,2*pi,100);
%     y = zeros(2,100);
%     for i = 1:100
%         y(:,i) =  C*predicted_states3(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
%     end
%     plot(y(1,:),y(2,:),'r');
% end
% 
% title("Gates Q3 = 10^2*I");
% grid on;
% 
% subplot(1,4,4);
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(output_estimated_states(1,:),output_estimated_states(2,:),'.');
for k = 2:length(time_steps)
    U = cholcov(output_predicted_innovation_covariance{k});
    theta = linspace(0,2*pi,100);
    y = zeros(2,100);
    for i = 1:100
        y(:,i) =  output_predicted_measurements(:,k) + sqrt(gate_threshold)*U*[cos(theta(i)) sin(theta(i))]';
    end
    plot(y(1,:),y(2,:),'r');
end

title("Gates of IMM");
grid on;


gatevolumes1 = zeros(1,length(time_steps));
gatevolumes2 = zeros(1,length(time_steps));
gatevolumes3 = zeros(1,length(time_steps));
gatevolumesimm = zeros(1,length(time_steps));
for k = 1:length(time_steps)
    gatevolumes1(k) = pi*gate_threshold*sqrt(det(predicted_innovation_covariance1{k}));
    gatevolumes2(k) = pi*gate_threshold*sqrt(det(predicted_innovation_covariance2{k}));
    gatevolumes3(k) = pi*gate_threshold*sqrt(det(predicted_innovation_covariance3{k}));
    gatevolumesimm(k) = pi*gate_threshold*sqrt(det(output_predicted_innovation_covariance{k}));
end

figure;
plot(time_steps,gatevolumes1);
hold on;
plot(time_steps,gatevolumes2);
plot(time_steps,gatevolumes3);
plot(time_steps,gatevolumesimm);
grid on;
title("Gate Volumes");
legend("Q1 = 0.1^2*I","Q2 = 1^2*I","Q3 = 10^2*I","IMM");



figure;
plot(x_true,y_true,LineWidth=1.5);
hold on;
plot(noisy_measurements(1,:),noisy_measurements(2,:),'r.');
title("True Target Trajectory, Noisy Measurements, and Estimated States from the IMM Filter");
plot(output_estimated_states(1,:),output_estimated_states(2,:),LineWidth=1.5);


ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Noisy Measurements","IMM Filter");
grid on;


% mode probabilities

figure;
scatter(time_steps,mode_probabilities(1,:));
hold on;
scatter(time_steps,mode_probabilities(2,:));
legend("Q1 = 0.1^2*I","Q2 = 10^2*I");
grid on;