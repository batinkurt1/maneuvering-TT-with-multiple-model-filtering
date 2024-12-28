function [estimated_states,estimated_covariance,predicted_states,predicted_innovation_covariance] = KalmanFilter(A,B,C,Q,H,R,time_steps,noisy_measurements,vmax,kappa,measurement_sigma)
estimated_states = zeros(4,length(time_steps));
predicted_states = zeros(4,length(time_steps));
estimated_covariance = cell(1,length(time_steps));
predicted_covariance = cell(1,length(time_steps));
predicted_innovation_covariance = cell(1,length(time_steps));

% single - point initialization
estimated_states(:,1) = [noisy_measurements(:,1);0;0];
estimated_covariance{1} = diag([measurement_sigma^2,...
    measurement_sigma^2,(vmax/kappa)^2,(vmax/kappa)^2]);



for k = 2:length(time_steps)
    % prediction update
    predicted_states(:,k) = A * estimated_states(:,k-1);
    predicted_covariance{k} = A * estimated_covariance{k-1} * A' + B*Q*B';

    % measurement update
    predicted_innovation_covariance{k} = C * predicted_covariance{k} * C' + ...
        H*R*H';
    K = predicted_covariance{k}*C'/predicted_innovation_covariance{k};
    estimated_states(:,k) = predicted_states(:,k) + ...
        K * (noisy_measurements(:,k) - C * predicted_states(:,k));
    estimated_covariance{k} = predicted_covariance{k} - ...
        K * predicted_innovation_covariance{k} * K';
    estimated_covariance{k} = 1/2 * (estimated_covariance{k}+estimated_covariance{k}');
end
end