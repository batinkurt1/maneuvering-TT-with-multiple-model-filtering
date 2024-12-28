function [output_estimated_states,output_estimated_covariance,output_predicted_measurements,output_predicted_innovation_covariance,mode_probabilities] = IMM(TPM,initial_mode_probabilities,time_steps,Nr,As,Bs,Cs,Qs,Hs,Rs,noisy_measurements,vmax,kappa,measurement_sigma)

estimated_states = cell(Nr,length(time_steps));
predicted_states = cell(Nr,length(time_steps));
estimated_covariance = cell(Nr,length(time_steps));
predicted_covariance = cell(Nr,length(time_steps));
predicted_innovation_covariance = cell(Nr,length(time_steps));
mode_probabilities = zeros(Nr,length(time_steps));
output_estimated_states = zeros(4,length(time_steps));
output_estimated_covariance = cell(1,length(time_steps));
output_predicted_innovation_covariance = cell(1,length(time_steps));
output_predicted_measurements = zeros(2,length(time_steps));


output_estimated_states(:,1) = [noisy_measurements(:,1);0;0];
output_estimated_covariance{1} = diag([measurement_sigma^2,...
    measurement_sigma^2,(vmax/kappa)^2,(vmax/kappa)^2]);

for j = 1:Nr
    % 1 step initialization
    estimated_states{j,1} = [noisy_measurements(:,1);0;0];
    estimated_covariance{j,1} = diag([measurement_sigma^2,...
        measurement_sigma^2,(vmax/kappa)^2,(vmax/kappa)^2]);
    mode_probabilities(j,1) = initial_mode_probabilities(j);
    predicted_states{j,1} = estimated_states{j,1};
    predicted_covariance{j,1} = estimated_covariance{j,1};
end


for k = 2:length(time_steps)
    % calculate the mixing probabilities
    mixing_probability_matrix = zeros(Nr,Nr);
    for i = 1:Nr
        for j = 1:Nr
            mixing_probability_matrix(j,i) = ...
                TPM(j,i)*mode_probabilities(j,k-1)/...
                (TPM(:,i)'*(mode_probabilities(:,k-1)));
        end
    end
    % calculate the mixed estimates and covariances
    mixed_estimates = cell(1,Nr);
    mixed_covariances = cell(1,Nr);
    for i = 1:Nr
        mixed_estimates{i} = zeros(4,1);
        mixed_covariances{i} = zeros(4,4);
        for j = 1:Nr
            mixed_estimates{i} = mixed_estimates{i} + ...
                mixing_probability_matrix(j,i)*estimated_states{j,k-1};
        end
        
        for j = 1:Nr
                        mixed_covariances{i} = mixed_covariances{i} + ...
                mixing_probability_matrix(j,i)*(estimated_covariance{j,k-1}+...
                (estimated_states{j,k-1} - mixed_estimates{i})*...
                (estimated_states{j,k-1} - mixed_estimates{i})');
        end
        % prediction update
        predicted_states{i,k} = As{i} * mixed_estimates{i};
        predicted_covariance{i,k} = As{i}*mixed_covariances{i}*As{i}'...
            +Bs{i}*Qs{i}*Bs{i}';
        % measurement update
        predicted_innovation_covariance{i,k} = ...
            Cs{i}*predicted_covariance{i,k}*Cs{i}'+Hs{i}*Rs{i}*Hs{i}';
        K = predicted_covariance{i,k}*Cs{i}'/ predicted_innovation_covariance{i,k};
        estimated_states{i,k} = predicted_states{i,k} + ...
            K*(noisy_measurements(:,k) - Cs{i}*predicted_states{i,k});
        estimated_covariance{i,k} = predicted_covariance{i,k} - ...
            K*predicted_innovation_covariance{i,k}*K';

        % update mode probabilities
        mode_probabilities(i,k) = mvnpdf(noisy_measurements(:,k),...
            Cs{i}*predicted_states{i,k},...
            predicted_innovation_covariance{i,k})*(TPM(:,i)'*mode_probabilities(:,k-1));

    end
    % normalize mode probabilities
    mode_probabilities(:,k) = mode_probabilities(:,k)/sum(mode_probabilities(:,k));
    
    % output estimate calculation
    output_estimated_covariances{k} = zeros(4,4);
    output_predicted_innovation_covariance{k} = zeros(2,2);
    output_predicted_measurements(:,k) = zeros(2,1);

    for i = 1:Nr
        output_estimated_states(:,k) = output_estimated_states(:,k) + mode_probabilities(i,k)*estimated_states{i,k};
        output_predicted_measurements(:,k) = output_predicted_measurements(:,k) + TPM(:,i)'*mode_probabilities(:,k-1)*Cs{i}*predicted_states{i,k};
    end
    
    for i = 1:Nr
        output_estimated_covariances{k} = output_estimated_covariances{k} ...
            + mode_probabilities(i,k)*(estimated_covariance{i,k} + ...
            (estimated_states{i,k} - output_estimated_states(:,k))...
            *(estimated_states{i,k} - output_estimated_states(:,k))');

        output_predicted_innovation_covariance{k} = ...
            output_predicted_innovation_covariance{k} + ...
            TPM(:,i)'*mode_probabilities(:,k-1)*...
            (predicted_innovation_covariance{i,k}+...
            (Cs{i}*predicted_states{i,k}-output_predicted_measurements(:,k))*...
            (Cs{i}*predicted_states{i,k}-output_predicted_measurements(:,k))');
    end

end





end