function noisy_measurements = generate_measurements(trueTarget,mu,R)
v_k = mvnrnd(mu,R)';
noisy_measurements = trueTarget + v_k;
end