%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Kriging By Hand   %%%%%%%%%%%
%%%%%%%%%% Troy Zangle %%%%%%%%%%%%


%% Set up the inputs over here
%% Ground truth will be a sine wave
n_samp = 4;
amp_noise = 0.2;
X_gt = linspace(0.0,1.0,200);
X_gt = transpose(X_gt);
X = [0.0,0.4,0.6,1.0];
X = transpose(X);
Y_gt = (6.*X_gt-2).^2.*sin(12.*X_gt-4);
Y = (6.*X-2).^2.*sin(12.*X-4);
disp(Y);

% Plot the ground truth next to the noisy sample points
% figure(1)
% plot(X_gt,Y_gt,'r-','LineWidth',2);
% hold on
% scatter(X,Y)
% xlabel('X');
% ylabel('Y');
% hold off


% Next create the semivariogram
variance = zeros(n_samp*(n_samp+1)*0.5);
distance = zeros(n_samp*(n_samp+1)*0.5);
iter = 1;
for i = 1:n_samp
    for j = i:n_samp
        distance(iter) = sqrt((X(i) - X(j))^2);
        variance(iter) = 0.5*(Y(i)-Y(j))^2;
        iter = iter+1;
    end
end

% figure(2)
% scatter(distance,variance);
% xlabel('distance');
% ylabel('variance');
% title('variogram');

% Calculate correlation matrix
psi_ij = zeros(n_samp,n_samp);
for i = 1:n_samp
    for j = 1:n_samp
        psi_ij(i,j)=exp(-abs(X(i)-X(j))^2);
    end
end
disp(psi_ij);
psi_inv = inv(psi_ij);

% Calculate mean
F = ones(n_samp,1);
disp(Y);
mu_hat = (transpose(F)*psi_inv*Y)/(transpose(F)*psi_inv*F);

% plot the MLE for n theta values
n_theta = 300;
theta_vec = linspace(0.1,6,n_theta);
mle = zeros(n_theta,1);
disp(mle);
firstsz = size(mle);
disp(firstsz);
for iter = 1:n_theta
    for i = 1:n_samp
        for j = 1:n_samp
            psi_ij(i,j)=exp(-theta_vec(iter)*abs(X(i)-X(j))^2);
        end
    end
    psi_inverse = inv(psi_ij);
    sigma = (1.0/n_samp)*transpose(Y-F*mu_hat)*psi_inverse*(Y-F*mu_hat);
    det_psi = det(psi_ij);
    mle(iter) = -0.5*(n_samp*log(sigma)+log(det_psi));
    
end
sz= size(mle);
disp(mle);
figure(3)
scatter(theta_vec,mle);
xlabel('theta');
ylabel('MLE');
title('Maximum Function');

theta = 3.3;

for i = 1:n_samp
    for j = 1:n_samp
        psi_ij(i,j)=exp(-theta*abs(X(i)-X(j))^2);
    end
end

inv_psi = inv(psi_ij);

% Calculate b
b = inv_psi*(Y-F*mu_hat);
disp(b);
% Set up kriging fit
y_pred = zeros(200);
infl = zeros(n_samp,200);
infl_add = zeros(200);
for ind = 1:200
    for i = 1:n_samp
        psi_new = exp(-theta*((X_gt(ind)-X(i))^2));
        infl(i,ind) = psi_new;
        infl_add(ind) = infl_add(ind)+psi_new;
        y_pred(ind) = y_pred(ind) + b(i)*psi_new;
    end
    y_pred(ind) = y_pred(ind)+mu_hat;
end

figure(5)
scatter(X_gt,infl);
xlabel('X');
ylabel('Infl');
title('Prediction');

figure(4)
plot(X_gt,Y_gt,'r-','LineWidth',2);
hold on
scatter(X,Y)
scatter(X_gt,y_pred(:,1));
xlabel('X');
ylabel('Y_hat');
title('Prediction');
