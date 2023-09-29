%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Co-kriging By Hand   %%%%%%%%%%%
%%%%%%%%%% Troy Zangle %%%%%%%%%%%%

%% Set up some sampling constants for graph purposes
n_test = 200;

%% Set up the inputs over here
%% Ground truth "Expensive" function from Forrester Kriging Paper
Xe = [0.0;0.4;0.6;1.0];
Ye = ExpensiveTestFunction(Xe);
ne = size(Xe,1);


%% Ground truth "Cheap" function from Forrester Kriging Paper
Xc = [0.0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9;1.0];
Yc = CheapTestFunction(Xc);
nc = size(Xc,1);

% Calculate the distance matrix
pre_psi_ij = CalculatePrePsiMatrix( X, n_samp);

% Calculate Kriging Constants
[psi_ij,inv_psi_ij] = CalculatePsiMatrix(pre_psi_ij,theta);
mu_hat = CalculateKrigingMean( inv_psi_ij, Y, n_samp);
b = CalculateConstantB(inv_psi_ij, Y, mu_hat, n_samp);

% Evaluate the value of the kriging model at each test sample point
X_gt = linspace(0,1,n_test);
Y_gt = TestFunction(X_gt);
y_pred = zeros(n_test,1);
for ind = 1:n_test
    y_pred(ind,1) = KrigingPrediction(X_gt(ind), X, theta, b, mu_hat, n_samp);
end

figure(4)
plot(X_gt,Y_gt,'r-','LineWidth',2);
hold on
scatter(X,Y)
scatter(X_gt,y_pred);
xlabel('X');
ylabel('Y_hat');
title('Prediction');

% Calculate sample points from the specified test function
% This function example is from Forrester 2008
function y_samp = TestFunction( x_samp)
    y_samp = (6.*x_samp-2).^2.*sin(12.*x_samp-4);
end

%Calculate the constant b for Kriging
function b = CalculateConstantB( psi_inv, y, mu_hat, n_samp)
    F = ones(n_samp,1);
    b = psi_inv*(y-F*mu_hat);
end

%Calculate the function mean for Kriging
function mu_hat = CalculateKrigingMean( inv_psi_ij, y, n_samp)
    F = ones(n_samp,1);
    mu_hat = (transpose(F)*inv_psi_ij*y)/(transpose(F)*inv_psi_ij*F);
end

% Calculate sigma
function sigma = CalculateSigma( inv_psi_ij, mu_hat, y, n_samp)
    F = ones(n_samp,1);
    sigma = (1.0/n_samp)*transpose(y-F*mu_hat)*inv_psi_ij*(y-F*mu_hat);
end

% Calculate the distance relation for the training locations
function pre_psi_ij = CalculatePrePsiMatrix( X, n_samp)
    pre_psi_ij = zeros(n_samp);
    for i = 1:n_samp
        for j = 1:n_samp
            pre_psi_ij(i,j)=abs(X(i)-X(j))^2;
        end
    end
end

% Calculate the psi correlation matrix
function [psi_ij,inv_psi_ij] = CalculatePsiMatrix( pre_psi_ij, theta)
    psi_ij = exp(-theta * pre_psi_ij);
    inv_psi_ij = inv(psi_ij);
end

% Maximum Likelihood estimation
function mle = CalculateMLE( pre_psi_ij,theta,y,n_samp)    
    disp(y);
    [psi_ij,inv_psi_ij] = CalculatePsiMatrix(pre_psi_ij,theta);
    mu_hat = CalculateKrigingMean(inv_psi_ij,y,n_samp);
    sigma = CalculateSigma(inv_psi_ij,mu_hat,y,n_samp);
    mle = -0.5*(n_samp*log(sigma)+log(det(psi_ij)));
end

% Evaluate Kriging Model
function y_pred = KrigingPrediction(x_new, X, theta, b, mu_hat, n_samp)
    y_pred = 0;
    for i = 1:n_samp
        psi_new = exp(-theta*((x_new-X(i))^2));
        y_pred = y_pred + b(i)*psi_new;
    end
    y_pred = y_pred+mu_hat;
end

% Evaluate Kriging influence of each point (Debugging fxn only)
function infl = KrigingPointInfluence(x_pred, X, theta, n_samp)
    infl = zeros(n_samp,1);
    for i = 1:n_samp
        infl(i) = exp(-theta*((x_pred-X(i))^2));
    end
end

function [dist,var] = GenerateSemiVariogram( X, Y, n_samp)
    % Next create the semivariogram
    var = zeros(n_samp*(n_samp+1)*0.5);
    dist = zeros(n_samp*(n_samp+1)*0.5);
    iter = 1;
    for i = 1:n_samp
        for j = i:n_samp
            dist(iter) = sqrt((X(i) - X(j))^2);
            var(iter) = 0.5*(Y(i)-Y(j))^2;
            iter = iter+1;
        end
    end
end