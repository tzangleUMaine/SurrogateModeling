%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Co-Kriging   %%%%%%%%%%%%%%%%
%%%%%%%%%% Troy Zangle %%%%%%%%%%%%


%% Set up some sampling constants for graph purposes
n_test = 200;

%% Set up the inputs over here
% Ground truth "Expensive" function from Forrester Kriging Paper
Xe = [0.0;0.4;0.;1.0];
Ye = ExpensiveTestFunction(Xe);

% Ground truth "Cheap" function from Forrester Kriging Paper
Xc = [0.0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9;1.0];
%Yc = CheapTestFunction(Xc);
Yc = LinearCheapFunction(Xc);

% Debugging Step
% Plot the expensive and cheap functions
[X_real,Yc_real,Ye_real]=PlotMultiFidelityFunctions(n_test);

%% Solve for the Co-kriging model
[inv_C,theta_c,theta_d,sigma_c,sigma_d,rho,mu_hat_ck,nc,ne]=CokrigingSetup(Xc,Xe,Yc,Ye,false);



%% Functions %%%%%%%%%%%%%%%%%%%%

% Co-Kriging Setup Function
function [inv_C,theta_c,theta_d,sigma_c,sigma_d,rho,mu_hat_ck,nc,ne]=CokrigingSetup(Xc,Xe,Yc,Ye,Verbose)

    %% Solve for the kriging model of the "Cheap" data    
    [theta_c, b_c, mu_hat_c, nc, sigma_c, psic_c, ~, ~] = KrigingSetup(Xc,Yc);
    
    % Debugging step
    % Plot the kriging model of the cheap function
    if Verbose
        PlotKrigingModel('Cheap Kriging Model', 0.0, 1.0, 300, Xc, Yc, theta_c, b_c, mu_hat_c, nc);
    end
    
    %% ONLY FOR DEBUG solve for the kriging model of the "Expensive" data
    if Verbose        
        [theta_e, b_e, mu_hat_e, ne, ~, ~, ~, ~] = KrigingSetup(Xe,Ye);
        % Plot the kriging model of the cheap function
    
        PlotKrigingModel('Expensive Kriging Model', 0.0, 1.0, 300, Xe, Ye, theta_e, b_e, mu_hat_e, ne);
    end

    %% Solve for the kriging model of the difference 
    %%      between the Expensive and Cheap data sets
    
    % Start by finding the value of yc_hat at all values of Xe
    ne = size(Xe,1);
    yc_hat_xe = zeros(ne,1);
    for ind = 1:ne
        yc_hat_xe(ind,1) = KrigingPrediction(Xe(ind), Xc, theta_c, b_c, mu_hat_c, nc);
    end
    
    [theta_d,b_d,mu_hat_d,rho,d,ne,sigma_d,psid_e,~,~] = DiffKrigingSetup(Xe,yc_hat_xe,Ye);
    
    % Debugging Step
    if Verbose
        % Plot the kriging model of the difference
        PlotKrigingModel('Diff Kriging Model', 0.0, 1.0, 300, Xe, d, theta_d, b_d, mu_hat_d, ne)
    end

    %% Assemble the Coefficients for Cokriging Evaluation
    % Calculate the psi_c(Xc,Xe) and psi_c(Xe,Xc) terms first
    [psic_ec,psic_ce] = CalculatePsiMatrix_MF(theta_c,Xe,Xc,ne,nc);
    
    % Calculate psic(Xe,Xe)
    pre_psi_e = CalculatePrePsiMatrix(Xe,ne);
    [psic_e, ~] = CalculatePsiMatrix(pre_psi_e, theta_c);
    
    % Assemble the C matrix from the above parts
    [~, inv_C] = CalculateCovariance(ne, nc, sigma_c, psic_c, rho, psic_ce, psic_ec, psic_e, sigma_d, psid_e);
    
    % Assemble the combined y vector of cheap and expensive values
    Yck = [Yc;Ye];
    
    %% Solve for the Co-kriging mean value
    % Find the cokriging mean value
    mu_hat_ck = CalculateKrigingMean(inv_C, Yck, nc+ne);    

    % Plot Comparison for debugging
    if true
        PlotCoKrigingModel('CoKriging Model', 0.0, 1.0, 300, Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
        %plotItAll(Xc, Yc, Xe, Ye, X_real, Yc_real, Ye_real, X_ke, Y_ke, X_kc, Y_kc, X_outck, Y_outck);
    end
end

% Kriging Setup Function
function [theta,b,mu_hat,n,sigma,psi,inv_psi,pre_psi] = KrigingSetup(X,Y)
    n = size(X,1);
    pre_psi = CalculatePrePsiMatrix( X, n);
    
%     PlotMLE(pre_psi,Y,n);
    
    fun = @(theta_solved)(-CalculateMLE(pre_psi,theta_solved,Y,n));
    theta = fminsearch(fun,1.0);
    
    % Calculate sigma^2, mu, and b constants
    [psi,inv_psi] = CalculatePsiMatrix(pre_psi,theta);
    mu_hat = CalculateKrigingMean( inv_psi, Y, n);
    sigma = CalculateSigma(inv_psi,mu_hat,Y,n);
    b = CalculateConstantB(inv_psi, Y, mu_hat, n);
end

% Difference Kriging Setup
% This includes a scaling factor output that is solved for
%   in the theta solver block
function [theta,b,mu_hat,rho,d,n,sigma,psi,inv_psi,pre_psi] = DiffKrigingSetup(X,Yce,Ye)
    n = size(X,1);

    % Calculate the distance matrix
    pre_psi = CalculatePrePsiMatrix(X, n);   
    
    fun = @(solved)(-CalculateMLEDiff(pre_psi,solved(1),Ye,Yce,solved(2),n));
    solved = [0.2, 1.9];
    options = optimset('MaxFunEvals',100000);
    output = fminsearch(fun,solved,options);
    theta = output(1);
    rho = output(2);

%     rho = 1.87;
%     fun = @(solved)(-CalculateMLEDiff(pre_psi,solved,Ye,Yce,rho,n));
%     theta = fminsearch(fun,1.0);
    
    % Calculate sigma, mu, and b constants
    [psi,inv_psi] = CalculatePsiMatrix(pre_psi,theta);
    d = Ye-rho*Yce;
    mu_hat = CalculateKrigingMean( inv_psi, d, n);
    sigma = CalculateSigma(inv_psi,mu_hat,d,n);
    b = CalculateConstantB(inv_psi, d, mu_hat, n);
end

% Calculate sample points from the specified "Expensive" test function
% This function example is from Forrester 2008
function y_samp = ExpensiveTestFunction(x_samp)
    y_samp = (6.*x_samp-2).^2.*sin(12.*x_samp-4);
end

% Calculate sample points from the specified "Cheap" test function
% This function example is from Forrester 2008
function y_samp = CheapTestFunction(x_samp)
    y_samp = 0.5*ExpensiveTestFunction(x_samp)+10*(x_samp-0.5)-5;
end

function y_samp = LinearCheapFunction(x_samp)
    x_lo = 0;
    x_hi = 1;
    y_lo = ExpensiveTestFunction(0);
    y_hi = ExpensiveTestFunction(1);
    y_samp = (y_hi-y_lo)/(x_hi-x_lo)*(x_samp-x_lo)-y_lo;
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

% Calculate the distance relation between multi-fidelity datasets
function [psi_ec, psi_ce] = CalculatePsiMatrix_MF( theta_c, Xe, Xc, ne, nc)
    psi_ec = zeros(ne,nc);
    psi_ce = zeros(nc,ne);
    for i = 1:ne
        for j = 1:nc
            psi_ec(i,j) = exp(-theta_c*((Xe(i)-Xc(j))^2));
            psi_ce(j,i) = psi_ec(i,j);
        end
    end
end

% Calculate the psi correlation matrix
function [psi_ij,inv_psi_ij] = CalculatePsiMatrix( pre_psi_ij, theta)
    psi_ij = exp(-theta * pre_psi_ij);
    inv_psi_ij = inv(psi_ij);
end

% Generate C matrix from calculated Psi matrices and other kriging
%    constants
function [C_mat, inv_C_mat] = CalculateCovariance(ne, nc, sigma_c, psic_c, rho, psic_ce, psic_ec, psic_e, sigma_d, psid_e)
    C_mat = zeros(ne+nc);
    C_mat(1:nc,1:nc)=sigma_c*psic_c;
    C_mat(1:nc,nc+1:nc+ne) = rho*sigma_c*psic_ce;
    C_mat(nc+1:nc+ne,1:nc) = rho*sigma_c*psic_ec;
    C_mat(nc+1:nc+ne,nc+1:nc+ne) = rho^2*sigma_c*psic_e+sigma_d*psid_e;
    inv_C_mat = inv(C_mat);
end

% Maximum Likelihood estimation for kriging
function mle = CalculateMLE( pre_psi_ij,theta,y,n_samp)
    [psi_ij,inv_psi_ij] = CalculatePsiMatrix(pre_psi_ij,theta);
    mu_hat = CalculateKrigingMean(inv_psi_ij,y,n_samp);
    sigma = CalculateSigma(inv_psi_ij,mu_hat,y,n_samp);
    mle = -0.5*(n_samp*log(sigma)+log(det(psi_ij)));
end

% Plotting method for debugging MLED diff calc
function PlotMLE(pre_psi_ij,y,n_samp)
    theta_vec = linspace(-1,100,300);
    mle = zeros(300,1);
    for i = 1:300
        [psi_ij,inv_psi_ij] = CalculatePsiMatrix(pre_psi_ij,theta_vec(i));
        mu_hat = CalculateKrigingMean(inv_psi_ij,y,n_samp);
        sigma = CalculateSigma(inv_psi_ij,mu_hat,y,n_samp);
        mle(i,1) = -0.5*(n_samp*log(sigma)+log(det(psi_ij)));    
    end
    % Plot the theta values assuming rho = 2
    figure('Name','Maximum Likelihood Estimation');
    scatter(theta_vec,mle);
    xlabel('Theta');
    ylabel('MLE');
    title('Maximum Likelihood Estimation');
end

% Maximum likelihood estimation for difference model in cokriging
function mle = CalculateMLEDiff(pre_psi_ij,theta,ye,yc_xe,rho,n_samp)
    [psi_ij,inv_psi_ij] = CalculatePsiMatrix(pre_psi_ij,theta);
    d = ye-rho*yc_xe;
    mu_hat = CalculateKrigingMean(inv_psi_ij,d,n_samp);
    sigma = CalculateSigma(inv_psi_ij,mu_hat,d,n_samp);
    mle = -0.5*(n_samp*log(sigma)+log(det(psi_ij)));
end

% Plotting method for debugging MLED diff calc
function PlotMLEDiff(pre_psi_ij,ye,yc_xe,n_samp)
    theta_vec = linspace(0,10,1000);
    mle = zeros(1000,1);
    for i = 1:1000
        [psi_ij,inv_psi_ij] = CalculatePsiMatrix(pre_psi_ij,theta_vec(i));
        d = ye-(2*yc_xe);
        mu_hat = CalculateKrigingMean(inv_psi_ij,d,n_samp);
        sigma = CalculateSigma(inv_psi_ij,mu_hat,d,n_samp);
        mle(i,1) = -0.5*(n_samp*log(sigma)+log(det(psi_ij)));    
    end
    % Plot the theta values assuming rho = 2
    figure('Name','Maximum Likelihood Estimation');
    scatter(theta_vec,mle);
    xlabel('Theta');
    ylabel('MLE');
    title('Maximum Likelihood Estimation Diff');
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

% Evaluate Kriging Model
function y_pred = CoKrigingPrediction(x_new, Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne)
    psic_c_new = zeros(nc,1);
    psic_e_new = zeros(ne,1);
    psid_e_new = zeros(ne,1);
    for i = 1:nc
        psic_c_new(i,1) = exp(-theta_c*((x_new-Xc(i))^2));
    end
    for i = 1:ne
        psic_e_new(i,1) = exp(-theta_c*((x_new-Xe(i))^2));
        psid_e_new(i,1) = exp(-theta_d*((x_new-Xe(i))^2));
    end
    c_vec = zeros(nc+ne,1);
    c_vec(1:nc,1) = rho*sigma_c*psic_c_new;
    c_vec(nc+1:ne+nc,1) = rho^2*sigma_c*psic_e_new + sigma_d*psid_e_new;    
    y_pred = mu_hat_ck + transpose(c_vec)*inv_C*(Yck-ones(nc+ne,1)*mu_hat_ck);
end

% Plot Co-kriging model
function [X_krig, Y_krig] = PlotCoKrigingModel(name, xmin, xmax, neval, Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne)
    X_eval = linspace(xmin,xmax,neval);
    Y_hat = zeros(neval,1);
    for ind = 1:neval
        Y_hat(ind,1) = CoKrigingPrediction(X_eval(ind), Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
    end
    figure('Name',name);
    plot(X_eval,Y_hat,'g-','LineWidth',2);
    hold on
    % Plot the cheap and expensive functions too
    Yc_real = CheapTestFunction(X_eval);
    Ye_real = ExpensiveTestFunction(X_eval);
    plot(X_eval,Yc_real,'r-','LineWidth',2,'DisplayName','Cheap');
    hold on
    plot(X_eval,Ye_real,'b-','LineWidth',2,'DisplayName','Expensive');
    hold on
    xlabel('X');
    ylabel('Y');
    title('CoKriging Prediction');
    X_krig = X_eval;
    Y_krig = Y_hat;
end

% Plot Kriging Model
function [X_krig, Y_krig] = PlotKrigingModel(name, xmin, xmax, neval, X, Y, theta, b, mu_hat, n_samp)
    X_eval = linspace(xmin,xmax,neval);
    Y_hat = zeros(neval,1);
    for ind = 1:neval
        Y_hat(ind,1) = KrigingPrediction(X_eval(ind), X, theta, b, mu_hat, n_samp);
    end
    figure('Name',name);
    plot(X_eval,Y_hat,'r-','LineWidth',2);
    hold on
    scatter(X,Y);
    xlabel('X');
    ylabel('Y');
    title('Kriging Prediction');
    X_krig = X_eval;
    Y_krig = Y_hat;
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

function [X_real, Yc_real, Ye_real] = PlotMultiFidelityFunctions(n_test)
    X_real = linspace(0,1,n_test);
    Yc_real = CheapTestFunction(X_real);
    Ye_real = ExpensiveTestFunction(X_real);
    figure('Name','Multi-Fidelity Functions')
    plot(X_real,Yc_real,'r-','LineWidth',2,'DisplayName','Cheap');
    hold on
    plot(X_real,Ye_real,'b-','LineWidth',2,'DisplayName','Expensive');
    hold on
    xlabel('X');
    ylabel('Y');
    title('Multi-Fidelity Functions');
end

function plotItAll(Xc, Yc, Xe, Ye, X_real, Yc_real, Ye_real, X_ke, Y_ke, X_kc, Y_kc, X_ck, Y_ck)
    figure('Name','MFS With Cokriging Method');
    title('MFS With Cokriging Method')
    xlabel('X');
    ylabel('Y');

    % Plot Expensive Fxn, Data Points, and Kriging model
    plot(X_real,Ye_real,'b-','LineWidth',2,'DisplayName','Expensive');
    hold on
    plot(X_ke,Y_ke,'b--','LineWidth',2,'DisplayName','Expensive Kriging');
    hold on
    plot(Xe,Ye,'o','Color','b','DisplayName','Expensive Data');
    hold on

    % Plot Cheap Fxn, Data Points, and Kriging model
    plot(X_real,Yc_real,'r-','LineWidth',1,'DisplayName','Cheap');
    hold on
    plot(X_kc,Y_kc,'r:','LineWidth',1,'DisplayName','Cheap Kriging');
    hold on
    plot(Xc,Yc,'diamond','Color','r','DisplayName','Cheap Data');
    hold on

    % Plot the Cokriging data fit
    plot(X_ck,Y_ck,'k-.','LineWidth',1,'DisplayName','Cokriging');
end