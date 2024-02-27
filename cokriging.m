%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Co-Kriging   %%%%%%%%%%%%%%%%
%%%%%%%%%% Troy Zangle %%%%%%%%%%%%


%% Set up some sampling constants for graph purposes
n_test = 300;

%% Model Inputs
% Expensive model samples
n_e = 4;
% Cheap model samples
n_c = 11;
% Function set choice, corresponds to a predefined set of LF/HF fxns
fxn_ind = 1;

%% Generate inputs
%Get bounds
[xmin,xmax,dim] = FunctionBounds(fxn_ind);

% Sample the expensive and cheap functions
%Xe = EqualSpacedSamples(n_e,xmin,xmax,dim);
Xe = RandSpacedSamples(n_e,xmin,xmax);
Xe = sort(Xe,1,"ascend");
Ye = ExpensiveTestFunction(fxn_ind,Xe);

%Xc = RandSpacedSamples(n_c,xmin,xmax);
Xc = RandSpacedSamples(n_c,xmin,xmax);
Xc = sort(Xc,1,"ascend");
Yc = CheapTestFunction(fxn_ind,Xc);



% Debugging Step
% Plot the expensive and cheap functions
%[X_real,Yc_real,Ye_real]=PlotMultiFidelityFunctions(fxn_ind,n_test);

%% Solve for the Co-kriging model
%[inv_C,theta_c,theta_d,sigma_c,sigma_d,rho,mu_hat_ck,nc,ne]=CokrigingSetup(Xc,Xe,Yc,Ye,xmin,xmax,false,true,fxn_ind);

%% Organized output methods
%TwoDimCoKriging();

GenerateCoKrigingModel(1,11,4)

%ForresterPaperResults();

%% Output Functions %%%%%%%%%%%%%
% Functions geared towards generating the plots for the report
function ForresterPaperResults()
    n_test = 300;
    fxn_ind = 1;
    [xmin,xmax,~] = FunctionBounds(fxn_ind);
    
    PlotMultiFidelityFunctions(fxn_ind,n_test);

    Xe = [0.0;0.4;0.6;1.0];
    Ye = ExpensiveTestFunction(fxn_ind,Xe);

    Xc = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
    Xc = transpose(Xc);
    Yc = CheapTestFunction(1,Xc);    

    rho_fixed = true;
    verbose = true;
    CokrigingSetup(Xc,Xe,Yc,Ye,xmin,xmax,rho_fixed,verbose,fxn_ind);
end

function TestDeltaNc(fxn_ind)

    n_test = 300;
    %% Model Inputs
    % Expensive model samples
    n_e = 4;

    %Get bounds
    [xmin,xmax,dim] = FunctionBounds(fxn_ind);

    % Generate evaluation locations
    X_eval = linspace(xmin,xmax,n_test);
    Yhat = zeros(n_test,10);
    error = zeros(10,1);

    % Generate Expensive Dataset for plotting comparison
    Y_reale = ExpensiveTestFunction(fxn_ind,X_eval);

    % Cheap model samples
    ind=1;
    for n_c = 6:15        
        % Sample the expensive and cheap functions
        Xe = EqualSpacedSamples(n_e,xmin,xmax,dim);
        Ye = ExpensiveTestFunction(fxn_ind,Xe);
        
        Xc = EqualSpacedSamples(n_c,xmin,xmax,dim);
        Yc = CheapTestFunction(fxn_ind,Xc);
        
        % Generate the cokriging model
        [inv_C,theta_c,theta_d,sigma_c,sigma_d,rho,mu_hat_ck,nc,ne]=CokrigingSetup(Xc,Xe,Yc,Ye,xmin,xmax,false,false,fxn_ind);
        Yck = [Yc;Ye];
        error(ind) = CalculateModelError( Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne, fxn_ind,n_test);
        for i = 1:n_test
            Yhat(i,ind) = CoKrigingPrediction(X_eval(i), Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
        end
        ind = ind+1;
    end    
    % Plot all of the co-kriging models
    figure('Name','Co-kriging with changing cheap evaluations');
    plot(X_eval,Y_reale,'b-','LineWidth',2,'DisplayName','Expensive')
    hold on
    plot(X_eval,Yhat(:,7),'r--','LineWidth',2,'DisplayName','nc = 12');
    hold on
    plot(X_eval,Yhat(:,8),'g:','LineWidth',2,'DisplayName','nc = 13');
    hold on
    plot(X_eval,Yhat(:,10),'k:','LineWidth',2,'DisplayName','nc = 15');
    hold on
    xlabel('X')
    ylabel('Y')

    % Plot the error
    n = linspace(6,15,10);
    figure('Name','Co-kriging RMS Error');
    plot(n,error,'b-','LineWidth',2);
    hold on
    xlabel('nc')
    ylabel('RMS Error')
end

function GenerateCoKrigingModel(fxn_ind,nc,ne)

    n_test = 300;

    %% Get bounds
    [xmin,xmax,dim] = FunctionBounds(fxn_ind);

    %% Generate evaluation locations
    X_eval = linspace(xmin,xmax,n_test);
    X_eval = transpose(X_eval);
    Yhat = zeros(n_test,1);

    %% Generate Expensive Dataset for plotting comparison
    Y_reale = ExpensiveTestFunction(fxn_ind,X_eval);

    %% Cheap model samples    
    % Sample the expensive and cheap functions
    % Sample the expensive and cheap functions
    Xe = EqualSpacedSamples(ne,xmin,xmax,dim);
    %Xe = RandSpacedSamples(ne,xmin,xmax);
    Xe = sort(Xe,1,"ascend");
    Ye = ExpensiveTestFunction(fxn_ind,Xe);
    
    Xc = EqualSpacedSamples(nc,xmin,xmax,dim);
    %Xc = RandSpacedSamples(nc,xmin,xmax);
    Xc = sort(Xc,1,"ascend");
    Yc = CheapTestFunction(fxn_ind,Xc);
    
    %% Generate the cokriging model
    [inv_C,theta_c,theta_d,sigma_c,sigma_d,rho,mu_hat_ck,nc,ne]=CokrigingSetup(Xc,Xe,Yc,Ye,xmin,xmax,false,false,fxn_ind);
    Yck = [Yc;Ye];
    error = CalculateModelError( Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne, fxn_ind,n_test);
    for i = 1:n_test
        Yhat(i) = CoKrigingPrediction(X_eval(i), Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
    end

    %% Plot all the co-kriging model
    figure('Name','Co-kriging Fit');
    plot(X_eval,Y_reale,'b-','LineWidth',2,'DisplayName','Expensive')
    hold on
    plot(X_eval,Yhat,'r--','LineWidth',2,'DisplayName','CK Model');
    hold on
    xlabel('X')
    ylabel('Y')

    %disp('The total error is: ', error)
end

function TwoDimCoKriging()
    %% Set up some sampling constants for graph purposes
    n_test = 300;

    %% Model Inputs
    % Expensive model samples
    n_e = 4;
    % Cheap model samples
    n_c = 11;
    % Function set choice, corresponds to a predefined set of LF/HF fxns
    fxn_ind = 5;
    
    %% Generate inputs
    %Get bounds
    [xmin,xmax,dim] = FunctionBounds(fxn_ind);
    
    % Sample the expensive and cheap functions
    Xe = EqualSpacedSamples(n_e,xmin,xmax,dim);
    Ye = ExpensiveTestFunction(fxn_ind,Xe);
    
    Xc = EqualSpacedSamples(n_c,xmin,xmax,dim);
    Yc = CheapTestFunction(fxn_ind,Xc);

    PlotMultiFidelityFunctions(fxn_ind,n_test);

    %% Solve for the Co-kriging model
    CokrigingSetup(Xc,Xe,Yc,Ye,xmin,xmax,false,true,fxn_ind);
end

%% Functions %%%%%%%%%%%%%%%%%%%%

% Space Samples
function Xsamp = EqualSpacedSamples(n,xmin,xmax,dim)
    grd = cell(1,dim);
    Xsamp = zeros(n^dim,dim);
    for i = 1:dim
        step = (xmax(i)-xmin(i))/(n-1);
        [grd{:}] = ndgrid(xmin(i):step:xmax(i));
        Xsamp(:,i) = grd{i}(:).';
    end
end

% Random spaced samples only works for 1dim currently
function Xsamp = RandSpacedSamples(n,xmin,xmax)
    Xsamp = (xmax-xmin).*rand(n,1) + xmin;
end
% Co-Kriging Setup Function
function [inv_C,theta_c,theta_d,sigma_c,sigma_d,rho,mu_hat_ck,nc,ne]=CokrigingSetup(Xc,Xe,Yc,Ye,xmin,xmax,rho_fixed,Verbose,fxn_ind)

    %% Solve for the kriging model of the "Cheap" data    
    [theta_c, b_c, mu_hat_c, nc, sigma_c, psic_c, invpsi_c, ~] = KrigingSetup(Xc,Yc);

    %%%% Debugging step %%%%%%%%%%%%%%%%%%%%
    writematrix(psic_c,'cheapPsi');
    writematrix(invpsi_c,'cheapInvPsi');
    %%%% Debugging step %%%%%%%%%%%%%%%%%%%%

    % Debugging step
    % Plot the kriging model of the cheap function
    if Verbose
        PlotKrigingModel('Cheap Kriging Model', xmin, xmax, 300, Xc, Yc, theta_c, b_c, mu_hat_c, nc);
    end
    
    %% ONLY FOR DEBUG solve for the kriging model of the "Expensive" data
    if Verbose        
        [theta_e, b_e, mu_hat_e, ne, ~, psi_e, invpsi_e, ~] = KrigingSetup(Xe,Ye);

        %%%% Debugging step %%%%%%%%%%%%%%%%%%%%
        writematrix(psi_e,'expPsi');
        writematrix(invpsi_e,'expInvPsi');
        %%%% Debugging step %%%%%%%%%%%%%%%%%%%%

        % Plot the kriging model of the cheap function    
        PlotKrigingModel('Expensive Kriging Model', xmin, xmax, 300, Xe, Ye, theta_e, b_e, mu_hat_e, ne);
    end

    %% Solve for the kriging model of the difference 
    %%      between the Expensive and Cheap data sets
    
    % Start by finding the value of yc_hat at all values of Xe
    ne = size(Xe,1);
    yc_hat_xe = zeros(ne,1);
    for ind = 1:ne
        yc_hat_xe(ind,1) = KrigingPrediction(Xe(ind), Xc, theta_c, b_c, mu_hat_c, nc);
    end
    
    [theta_d,b_d,mu_hat_d,rho,d,ne,sigma_d,psid_e,invpsid_e,~] = DiffKrigingSetup(Xe,yc_hat_xe,Ye,rho_fixed);
    
    %%%% Debugging step %%%%%%%%%%%%%%%%%%%%
    %writematrix(psid_e,'diffPsi');
    %writematrix(invpsid_e,'diffInvPsi');
    %%%% Debugging step %%%%%%%%%%%%%%%%%%%%

    % Debugging Step
    if Verbose
        % Plot the kriging model of the difference
        PlotKrigingModel('Diff Kriging Model', xmin, xmax, 300, Xe, d, theta_d, b_d, mu_hat_d, ne)
    end

    %% Assemble the Coefficients for Cokriging Evaluation
    % Calculate the psi_c(Xc,Xe) and psi_c(Xe,Xc) terms first
    [psic_ec,psic_ce] = CalculatePsiMatrix_MF(theta_c,Xe,Xc,ne,nc);
    
    % Calculate psic(Xe,Xe)
    pre_psi_e = CalculatePrePsiMatrix(Xe,ne);
    [psic_e, ~] = CalculatePsiMatrix(pre_psi_e, theta_c);
    
    % Assemble the C matrix from the above parts
    [Cmat, inv_C] = CalculateCovariance(ne, nc, sigma_c, psic_c, rho, psic_ce, psic_ec, psic_e, sigma_d, psid_e);

    %%%% Debugging step %%%%%%%%%%%%%%%%%%%%
    writematrix(Cmat,'CMat')
    writematrix(inv_C,'invCMat');
    %%%% Debugging step %%%%%%%%%%%%%%%%%%%%

    % Assemble the combined y vector of cheap and expensive values
    Yck = [Yc;Ye];
    
    %% Solve for the Co-kriging mean value
    % Find the cokriging mean value
    mu_hat_ck = CalculateKrigingMean(inv_C, Yck, nc+ne);    

    % Plot Comparison for debugging
    if Verbose
        PlotCoKrigingModel('CoKriging Model', xmin, xmax, 300, Xc, Yc, Xe, Ye, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne, fxn_ind, rho_fixed);
        %plotItAll(Xc, Yc, Xe, Ye, X_real, Yc_real, Ye_real, X_ke, Y_ke, X_kc, Y_kc, X_outck, Y_outck);
    end
end

% Kriging Setup Function
function [theta,b,mu_hat,n,sigma,psi,inv_psi,pre_psi] = KrigingSetup(X,Y)
    n = size(X,1);
    pre_psi = CalculatePrePsiMatrix( X, n);
    
%     PlotMLE(pre_psi,Y,n);
    
    fun = @(theta_solved)(-CalculateMLE(pre_psi,theta_solved,Y,n));
    theta = fminsearch(fun,2.0);
    
    if theta < 1e-3
        theta = 1e-3;
    end

    % Calculate sigma^2, mu, and b constants
    [psi,inv_psi] = CalculatePsiMatrix(pre_psi,theta);
    mu_hat = CalculateKrigingMean( inv_psi, Y, n);
    sigma = CalculateSigma(inv_psi,mu_hat,Y,n);
    b = CalculateConstantB(inv_psi, Y, mu_hat, n);
end

% Difference Kriging Setup
% This includes a scaling factor output that is solved for
%   in the theta solver block
function [theta,b,mu_hat,rho,d,n,sigma,psi,inv_psi,pre_psi] = DiffKrigingSetup(X,Yce,Ye,rho_fixed)
    n = size(X,1);

    % Calculate the distance matrix
    pre_psi = CalculatePrePsiMatrix(X, n);   
    
    fun = @(solved)(-CalculateMLEDiff(pre_psi,solved(1),Ye,Yce,solved(2),n));
    solved = [1.0, 1.0];
    options = optimset('MaxFunEvals',10000);
    output = fminsearch(fun,solved,options);
    theta = output(1);
    if theta < 1e-3
        theta = 1e-3;
    end
    rho = output(2);
    
    % Sets rho to value used in contrived forrester paper output
    if rho_fixed
        rho = 1.87;
        fun = @(solved)(-CalculateMLEDiff(pre_psi,solved,Ye,Yce,rho,n));
        theta = fminsearch(fun,1.0);
    end

    % Calculate sigma, mu, and b constants
    [psi,inv_psi] = CalculatePsiMatrix(pre_psi,theta);
    d = Ye-rho*Yce;
    mu_hat = CalculateKrigingMean( inv_psi, d, n);
    sigma = CalculateSigma(inv_psi,mu_hat,d,n);
    b = CalculateConstantB(inv_psi, d, mu_hat, n);
end

% Calculate sample points from the specified "Expensive" test function
% This function example is from Forrester 2008
function y_samp = ExpensiveTestFunction(fxn_ind,Xin)
    if fxn_ind == 1
        y_samp = (6.*Xin-2).^2.*sin(12.*Xin-4);
    elseif fxn_ind == 2
        y_samp = sin(2*pi.*(Xin-0.1))+Xin.^2;
    elseif fxn_ind == 3
        y_samp = Xin.*sin(Xin)./10;
    elseif fxn_ind == 4
        y_samp = cos(3.5*pi.*Xin).*exp(-1.4.*Xin);
    elseif fxn_ind == 5
        y_samp = 4-2.1.*Xin(:,1).^4+(1/3).*Xin(:,1).^6+Xin(:,1).*Xin(:,2)-4.*Xin(:,2).^2+4.*Xin(:,2).^4;
    else
        disp('invalid function index');
    end
end

% Calculate sample points from the specified "Cheap" test function
% This function example is from Forrester 2008
function y_samp = CheapTestFunction(fxn_ind,Xin)
    if fxn_ind == 1
        y_samp = 0.5*ExpensiveTestFunction(fxn_ind,Xin)+10*(Xin-0.5)-5;
    elseif fxn_ind == 2
        y_samp = sin(2*pi.*(Xin-0.1));
    elseif fxn_ind == 3
        y_samp = Xin.*sin(Xin)./10 + Xin./10;
    elseif fxn_ind == 4
        y_samp = cos(3.5*pi.*Xin).*exp(-1.4.*Xin) + 0.75.*Xin.^2;
    elseif fxn_ind == 5
        y_samp = 4-0.5.*Xin(:,1).^4+(0.04).*Xin(:,1).^6+0.5.*Xin(:,1).*Xin(:,2)-1.96.*Xin(:,2).^2+0.96.*Xin(:,2).^4;
    else
        disp('invalid function index');
    end
end

% Function returns the bounds and input dimensions of the function of interest
function [xmin,xmax,dim] = FunctionBounds(fxn_ind)
    if fxn_ind == 1
        dim = 1;
        xmin = 0;
        xmax = 1;
    elseif fxn_ind == 2
        dim = 1;
        xmin = 0;
        xmax = 1;
    elseif fxn_ind == 3
        dim = 1;
        xmin = 0;
        xmax = 10;
    elseif fxn_ind == 4
        dim = 1;
        xmin = 0;
        xmax = 1;
    elseif fxn_ind == 5
        xmin = zeros(2,1);
        dim = 2;
        xmin(1) = -2;
        xmin(2) = -2;
        xmax(1) = 2;
        xmax(2) = 2;
    end
end

function y_samp = LinearCheapFunction(x_samp)
    x_lo = 0;
    x_hi = 1;
    y_lo = ExpensiveTestFunction(1,0);
    y_hi = ExpensiveTestFunction(1,1);
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
            diffX = X(i,:) - X(j,:);
            pre_psi_ij(i,j)=dot(diffX,diffX);
        end
    end
end

% Calculate the distance relation between multi-fidelity datasets
function [psi_ec, psi_ce] = CalculatePsiMatrix_MF( theta_c, Xe, Xc, ne, nc)
    psi_ec = zeros(ne,nc);
    psi_ce = zeros(nc,ne);
    for i = 1:ne
        for j = 1:nc
            diffX = Xe(i,:)-Xc(j,:);
            psi_ec(i,j) = exp(-theta_c*dot(diffX,diffX));
            psi_ce(j,i) = psi_ec(i,j);
        end
    end
end

% Calculate the psi correlation matrix
function [psi_ij,inv_psi_ij] = CalculatePsiMatrix( pre_psi_ij, theta)
    psi_ij = exp(-theta * pre_psi_ij);
    disp(psi_ij);
    inv_psi_ij = inv(psi_ij);
    disp(inv_psi_ij);
    disp(inv_psi_ij(1,1));
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
        diffX = x_new-Xc(i,:);
        psic_c_new(i,1) = exp(-theta_c*dot(diffX,diffX));
    end
    for i = 1:ne
        diffX = x_new-Xe(i,:);
        psic_e_new(i,1) = exp(-theta_c*dot(diffX,diffX));
        psid_e_new(i,1) = exp(-theta_d*dot(diffX,diffX));
    end
    c_vec = zeros(nc+ne,1);
    c_vec(1:nc,1) = rho*sigma_c*psic_c_new;
    c_vec(nc+1:ne+nc,1) = rho^2*sigma_c*psic_e_new + sigma_d*psid_e_new;    
    y_pred = mu_hat_ck + transpose(c_vec)*inv_C*(Yck-ones(nc+ne,1)*mu_hat_ck);
end

% Plot Co-kriging model
function [X_krig, Y_krig] = PlotCoKrigingModel(name, xmin, xmax, neval, Xc, Yc, Xe, Ye, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne, fxn_id, rho_fixed)    
    dim = size(Xc,2);
    if dim == 1
        X_eval = linspace(xmin(1),xmax(1),neval);
        Y_reale = ExpensiveTestFunction(fxn_id,X_eval);
        Y_realc = CheapTestFunction(fxn_id,X_eval);
        Y_hat = zeros(neval,1);
        for ind = 1:neval
            Y_hat(ind,1) = CoKrigingPrediction(X_eval(ind), Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
        end
        figure('Name',name);
        plot(X_eval,Y_reale,'b-','LineWidth',2,'DisplayName','Expensive')
        hold on        
        plot(X_eval,Y_realc,'r-','LineWidth',2,'DisplayName','Cheap')
        hold on        
        plot(X_eval,Y_hat,'g--','LineWidth',2,'DisplayName','Co-kriging');
        hold on
        if rho_fixed
            csv_data = readmatrix('cokrigingdata.txt');
            plot(csv_data(:,1),csv_data(:,2),'k:','LineWidth',2,'DisplayName','Forrester');
        end
        % Plot the cheap and expensive data sets
        scatter(Xc,Yc,'DisplayName','Data C','ColorVariable','b');
        hold on
        scatter(Xe,Ye,'DisplayName','Data E');
        hold on
        xlabel('X');
        ylabel('Y');
        title('CoKriging Prediction');
    elseif dim == 2
        X_eval = zeros(neval*neval,2);
        x1_vec = linspace(xmin(1),xmax(1),neval);
        x2_vec = linspace(xmin(2),xmax(2),neval);
        Y_hat = zeros(neval,1);
        for col = 1:neval
            for row = 1:neval
                ind = row + (col-1)*neval;
                X_eval(ind,:) = [x1_vec(row),x2_vec(col)];
                Y_hat(ind,1) = CoKrigingPrediction(X_eval(ind,:), Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
            end
        end
        figure('Name',name);
        plot3(X_eval(:,1),X_eval(:,2),Y_hat,'bo','LineWidth',2,'DisplayName','Cokriging');
        hold on
        % Plot the cheap and expensive data sets
        scatter3(Xc(:,1),Xc(:,2),Yc,'DisplayName','Cheap');
        hold on
        scatter3(Xe(:,1),Xe(:,2),Ye,'DisplayName','Expensive');
        hold on
        xlabel('X');
        ylabel('Y');
        title('CoKriging Prediction');
    
    else
        disp('Cannot plot in greater than 3 dimensions')
    end
    X_krig = X_eval;
    Y_krig = Y_hat;
end

% Plot Kriging Model
function [X_krig, Y_krig] = PlotKrigingModel(name, xmin, xmax, neval, X, Y, theta, b, mu_hat, n_samp)
    dim = size(X,2);
    if dim == 1
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
    elseif dim == 2
        X_eval = EqualSpacedSamples(neval,xmin,xmax,dim);
        Y_hat = zeros(neval^2,1);
        for ind = 1:neval^2
            Y_hat(ind,1) = KrigingPrediction(X_eval(ind), X, theta, b, mu_hat, n_samp);
        end
        figure('Name',name);
        scatter3(X_eval(:,1),X_eval(:,2),Y_hat);
        hold on
        scatter3(X(:,1),X(:,2),Y);
        xlabel('X');
        ylabel('Y');
        title('Kriging Prediction');
        X_krig = X_eval;
        Y_krig = Y_hat;
    else
        disp('Cannot plot higher dimensions')
    end
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

function error = CalculateModelError( Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne, fxn_ind,n_test)
    error = 0;
    [xmin,xmax,~] = FunctionBounds(fxn_ind);
    X_eval = linspace(xmin(1),xmax(1),n_test);
    Y_reale = ExpensiveTestFunction(fxn_ind,X_eval);
    for ind = 1:n_test
       Y_hat = CoKrigingPrediction(X_eval(ind), Xc, Xe, Yck, inv_C, theta_c, theta_d, sigma_c, sigma_d, rho, mu_hat_ck, nc, ne);
       error = error + sqrt((Y_hat-Y_reale(ind))^2);
    end
    error = error/n_test;
end

function [X_real, Yc_real, Ye_real] = PlotMultiFidelityFunctions(fxn_ind, n_test)
    [xmin,xmax,dim] = FunctionBounds(fxn_ind);
    if dim == 1
        X_real = EqualSpacedSamples(n_test,xmin,xmax,dim);
        Yc_real = CheapTestFunction(fxn_ind,X_real);
        Ye_real = ExpensiveTestFunction(fxn_ind,X_real);
        figure('Name','Multi-Fidelity Functions')
        plot(X_real,Yc_real,'r-','LineWidth',2,'DisplayName','Cheap');
        hold on
        plot(X_real,Ye_real,'b-','LineWidth',2,'DisplayName','Expensive');
        hold on
        xlabel('X');
        ylabel('Y');
        title('Multi-Fidelity Functions');
    elseif dim == 2        
        X_real = EqualSpacedSamples(n_test,xmin,xmax,dim);
        Yc_real = CheapTestFunction(fxn_ind,X_real);
        Ye_real = ExpensiveTestFunction(fxn_ind,X_real);
        figure('Name','Multi-Fidelity Functions')
        scatter3(X_real(:,1),X_real(:,2),Yc_real,'DisplayName','Cheap');
        hold on
        scatter3(X_real(:,1),X_real(:,2),Ye_real,'DisplayName','Expensive');
        hold on
        xlabel('X');
        ylabel('Y');
        title('Multi-Fidelity Functions');
    else
        disp('Dim > 2: No visual available');
    end
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