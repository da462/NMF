function [W, H, alphas] = is_nmf_gibbs_IG_param(V, W, H)

% Gibbs converges on the order of a few thousand iterations 

% INPUTS
% V: positive matrix data (F x N)
% W: basis matrix initialisation (F x K)
% H: gains matrix initialisation (K x N)

% OUTPUTS
% W & H such that V \approx W*H
% alphas for parameter learning 

% Code based on the Gibbs Sampler outlined in the following paper
% https://www.irit.fr/Cedric.Fevotte/publications/proceedings/eusipco09a.pdf

%% DEFINE INITIAL PARAMETERS

% get the dimensions
[F,N] = size(V);
K = size(W,2);

% set the IG parameters
a = 10;

% modified c values for sampling
ch = zeros(K,F,N);
cw = zeros(K,F,N);

% number of runs
runs = 100;

% parameter learning
alphas = zeros(1,runs);

for iter=1:runs
    fprintf('I am on iteration %d \n',iter);

    % Compute data approximate (A) with current W and H
    A = W*H;

%% SAMPLE C

    % loop over every c_{f,n}
    for f=1:F
        for n=1:N

            % temporary (Kx1 vector)
            c = zeros(K,1);
            
            % find full conditional parameters 
            lam = W(f,1:(K-1))'.*H(1:(K-1),n); % (K-1)x1
            mu = (lam/A(f,n)) * V(f,n);
            Sig = diag(lam) -  (1/A(f,n) * lam)*lam';

            % sample from a complex multivariate normal (1xK-1) 
            sample = (1/sqrt(2))*(mvnrnd(zeros(K-1,1),Sig) + 1i*mvnrnd(zeros(K-1,1),Sig));
            % store c_{fn}^(i) as a column vector (Kx1)
            c(1:K-1) = mu + sample'; 
            
            % restrict final degree of freedom
            c(K) = V(f,n) - sum(c);

            % store values for W & H sampling
            cw(:,f,n) = (abs(c).^2)./H(:,n);
            ch(:,f,n) = (abs(c).^2)./W(f,:)';
            
        end        
    end

%% SAMPLE W & H

    % loop over all k values as done in the paper
    for k=1:K

        % sample h_{k,:}^(i)
        
        % sample from Generalized Inverse Gaussian full conditional
        H(k,1) = gigrnd(a-F,2*(a+1)/H(k,2),2*(sum(ch(k,:,1))),1);
        
        for n=2:(N-1)
             % sample from Generalized Inverse Gaussian full conditional
             % n-1 from this but n+1 from last?
             H(k,n) = gigrnd(-F,2*(a+1)/H(k,n+1),2*((a+1)*H(k,n-1) + sum(ch(k,:,n))), 1); 
        end
        
        % sample from Inverse Gamma full conditional
        H(k,N) = 1/(gamrnd( ( a + F ), 1/( (a+1)*H(k,n-1) + sum(ch(k,:,N)) ) ) );

        % sample w_{:,k}^(i)
        for f=1:F
            % sample from Inverse Gamma full conditional
            W(f,k) = 1/(gamrnd( N , 1/( sum(cw(k,f,:)) ) ) );
        end
        
    end 
    
    ch = zeros(K,F,N);
    cw = zeros(K,F,N);
    
%% SAMPLE alpha using the first row of H
    h1 = H(1:N-1);
    h2 = H(2:N);
    
    % sample from prior:
    g = gamrnd(8.64,1/48.2);
    z = 1/g;
    
    % sample from U[0,1]
    l_u = log(unifrnd(0,1));
    
    % forward direction
    q = (a*(N-1))*log(a+1) - (N-1)*log(gamma(a));
    s = -(a+1)* sum(h1./h2);
    t = log(1/gampdf(z,8.64,1/48.2));
    disp(t)
    l_for = q + s + t;
    
    % reverse direction
    q = (z*(N-1))*log(z+1) - (N-1)*log(gamma(z));
    s = -(z+1)* sum(h1./h2);
    t = log(1/gampdf(a,8.64,1/48.2));
    l_rev = q + s + t;
    
    % calculate the acceptance probability
    if l_u <= min(l_rev - l_for, 0)
        a = z;
    end

    disp(l_rev - l_for)
    
    alphas(iter) = a;
    disp(a)

end