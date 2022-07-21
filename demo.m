clear variables;
close all;
clc;

%% Create signal
addpath('wav_files')
addpath('ltfat')
addpath('is_nmf')
ltfatstart;

%[xx,Fs] = audioread('glockenspiel.wav');
[xx,Fs] = audioread('Piano.wav');
%[xx,Fs] = audioread('harpsichord2.wav'); %Fs is the sample rate

% load wav file and resample
new_Fs = 22050;
xx = resample(xx,new_Fs,Fs);
T = length(xx);
Time_axis = linspace(0,T/22050,T);

rng(0);
%sound(xx, new_Fs)

%% Add noise
snrlvl = 15;
sigma2 = var(xx)*10^(-snrlvl/10);
%sound(sqrt(sigma2)*randn(size(xx)), new_Fs)
x = xx + sqrt(sigma2)*randn(size(xx));
size(x)

snrin = calculate_snr(xx, x);

fprintf('SNRin is %d \n',snrin);

%% IS-NMF

% Gabor parameters
M = 1024;
a = M/2;
g = gabwin({'tight', 'hann'}, a, M);
G = dgtreal(x, g, a, M);
[nf,nt] = size(G);

% analysis coefficients given by gabor transform
% synthesis coefficients given by inverse gabor
op.analysis = @(x) dgtreal(x,g,a,M);
op.synthesis = @(x) idgtreal(x,g,a,M,T); 
XX = op.analysis(x);

% arbitrarily set to 10
% common sense suggests same number as number of notes
options.rank = 10;

% Initialise the approximation with the SVD
[U,D,V] = svd(XX);
V = D*V';
W_svd = abs(U(:,1:options.rank));
H_svd = abs(V(1:options.rank,:));
W_init = W_svd;
H_init = H_svd;


% IS NMF step
%W_IS = W_init;
%H_IS = H_init;
%[W_IS, H_IS] = is_nmf_mu(abs(XX).^2, 1000, W_init, H_init,5e-5);
%[W_IS, H_IS, cost] = is_nmf_ig(abs(XX).^2, 1000, W_init, H_init,10*ones(options.rank,1)); %arbitrarily set to 10
%[W_IS, H_IS, cost] = is_nmf_gig(abs(XX).^2, 1000,  W_init, H_init);
%[W_IS, H_IS] = is_nmf_gibbs(abs(XX).^2, W_init, H_init);
[W_IS, H_IS, alphas] = is_nmf_gibbs_IG_param(abs(XX).^2, W_init, H_init);
%[W_IS, H_IS, h] = is_nmf_gibbs_IG(abs(XX).^2, W_init, H_init);
%[W_IS, H_IS] = is_nmf_gibbs_GIG(abs(XX).^2, W_init, H_init);


%% Reconstruction

S = zeros(options.rank,T);
for k=1:options.rank
    XX1 = ( (W_IS(:,k)*H_IS(k,:))./(W_IS*H_IS+1e-6) ).*XX;
    s1 = op.synthesis(XX1);
    S(k,:) = s1(:)';
end

nrj = sum(abs(S).^2,2);
[~,Isort] = sort(nrj,'descend');
S = S(Isort,:);

W_IS = W_IS(:,Isort);
H_IS = H_IS(Isort,:);

figure
plot(H_IS(1,:));
title('Temporal structure of activation coefficients: row #1')
xlabel('Frame')
ylabel('Magnitude')

figure
hold on;
title('Reconstruction of 10 bases');
for k=1:options.rank
    s1 = S(k,:);
    subplot(ceil(options.rank/2),2,k);
    plot(Time_axis,s1);
    title(num2str(k))
    axis normal
    xlim([0,length(xx)/Fs]);
    ylim([-0.5,0.5]);
end

fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',10)

%% 
k = 1;
%sound(S(k,:), new_Fs)

%% 
k = 2;
%sound(S(k,:), new_Fs)

%% 
k = 3;
%sound(S(k,:), new_Fs)

%% 
k = 4;
%sound(S(k,:), new_Fs)

%% 
k = 5;
%sound(S(k,:), new_Fs)

%%
k = 6;
%sound(S(k,:), new_Fs)

%% 
k = 7;
%sound(S(k,:), new_Fs)

%% 
k = 8;
%sound(S(k,:), new_Fs)

%% 
k = 9;
%sound(S(k,:), new_Fs)

%% 
k = 10;
sound(S(k,:), new_Fs)

%%
K = options.rank;
indx = 1:K;
exclude = [10];
indx(exclude) = [];
sound(sum(S(indx,:)),new_Fs)
w = sum(S(indx,:))';

snrout = calculate_snr(xx, w);

fprintf('SNRout is %d \n',snrout);

%%
%close all;
%for k=1:10
%    plot(h(:,k))
%    hold on;
%end
%plot(h(:,100))
%hold off;

%%
%close all;
%x = linspace(101,115,15);

%for k=1:100
%   drawnow
%    pause(0.01)
%    plot(x,h(:,k))
%end