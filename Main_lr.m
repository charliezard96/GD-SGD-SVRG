%% Homework 1 -- Optimization for data science 2020 -- final version 17th may 2020
% developed by Andrea Michielon, Lisa Facciolo, Carlo Alberto Bernardini

%MAIN
clc, clear; rng(0);
%% dataset extraction
[x,y,beta]=get_data_lr();
m=size(x,1);
% w=ones(size(x,2),1);
w=[ones(size(x,2)/2,1);-ones(size(x,2)/2,1)];
%random
% w=(-1)+2.*rand(size(x,2),1);
% [x,y]=create_data_lr(10000,50); % sample/features
% w=[ones(floor(size(x,2)/2),1);-ones(ceil(size(x,2)/2),1)];
%% parameters for algs
lambda = 0.1;
%estimation of the L constant coming from L<=delta*beta+lambda
% lcGD = 4.291180019125216e+04;
compL = false;

lcGD = 2e7;
lcSGD = 0.0001;
lcSVRG = 0.001;

maxitGD = 30000;
maxitSGD = 200000;
maxitSVRG = 200000;
nepochsSVRG = 1000;

%Comment the three line below if dataset is not Telescope
stopCondGD = 1;
stopCondSGD = 1;
stopCondSVRG = 1;

%% runs of algs
disp('*****************');
disp('*  GD STANDARD  *');
disp('*****************');

[wGD, time_vec_GD, h_vec_GD, acc_vec_GD]=alg_GD(x,y,maxitGD,lambda,lcGD,w,stopCondGD);

disp('*****************');
disp('*  SGD STANDARD *');
disp('*****************');

[wSGD, time_vec_SGD, h_vec_SGD, acc_vec_SGD]=alg_SGD(x,y,maxitSGD,lambda,lcSGD,w,stopCondSGD);


disp('*******************');
disp('*  SVRG STANDARD  *');
disp('*******************');

[wSVRG, time_vec_SVRG, h_vec_SVRG, acc_vec_SVRG]=alg_SVRG(x,y,maxitSVRG,nepochsSVRG,lambda,lcSVRG,w,stopCondSVRG);

%% plots loss function
figure()
semilogy(time_vec_GD,h_vec_GD,'r-')
hold on;
semilogy(time_vec_SGD,h_vec_SGD,'b-')
hold on;
semilogy(time_vec_SVRG,h_vec_SVRG,'g-')
hold on;
xlabel('time');
ylabel('loss');
legend('GD', 'SGD','SVRG')

%% plots accuracy vs cputime
figure()
plot(time_vec_GD,acc_vec_GD,'r-')
hold on;
plot(time_vec_SGD,acc_vec_SGD,'b-')
hold on;
plot(time_vec_SVRG,acc_vec_SVRG,'g-')
hold on;
xlabel('time');
ylabel('acc');
legend('GD', 'SGD','SVRG')
%% Matrix P
% for gradient method
if(compL)
    P = zeros(m);
    max = 0;
    for i=1:m
        P(i,i) = 1/(1+exp(-y(i)*wGD'*x(i,:)'));
        if(P(i,i)>max)
            max=P(i,i);
        end
    end
    delta = max;
    L_upper_bound = delta*beta+lambda;
end
%% final accuracy
accGD=get_accuracy_lr(wGD,x,y);
accSGD=get_accuracy_lr(wSGD,x,y);
accSVRG=get_accuracy_lr(wSVRG,x,y);