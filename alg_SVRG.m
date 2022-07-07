function [w, time_vec, h_vec, acc_vec]=alg_SVRG(x,y,maxit,nepochs,lambda,lc,wi,stopCond)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Implementation of the Stochastic Gradient Method
% for f(w)= summ in i=1 to m of log(1+exp(y_i*w^T*x^i))
%
%INPUTS:
%x: matrix
%y: class
%lambda: regularization
%lc: Lipschitz constant of the gradient
%maxit: maximum number of iterations
%nepochs: number of iterations per epochs
%
%OUTPUTS:
%w: last weight vector
%h: last iter objective func value
%time_vec: time vector (time of each iter)
%err_vec: least square error for iteration
%h_vec: objective func evoulution at each iter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m,n] = size(x);
w=wi;

h_vec=zeros(1,maxit);
time_vec=zeros(1,maxit);
acc_vec = zeros(1,maxit);

tic;
time_vec(1) = 0;

%smart computation of the objective function/gradient
h = loss_lr(x,y,w,lambda);
w_tilde = w;
w_k = w;
g_tilde=grad_lr(x,y,w_tilde,lambda);

it=1;

while (1)
    % stopping criteria and test for termination
    if (it>maxit)
        break;
    end
    if (stopCond)
       if(h<10e-6)
           break;
       end
    end
    
    %vectors updating
    if (it>1)
        time_vec(it) = toc;
    end
    
    % gradient evaluation
    ind=randi(m);
    g_tilde_k=grad_lr(x(ind,:),y(ind),w_tilde,lambda);
    g_ind_k=grad_lr(x(ind,:),y(ind),w_k,lambda);
    
    %alpha selection
    alpha = lc;
    w_k= w_k-alpha*(g_ind_k-g_tilde_k+(1/m*(g_tilde)));
    
    % update gradient at the end of epoch (10000 iterations per epoch)
    if(mod(it,nepochs)==0)
        w_tilde = w_k;
        h = loss_lr(x,y,w_k,lambda);
        g_tilde=grad_lr(x,y,w_tilde,lambda);
    end
    
    h_vec(it)=h;
    acc_vec(it)=get_accuracy_lr(w_k,x,y);
    it = it+1;    
end
w=w_k;

if(it<maxit)
    h_vec(it:maxit)=h_vec(it-1);
    acc_vec(it:maxit)=acc_vec(it-1);
    time_vec(it:maxit)=time_vec(it-1);
end

end