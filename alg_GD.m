function [w, time_vec, h_vec, acc_vec]=alg_GD(x,y,maxit,lambda,lc,wi,stopCond)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Implementation of the Gradient Method
% for f(w)= summ in i=1 to m of log(1+exp(y_i*w^T*x^i))
%
%INPUTS:
%x: matrix
%y: class
%lambda: regularization
%lc: Lipschitz constant of the gradient
%maxit: maximum number of iterations
%
%OUTPUTS:
%w: last weight vector
%h: last iter objective func value
%time_vec: time vector (time of each iter)
%err_vec: least square error for iteration
%h_vec: objective func evoulution at each iter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w=wi;
h_vec=zeros(1,maxit);
time_vec=zeros(1,maxit);
acc_vec = zeros(1,maxit);

tic;
time_vec(1) = 0;

h = loss_lr(x,y,w,lambda);
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
    
    h_vec(it)=h;
    
    % gradient evaluation
    g=grad_lr(x,y,w,lambda);
    
    %fixed alpha
    alpha=1/lc;
    new_w= w-alpha*g;
    new_h = loss_lr(x,y,new_w,lambda);
    w= new_w;
    h = new_h;
    acc_vec(it)=get_accuracy_lr(w,x,y);
    it = it+1;
end

if(it<maxit)
    h_vec(it:maxit)=h_vec(it-1);
    acc_vec(it:maxit)=acc_vec(it-1);
    time_vec(it:maxit)=time_vec(it-1);
end

end