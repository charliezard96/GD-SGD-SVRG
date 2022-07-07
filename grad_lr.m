function [g]=grad_lr(x,y,w,lambda)
g=-sum(((y'.*x).*exp(-y'.*(x*w)))./(1+exp(-y'.*(x*w)))) + lambda*w';
g=g';
end