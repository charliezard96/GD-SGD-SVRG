function [h]=loss_lr(x,y,w,lambda)
h2=log(1+exp(-y'.*(x*w)));
h=sum(h2)+lambda/2*norm(w)^2;
end