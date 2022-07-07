function [accuracy]=get_accuracy_lr(w,x,y)
m = size(x,1);
preds=w'*x';
ress=y.*preds;
tmp = find(ress>0);
accuracy = size(tmp,2)/m; 
end