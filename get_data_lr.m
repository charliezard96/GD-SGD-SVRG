function [x,y,beta] = get_data_lr()

whole_dataset= readtable('telescope_data.csv','HeaderLines',1);
whole_dataset= whole_dataset(:,2:end);
x=table2array(whole_dataset(:,1:end-1));
x(isnan(x))=0;
x= rescale(x,-1,1);
y_cat=table2array(whole_dataset(:,end));
y=zeros(size(y_cat));
for i = 1:1:size(y_cat,1)
    if y_cat{i}== 'g'
        y(i)=1;
    else
        y(i)=-1;
    end
end
xy_concat = [x,y];
random_xy = xy_concat(randperm(size(xy_concat, 1)), :);

x=random_xy(:,1:end-1);
y=random_xy(:,end);
y=y';

%calculation of L
% XtX = x*x';
% eigenvaluesOfX = eig(XtX);
% beta=max(eigenvaluesOfX);

%for speed's sake I avoid to use the above part of code
beta = 4.314929935996378e+04;



end