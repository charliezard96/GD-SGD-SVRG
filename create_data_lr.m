function [x,y]=create_data_lr(nsamp,nfeat)

y=-3.+2.*randi(2,1,nsamp);
x=zeros(nsamp,nfeat);
for i=1:nsamp
    check=y(i);
    if(check==1)
        x(i,:)=rand(1,nfeat);
    else
        x(i,:)=-rand(1,nfeat);
    end
end

end