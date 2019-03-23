clc;
clear all;

data = readtable('iris.csv');
d = table2array(data(:,1:4));
label = data(:,5);
Y = zeros(150,3);
%disp(label);

for i= 1:150
    if strcmp('Iris-setosa', label{i,1})
        Y(i,1:3)=[1 0 0];
    elseif strcmp('Iris-versicolor', label{i,1})
        Y(i,1:3)=[0 1 0];
    elseif strcmp('Iris-virginica', label{i,1})
        Y(i,1:3)=[0 0 1]; 
    end
end

data = [d Y];
%disp(data);
data = data(randperm(size(data,1)),:);

X = data(:,1:4);
Y = data(:,5:7);

k = 4;
C = cluster(k,X);
disp(C);

sigma = zeros(k,1);
for i=1:k
    sum = 0;
    for j=1:k
        sum = sum + (norm(C(i,:) - C(j,:))^2);
    end  
    sigma(i,:) = sqrt((1/k)*(sum));
end

Phi_X = zeros(size(X,1),k);
for i=1:size(X,1)
    for j=1:k
        Phi_X(i,j) = exp(-(norm(X(i,:)-C(j,:)))^2/(2*sigma(j,:)*sigma(j,:)));
    end
end

%{
SVMModels = cell(3,1);
classes = unique(Y);
%disp(classes);
for j = 1:numel(classes)
    indx = zeros(size(Y,1),1);
    for i =  1:size(Y,1)
        if Y(i,:) == i
            indx(i,1) = 1;
        else
            indx(i,1) = 0;
        end
    end
    
    SVMModels{j} = fitcsvm(Phi_X,indx,'ClassNames',[false true],'Standardize',true,'KernelFunction','rbf','BoxConstraint',1);
end


for j=1:numel(classes)
    CVSVMModel = crossval(SVMModels{j})
end

%}
%disp(size(Y));
weights = SLP(Phi_X,k,size(Y,2),Y)

function x = SLP(X,k,n,Y)
lr = 0.01;
train_X = [ones(size(X,1),1) X];
train_Y = Y;
theta = rand(k+1,n);
maxIter = 1000;
error = zeros(maxIter,1);
for i=1:maxIter
            pred = train_X*theta;

            theta = theta - (train_X'*(pred-train_Y)*lr/size(X,1));
            error(i,1) = sum(sum((pred-train_Y)).^2)/size(X,1);

            if error(i,1) < 0.0001
                break;
            end
end

x = theta;
end





function c = cluster(k,X)
initial_c = X(randperm(size(X,1),k),:);
m = 2;
U = rand(size(X,1),k);
U = bsxfun(@rdivide, U, sum(U,2));
%disp(sum(U,2));
C = update(initial_c,U,m,X);

maxItr = 1000;

for it=1:maxItr
    
    U = findU(C,X,k,m);
    %disp(sum(U,2));
    %disp(U);
    new_C = update(C,U,m,X);
    x = 0;
    for i=1:k
        for j=1:size(C,2)
            if abs(new_C(i,j)-C(i,j)) <= 0.0001
                x =x+1;
            end
        end
    end
    if x == k*size(C,2)
        break
    end
    
    C = new_C;

end

c = C;

end




function x = update(c,U,m,X);

new_C = zeros(size(c,1),size(c,2));
U = U';
%disp(size(X))

for i=1:size(c,1)
    new_C(i,:) = (U(i,:).^m * X)/sum(U(i,:).^m);
end
%disp(new_C);

x = new_C;
end


function U = findU(C,X,k,m)

u = zeros(size(X,1),k);
for i=1:size(X,1)
    
    
    for j = 1:k
        sum = 0;
        for l = 1:k
            sum = sum + ((norm(X(i,:) - C(j,:))/norm(X(i,:) - C(l,:))))^(2/(m-1));
        end
        %disp(sum);
        u(i,j) = 1/sum;
    end
    
end

%disp(size(u))
U = u;
end