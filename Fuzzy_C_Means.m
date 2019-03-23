clc;
clear all;

data = readtable('iris.csv');
d = table2array(data(:,1:4));
label = data(:,5);
Y = zeros(150,1);
%disp(label);

for i= 1:150
    if strcmp('Iris-setosa', label{i,1})
        Y(i,1)=1;
    elseif strcmp('Iris-versicolor', label{i,1})
        Y(i,1:3)=2;
    elseif strcmp('Iris-virginica', label{i,1})
        Y(i,1:3)=3; 
    end
end

data = [d Y];
%disp(data);
data = data(randperm(size(data,1)),:);

X = data(:,1:4);
Y = data(:,5:5);

k = 3;
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

assign = zeros(size(X,1),1);
U = findU(C,X,k,m);

for i=1:size(U,1)
    
    [~,indx] = max(U(i,:));

    assign(i) = indx;
end

%disp([U assign])
disp('Confusion Matrix : ');
confusion_mat = zeros(k,k);

for i=1:k
    class = zeros(k,1);
    for j=1:size(assign,1)
        if assign(j,1) == i
            for l =1:k
                if Y(j,1) == l
                    class(l,1) = class(l,1)+1;
                end
            end
        end
    end 
    confusion_mat(i,:) = class;
end

disp(confusion_mat');
accuracy = sum(max(confusion_mat,[],2))*100/size(data,1);
disp('Accuracy is : ');
disp(accuracy);

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


