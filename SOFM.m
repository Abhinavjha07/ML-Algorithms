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

%disp(Y);

k = 3;

n = size(X,2);
w = rand(k,n);

w_new = zeros(k,n);
w_prev = zeros(k,n);
lr = 0.1;
maxItr = 1000;
sigma = 1;
flag = 0;
for it = 1:maxItr
    i_lr = lr * exp(-it/maxItr);
    i_sigma = sigma * exp(-it/maxItr);
    for i=1:size(X,1)
        win = competition(X(i,:),w);

        h = cooperation(k,win,i_sigma);

        new_w = weight_update(w,h,i_lr,X(i,:));

        
        w = new_w;
    end
    
    w_new = w;
    
    x = 0;
    for i=1:k
        for j=1:size(w,2)
            if abs(w_new(i,j)-w_prev(i,j)) <= 0.0001
                x =x+1;
            end
        end
    end
    if x == k*size(w,2)
        break
    end
    
    w_prev = w;
end

assign = zeros(size(X,1),1);
for i=1:size(X,1)
    win = competition(X(i,:),w);
    assign(i,1) = win ;
end

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


disp(confusion_mat')

accuracy = sum(max(confusion_mat,[],2))*100/size(data,1);
disp('Accuracy is : ');
disp(accuracy);


function x = competition(x,initial_c)

dist = zeros(size(initial_c,1),1);
for i=1:size(initial_c,1)
    dist(i,:) = norm(x-initial_c(i,:));
end

[z,idx] = min(dist);
x = idx;
end


function z = cooperation(k,win,sigma)
co_op = zeros(k,2);
for i=1:k
    co_op(i,1)= i-1;
end
d = zeros(k,1);
for i=1:k
    d(i,:) = norm(co_op(i,:)-co_op(win,:));
end

h = zeros(k,1);

for i=1:k
    h(i,1) = exp(-d(i,1)^2/(2*sigma^2)); 
end

z = h;
end


function w = weight_update(w,h,lr,x)

z = zeros(size(w,1),size(w,2));

for i=1:size(w,1)
    z(i,:) = w(i,:) + (x-w(i,:)).*(lr*h(i,:));
    
end

w = z;
end

