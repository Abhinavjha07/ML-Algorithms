clc;
clear all;
data = load('input.csv');
X = data(:,[1:(size(data,2)-1)]);
Y = data(:,[size(data,2)]);
n = size(X,2);
m = length(Y);
Y = reshape(Y,m,1);
theta = rand(n+1,1);
disp(theta);

iter = 10;
Cost = zeros(iter,1);
X = [ones(m,1) X];


error = zeros(iter,1);
lr = 0.01;
for i = 1:iter
    e = 0 ;
    for j= 1:m
        temp = X(j,:)*theta;
        theta = theta - (X(j,:)'*(temp-Y(j))*lr);
        e =e + ((temp-Y(j))*(temp-Y(j)));
    end
    e = e/2/m;
    error(i) = e;
    
    if error(i) <= 0.0001
        break
    end
   
end

disp('Theta : ')
disp(theta);

disp('Error : ')
disp(error(error > 0));

%{
disp('Prediction : (1,11)')
d = [1,1,11];
disp(d*theta);

disp('Prediction : (11,1)')
d = [1,11,1];
disp(d*theta);
%}

figure;
d = X * theta
scatter([1:5],Y);
hold on;
scatter([1:5],d);
figure(2);
plot([1:iter],error,'color','r')

