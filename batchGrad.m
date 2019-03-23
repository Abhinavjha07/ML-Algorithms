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
    temp = X*theta;
    
    theta = theta - (X'*(temp-Y)*lr/m);
    error(i) = ((temp-Y)'*(temp-Y))/2/m;
    
    if error(i) <= 0.0001
        break
    end
end

disp('Theta : ')
disp(theta);

disp('Errors : ')
disp(error);

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



   


