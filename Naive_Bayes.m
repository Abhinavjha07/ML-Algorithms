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
train_percent = 30;

x = train_percent*50/100;

A = randperm(50,x);
B = 50+randperm(50,x);
C = 100+randperm(50,x);

train_X = data([A;B;C],1:4);
train_Y  = data([A;B;C],5:5);
train_data = [train_X train_Y];

train_data = train_data(randperm(size(train_data,1)),:);
train_X = train_data(:,1:4);
train_Y = train_data(:,5:5);
A = [A B C];
test_X =[];
test_Y = [];
for i=1:size(data,1)
    if ~ismember(A,i)
        test_X = [test_X ;data(i,1:4)];
        test_Y = [test_Y ;data(i,5:5)];
    end
end

test_data = [test_X test_Y];

test_data = test_data(randperm(size(test_data,1)),:);
test_X = test_data(:,1:4);
test_Y = test_data(:,5:5);

Mdl = fitcnb(train_X,train_Y)
%Mdl.DistributionParameters
isLabels = predict(Mdl,test_X);
confusion_mat = confusionmat(test_Y,isLabels)
accuracy = sum(max(confusion_mat,[],2))*100/(size(test_data,1));
disp('Accuracy is : ');
disp(accuracy);


