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

data = data(randperm(size(data,1)),:);
X = data(:,1:4);
Y = data(:,5:5);
k = 3;

assign = kmeans(X,k);

%disp(assign);

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

disp(confusion_mat);


accuracy = sum(max(confusion_mat))*100/size(data,1);
disp('Accuracy is : ');
disp(accuracy);