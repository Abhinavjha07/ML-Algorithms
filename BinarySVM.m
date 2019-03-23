clc;
clear all;
data = readtable('iris.csv');
d = table2array(data(:,1:4));
label = data(:,5);
Y = zeros(150,1);
%disp(label);

for i= 1:150
    if strcmp('Iris-setosa', label{i,1})
        Y(i,1)=2;
    elseif strcmp('Iris-versicolor', label{i,1})
        Y(i,1:3)=0;
    elseif strcmp('Iris-virginica', label{i,1})
        Y(i,1:3)=1; 
    end
end

data = [d Y];
%disp(data);
data = data(51:150,:);
data = data(randperm(size(data,1)),:);
accuracy = zeros(5,1);
for sim=1:5
    
    train_X =[];
    train_Y = [];
    %testing and training split
    test_X = data(((sim-1)*20+1):(sim*20),1:4);
    test_Y = data(((sim-1)*20+1):(sim*20),5:5);
    A = (((sim-1)*20+1):(sim*20));
    for i=1:size(data,1)
        if ~ismember(A,i)
            train_X = [train_X ;data(i,1:4)];
            train_Y = [train_Y ;data(i,5:5)];
        end
    end
    
    
    SVMModel = fitcsvm(train_X,train_Y);
    sv = SVMModel.SupportVectors;
    
    figure
    gscatter(train_X(:,3),train_X(:,4),train_Y)
    hold on
    plot(sv(:,3),sv(:,4),'ko','MarkerSize',10)
    legend('versicolor','virginica','Support Vector')
    hold off
    
    
        
    [prediction,score] = predict(SVMModel,test_X);
    
        
    %disp(prediction)
    confusion_mat = confusionmat(test_Y,prediction);
    
    accuracy(sim,1) = sum(max(confusion_mat,[],2))*100/(size(test_X,1));
    
end
disp('Accuracy : ');
disp(mean(accuracy))
