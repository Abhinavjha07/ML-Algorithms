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


n_classes = 3;
k = 5;
m = 2;
accuracy = zeros(5,1);
for sim=1:5
    train_X =[];
    train_Y = [];
    %testing and training split
    test_X = X(((sim-1)*30+1):(sim*30),1:4);
    test_Y = Y(((sim-1)*30+1):(sim*30),:);
    A = (((sim-1)*30+1):(sim*30));
    for i=1:size(X,1)
        if ~ismember(A,i)
            train_X = [train_X ;data(i,1:4)];
            train_Y = [train_Y ;data(i,5:5)];
        end
    end
    
    U_train = zeros(size(train_X,1),3);
    
    for i = 1:size(train_X,1)
        
        U_train(i,int8(train_Y(i,1))) = 1;
        
    end
    
    
    
    U_test = zeros(size(test_X,3));
    for i=1:size(test_X,1)
        count = zeros(3,1);
        indx = distance(test_X(i,:),train_X,k);
        %disp(indx);
        
        %finding memebership of testing data
        for l =1 : n_classes
            num = 0;
            denom = 0;
            for j = 1:size(indx,1)
                num = num + U_train(indx(j,1),l)*(norm(test_X(i,:) - train_X(indx(j,1),:)))^(2/(m-1));
                denom = denom + norm(test_X(i,:) - train_X(indx(j,1),:))^(2/(m-1));
            end
            
            U_test(i,l) = num/denom;
        end
    end
    
    %checking assignments
    assign = zeros(size(test_X,1),1);
    %disp(U_test);
    for i=1:size(U_test,1)
        [~,indx] = max(U_test(i,:));
        if(U_test(i,indx) >=0.7)
            assign(i) = indx;
        end
    end
    
    %finding accuracy
    count = 0;
    for i=1:size(assign,1)
        if assign(i,1) == test_Y(i,1)
            count = count+1;
        end
    end
    accuracy(sim,1) =  count*100/size(test_Y,1);
    count = 0;
    for i=1:size(assign,1)
        if assign(i,1) == 0
            count = count+1;
        end
    end
    
    disp('Unassigned : ');
    disp(count)
    
    disp('Accuracy of this simlation : ')
    disp(accuracy(sim,1));
end
disp('Mean Accuracy : ');
disp(mean(accuracy));
    
 
%returning the indx of k nearest training samples
function d = distance(c,train_X,k)

x = zeros(size(train_X,1),1);

for i=1:size(train_X,1)  
    x(i,1) = norm(c-train_X(i,:));
end

[~,indx] = mink(x,k);

d = indx;

end    
