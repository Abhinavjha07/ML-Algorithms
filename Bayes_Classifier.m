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

count = 0;
k = 3;
assign = zeros(size(test_X,1),1);

for i=1:size(test_X,1)
   z = findClass(train_X,train_Y,test_X(i,:),k);
   %disp(test_Y(i,1));
   assign(i,1) = z;
   if z == test_Y(i,1)
       count = count + 1;
   end
end

confusion_mat = confusionmat(test_Y,assign)


disp('Accuracy : ')
accuracy = sum(max(confusion_mat,[],2))*100/(size(test_data,1));
disp(accuracy);


function c = findClass(train_X,train_Y,x,k)

prob = zeros(1,k);
for i=1:k
    X = [];
    for j=1:size(train_Y,1)
        if train_Y(j,1) == i
            X = [X ; train_X(j,:)];
        end
    end
    
    mean = zeros(1,size(train_X,2));
    variance = zeros(1,size(train_X,2));
    
    for j=1:size(X,2)
        pd = fitdist(X(:,j),'Normal');
        mean(1,j) = pd.mu;
        variance(1,j) = pd.sigma;   
    end
    prob(1,i)=1;
    for j=1:size(x,2)
        prob(1,i) = prob(1,i)*((1/((2*3.14*((variance(1,j))^2))^(1/2))) * exp(-((x(1,j)-mean(1,j))^2)/(2*(variance(1,j))^2)));
    end
    
    prob(1,i) = prob(1,i)*(1/3);
end
    %disp(prob)
    [~,idx] = max(prob);
    c = idx;
end

