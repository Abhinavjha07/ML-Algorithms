clc;
clear all;

data = load('vowel.csv');
train = data(1:528,:);
test = data(529:990,:);
train = train(randperm(size(train,1)),:);
train_X = train(1:528,4:13);
train_Y = train(1:528,14:14);

test_X = data(529:990,4:13);
test_Y = data(529:990,14:14);

n_classes = 11;
%disp(size(test_X));
for i=1:size(train_Y,1)
    train_Y(i,1) = train_Y(i,1)+1;
end

for i=1:size(test_Y,1)
    test_Y(i,1) = test_Y(i,1)+1;
end

Y = zeros(size(train_Y,1),n_classes);
for i=1:size(train_Y,1)
    Y(i,train_Y(i,1))=1;
end

train_Y = Y;
%disp(size(train_Y));
Y = zeros(size(test_Y,1),n_classes);
for i=1:size(test_Y,1)
    Y(i,test_Y(i,1))=1;
end

test_Y = Y;
%disp(train_Y);
n_features = 10;
max_itr = 1000;

error = zeros(max_itr,1);
theta = rand(n_features,n_classes);
lr = 0.1;
m = size(train_Y,1);
for i=1:max_itr
   pred = train_X*theta;
    
    theta = theta - (train_X'*(pred-train_Y)*lr/m);
    error(i) = sum(sum(((pred-train_Y)'*(pred-train_Y))/2/m));
    
    if error(i) <= 0.000001
        break
    end 
end


pred1 = train_X*theta;
c = 0;
[~,idx]= max(pred1,[],2);
for i=1:size(pred1,1)
    if train_Y(i,idx(i))==1
        c =c+1;
    end
end
train_acc = c*100/528;
disp('Training Accuracy : ')
disp(train_acc);

pred1 = test_X*theta;
c = 0;
[val,idx]= max(pred1,[],2);
for i=1:size(pred1,1)
    if test_Y(i,idx(i))==1
        c =c+1;
    end
end
test_acc = c*100/462;
disp('Testing Accuracy : ')
disp(test_acc);




