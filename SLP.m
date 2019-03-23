clc;
clear all;
data = readtable('iris.csv');
d = table2array(data(:,1:4));
label = data(:,5);
Y = zeros(150,1);
%disp(label);

for i= 1:150
    if strcmp('Iris-setosa', label{i,1})
        Y(i,1:3)=[1 0 0];
    elseif strcmp('Iris-versicolor', label{i,1})
        Y(i,1:3)=[0 1 0];
    elseif strcmp('Iris-virginica', label{i,1})
        Y(i,1:3)=[0 0 1]; 
    end
end

data = [d Y];
m = size(data,1);
n = 5;
n_classes = 3;
theta = rand(5,n_classes);
maxIter = 3000;
D = [10;20;30;40;50;60];
Train_acc = zeros(size(D,1),1);
Test_acc = zeros(size(D,1),1);
fid = fopen('output.csv','wt');
fprintf(fid,'%s,%s,%s\n','Percentage of training data','Training Accuracy','Testing Accuracy');

for ep=1:size(D,1)
    train_acc=zeros(10,1);
    test_acc = zeros(10,1);
    for sim=1:10
        
        x = D(ep)*50/100;

        A = randperm(50,x);
        B = 50+randperm(50,x);
        C = 100+randperm(50,x);



        train_X = data([A;B;C],1:4);
        train_Y  = data([A;B;C],5:7);

        A = [A B C];
        test_X =[];
        test_Y = [];
        for i=1:size(data)
            if ~ismember(A,i)
                test_X = [test_X ;data(i,1:4)];
                test_Y = [test_Y ;data(i,5:7)];
            end
        end


        train_X = [ones(x*3,1) train_X];
        test_X  = [ones(size(data,1)-x*3,1) test_X];
        error = zeros(maxIter,1);
        lr = 0.01;
        for i=1:maxIter
            pred = train_X*theta;

            theta = theta - (train_X'*(pred-train_Y)*lr/x/3);
            error(i) = sum(sum((pred-train_Y)).^2)/x/3;

            if error(i) < 0.0001
                break;
            end
        end

        %disp(theta);



        %plot(1:size(error,1),error,'color','r');

        pred1 = test_X*theta;
        pred2 = train_X*theta;  
        c = 0;
        [~,idx]= max(pred2,[],2);
        for j = 1:size(pred2,1)
            if train_Y(j,idx(j))==1
                c = c+1;
            end
        end

        %disp(c*100/x/3);
        train_acc(sim,1) = c*100/x/3;
        c = 0;
        [val,idx]= max(pred1,[],2);
        for j = 1:size(pred1,1)
            if test_Y(j,idx(j))==1
                c = c+1;
            end
        end
        x = size(data,1)-(x*3);
        %disp(c*100/x);
        test_acc(sim,1)= c*100/x;
        
    end
    %disp(train_acc);
    %disp(test_acc);
    txt = sprintf('Train & Test Accuracy for %d%% of training data',D(ep));
    disp(txt);
    disp(mean(train_acc(:,1)));
    disp(mean(test_acc(:,1)));
    txt = sprintf('%d%%',D(ep));
    Train_acc(ep,1) = mean(train_acc(:,1));
    Test_acc(ep,1) =  mean(test_acc(:,1));
    
    fprintf(fid,'%s,%f,%f\n',txt,mean(train_acc(:,1)),mean(test_acc(:,1)));
    
end
fclose(fid);
figure;
plot(D,Train_acc,'color','r');
xlabel('Percentage of Data');
ylabel('Accuracy');

hold on;
plot(D,Test_acc,'color','g');
xlabel('Percentage of Data');
ylabel('Accuracy');
title('Training / Testing Accuracy');







    

    





