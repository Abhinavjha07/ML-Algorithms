clc;
clear all;
data = readtable('iris.csv');
d = table2array(data(:,1:4));
label = data(:,5);
Y = zeros(150,1);
%disp(label);
fid = fopen('output.csv','wt');
fprintf(fid,'%s,%s\n','Percentage of training data','Testing Accuracy');


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
D = [10;20;30;40;50;60];

Test_acc = zeros(size(D,1),1);
simulations = 1;
for ep=1:size(D,1)
    test_acc = zeros(simulations,1);
    for s=1:simulations
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


       
        net = feedforwardnet([10],'traingd');
        net.layers{1}.transferFcn = 'tansig'; 
        net.divideFcn='';
        %{
        net.divideParam.trainRatio = 1;
        net.divideParam.valRatio = 0;
        net.divideParam.testRatio = 0;
        %}
        net = configure(net,train_X',train_Y');
        net.trainParam.lr = 0.01;
        
        net = train(net,train_X',train_Y');
        pred = sim(net,test_X');
        pred = pred';
        [~,idx]= max(pred,[],2);
        c = 0;
        for j = 1:size(pred,1)
            if test_Y(j,idx(j))==1
                c = c+1;
            end
        end
        test_acc(s,1) = c*100/(150-3*x);
        %disp(test_acc(s,1));
    end
    disp('Accuracy after simulations : ')
    disp(test_acc);
    Test_acc(ep,1) = mean(test_acc(:,1));
end
disp('Test Accuracy : ');
disp(Test_acc);

for i=1:size(D,1)
    fprintf(fid,'%f,%f\n',D(i),Test_acc(i));
end

fclose(fid);


    