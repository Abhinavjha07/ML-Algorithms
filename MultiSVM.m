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

D = [30;40;50;60];
accuracy = zeros(size(D,1),1);
for ep=1:size(D,1)
        x = D(ep)*50/100;

        A = randperm(50,x);
        B = 50+randperm(50,x);
        C = 100+randperm(50,x);
        
        
        train_X = data([A;B;C],3:4);
        train_Y  = data([A;B;C],5:5);

        A = [A B C];
        test_X =[];
        test_Y = [];
        for i=1:size(data)
            if ~ismember(A,i)
                test_X = [test_X ;data(i,3:4)];
                test_Y = [test_Y ;data(i,5:5)];
            end
        end
        
        SVMModels = cell(3,1);
        classes = 3;


        for j = 1:classes
            Y = zeros(size(train_Y,1),1);
            for i=1:size(train_Y,1)
                if train_Y(i,1) == j
                    Y(i,1) = 1;
                end
            end
            
            SVMModels{j} = fitcsvm(train_X,Y,'ClassNames',[false true],'Standardize',true,'KernelFunction','rbf');
        end
        
        d = 0.01;
        [x1Grid,x2Grid] = meshgrid(min(test_X(:,1))-1:d:max(test_X(:,1))+1,min(test_X(:,2))-1:d:max(test_X(:,2))+1);
        xGrid = [x1Grid(:),x2Grid(:)];
        N = size(xGrid,1);
        scoreAll = zeros(N,classes);

        for j = 1:classes
            [~,score] = predict(SVMModels{j},xGrid);
            scoreAll(:,j) = score(:,2); 
        end
        [~,pred] = max(scoreAll,[],2);
        
        figure
        h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),pred,[0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
        hold on
        h(4:6) = gscatter(test_X(:,1),test_X(:,2),test_Y);
        
        hold off
        %disp(pred)
        scoreAll = zeros(size(test_X,1),classes);
        for j = 1:classes
            [~,score] = predict(SVMModels{j},test_X);
            scoreAll(:,j) = score(:,2); 
        end
        [~,pred] = max(scoreAll,[],2);
        confusion_mat = confusionmat(test_Y,pred)
        disp('Claass-wise Accuracy : ')
        disp(max(confusion_mat,[],2)*3*100/size(test_X,1))
        accuracy(ep,1) = sum(max(confusion_mat,[],2))*100/(size(test_X,1));
end
disp('Accuracies : ')
disp(accuracy)
disp('Mean Accuracy : ')
disp(mean(accuracy))