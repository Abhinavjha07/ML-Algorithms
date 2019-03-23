clc;
clear all;
load wine_dataset;
Y = vec2ind(wineTargets);


rng(0);
d = wineInputs';

Y = Y';



data = [d Y];

D = [30;40;50;60];
accuracy = zeros(size(D,1),1);
for ep=1:size(D,1)
        x = D(ep)*5/10;

        A = randperm(48,x);
        B = 59+randperm(48,x);
        C = 130+randperm(48,x);
        
        
        train_X = data([A;B;C],1:13);
        train_Y  = data([A;B;C],14:14);

        A = [A B C];
        test_X =[];
        test_Y = [];
        for i=1:size(data)
            if ~ismember(A,i)
                test_X = [test_X ;data(i,1:13)];
                test_Y = [test_Y ;data(i,14:14)];
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
            
            SVMModels{j} = fitcsvm(train_X,Y,'ClassNames',[0 1],'Standardize',true,'KernelFunction','rbf');
        end
        figure;
        X = tsne(test_X);
        gscatter(X(:,1),X(:,2),test_Y);
        
        
        %disp(pred)
        scoreAll = zeros(size(test_X,1),classes);
        for j = 1:classes
            [~,score] = predict(SVMModels{j},test_X);
            scoreAll(:,j) = score(:,2); 
        end
        [~,pred] = max(scoreAll,[],2);
        confusion_mat = confusionmat(test_Y,pred)
        
        accuracy(ep,1) = sum(max(confusion_mat,[],2))*100/(size(test_X,1));
end
disp('Accuracies : ')
disp(accuracy)
disp('Mean Accuracy : ')
disp(mean(accuracy))