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
d = data;
data = d(:,1:4);
Y = d(:,5:5);

k = 3;

inds = randperm(size(data,1),k);
c = data(inds,:);
disp('Initial Clusters : ');
disp(c);

maxItr = 800;
it = 0;
for itr = 1:maxItr
    it = it+1;
    assign = zeros(size(data,1),1);
    for i=1:size(data,1)
       dist = distance(c,data(i,:),k); 
       [~,indx] = min(dist);
       assign(i) = indx;
    end
    
    new_c = update(c,data,assign,k);
    x = 0;
    for i=1:k
        for j=1:size(c,2)
            if new_c(i,j)-c(i,j) == 0
                x =x+1;
            end
        end
    end
    if x == k*size(c,2)
        break
    end
    
    c = new_c;
end

disp('Updated Cluster Centers : ')
disp(c);
%disp(it);


for i=1:size(data,1)
       dist = distance(c,data(i,:),k); 
       [~,indx] = min(dist);
       assign(i) = indx;
end

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
accuracy = sum(max(confusion_mat,[],2))*100/size(data,1);
disp('Accuracy is : ');
disp(accuracy);



function d = distance(c,data,k)

x = zeros(k,1);

for i=1:size(c,1)
    for j=1:size(c,2)
        x(i,1) = x(i,1) + (c(i,j)-data(1,j))^2;
    end
    x(i,1) = sqrt(x(i,1));
end

d = x;

end


function x = update(c,data,assign,k)

for i=1:k
    cnt =0;
    p = zeros(1,size(data,2));
    for j=1:size(data,1)
        if assign(j,1) == i
            cnt=cnt+1;
            p = p + data(j,:);
        end
    end

    c(i,:) = p/cnt;
    
end

x = c;

end

