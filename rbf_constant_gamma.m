load 'SVM_data_nonlinear.mat'
k = 5;

indices = randperm(60, k);
% randomly chosen centroids
centroids = x(indices, :);
prev_c = 0;
cluster = zeros(60, 1);
% to find closest centroid for each data point
dist = zeros(60, k);

while(prev_c ~= centroids)
    prev_c = centroids;
    
    % find nearest centroid for each point
    for i=1:60
        for j=1:k
            dist(i,j) = norm(x(i,:)-centroids(j,:));
        end
    end
    
    % cluster the data st points closest to a center are part of that
    % cluster
    for i=1:60
        cluster(i) = find(dist(i,:)==min(dist(i,:)));
    end
    
    %reposition centroid to be at the center of its cluster
    for i=1:k
        centroids(i, :) = mean(x(cluster==i,:));
    end
    
end

% use the Euclidean distance of each centroid from its group to determine
% gamma
sigmas = zeros(k,1);
for i=1:k
   group = x((cluster == i), :);
   sigmas(i) = norm(group - centroids(i,:));
end
gamma_opt = 1 ./ (sigmas);

% using pseudo-inverse equation to find optimal weights
h = calcKernel(gamma_opt, x, centroids);
weights = pinv(h'*h)*h'*y;

%plot data and decision boundary
m = 60;
x1range = [min(x(:,1))-1, max(x(:,1))+1];
x2range = [min(x(:,2))-1, max(x(:,2))+1];
d = 0.05;
[x1Grid,x2Grid] = meshgrid(x1range(1):d:x1range(2),...
    x2range(1):d:x2range(2));
xGrid = [x1Grid(:) x2Grid(:)];

h = calcKernel(gamma_opt, xGrid, centroids);
y_p = h*weights;
y_p = reshape(y_p,size(x1Grid));
figure()
contour(x1Grid,x2Grid,y_p,5);
hold on;
set(gca, 'ydir', 'reverse');
plot(x(1:m/2,1), x(1:m/2,2),'+');
for i=1:k
    plot(centroids(i,1),centroids(i,2), 'o');
end
plot(x((m/2)+1:m,1),x((m/2)+1:m,2),'r*');
ylim([-5,4]);
xlim([-5,4]);
pbaspect([1 1 1])

function H = calcKernel(gamma, x, centroids)
    m = length(x);
    k = length(centroids);
    H = zeros(m, k);
    for i=1:m
        for j=1:k
            H(i,j) = exp(-gamma(j).*norm(x(i,:)-centroids(j,:))^2);
        end
    end
end
