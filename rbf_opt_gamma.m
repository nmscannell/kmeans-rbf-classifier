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


gamma = rand(k,1);
error_opt = 1;
iter = 1;
while error_opt > .8
    h = calcKernel(gamma, x, centroids);
    % use pseudo-inverse equation to find optimal weights with this gamma
    weights = pinv(h'*h)*h'*y;
    % calc error
    out = predict(h*weights);
    error = out - y;
    J = calcJacobian(x, centroids, weights, h);
    % update gamma using error
    gamma_opt = gamma - pinv(J'*J)*J'*error;
    h_opt = calcKernel(gamma_opt, x, centroids);
    % find mse of current optimal
    out_opt = predict(h_opt*weights);
    error_opt = out_opt - y;
    error_opt = mean(error_opt.^2);
    iter = iter + 1;
    if iter > 100
        break;
    end
end

% plot contours / decisison boundary
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

function J = calcJacobian(x, centroids, weights, h)
    m = length(x);
    k = length(centroids);
    J = zeros(m,k);
    for i=1:m
        for j=1:k
            J(i,j) = (-weights(j).*norm(x(i,:)-centroids(j,:))^2)*h(i,j);
        end
    end
end

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

function s = predict(z)
    for i=1:length(z)
        if z(i) <= 0
            z(i) = -1;
        else
            z(i) = 1;
        end
    end
    s = z;
end