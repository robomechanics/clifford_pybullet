% Obstacle Coordinates
% B1 = [2, 2.4, 0.1, .2];
% B2 = [-1, 2, 2, 1];
% B3 = [2, 1.4, 0.1, .2];

B1 = [-3, 0, 1, 5];
B2 = [-1, 2, 2, 1];
B3 = [2, 0, 1, 5];

% Initialize
target = [0 4];
N = 50;
rarParam = 0.96;
Ne = cast(N*rarParam, 'uint32');
numInputs = 42;
gamma_t = 0;
k = 1;

% eliteSetX = zeros(1,numInputs);
% eliteNumX = 1;
% eliteSetY = zeros(1,numInputs);
% eliteNumY = 1;

mu_0 = zeros(1,numInputs);
covar_0 = eye(numInputs);
muX_t = mu_0;
muY_t = mu_0;
covarX_t = covar_0;
covarY_t = covar_0;

centr = [mu_0; mu_0];
vars = [covar_0; covar_0];

a = 0.1;
rho = N - Ne + 1;
t = 0;
iterate = true;
tic;
while iterate
    t = t + 1
    
    % Generate Samples
    thrustsX = sampleFromDist(muX_t, covarX_t, N, numInputs);
    thrustsY = sampleFromDist(muY_t, covarY_t, N, numInputs);
%     thrustsX = sampleKDE(eliteSetX, eliteNumX, N, numInputs);
%     thrustsY = sampleKDE(eliteSetY, eliteNumY, N, numInputs);
%     thrustsX = sampleMixture(centr, vars, N, numInputs);
%     thrustsY = sampleMixture(centr, vars, N, numInputs);
    
    % Simulate Paths
    figure(t)
    rectangle('Position', [B1(1,1) B1(1,2) B1(1,3) B1(1,4)])
    hold on
    rectangle('Position', [B2(1,1) B2(1,2) B2(1,3) B2(1,4)])
    rectangle('Position', [B3(1,1) B3(1,2) B3(1,3) B3(1,4)])
    plot(target(1,1), target(1,2), 'r*')
    xlim([-5 6])
    ylim([-5 6])
    simulating = toc
    endPts = simulate(thrustsX, thrustsY, N, numInputs, B1, B2, B3);
    
    costs = toc
    % Calculate Costs
    % costs = getCosts(endPts, N, target);
    costs = getCosts(endPts, N, target) + 0.1*getThrustMag(thrustsX, thrustsY);
    
    % Update Gamma
    gamma_0 = gamma_t;
    gamma_t = updateGamma(costs, rho);
    if abs(gamma_t-gamma_0) < 0.001
        iterate = false;
    end
    
    updateDistr = toc
    % Update Distributions
    ind = costs < gamma_t;
    eliteSetX = thrustsX(ind,:);
    eliteSetY = thrustsY(ind,:);
    eliteNumX = sum(ind);
    eliteNumY = sum(ind);    
    eliteSumX = sum(eliteSetX,1);
    eliteSumY = sum(eliteSetY,1);
%     
%     [centrX, indX] = kMeans(eliteSetX, eliteNumX);
%     [centrY, indY] = kMeans(eliteSetY, eliteNumY);
%     varsX = kVars(centrX, eliteSetX, eliteNumX, numInputs, indX);
%     varsY = kVars(centrY, eliteSetY, eliteNumY, numInputs, indY);
%     
    muX_t = a*muX_t + (1-a)*eliteSumX/eliteNumX
    muY_t = a*muY_t + (1-a)*eliteSumY/eliteNumY;
    
    eliteArrayX = zeros(numInputs);
    eliteArrayY = zeros(numInputs);
    for k = 1:N
        if costs(k,1) < gamma_t
            vX = thrustsX(k,:) - muX_t;
            eliteArrayX = eliteArrayX + vX*vX';
            vY = thrustsY(k,:) - muY_t;
            eliteArrayY = eliteArrayY + vY*vY';            
        end
    end
    covarX_t = a*covarX_t + (1-a)*eliteArrayX/(eliteNumX);
    covarY_t = a*covarY_t + (1-a)*eliteArrayY/(eliteNumY);
end
x = 'converged'


function thrust = sampleFromDist(mu, covar, N, numInputs)
thrust = mvnrnd(mu, covar, N);
end

function thrust = sampleKDE (eliteSet, eliteNum, N, numInputs)
r = 1 + (eliteNum-1)*rand(N, 1);
r = round(r);
means = eliteSet(r, :);
thrust = zeros(N, numInputs);
for i = 1:N
    thrust(i,:) = normrnd(means(i,:), 1);
end

end

function thrust = sampleMixture(centr, vars, N, numInputs)
centr1 = centr(1,:);
centr2 = centr(2,:);
vars1 = vars(1:numInputs, :);
vars2 = vars(numInputs+1:end,:);
thrust1 = mvnrnd(centr1, vars1, N/2);
thrust2 = mvnrnd(centr2, vars2, N/2);
thrust = [thrust1; thrust2];

end

function endPts = simulate(thrustsX, thrustsY, N, numInputs, B1, B2, B3)
options = odeset('Events', @wallEvents, 'Maxstep', 0.0001);
z0 = zeros(1, 4*N);
tstart = 0;
tend = 8;
ztot = [];

while tstart < tend
    tspan = linspace(tstart, tend, numInputs);
    [t, z] = ode45(@odefun1, tspan, z0, options, thrustsX, thrustsY, N, tend/(numInputs-2), B1, B2, B3);
    endODE = toc
    ztot = [ztot; z];
    tstart = t(end,1);
    z0 = bounce(z(end,:), N, B1, B2, B3);
    endbounce = toc
end
endpath = toc
for i = 1:N
    plot(ztot(:,i),ztot(:,i+N))
end
plotting = toc
drawnow
endPts = [ztot(end,1:N)' ztot(end,N+1:2*N)'];
end

function costs = getCosts(endPts, N, target)
costs = zeros(N, 1);
for i = 1:N
    costs(i, 1) = (target(1,1)-endPts(i,1))^2 + (target(1,2)-endPts(i,2))^2;
end
end

function costs = getThrustMag (thrustX, thrustY)
magnitude = (thrustX.*thrustX + thrustY.*thrustY).^0.5;
costs = sum(magnitude, 2);
end

function gamma = updateGamma(costs, rho)
sorted = sort(costs);
gamma = sorted(rho, 1);
end

function [centr, ind] = kMeans(eliteSet, eliteNum)
cont = 5;
% initialize centr1 and centr2
centr = randi([1 eliteNum], 1 ,2);
centr1 = eliteSet(centr(1,1),:);
centr2 = eliteSet(centr(1,2),:);
while cont > 0.01
    centr1_0 = centr1;
    centr2_0 = centr2;
    % compute sum of squared distances from each centroid
    diff1 = (eliteSet - ones(eliteNum, 1) * centr1).^2;
    diff2 = (eliteSet - ones(eliteNum, 1) * centr2).^2;
    sum1 = sum(diff1, 2);
    sum2 = sum(diff2, 2);
    % assign data points to each centroid
    categorize = sum1 - sum2;
%     ind1 = categorize < 0;
%     ind2 = categorize > 0;
    ind1 = ones(eliteNum, 1);
    ind2 = zeros(eliteNum,1);
    cat1 = eliteSet(ind1, :);
    cat2 = eliteSet(ind2, :);
    % calculate new centroids
    centr1 = sum(cat1, 1)/sum(ind1);
    centr2 = sum(cat2, 1)/sum(ind2);
    cont = sum((centr1 - centr1_0).^2 + (centr2 - centr2_0).^2);
end 
centr = [centr1; centr2];
ind = [ind1 ind2];
end

function vars = kVars(centr, eliteSet, eliteNum, numInputs, ind)
centr1 = centr(1,:);
centr2 = centr(2,:);
ind1 = ind(:,1);
ind2 = ind(:,2);

matr1 = eliteSet(ind1,:) - ones(sum(ind1),1)*centr1;
covar1 = zeros(numInputs);
for k = 1:sum(ind1)
    x = matr1(k,:);
    covar1 = covar1 + x*x';
end

matr2 = eliteSet(ind2,:) - ones(sum(ind2),1)*centr2;
covar2 = zeros(numInputs);
for k = 1:sum(ind2)
    x = matr2(k,:);
    covar2 = covar2 + x*x';
end

vars = [covar1; covar2];
end

function [value,isterminal,direction] = wallEvents(t, z, thrustX, thrustY, N, numInputs, B1, B2, B3)
x = z(1:N, 1);
y = z(N+1:2*N, 1);
val1 = (x > (B1(1, 1)+B1(1,3)) | x < B1(1, 1) | y > (B1(1, 2)+B1(1,4)) | y < B1(1, 2));
val2 = (x > (B2(1, 1)+B2(1,3)) | x < B2(1, 1) | y > (B2(1, 2)+B2(1,4)) | y < B2(1, 2));
val3 = (x > (B3(1, 1)+B3(1,3)) | x < B3(1, 1) | y > (B3(1, 2)+B3(1,4)) | y < B3(1, 2));

value = [val1; val2; val3];
isterminal = ones(3*N, 1);    
direction = -1*ones(3*N, 1);
end

function zdot = odefun(t, z, thrustX, thrustY, N, interval, B1, B2, B3)
i = fix(t/interval)+1;
dim = 2*N;
A = [zeros(dim), eye(dim); zeros(dim), zeros(dim)];
B = [zeros(dim, 1); thrustX(:,i); thrustY(:,i)];
zdot = A*z + B;

end

% 1st order hold
% numInputs = num + 2
function zdot = odefun1(t, z, thrustX, thrustY, N, interval, B1, B2, B3)
i = fix(t/interval)+1;
x = mod(t, interval);
inputX = (thrustX(:,i+1)-thrustX(:,i))./interval .* x + thrustX(:,i);
inputY = (thrustY(:,i+1)-thrustY(:,i))./interval .* x + thrustY(:,i);
dim = 2*N;
A = [zeros(dim), eye(dim); zeros(dim), zeros(dim)];
B = [zeros(dim, 1); inputX; inputY];
zdot = A*z + B;
end

function z0 = bounce(z, N, B1, B2, B3)
tol = .05;
x = z(1,1:N);
y = z(1, N+1:2*N);
x_dot = z(1, 2*N+1:3*N);
y_dot = z(1, 3*N+1:4*N);

ind1x = (abs(x-B1(1,1)) < tol | abs(x-B1(1,1)-B1(1,3)) < tol) & y > B1(1,2) & y < B1(1,2)+B1(1,4);
x_dot(ind1x) = -x_dot(ind1x);
ind1y = (abs(y-B1(1,2)) < tol | abs(y-B1(1,2)-B1(1,4)) < tol) & x > B1(1,1) & x < B1(1,1)+B1(1,3);
y_dot(ind1y) = -y_dot(ind1y);

ind2x = (abs(x-B2(1,1)) < tol | abs(x-B2(1,1)-B2(1,3)) < tol) & y > B2(1,2) & y < B2(1,2)+B2(1,4);
x_dot(ind2x) = -x_dot(ind2x);
ind2y = (abs(y-B2(1,2)) < tol | abs(y-B2(1,2)-B2(1,4)) < tol) & x > B2(1,1) & x < B2(1,1)+B2(1,3);
y_dot(ind2y) = -y_dot(ind2y);

ind3x = (abs(x-B3(1,1)) < tol | abs(x-B3(1,1)-B3(1,3)) < tol) & y > B3(1,2) & y < B3(1,2)+B3(1,4);
x_dot(ind3x) = -x_dot(ind3x);
ind3y = (abs(y-B3(1,2)) < tol | abs(y-B3(1,2)-B3(1,4)) < tol) & x > B3(1,1) & x < B3(1,1)+B3(1,3);
y_dot(ind3y) = -y_dot(ind3y);


z0 = [x, y, x_dot, y_dot];
end


