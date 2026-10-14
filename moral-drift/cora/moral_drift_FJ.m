function res = moral_drift_FJ()
% moral_drift_FJ - set-based moral drift of Moral Machine cultural clusters
%    under Friedkin-Johnsen opinion dynamics with interval-uncertain
%    coupling, and the alignment horizon of a classifier trained on one
%    cluster's present-day preferences.
%
%    Reads ../data/cora_params.json, written by generate.py: the fitted
%    preference weights, their standard errors, the coupling interval and
%    the 2,295 dilemmas. The coupling a_ij and gamma are ASSUMPTIONS (see
%    generate.py); the result holds for this model and these intervals.
%
% Model (continuous-time Friedkin-Johnsen, per moral dimension d):
%
%    dx_i/dt = gamma * ( -x_i + sum_j a_ij x_j + (1 - sum_j a_ij) u_i )
%
%    x_i   - preference weight of cluster i on dimension d
%    u_i   - anchor / prejudice of cluster i (= its initial preference)
%    a_ij  - influence of cluster j on cluster i, a_ij = lambda_i * w_ij,
%            so that lambda_i = sum_j a_ij is cluster i's susceptibility
%    gamma - drift rate (1/years)
%
%    Writing the model in a_ij instead of (lambda_i, w_ij) makes it affine
%    in the uncertain parameters, so each a_ij becomes one generator of a
%    matrix zonotope and its appearance in both the x- and u-terms stays
%    correlated. The anchors u are carried as constant states (du/dt = 0).
%
% Alignment check:
%    A dilemma k is described by phi_k (attribute differences between the
%    two outcomes, one entry per dimension, the first being the action
%    bias) and offset b_k. The cluster's choice is sign(phi_k' * x_c + b_k)
%    (+1 = swerve). The classifier is provably aligned on dilemma k at time
%    t iff every preference vector in the reachable set yields the
%    classifier's label.
%
% Syntax:
%    res = moral_drift_FJ()
%
% Outputs:
%    res - struct with reachable set, alignment horizons and sensitivities
%
% References:
%    [1] E. Awad et al. "The Moral Machine experiment", Nature, 2018.
%    [2] N. E. Friedkin, E. C. Johnsen. "Social influence and opinions",
%        J. Math. Sociology, 1990.

% ------------------------------ BEGIN CODE -------------------------------

% Data --------------------------------------------------------------------

P = jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'data', 'cora_params.json')));

clusters   = P.clusters';
dimensions = P.dimensions';
N = numel(clusters);
D = numel(dimensions);

% fitted preference weights (rows = clusters, columns = dimensions) and
% their standard errors, the radius of the initial set
x0 = P.x0;
x0_rad = P.x0_rad;

% influence matrix a_ij (row i listens to column j), centre and radius of
% the interval. Row sums are the susceptibilities lambda_i and must stay
% below 1 over the whole interval. ASSUMPTION.
a_c = P.a_c;
a_r = P.a_r;
assert(all(sum(a_c + a_r,2) < 1), 'susceptibility must stay below 1');

% drift rate in 1/years. ASSUMPTION.
gamma = P.gamma;

% dilemmas: one row of Phi per dilemma
names = {P.dilemmas.name}';
Phi = [P.dilemmas.phi]';
b = [P.dilemmas.b]';
K = numel(names);

% cluster the classifier was trained on
c = 2;  % eastern


% System Dynamics ---------------------------------------------------------

matZ_A = aux_buildFJ(a_c, a_r, gamma, D);
n = 2*N*D;                         % states [x; u]
sys = linParamSys(matZ_A, zeros(n,1), 'constParam');
% FUTURE WORK: P.sigma bounds the unexplained yearly change. As a bounded
% disturbance it would enter as an input on the x-states, e.g.
% linParamSys(matZ_A, [eye(N*D); zeros(N*D)], ...) with params.U a
% zonotope of radius sigma per dimension.

% state index of cluster i on dimension d
idx = @(i,d) (d-1)*N + i;
idx_c = arrayfun(@(d) idx(c,d), 1:D);


% Parameters --------------------------------------------------------------

% initial set: x(0) = u = fitted weights +- standard error (x and u share
% generators)
x0v = x0(:); r0v = x0_rad(:);
params.R0 = zonotope([x0v; x0v], [diag(r0v); diag(r0v)]);
params.U = zonotope(0);
params.tFinal = P.tFinal;          % years


% Reachability Settings ---------------------------------------------------

options.timeStep = 0.1;
options.taylorTerms = 6;
options.zonotopeOrder = 50;
options.intermediateTerms = 2;
options.compTimePoint = false;


% Classifier ----------------------------------------------------------------

% mock: a classifier frozen at deployment, which reproduces cluster c's
% present-day choices. FUTURE WORK: a trained network over phi, loaded
% with neuralNetwork.readONNXNetwork. The moral-classification networks
% take sentence embeddings, not phi, so they cannot be used here directly.
label = sign(Phi*x0(c,:)' + b);


% Reachability Analysis ---------------------------------------------------

tic;
R = reach(sys, params, options);
disp(['Reachable set computed in ', num2str(toc), ' s']);

[horizon, flipped, margin, tGrid] = aux_alignment(R, idx_c, Phi, b, label);

fprintf('\nClassifier frozen on %s preferences, %d dilemmas, %g years:\n', ...
    clusters{c}, K, P.tFinal);
fprintf('  provably aligned throughout: %4d\n', sum(isinf(horizon)));
fprintf('  may disagree at some point:  %4d\n', sum(isfinite(horizon)));
fprintf('  surely disagree at some point: %2d\n', sum(isfinite(flipped)));

[~, order] = sort(horizon);
nShow = min(5, sum(isfinite(horizon)));
fprintf('\nEarliest possible disagreements:\n');
for k = order(1:nShow)'
    fprintf('  %s  label %+d   aligned until t = %5.2f   surely flipped at t = %5.2f\n', ...
        names{k}, label(k), horizon(k), flipped(k));
end


% Sensitivity Analysis ----------------------------------------------------

% fix each coupling a_ij to its centre in turn and see how far the
% alignment horizon of the earliest dilemma moves
kSens = order(1);
sens = nan(N,N);
for i = 1:N
    for j = 1:N
        a_r_ij = a_r; a_r_ij(i,j) = 0;
        sys_ij = linParamSys(aux_buildFJ(a_c, a_r_ij, gamma, D), zeros(n,1), 'constParam');
        R_ij = reach(sys_ij, params, options);
        h = aux_alignment(R_ij, idx_c, Phi(kSens,:), b(kSens), label(kSens));
        sens(i,j) = h - horizon(kSens);
    end
end
disp(['Horizon gain [years] for ', names{kSens}, ...
    ' when a_ij is known exactly (row i listens to column j):']);
disp(array2table(sens, 'RowNames', clusters, 'VariableNames', clusters));


% Simulation --------------------------------------------------------------

simOpt.points = 20;
traj = simulateRandom(sys, params, simOpt);


% Visualization -----------------------------------------------------------

colors = lines(max(N, nShow));

% PLOT 1: reachable band of each cluster per dimension
figure;
tiledlayout('flow');
for d = 1:D
    nexttile; hold on; box on;
    for i = 1:N
        plotOverTime(R, idx(i,d), 'FaceColor', colors(i,:), ...
            'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', clusters{i});
    end
    for i = 1:N
        plotOverTime(traj, idx(i,d), 'Color', 0.3*colors(i,:), ...
            'HandleVisibility', 'off');
    end
    xlabel('t [years]'); ylabel('weight');
    title(dimensions{d});
end
legend();

% PLOT 2: alignment margin of the trained cluster on the earliest dilemmas
figure; hold on; box on;
for s = 1:nShow
    k = order(s);
    m = squeeze(margin(:,k,:));
    fill([tGrid; flipud(tGrid)], [m(:,1); flipud(m(:,2))], colors(s,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.4, 'DisplayName', names{k});
    xline(horizon(k), '--', 'Color', colors(s,:), 'HandleVisibility', 'off');
end
yline(0, 'k', 'HandleVisibility', 'off');
xlabel('t [years]');
ylabel(['label \times (\phi^T x_{', clusters{c}, '} + b)']);
title(['Alignment margin, classifier trained on ', clusters{c}]);
legend();

res.R = R;
res.names = names;
res.horizon = horizon;
res.flipped = flipped;
res.sensitivity = sens;

end


% Auxiliary functions -----------------------------------------------------

function matZ_A = aux_buildFJ(a_c, a_r, gamma, D)
% matrix zonotope of the augmented FJ system [x; u] with one generator per
% uncertain coupling a_ij; states ordered dimension-major

N = size(a_c,1);
I_D = eye(D);
blk = @(Mx, Mu) gamma*[kron(I_D,Mx) kron(I_D,Mu); zeros(N*D,2*N*D)];

C = blk(-eye(N) + a_c, diag(1 - sum(a_c,2)));

[ii,jj] = find(a_r);
G = zeros(2*N*D, 2*N*D, numel(ii));
for g = 1:numel(ii)
    E = zeros(N); E(ii(g),jj(g)) = a_r(ii(g),jj(g));
    G(:,:,g) = blk(E, -diag(sum(E,2)));
end
matZ_A = matZonotope(C, G);

end

function [horizon, flipped, margin, tGrid] = aux_alignment(R, idx_c, Phi, b, label)
% alignment margin label*(phi'*x + b) over each time-interval reachable
% set, for all dilemmas at once; aligned while the lower bound is
% positive, surely flipped once the upper bound is negative. For a
% zonotope with centre z and generators G, the margin in direction m lies
% in m'*z +- sum(abs(m'*G)), which is its exact support function.

sets = R.timeInterval.set;
times = R.timeInterval.time;
tGrid = cellfun(@(t) infimum(t), times);
M = label .* Phi;                   % one direction per dilemma (rows)
off = label .* b;
margin = zeros(numel(sets), numel(label), 2);
for s = 1:numel(sets)
    Z = zonotope(project(sets{s}, idx_c));
    mid = M*center(Z) + off;
    rad = sum(abs(M*generators(Z)), 2);
    margin(s,:,1) = mid - rad;
    margin(s,:,2) = mid + rad;
end

K = numel(label);
horizon = inf(K,1); flipped = inf(K,1);
for k = 1:K
    first = find(margin(:,k,1) <= 0, 1);
    if ~isempty(first), horizon(k) = tGrid(first); end
    first = find(margin(:,k,2) < 0, 1);
    if ~isempty(first), flipped(k) = tGrid(first); end
end

end

% ------------------------------ END OF CODE ------------------------------
