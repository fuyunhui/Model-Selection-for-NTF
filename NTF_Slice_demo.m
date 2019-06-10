cd /path/to/MATLAB/tensor_toolbox/
addpath(pwd)
cd met; addpath(pwd)
cd /path/to/rank/selection/

load /sliced/tensor/X.mat
Size = size(X);
% Our synthetic data tensor has a size like 2000 x 1000 x 5
V3 = cell([Size(3) 1]); 
Wstore3 = cell([Size(3) 1]);
Hstore3 = cell([Size(3) 1]);
E3 = cell([Size(3) 1]);
rValues = 1:100; % Set the range of candidate ranks
precision = 0.001; tW = 1; tH = 1; rank = zeros([Size(3) 6]);

for i = 1:Size(3)
    V3{i,1} = double(X(:,:,i));
    Wtmp = cell([length(rValues) 1]);
    Htmp = cell([length(rValues) 1]);
    Etmp = zeros(length(rValues),1);
    parfor k = 1:length(rValues)
        ops = [];
        [ H_best, his_best ] = AOadmm(V3{i,1},rValues(k),ops);
        loss_best = his_best.err(end);
        for j = 1:9
            ops = [];
            [ H, his ] = AOadmm(V3{i,1},rValues(k),ops);
            loss = his.err(end);
            if loss < loss_best
                H_best = H;
                loss_best = loss;
            end
        end
        Etmp(k) = loss_best;
        Wtmp{k} = H_best{1};
        Htmp{k} = H_best{2}';
    end
    Wstore3{i,1} = Wtmp;
    Hstore3{i,1} = Htmp;
    E3{i,1} = Etmp;
    [LdistSet,LvalsSet,AICdist,BICdist,AIChist,BIChist]=MDLTensorSlice(V3{i},precision,Wstore3{i},Hstore3{i},rValues,tW,tH);
    [~,idx_dist] = min(LdistSet(:,1)); [~,idx_vals] = min(LvalsSet(:,1)); [~,idx_AICd] = min(AICdist);
    [~,idx_BICd] = min(BICdist); [~,idx_AICh] = min(AIChist); [~,idx_BICh] = min(BIChist);
    rank(i,1) = rValues(idx_dist); rank(i,2) = rValues(idx_vals); rank(i,3) = rValues(idx_AICd); 
    rank(i,4) = rValues(idx_BICd); rank(i,5) = rValues(idx_AICh); rank(i,6) = rValues(idx_BICh);
end

R_NMLHist = max(rank(:,2))
R_2stDist = max(rank(:,1))
R_AIC1 = max(rank(:,5))
R_AIC2 = max(rank(:,3))
R_BIC1 = max(rank(:,6))
R_BIC2 = max(rank(:,4))
