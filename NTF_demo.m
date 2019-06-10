cd /path/to/MATLAB/tensor_toolbox/
addpath(pwd)
cd met; addpath(pwd)
cd /path/to/rank/selection/

load /tensor/X.mat
X = sptensor(X);
rValues = 1:100;
precision = 0.001; tT = 1.0; tU = 1.0; tV = 1.0;
Tstore = cell([length(rValues) 1]);
Ustore = cell([length(rValues) 1]);
Vstore = cell([length(rValues) 1]);
err = zeros(length(rValues),1);

parfor i = 1:length(rValues)
    ops = [];
    [ H_best, his_best ] = AOadmm(X,rValues(i),ops);
    loss_best = his_best.err(end);
    for j = 1:9
        ops = [];
        [ H, his ] = AOadmm(X,rValues(i),ops);
        loss = his.err(end);
        if loss < loss_best
            H_best = H;
            loss_best = loss;
        end
    end
    err(i) = loss_best;
    Tstore{i} = H_best{3};
    Ustore{i} = H_best{1};
    Vstore{i} = H_best{2};
end

[LdistSet,LvalsSet,AICd,BICd,AICh,BICh]=MDLTensor(X,precision,Tstore,Ustore,Vstore,rValues,tT,tU,tV);
[~,idx_dist] = min(LdistSet(:,1));[~,idx_vals] = min(LvalsSet(:,1));
[~,idx_AICd] = min(AICd); [~,idx_BICd] = min(BICd);[~,idx_AICh] = min(AICh); [~,idx_BICh] = min(BICh);

R_NMLHist = rValues(idx_vals)
R_2stDist = rValues(idx_dist)
R_AIC1 = rValues(idx_AICh)
R_AIC2 = rValues(idx_AICd)
R_BIC1 = rValues(idx_BICh)
R_BIC2 = rValues(idx_BICd)
