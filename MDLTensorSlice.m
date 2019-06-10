function [Ldist,Lvals,AICdist,BICdist,AIChist,BIChist]=MDLTensorSlice(V,precision,Wstore,Hstore,rValues,tW,tH)
nRuns=length(Wstore);
Ldist=zeros(nRuns,6);
Lvals=zeros(nRuns,6); 
AICdist = zeros(nRuns,1); BICdist = zeros(nRuns,1);
AIChist = zeros(nRuns,1); BIChist = zeros(nRuns,1); 
[m,n]=size(V);
plotNum=1;
zeroValW=tW*precision;zeroValH=tH*precision;
for i=1:nRuns
    W=Wstore{i,1};H=Hstore{i,1};
    r=rValues(i);
    %%%%%%%%%%%%%%%%%%%%%%% FOR W %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    W2=reshape(W,[],1);
    %%%%% FIRST SORT OUT THE W0 TERMS
    W0sLoc=find(W2(:,1)<zeroValW);numW0s=length(W0sLoc);
    numW1s=length(W2)-numW0s;numWTotal=length(W2);
    W3=W2;W2(W0sLoc,:)=[];
    if(numW0s~=0)
        Lw0=-numW0s*log2(numW0s/numWTotal)-numW1s*log2(numW1s/numWTotal)+0.5*log2(pi*numWTotal/2);
    else
        Lw0=0;
    end
    %%% THE W+ TERMS
    [Wvals,Wlocs]=histcounts(W2,'binwidth',precision); 
    k = length(Wvals);
    lenkW = LenInteger(k);
    lenWNML = (k/2)*log2(numWTotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkW;
    
    WvalsProb=Wvals/(sum(Wvals)); 
    Wlocs2=[(Wlocs(1)-precision),Wlocs(1:end)+precision]; 
    Wbinned=interp1(Wlocs2,Wlocs2,W3,'nearest');
    Wbinned(W0sLoc)=0;Wbinned=reshape(Wbinned,m,r);
    [parW,~] = gamfit(Wlocs2(2:end-1),[],[],Wvals); % The parameters of the gamma function are found using the frequency of terms in each bin
    yW=gampdf(Wlocs2(2:end-1),parW(1),parW(2));nBinsW=length(Wvals); % The probability density for each bin is found

    lenWdist = LenRealNumber(precision) + 0.5*log2(numWTotal) + log2(numW1s) + lenkW;

    WdistProb=precision*yW; % The probability of being in each bin from the gamma distribution is calculated
    % normalization
    WdistProb = WdistProb/sum(WdistProb);
    WdistProb3=zeros(length(Wvals),1);WvalsProb3=zeros(length(Wvals),1); % *dist* are for distributions, *vals* are for terms drawn directly from histograms
    WdistProb4=zeros(length(Wvals),1);WvalsProb4=zeros(length(Wvals),1); % *Prob3 contains the description length for each bin, *Prob4 the cumulative description length
    for j=1:nBinsW
        if(WdistProb(j)==0)
            WdistProb(j)=min(WdistProb(WdistProb>0)); % This sets zero probability terms to the smallest non-zero probability, this is an approximation.
        end
        if(WvalsProb(j)==0)
            WvalsProb(j)=min(WvalsProb(WvalsProb>0)); % For the histograms this is just to prevent taking the log of a zero.
        end
        WdistProb2=-log2(WdistProb(j));WvalsProb2=-log2(WvalsProb(j));
        WdistProb3(j,1)=Wvals(j)*WdistProb2;WvalsProb3(j,1)=Wvals(j)*WvalsProb2;
        if(j>1)
            WdistProb4(j,1)=WdistProb3(j,1)+WdistProb4(j-1,1);
            WvalsProb4(j,1)=WvalsProb3(j,1)+WvalsProb4(j-1,1);
        else
            WdistProb4(j,1)=WdistProb3(j,1);
            WvalsProb4(j,1)=WvalsProb3(j,1);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%% END OF W %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% FOR H %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H2=reshape(H,[],1);
    %%%%% FIRST SORT OUT THE H0 TERMS
    H0sLoc=find(H2(:,1)<zeroValH);numH0s=length(H0sLoc);
    numH1s=length(H2)-numH0s;numHTotal=length(H2);
    H3=H2;H2(H0sLoc,:)=[];
    if(numH0s~=0)
        Lh0=-numH0s*log2(numH0s/numHTotal)-numH1s*log2(numH1s/numHTotal)+0.5*log2(pi*numHTotal/2);
    else
        Lh0=0;
    end
    %%% THE H+ TERMS
    [Hvals,Hlocs]=histcounts(H2,'binwidth',precision);
    k = length(Hvals);
    lenkH = LenInteger(k);
    lenHNML = (k/2)*log2(numHTotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkH;
    HvalsProb=Hvals/(sum(Hvals));
    Hlocs2=[(Hlocs(1)-precision),Hlocs(1:end)+precision];
    Hbinned=interp1(Hlocs2,Hlocs2,H3,'nearest');
    Hbinned(H0sLoc)=0;Hbinned=reshape(Hbinned,r,n);
    [parH,~] = gamfit(Hlocs2(2:end-1),[],[],Hvals);
    lenHdist = LenRealNumber(precision) + 0.5*log2(numHTotal) + log2(numH1s) + lenkH;
    yH=gampdf(Hlocs2(2:end-1),parH(1),parH(2));nBinsH=length(Hvals);
    HdistProb=precision*yH;
    HdistProb = HdistProb/sum(HdistProb);
    HdistProb3=zeros(length(Hvals),1);HvalsProb3=zeros(length(Hvals),1);
    HdistProb4=zeros(length(Hvals),1);HvalsProb4=zeros(length(Hvals),1);
    for j=1:nBinsH
        if(HdistProb(j)==0)
            HdistProb(j)=min(HdistProb(HdistProb>0));
        end
        if(HvalsProb(j)==0)
            HvalsProb(j)=min(HvalsProb(HvalsProb>0));
        end
        HdistProb2=-log2(HdistProb(j));HvalsProb2=-log2(HvalsProb(j));
        HdistProb3(j,1)=Hvals(j)*HdistProb2;HvalsProb3(j,1)=Hvals(j)*HvalsProb2;
        if(j>1)
            HdistProb4(j,1)=HdistProb3(j,1)+HdistProb4(j-1,1);
            HvalsProb4(j,1)=HvalsProb3(j,1)+HvalsProb4(j-1,1);
        else
            HdistProb4(j,1)=HdistProb3(j,1);
            HvalsProb4(j,1)=HvalsProb3(j,1);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%% END OF H %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%% FOR E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Vpred=Wbinned*Hbinned;
    E=V-Vpred;Ecol=reshape(E,1,[]);
    [Evals,Elocs]=histcounts(Ecol,'binwidth',precision);%%%% Evals CONTAINS THE NUMBER OF COUNTS IN EACH BIN, Elocs THE EDGES OF THE BINS
    numETotal=length(Ecol);
    k = length(Evals);
    lenkE = LenInteger(k);
    lenENML = (k/2)*log2(numETotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkE;
    Elocs2=Elocs(1:end-1)+precision;
    EvalsProb=Evals/(length(Ecol));
    [muE,sigE,~,~] = normfit(Elocs2,[],[],Evals);% muE IS THE MEAN
    lenEdist = LenRealNumber(precision) + 1.5*log2(numETotal) + lenkE;
    yE = normpdf(Elocs2,muE,sigE);nBinsE=length(Evals);
    EdistProb=precision*yE;
    EdistProb = EdistProb/sum(EdistProb);
    EdistProb3=zeros(nBinsE,1);EvalsProb3=zeros(nBinsE,1);
    EdistProb4=zeros(nBinsE,1);EvalsProb4=zeros(nBinsE,1);
    for j=1:nBinsE
        if(EdistProb(j)==0)
            EdistProb(j)=min(EdistProb(EdistProb>0));
        end
        if(EvalsProb(j)==0)
            EvalsProb(j)=min(EvalsProb(EvalsProb>0));
        end
        EdistProb2=-log2(EdistProb(j));EvalsProb2=-log2(EvalsProb(j));
        EdistProb3(j,1)=Evals(j)*EdistProb2;EvalsProb3(j,1)=Evals(j)*EvalsProb2;
        if(j>1)
            EdistProb4(j,1)=EdistProb3(j,1)+EdistProb4(j-1,1);
            EvalsProb4(j,1)=EvalsProb3(j,1)+EvalsProb4(j-1,1);
        else
            EdistProb4(j,1)=EdistProb3(j,1);
            EvalsProb4(j,1)=EvalsProb3(j,1);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%% END OF E %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    AICdist(i)=EdistProb4(end)+k; BICdist(i)=EdistProb4(end)+0.5*k*log2(m*n);
    Ldist(i,2)=EdistProb4(end)+lenEdist;Ldist(i,3)=WdistProb4(end)+lenWdist;
    Ldist(i,4)=Lw0;Ldist(i,5)=HdistProb4(end)+lenHdist;
    Ldist(i,6)=Lh0;Ldist(i,1)=sum(Ldist(i,:));
    
    AIChist(i)=EvalsProb4(end)+k; BIChist(i)=EvalsProb4(end)+0.5*k*log2(m*n);
    Lvals(i,2)=EvalsProb4(end)+lenENML;Lvals(i,3)=WvalsProb4(end)+lenWNML;
    Lvals(i,4)=Lw0;Lvals(i,5)=HvalsProb4(end)+lenHNML;
    Lvals(i,6)=Lh0;Lvals(i,1)=sum(Lvals(i,:));
end
lValuesPlot(rValues,Ldist,Lvals,plotNum)
end


function lValuesPlot(rValues,Ldist,Lvals,plotNum)
x=rValues;
y1=Ldist(:,1);y2=Ldist(:,2);y3=Ldist(:,3)+Ldist(:,4);
y4=Ldist(:,5)+Ldist(:,6);
y11=Lvals(:,1);y22=Lvals(:,2);y33=Lvals(:,3)+Lvals(:,4);
y44=Lvals(:,5)+Lvals(:,6);

figure(plotNum)
plot(x,y1,'k-')
hold on
plot(x,y2,'k--')
plot(x,y3,'k:')
plot(x,y4,'k-.')
plot(x,y11,'r-')
plot(x,y22,'r--')
plot(x,y33,'r:')
plot(x,y44,'r-.')
legend('L_{tot}','L_{E}','L_{W}','L_{H}','location','northeast')
legend boxoff
xlabel('r')
ylabel('Description length')
title('Description lengths')
set(gca,'FontSize',16)
end

function lenk = LenInteger( k )
lenk = log2(2.865);
tmp = log2(k);
while tmp > 0
    lenk = lenk + tmp;
    tmp = log2(tmp);
end
end

function lenk = LenRealNumber( k )
r = 0;
if k < 0
    r = 1;
    k = -k;
end
d = round(k);
k2 = k;
while d ~= k2
    r = r+1;
    k2 = 10*k2;
    d = round(k2);
end
lenk = r * log2(10) + log2(2.865);
tmp = log2(ceil(k));
while tmp > 0
    lenk = lenk + tmp;
    tmp = log2(tmp);
end
end
