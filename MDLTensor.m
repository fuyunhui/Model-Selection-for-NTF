function [Ldist,Lvals,AICdist,BICdist,AIChist,BIChist] = MDLTensor(X,precision,Tstore,Ustore,Vstore,rValues,tT,tU,tV)
nRuns=length(Tstore);
Ldist=zeros(nRuns,8); Lvals=zeros(nRuns,8);
AICdist = zeros(nRuns,1); BICdist = zeros(nRuns,1);
AIChist = zeros(nRuns,1); BIChist = zeros(nRuns,1);
s=size(X);
zeroValT=tT*precision; zeroValU=tU*precision; zeroValV=tV*precision;
for i=1:nRuns
    T=Tstore{i,1}; U=Ustore{i,1}; V=Vstore{i,1};
    r=rValues(i);
    %%%%%%%%%%%%%%%%%%%%%%% FOR T %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T2=reshape(T,[],1);
    %%%%% FIRST SORT OUT THE T0 TERMS
    T0sLoc=find(T2(:,1)<zeroValT);numT0s=length(T0sLoc);
    numT1s=length(T2)-numT0s;numTTotal=length(T2);
    T3=T2;T2(T0sLoc,:)=[];
    if(numT0s~=0)
        Lt0=-numT0s*log2(numT0s/numTTotal)-numT1s*log2(numT1s/numTTotal)+0.5*log2(pi*numTTotal/2);
    else
        Lt0=0;
    end
    %%% THE T+ TERMS
    [Tvals,Tlocs]=histcounts(T2,'binwidth',precision);
    k = length(Tvals);
    lenkT = LenInteger(k);
    lenTNML = (k/2)*log2(numTTotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkT;
    TvalsProb=Tvals/(sum(Tvals));
    Tlocs2=[(Tlocs(1)-precision),Tlocs(1:end)+precision];
    Tbinned=interp1(Tlocs2,Tlocs2,T3,'nearest');
    Tbinned(T0sLoc)=0;Tbinned=reshape(Tbinned,s(3),r);
    [parT,~] = gamfit(Tlocs2(2:end-1),[],[],Tvals);
    yT=gampdf(Tlocs2(2:end-1),parT(1),parT(2));nBinsT=length(Tvals);
    lenTdist = LenRealNumber(precision) + 0.5*log2(numTTotal) + log2(numT1s) + lenkT;
    TdistProb=precision*yT;
    TdistProb = TdistProb/sum(TdistProb);
    TdistProb3=zeros(length(Tvals),1);TvalsProb3=zeros(length(Tvals),1);
    TdistProb4=zeros(length(Tvals),1);TvalsProb4=zeros(length(Tvals),1);
    for j=1:nBinsT
        if(TdistProb(j)==0)
            TdistProb(j)=min(TdistProb(TdistProb>0));
        end
        if(TvalsProb(j)==0)
            TvalsProb(j)=min(TvalsProb(TvalsProb>0));
        end
        TdistProb2=-log2(TdistProb(j));TvalsProb2=-log2(TvalsProb(j));
        TdistProb3(j,1)=Tvals(j)*TdistProb2;TvalsProb3(j,1)=Tvals(j)*TvalsProb2;
        if(j>1)
            TdistProb4(j,1)=TdistProb3(j,1)+TdistProb4(j-1,1);
            TvalsProb4(j,1)=TvalsProb3(j,1)+TvalsProb4(j-1,1);
        else
            TdistProb4(j,1)=TdistProb3(j,1);
            TvalsProb4(j,1)=TvalsProb3(j,1);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%% END OF T %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% FOR U %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U2=reshape(U,[],1);
    %%%%% FIRST SORT OUT THE U0 TERMS
    U0sLoc=find(U2(:,1)<zeroValU);numU0s=length(U0sLoc);
    numU1s=length(U2)-numU0s;numUTotal=length(U2);
    U3=U2;U2(U0sLoc,:)=[];
    if(numU0s~=0)
        Lu0=-numU0s*log2(numU0s/numUTotal)-numU1s*log2(numU1s/numUTotal)+0.5*log2(pi*numUTotal/2);
    else
        Lu0=0;
    end
    %%% THE U+ TERMS
    [Uvals,Ulocs]=histcounts(U2,'binwidth',precision);
    k = length(Uvals);
    lenkU = LenInteger(k);
    lenUNML = (k/2)*log2(numUTotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkU;
    UvalsProb=Uvals/(sum(Uvals));
    Ulocs2=[(Ulocs(1)-precision),Ulocs(1:end)+precision];
    Ubinned=interp1(Ulocs2,Ulocs2,U3,'nearest');
    Ubinned(U0sLoc)=0;Ubinned=reshape(Ubinned,s(1),r);
    [parU,~] = gamfit(Ulocs2(2:end-1),[],[],Uvals);
    lenUdist = LenRealNumber(precision) + 0.5*log2(numUTotal) + log2(numU1s) + lenkU;
    yU=gampdf(Ulocs2(2:end-1),parU(1),parU(2));nBinsU=length(Uvals);
    UdistProb=precision*yU;
    UdistProb = UdistProb/sum(UdistProb);
    UdistProb3=zeros(length(Uvals),1);UvalsProb3=zeros(length(Uvals),1);
    UdistProb4=zeros(length(Uvals),1);UvalsProb4=zeros(length(Uvals),1);
    for j=1:nBinsU
        if(UdistProb(j)==0)
            UdistProb(j)=min(UdistProb(UdistProb>0));
        end
        if(UvalsProb(j)==0)
            UvalsProb(j)=min(UvalsProb(UvalsProb>0));
        end
        UdistProb2=-log2(UdistProb(j));UvalsProb2=-log2(UvalsProb(j));
        UdistProb3(j,1)=Uvals(j)*UdistProb2;UvalsProb3(j,1)=Uvals(j)*UvalsProb2;
        if(j>1)
            UdistProb4(j,1)=UdistProb3(j,1)+UdistProb4(j-1,1);
            UvalsProb4(j,1)=UvalsProb3(j,1)+UvalsProb4(j-1,1);
        else
            UdistProb4(j,1)=UdistProb3(j,1);
            UvalsProb4(j,1)=UvalsProb3(j,1);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%% END OF U %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% FOR V %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    V2=reshape(V,[],1);
    %%%%% FIRST SORT OUT THE V0 TERMS
    V0sLoc=find(V2(:,1)<zeroValV);numV0s=length(V0sLoc);
    numV1s=length(V2)-numV0s;numVTotal=length(V2);
    V3=V2;V2(V0sLoc,:)=[];
    if(numV0s~=0)
        Lv0=-numV0s*log2(numV0s/numVTotal)-numV1s*log2(numV1s/numVTotal)+0.5*log2(pi*numVTotal/2);
    else
        Lv0=0;
    end
    %%% THE V+ TERMS
    [Vvals,Vlocs]=histcounts(V2,'binwidth',precision);
    k = length(Vvals);
    lenkV = LenInteger(k);
    lenVNML = (k/2)*log2(numVTotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkV;
    VvalsProb=Vvals/(sum(Vvals));
    Vlocs2=[(Vlocs(1)-precision),Vlocs(1:end)+precision];
    Vbinned=interp1(Vlocs2,Vlocs2,V3,'nearest');
    Vbinned(V0sLoc)=0;Vbinned=reshape(Vbinned,s(2),r);
    [parV,~] = gamfit(Vlocs2(2:end-1),[],[],Vvals);
    lenVdist = LenRealNumber(precision) + 0.5*log2(numVTotal) + log2(numV1s) + lenkV;
    yV=gampdf(Vlocs2(2:end-1),parV(1),parV(2));nBinsV=length(Vvals);
    VdistProb=precision*yV;
    VdistProb = VdistProb/sum(VdistProb);
    VdistProb3=zeros(length(Vvals),1);VvalsProb3=zeros(length(Vvals),1);
    VdistProb4=zeros(length(Vvals),1);VvalsProb4=zeros(length(Vvals),1);
    for j=1:nBinsV
        if(VdistProb(j)==0)
            VdistProb(j)=min(VdistProb(VdistProb>0));
        end
        if(VvalsProb(j)==0)
            VvalsProb(j)=min(VvalsProb(VvalsProb>0));
        end
        VdistProb2=-log2(VdistProb(j));VvalsProb2=-log2(VvalsProb(j));
        VdistProb3(j,1)=Vvals(j)*VdistProb2;VvalsProb3(j,1)=Vvals(j)*VvalsProb2;
        if(j>1)
            VdistProb4(j,1)=VdistProb3(j,1)+VdistProb4(j-1,1);
            VvalsProb4(j,1)=VvalsProb3(j,1)+VvalsProb4(j-1,1);
        else
            VdistProb4(j,1)=VdistProb3(j,1);
            VvalsProb4(j,1)=VvalsProb3(j,1);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%% END OF V %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%% FOR E %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xpred=ktensor({Ubinned,Vbinned,Tbinned});
    E=X-tensor(Xpred);Ecol=reshape(double(E),1,[]);
    [Evals,Elocs]=histcounts(Ecol,'binwidth',precision);
    numETotal=length(Ecol);
    k = length(Evals);
    lenkE = LenInteger(k);
    lenENML = (k/2)*log2(numETotal/(2*pi)) + k*0.5*log2(pi) - gammaln(k/2)/log(2) + lenkE;
    Elocs2=Elocs(1:end-1)+precision;
    EvalsProb=Evals/(length(Ecol));
    [muE,sigE,~,~] = normfit(Elocs2,[],[],Evals);
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
    AICdist(i)=EdistProb4(end)+r*sum(s)+k; BICdist(i)=EdistProb4(end)+0.5*(r*sum(s)+k)*log2(prod(s));
    Ldist(i,2)=EdistProb4(end)+lenEdist;Ldist(i,3)=TdistProb4(end)+lenTdist;
    Ldist(i,4)=Lt0;Ldist(i,5)=UdistProb4(end)+lenUdist;
    Ldist(i,6)=Lu0;Ldist(i,7)=VdistProb4(end)+lenVdist;
    Ldist(i,8)=Lv0;Ldist(i,1)=sum(Ldist(i,:));

    AIChist(i)=EvalsProb4(end)+r*sum(s)+k; BIChist(i)=EvalsProb4(end)+0.5*(r*sum(s)+k)*log2(prod(s));
    Lvals(i,2)=EvalsProb4(end)+lenENML;Lvals(i,3)=TvalsProb4(end)+lenTNML;
    Lvals(i,4)=Lt0;Lvals(i,5)=UvalsProb4(end)+lenUNML;
    Lvals(i,6)=Lu0;Lvals(i,7)=VvalsProb4(end)+lenVNML;
    Lvals(i,8)=Lv0;Lvals(i,1)=sum(Lvals(i,:));
end
plotNum=1;
lValuesPlot(rValues,Ldist,Lvals,plotNum)
end

function lValuesPlot(rValues,Ldist,Lvals,plotNum)
x=rValues;
y1=Ldist(:,1);y2=Ldist(:,2);y3=Ldist(:,3)+Ldist(:,4);
y4=Ldist(:,5)+Ldist(:,6);y5=Ldist(:,7)+Ldist(:,8);
y11=Lvals(:,1);y22=Lvals(:,2);y33=Lvals(:,3)+Lvals(:,4);
y44=Lvals(:,5)+Lvals(:,6);y55=Lvals(:,7)+Lvals(:,8);
figure(plotNum)
plot(x,y1,'b-')
hold on
plot(x,y2,'k--')
plot(x,y3,'k:')
plot(x,y4,'k-.')
plot(x,y5,'k-')
plot(x,y11,'y-')
plot(x,y22,'r--')
plot(x,y33,'r:')
plot(x,y44,'r-.')
plot(x,y55,'r-')
legend('L_{tot}','L_{E}','L_{T}','L_{U}','L_{V}','location','northeast')
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
