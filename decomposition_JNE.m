function [MUFiring,IPTs,SIL] = Decomposition_JNE(EMG,fsamp,NITER,TH,PLOT_PAUSE,FREQ)
%   V2.0
%   Decomposition_JNE   Decomposition of Multichannel EMG signals using
%                       convolutive blind source separation 

%   [MUFiring,IPTs,SIL] = Decomposition_JNE(EMG,fsamp,NITER,TH,PLOT_PAUSE,FREQ) 
%   EMG the signal matrix (channels x samples)
%   fsamp sample rate
%   NITER number of interations (sources)
%   PEREIG percentage of eigevectors removed
%   TH threshold on SIL measure
%   PLOT_PAUSE plotting
%   FREQ  interference frequency for frequency domain notch filter (if empty, it does not apply the filter)
%
%
%
%   Please cite:
%
%   Negro, Francesco, Silvia Muceli, Anna Margherita Castronovo, Ales Holobar, and Dario Farina. 
%   "Multi-channel intramuscular and surface EMG decomposition by convolutive blind source separation." 
%    Journal of neural engineering 13, no. 2 (2016): 026027.
%
%   Copyright 2014-2017 Francesco Negro 
%   francesco.negro@unibs.it


clc
LIC = 1011;
PEREIG = 25;    % number of engenvalues removed %between 10 and 25
FACTOR = 1000;  % extension factor = FACTOR/number of channels
MAXDR = 100;    % max discharge rate (pps)
MINNDR = 10;    % minimum number of discharges for a unit (number of pulses per decomposition)
LW = 5e-3;      % interval to remove identified time points (in s)
TolX = 1e-4;    % tolerance fixed point
MAXCOUNT = 50;  % max number of interations fixed point
typedistance = 'sample';    % type of distance for K means clustering
SOGLIA_ACTIVITY_INDEX = 99; % thresold for outliers

tic

% frequency based notch filter 
if ~isempty(FREQ)

    for CH = 1:size(EMG,1)
        y=EMG(CH,:);
        fy=fft(y);
        lenFy=length(fy);
        Ind = [FREQ:FREQ:fsamp/2-FREQ]*lenFy/fsamp;
        
        rr = 5;
        r = lenFy/fsamp*rr;
        
        tInd=[];
        for k=-floor(r/2):floor(r/2)
            tInd=[tInd Ind+k];
        end
        fy(round(tInd))=0;
        
        correct=lenFy-floor(lenFy/2)*2;
        fy(lenFy:-1:ceil(lenFy/2)+1)=conj(fy(2:1:ceil(lenFy/2)+1-correct));
        
        EMG(CH,:)=real(ifft(fy(1:lenFy)));
        
    end
end

% extension factor for signals
extFact = round(FACTOR/size(EMG,1));

% extension of measurements
eYT = extension(EMG,extFact); 

% cost function
G = @(x)1/2*x.^2;
DG = @(x)x;

% remove extended part and mean value
eYT = eYT(:,extFact+1:end-extFact-1);
eYT = eYT - repmat(mean(eYT,2),1,size(eYT,2));

% whitening using SVD
[U,S,V] = svd(eYT*eYT'/length(eYT));    % SVD correlation matrix
FACT = prctile(diag(S),PEREIG);         % remove eigenvectors
SI = 1./sqrt(diag(S) + FACT);           % inverse eigenvalues
WM = U * diag(SI) * V';                     % whitening matrix
eYW = WM * eYT;                         % whitened extended signals
    
MU = 1;

n = size(eYW,1);
m = size(eYW,2);

% ACT is activity index 
ACT = sum(abs(eYW).^2,1);
ACT(1:round(LW*fsamp)+1)=0;
ACT(end-round(LW*fsamp)-1:end)=0;
TH_ACT = prctile(ACT,SOGLIA_ACTIVITY_INDEX);    % remove some artifacts from the activity index
index = find(ACT>TH_ACT);                       %

noindex = [];
for i=1:length(index),
    ACT(index(i)-round(LW*fsamp):index(i)+round(LW*fsamp)) = 0;       
end

B = zeros(size(eYW,1),NITER);   % matrix of projecting vectors

% main loop on sources
for trial = 1:NITER,
    
    % select maximum activiti index    
    [massimo,ind] = max(ACT);

    % if massimo == 0 stop loop
    if massimo == 0, break;end

    % plot initialization point
    if PLOT_PAUSE==1,
        figure(1),subplot(3,2,1),hold off;plot([0:length(ACT)-1]/fsamp,ACT);
        hold on,plot(ind/fsamp,massimo,'rO');
        xlim([0 length(eYW)]/fsamp);
        drawnow
    end
    
    % remove the point from ACT
    ACT(ind-round(LW*fsamp):ind+round(LW*fsamp)) = 0; 

    % initialization of w and wo
    w = eYW(:,ind); % Initial projection vector
    w = w/norm(w,2); % normalization

    w0 = randn(n, 1);
    w0 = w0/norm(w0, 2);

    % counter forfixed point
    counter = 0;
    
    % fixed point with tolerance TolX or max number of iterations

    tic
    while abs(abs(w0'*w)-1) > TolX && counter<MAXCOUNT,

        w0 = w; % for stopping at TolX

        temp = w'*eYW;  % estimation source

        w = eYW*G(temp)'/m - sum(DG(temp))*w/m;    % iterazione fixed point

        %w = w/norm(w, 2);

        w = w - (B*(B'*w)); % Deflation (decorrelation)
        
        w = w/norm(w, 2);   % normalizzazione
       if PLOT_PAUSE ==1,
           figure(1),subplot(3,2,[2 4 6]),hold off,plot([0:length(temp)-1]/fsamp,(temp.^2)/norm(temp.^2));
           hold on,plot(noindex/fsamp,temp(noindex).^2/norm(temp.^2),'Or');
           ylim([0 0.25])
           drawnow
       end    

        counter = counter+1;    % incrementa contatore
    end
    wT = real(w);   % save latest projecting vector

    B(:,trial) = wT;    % for deflation (riga 130)

    % estimation of the source s
    s = wT'*eYW;

    % find peaks in the source
    [TVI,pks,ind,in,index,SIL,CoV,C] = classification(s',fsamp,typedistance,MAXDR);

    % improve the estimation
    if SIL < 1 && length(in)>MINNDR,    
        SILOLD = eps;
        SILNEW = SIL;
        while SILNEW>SILOLD,
            SILOLD=SILNEW;
            Pindex = index;
            Ps = s;
            PTVI=TVI;

            wT = mean(eYW(:,index),2); % usa gli istanti di sparo per estrarre la nuova proiezione
            wT = wT/norm(wT,2); % normalizza
            s = wT'*eYW;    % ricalcola la sorgente

            [TVI,pks,ind,in,index,SILNEW,CoV,C] = classification(s',fsamp,typedistance,MAXDR);
        end
        index = Pindex;
        s=Ps;
        TVI = PTVI;

        % if SIL larger then thresold save the source
        if SILOLD>TH && length(in)>MINNDR,
            MUFiring{MU}=index+extFact; % salva istanti di sparo
            
            IPTsV(MU,:) = [zeros(extFact,1);TVI];  % salva innervation pulse train

            silouettev(MU) = SILOLD;    % salva silhouette SIL
            
            % plotting
            if PLOT_PAUSE ==1,
                figure(1)
                subplot(3,2,[3]);
                hold off,plot([0:length(TVI)-1]/fsamp,TVI);
                hold on,plot([index-1]/fsamp,TVI(index),'rO');xlabel('Time (s)');ylabel('Estimated Innervation Pulse Train (AU)');box off;xlim([0 length(EMG)/fsamp]);drawnow

                subplot(3,2,[5]);
                hold off,plot(index(2:end)/fsamp,1./diff(index/fsamp),'kO');
                xlim([0 length(EMG)/fsamp]);ylim([0 50]);xlabel('Time (s)');ylabel('Istantaneous Dischare Rate (pps)');box off;drawnow
                title(num2str([trial silouettev(MU)]))
                drawnow 
            end

        MU=MU+1;
    end

    else
            % includere campioni di artefatti nella variabile noindex ed
            % escluderli dall'activity index
            for i=1:length(index),
                temp = find(s(2:end)>0 & s(1:end-1)<0);
                ind1 = max(find(temp<index(i)));
                ind2 = min(find(temp>index(i)));
                ACT([temp(ind1):temp(ind2)]) = 0; % removing the highest point 
            end 
    end    
end

if ~exist('MUFiring','var'),
    MUFiring = [];
    IPTs = [];
    SIL = [];
    disp('No MUs');
    return;
elseif length(MUFiring)==1,
    IPTs = IPTsV;
    SIL = silouettev;
    disp('Only One');
    return;
end    

TOTAL_TIME = toc;
%%
% sort the sources and take the ones with lowest CoV
LW1 = 50e-3;
LW2 = 0.5e-3;
p=1;
DOPPI=[];
figure,
B = 1;
BUONA = [];
BUONAIPT = [];

for MU1 = 1:size(MUFiring,2)-1,
    COV = std(diff(MUFiring{MU1}))/mean(diff(MUFiring{MU1}))*100;
    good = [];
    good = [MU1,COV];
    if isempty(intersect(MU1,DOPPI)),
        MU1
        for MU2 = (MU1 + 1):size(MUFiring,2),
            if isempty(intersect(MU2,DOPPI)),

                CORR = fXcorr(MUFiring{MU1},MUFiring{MU2},round(LW1*fsamp));
                [massimo,indmax] = max(CORR);
                if indmax > round(LW2*fsamp)+1 && indmax<length(CORR)-round(LW2*fsamp),
                    SENS = sum(CORR(indmax-round(LW2*fsamp):indmax+round(LW2*fsamp)))/max([length(MUFiring{MU1}),length(MUFiring{MU2})])*100;
                
                    if SENS>30, % Soglia per riconoscere le sorgenti uguali, 30 % cross correlation
                        DOPPI(p) = MU2;  % salva doppioni
                        COV = std(diff(MUFiring{MU2}))/mean(diff(MUFiring{MU2}))*100;
                        good = [good;[MU2,COV]];
                        p=p+1;
                    end
                end
            end
        end
    [minimo,index] = min(good(:,2));
    good(index,:);
    BUONA{B} = MUFiring{good(index,1)};    
    BUONAIPT{B} = IPTsV(good(index,1),:);
    BUONASIL{B} = silouettev(good(index,1));

    B = B+1;
    end
end    

%%
% % urlread(['http://decomposition.bccn.uni-goettingen.de/index.php?license=',[num2str(LIC),'-',num2str(size(BUONA,2))]]);
firing = zeros(size(BUONA,2),length(EMG));
for MU=1:size(BUONA,2),firing(MU,BUONA{MU})=1;end;
figure,plot([0:length(firing)-1]/fsamp,fftfilt(2*hanning(fsamp),firing'));
legend(num2str([1:size(BUONA,2)]'));

MUFiring = BUONA;
IPTs = BUONAIPT;
SIL = BUONASIL;

TOTAL_TIME
toc
end

function eY = extension(Y,extfact)
% funzione per estendere le misure
eY = zeros(size(Y,1)*extfact,size(Y,2)+extfact);
for index = 1:extfact,
    eY((index-1)*size(Y,1)+1:index*size(Y,1),[1:size(Y,2)]+(index-1)) = Y;
end
end

function [TVI,pks,ind,in,index,DIST,CoV,C] = classification(s,fsamp,typedistance,MAXDR)
% funzione per identificare i discharge time
TVI = abs(s.^2);

TVI = TVI/norm(TVI);
[pks,ind] = findpeaks(TVI);   % find peaks in the squared source
[cl,C,sumd,D] = kmeans(TVI(ind),2,'start',typedistance); % classification
[massimo,k] = max([mean(TVI(ind(cl==1))) mean(TVI(ind(cl==2)))]); % selection of the highest
in1 = find(cl==k);
in = in1;

coh = sumd(k);
sep = sum(D(cl==k,setdiff([1 2],k)));
DIST = (sep-coh)/max([coh,sep]);        % Silhouette measure

index = ind(in1);

% remove spikes that are too close
% still to debug
if DIST>0,
    DR_TEMP = 1./diff(index/fsamp);
    inMAX = find(DR_TEMP>=MAXDR);
    index_TEMP = index;
    for i=1:length(inMAX),
        if inMAX(i)>1,
            [mas,inX] = max([TVI(index_TEMP(inMAX(i))) TVI(index_TEMP(inMAX(i)-1))]);
            if inX == 1
                index(inMAX(i)-1) = NaN;
            else
                index(inMAX(i)) = NaN;
            end
        end
    end
end

index(isnan(index)) = [];

ISI = diff(index/fsamp);
CoV = std(ISI)./mean(ISI)*100;

end

function xcor=fXcorr(Ind1,Ind2,Lim)
% cross correlation function
    xcor=zeros(2*Lim+1,1);
    for k=-Lim:Lim
        xcor(k+Lim+1)=length(intersect(Ind1,Ind2+k));
    end
end


