% LGN learning -- allowing overlapping nodes and WTA behavior

clear
clc
close all

% Initialize Retinal nodes and their properties

numRetina = 1;
totNeurons_Retina = 3000;
squareLength = 35;
retinaParams_old = {};

for i = 1:numRetina
    
    retinaParams_old(i).numNeurons = totNeurons_Retina;
    re = rand(totNeurons_Retina,1);
    
    retinaParams_old(i).x = squareLength*rand(totNeurons_Retina,2);
    
    centroid_RGC = mean(retinaParams_old(i).x);
    dist_center_to_all = bsxfun(@minus, retinaParams_old(i).x, centroid_RGC);
    
    dist_center_to_all = pdist2(retinaParams_old(i).x, centroid_RGC);
    gaussian_val = 6*exp(-(dist_center_to_all)/10);
    
    retinaParams_old(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams_old(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams_old(i).c = [-65+15*re.^2];
    %retinaParams(i).d = [8-6*re.^2];
    retinaParams_old(i).d = bsxfun(@minus, 8, gaussian_val);
    
    retinaParams_old(i).D = squareform(pdist(retinaParams_old(i).x));
    D = retinaParams_old(i).D;
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>10).*exp(-D/10); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end

retinaParams = retinaParams_old;

figure;
scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),'k','filled')
pause(0.2)

LGN_num = [400];%, 500];

for outerRadius = 8%:2:24

for numLGN= LGN_num
% Change wave-size!
retinaParams(i).Dk = 5*(D<2)- 2*(D>outerRadius).*exp(-D/10); 
retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));

x_3d = [retinaParams(i).x, zeros(size(retinaParams(i).x,1),1)];

%% Parameters of the LGN

eta = 0.1;
decay_rt = 0.01;
maxInnervations = totNeurons_Retina;

LGN_params = {};
connectedNeurons = [];
initConnections  = [];

mu_wts = 2.5;
sigma_wts = 0.14;

synapticMatrix_retinaLGN = zeros(totNeurons_Retina, numLGN);

% Choose random nodes on the arbitGeometry REtina -- and layer up!
layer_LGN = randi([1, totNeurons_Retina],numLGN,1);
LGN_pos2d = retinaParams(1).x(layer_LGN,:);

LGN_pos3d = [LGN_pos2d, ones(size(LGN_pos2d,1),1)];

di = pdist2(x_3d, LGN_pos3d);

% Normalizing synaptic matrix
for i = 1:numLGN
    synapticMatrix_retinaLGN(:,i) = normrnd(mu_wts, sigma_wts, [totNeurons_Retina,1]);
    synapticMatrix_retinaLGN(:,i) = synapticMatrix_retinaLGN(:,i)/mean(synapticMatrix_retinaLGN(:,i))*mu_wts;
end

LGN_synapticChanges = zeros(numLGN,1);
LGN_threshold = normrnd(70,2,numLGN,1);

LGNactivity = [];
initSynapticMatrix_retinaLGN = synapticMatrix_retinaLGN;

heatMap_wave = zeros(totNeurons_Retina,1); % # of times each neuron spikes

rfSizes = [150:50:750]';
   
%% Spontaneous synchronous bursts of Retina(s)
t = 0;

tic
rgc_connected = [];
time_rf = zeros(size(rfSizes,1),1);
numRfcompInit = 0;

while(1)
    t = t+1;
    
    if (t>1e6)
        break
    end

    if mod(t,1000) == 0
    
        s2Matrix= synapticMatrix_retinaLGN;
        s2Matrix(s2Matrix<0.1) = NaN;
        s2Matrix = ~isnan(s2Matrix);
        
        for ind_rf = 1:length(rfSizes)
            rf = rfSizes(ind_rf);
            temp1 = find(sum(s2Matrix)<rf);
            
            if and(length(temp1)>0.9*numLGN, time_rf(ind_rf) ==0)
                time_rf(ind_rf) = t;
            end
        end
        
        numRfcomp = length(time_rf)-length(find(time_rf ==0))
        
        if numRfcomp>=1
            
        if numRfcomp>numRfcompInit
            
            u = find(time_rf ~=0);
            minRFsize = u(1);
            
            s2Matrix= synapticMatrix_retinaLGN;
            s2Matrix(s2Matrix<0.1) = NaN;
            s2Matrix = ~isnan(s2Matrix);        
            
            temp1 = find(sum(s2Matrix)<rfSizes(minRFsize));

            figure;
            ctr = 1;
            for j = 1:20%numLGN        
                    subplot(4,5,j)
                    clear l

                    l = find(synapticMatrix_retinaLGN(:,temp1(j))>0.1);
                    hold on
                    scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],'k','filled')
                    scatter(retinaParams(1).x(l,2),retinaParams(1).x(l,1),[],'r','filled')
                    hold on
                    title(sprintf('LGN %d',temp1(j)))

                    ctr = ctr + 1;
            end 

            for j = temp1%1:numLGN
                rgc_connected = [rgc_connected, find(~isnan(s2Matrix(:,j)))'];
            end
            percent_node = length(unique(rgc_connected))/totNeurons_Retina;
        end
        numRfcompInit = numRfcomp;
        end
        
        if numRfcomp == length(rfSizes)
            break
        end
        
    
        % AFter every 1 sec (1000 ms) -- update the threshold to prevent random shifting % LGN_params.threshold
        for i = 1:numLGN
            
            if LGN_synapticChanges(i) < 200
                LGN_threshold(i) = max(LGNactivity(:,i))*1/5;
            end
            
            
        end
        LGNactivity = max(LGNactivity);
        disp(t)
        
    end
    
    spikeMat = zeros(numRetina*totNeurons_Retina,1);
    
    for i = 1:numRetina
       
        retinaParams(i).array_act = [];
        retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; 
        
        retinaParams(i).fired = find(retinaParams(i).v >= 30);
        fired = retinaParams(i).fired;
        
        retinaParams(i).array_act = retinaParams(i).x(fired,:);
        
        %retinaParams(i).firings = [retinaParams(i).firings; t+0*fired, fired];
        retinaParams(i).v(fired) = retinaParams(i).c(fired);
        retinaParams(i).u(fired) = retinaParams(i).u(fired) + retinaParams(i).d(fired);
        retinaParams(i).I = retinaParams(i).I + sum(retinaParams(i).Dk(:,fired),2);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).u = retinaParams(i).u + retinaParams(i).a.*(retinaParams(i).b.*retinaParams(i).v - retinaParams(i).u); 

        spikeMat(fired+(i-1)*totNeurons_Retina,1) = ones(length(fired),1);
        
        if length(fired)>30
            heatMap_wave(fired) = heatMap_wave(fired)+1;

        end
        
    end

    % Hebbian learning for LGN nodes
      
    y1_allLGN = [];
    thresh_LGN = [];
    y1_allLGN = spikeMat'*synapticMatrix_retinaLGN;
    y1_allLGN(y1_allLGN<0) = 0;
    thresh_LGN = LGN_threshold';
    
    LGNactivity(end+1,:) = y1_allLGN;

    yAct_allLGN = bsxfun(@minus, y1_allLGN, thresh_LGN);
    yAct_allLGN(yAct_allLGN<0) = 0;
    [maxAct, maxInd_LGN] = max(yAct_allLGN);
    
    % Check if max node is greater than threshold
    
    if yAct_allLGN(maxInd_LGN) > 0
    
    % Modify weights ONLY for maxInd_LGN
    
    x_input = spikeMat;
    wt_input = synapticMatrix_retinaLGN(:,maxInd_LGN);
    
    wt_input = wt_input + 0.5*(eta*(yAct_allLGN(maxInd_LGN))*x_input);
    wt_input = wt_input + 0.5*(eta*(yAct_allLGN(maxInd_LGN))*x_input);
    
    synapticMatrix_retinaLGN(:,maxInd_LGN) = wt_input;
    
    % Keep track of number of synapses added/pruned!
    LGN_synapticChanges(maxInd_LGN) = LGN_synapticChanges(maxInd_LGN) + 1;
    
    % Modifying threshold! If threshold is much larger than activity :reduce ELSE increase
    LGN_threshold(maxInd_LGN) = LGN_threshold(maxInd_LGN) + 0.005*yAct_allLGN(maxInd_LGN);
    
    
    % Normalize weights to a constant strength
    synapticMatrix_retinaLGN(:,maxInd_LGN) = synapticMatrix_retinaLGN(:,maxInd_LGN)/mean(synapticMatrix_retinaLGN(:,maxInd_LGN))*mu_wts;
    
    end
    
end
end

end

