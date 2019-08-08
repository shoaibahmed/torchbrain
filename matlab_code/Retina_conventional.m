%% Randomly damage the input nodes!

clear;
clc;
close all

% Initialize Retinal nodes and their properties

numRetina = 1;
totNeurons_Retina = 7000;
squareLen = 50;
retinaParams_old = {};

for i = 1:numRetina
    
    retinaParams_old(i).numNeurons = totNeurons_Retina;
    re = rand(totNeurons_Retina,1);
    
    retinaParams_old(i).x = squareLen*rand(totNeurons_Retina,2);

    retinaParams_old(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams_old(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams_old(i).c = [-65+15*re.^2];
    retinaParams_old(i).d = [8-6*re.^2];

    retinaParams_old(i).D = squareform(pdist(retinaParams_old(i).x));
    D = retinaParams_old(i).D;
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>4).*exp(-D); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end

fnum = 1;
retinaParams = retinaParams_old;

trialNum = 0;
a = [4];
for outerRadius = a

retinaParams(i).Dk = 2*(D<4).*exp(-D/100)- 3*(D>5).*exp(-D/10);
retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));

totTime = 2000; %Total time of simulation

pairWise_allRGC = sum(pdist(retinaParams(1).x));
heatMap_wave = zeros(totNeurons_Retina,1); % # of times each neuron spikes

numActiveNodes_wave = [];
clusterDisp_wave = [];
radius_wave = [];

for t = 1:totTime % simulation of totTime (ms)
    
    spikeMat = zeros(totNeurons_Retina,1);
    
    for i = 1:numRetina
        
        retinaParams(i).array_act = []; %Active nodes
        retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; % Noisy input
        
        retinaParams(i).fired = find(retinaParams(i).v >= 30);
        fired = retinaParams(i).fired; 

        retinaParams(i).array_act = retinaParams(i).x(fired,:);

        spikeMat(fired + (i-1)*totNeurons_Retina,1) = 1;
        
        %Find number of nodes that fired AND cluster contiguity
        pairWise_firingNode = sum(pdist(retinaParams(i).array_act));
        contig_firing = -log(pairWise_firingNode/pairWise_allRGC);

        if length(fired)>30
             heatMap_wave(fired) = heatMap_wave(fired)+1;
        
        
        figure(2);
        scatter(retinaParams(i).x(:,2),retinaParams(i).x(:,1),'k','filled')
        hold on
  
        if size(retinaParams(i).array_act,2) ~=0
            scatter(retinaParams(i).array_act(:,2),retinaParams(i).array_act(:,1),[],'r','filled')
            axis off
            pause(0.3)
        end

        end

        retinaParams(i).firings = [retinaParams(i).firings; t+0*fired, fired];
        retinaParams(i).v(fired) = retinaParams(i).c(fired);
        retinaParams(i).u(fired) = retinaParams(i).u(fired) + retinaParams(i).d(fired);
        retinaParams(i).I = retinaParams(i).I + sum(retinaParams(i).Dk(:,fired),2);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).u = retinaParams(i).u + retinaParams(i).a.*(retinaParams(i).b.*retinaParams(i).v - retinaParams(i).u); 
        
        % Keeping track of certain parameters: 
        % (1): Number of active nodes; (2): active nodes contiguity; (3): wave radius 
        
        numActiveNodes_wave(t) = length(fired);
        clusterDisp_wave = [clusterDisp_wave, contig_firing'];
        
        centroid_wave = mean(retinaParams(i).array_act,1);
        
        dist = pdist2(retinaParams(i).array_act, centroid_wave);
        radius_wave(t,:) = [centroid_wave, prctile(dist, 80)];

    end
    
end

end

%% PLOTS

% close all

% RASTER PLOT!

figure;
for i = 1:numRetina
    plot(retinaParams(i).firings(:,1),retinaParams(i).firings(:,2)+(i-1)*1000,'.')
    hold on
end
xlabel('Time')
ylabel('Neurons')
title('Raster plot')

% HEATMAP WITH TIME (CUMULATIVE SPIKES OVER ENTIRE SIMULATION)

figure;
scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],heatMap_wave(:,end),'filled')
colorbar
title('probability of neuron firing - hotspots of wave')
