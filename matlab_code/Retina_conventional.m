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
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>4).*exp(-D); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end




fnum = 1;
retinaParams = retinaParams_old;

% figure;
% scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),'k','filled')

%% 
trialNum = 0;
%a = repmat(6:2:30,1,3);
a = [4];
for outerRadius = a

% clearvars -except trialNum outerRadius
% clc
% close all
% 
% load('retinaParameters_10000_50x50.mat')
% 
% i = 1;
% numRetina = 1;
% D = retinaParams(i).D;
% totNeurons_Retina = size(retinaParams(i).x,1);
% 
% Change wave-size!
%retinaParams(i).Dk = 5*(D<2)- 2*(D>4).*exp(-D/10); 
retinaParams(i).Dk = 2*(D<4).*exp(-D/100)- 3*(D>5).*exp(-D/10); 

retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));

    
totTime = 2000; %Total time of simulation

%trackActivity = zeros(totTime, totNeurons_Retina);


pairWise_allRGC = sum(pdist(retinaParams(1).x));
heatMap_wave = zeros(totNeurons_Retina,1); % # of times each neuron spikes

numActiveNodes_wave = [];
clusterDisp_wave = [];
radius_wave = [];

% Col 1,2 -> centroid of spiking neurons; Col 3 -># of spiking neurons; Col 4 -> Contiguity of cluster
%centroid_wave = zeros(totTime,4); 

for t = 1:totTime % simulation of totTime (ms)
    
    spikeMat = zeros(totNeurons_Retina,1);
    
    for i = 1:numRetina
        
        retinaParams(i).array_act = []; %Active nodes
        retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; % Noisy input
        
        retinaParams(i).fired = find(retinaParams(i).v >= 30);
        fired = retinaParams(i).fired; 
        
%         if length(fired)>0
%             trackActivity(t,fired) = retinaParams(i).v(fired);
%         end
        
        retinaParams(i).array_act = retinaParams(i).x(fired,:);

        spikeMat(fired + (i-1)*totNeurons_Retina,1) = 1;
        
        %Find number of nodes that fired AND cluster contiguity
        pairWise_firingNode = sum(pdist(retinaParams(i).array_act));
        contig_firing = -log(pairWise_firingNode/pairWise_allRGC);
        
%         if length(fired) == 0
%             centroid_wave(t,:) = [NaN, NaN, length(fired),contig_firing];
%         else
%             centroidVal = spikeMat'*retinaParams(i).x/sum(spikeMat);
%             centroid_wave(t,:) = [centroidVal, length(fired),contig_firing];
%         end

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
         
        %x_3d = [retinaParams(i).array_act,zeros(length(retinaParams(i).array_act),1)];
        
        %fname = sprintf('waveMatFiles/FiringNodes_%d.mat',fnum);
        %save(fname, 'x_3d');
        
        %fnum = fnum + 1;
        
        end
        
        %fname = sprintf('retWaves_img/wave_outerR=%d_Time=%d.png',outerRadius,t);
        %saveas(gca, fname);
        
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
        
%         figure(3);
%         hold on
%         plot(clusterDisp_wave,'b--')
%         plot(radius_wave(:,3),'r')
%         legend('clusterDispersion', 'wave radius')
        
    end
    
%     if (mod(t,1000) == 0)
%         
%         % HEATMAP WITH TIME (CUMULATIVE SPIKES OVER ENTIRE SIMULATION)
%         
%         figure;
%         
%         subplot(1,2,1)
%         
%         heatMap_wave(heatMap_wave == 0) = 0.1;
%         scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),heatMap_wave(:,end),'filled')
%         title('probability of neuron firing - hotspots of wave')
%         
%         % CENTROID OF WAVE -- AS A FUNCTION OF NUMBER OF NEURONS FIRING AND CLUSTER CONTIGUITY 
%         indices_wave = find(centroid_wave(:,3)>20);
%         %figure;
%         
%         subplot(1,2,2)
%         hold on
%         scatter(retinaParams(1).x(:,2), retinaParams(1).x(:,1),'k','filled')
%         scatter(centroid_wave(indices_wave,2), centroid_wave(indices_wave,1),[],centroid_wave(indices_wave,3),'filled')
%         colorbar
%         title('Centroid of wave = colored based on # of neurons in cluster')
%         hold off
%     end
    
    
end
% useful = find(numActiveNodes_wave>30);
% 
% fsave = sprintf('4DecWaveTopology_10000_50x50/retinalWaves_topology_OuterRadius_%d_trial_%d',outerRadius, trialNum);
% save(fsave)
% trialNum = trialNum + 1;

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

% figure;
% hold on
% plot(numActiveNodes_wave)
% plot(clusterDisp_wave, 'b--')
% plot(radius_wave(:,3),'r')
% legend('clusterDispersion', 'wave radius')

% figure(10);
% hold on
% cdfplot(radius_wave(useful,3))

% PLOT CENTROID OF PROPAGATING WAVE

% indices_wave = find(centroid_wave(:,3)>20);
% 
% figure; 
% hold on
% scatter(retinaParams(1).x(:,2), retinaParams(1).x(:,1),'k','filled')
% scatter(centroid_wave(indices_wave,2), centroid_wave(indices_wave,1),'b','filled')
% title('Centroid of propagating wave')


% HEATMAP WITH TIME (CUMULATIVE SPIKES OVER ENTIRE SIMULATION)

figure;
% heatMap_wave(heatMap_wave == 0) = 0.1;
scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],heatMap_wave(:,end),'filled')
colorbar
title('probability of neuron firing - hotspots of wave')

% figure;
% heatMap_wave(heatMap_wave == 0) = 0.1;
% scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),heatMap_wave(:,end),'filled')
% title('probability of neuron firing - hotspots of wave')


% CENTROID OF WAVE -- AS A FUNCTION OF NUMBER OF NEURONS FIRING AND CLUSTER CONTIGUITY 
% indices_wave = find(centroid_wave(:,3)>20);
% figure;
% hold on
% scatter(retinaParams(1).x(:,2), retinaParams(1).x(:,1),'k','filled')
% scatter(centroid_wave(indices_wave,2), centroid_wave(indices_wave,1),[],centroid_wave(indices_wave,3),'filled')
% colorbar
% title('Centroid of wave = colored based on # of neurons in cluster')

% % CENTROID OF WAVE -- AS A FUNCTION OF NUMBER OF NEURONS FIRING AND CLUSTER CONTIGUITY 
% indices_wave = find(centroid_wave(:,3)>20);
% figure;
% hold on
% scatter(retinaParams(1).x(:,2), retinaParams(1).x(:,1),'k','filled')
% scatter(centroid_wave(indices_wave,2), centroid_wave(indices_wave,1),[],centroid_wave(indices_wave,4),'filled')
% colorbar
% title('Centroid of wave = colored based on cluster contiguity')


% REFRACTORY PERIODS OF NEURONS 

% figure;
% scatter(retinaParams(1).x(:,2), retinaParams(1).x(:,1),[], retinaParams(1).d,'filled')
% colorbar
% title('Refractory period (parameter) reset of neurons')


% for t = 1:totTime
%     scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),heatMap_wave(:,t),'filled')
%     pause(0.2)
% end


% figure;
% for i = 1:numRetina
%     
%     plot(retinaParams(i).dia_spatWave)
%     hold on
% end
% xlabel('Time')
% ylabel('Spatial Wave diameter')

