
%% eval_learned_grf_models.m
%
% Description:
%   Evaluates (a? both? all? any?) learned ground reaction force (GRF)
%   model by computing GRFs over a range of foot states and comparing to
%   (either? both?) discrete-element method (DEM) simulations (from Chrono)
%   (or? and?) an analytical GRF model based on resistive force theory
%   (RFT).
%
% Inputs:
%   none
%
% Outputs:
%   none

function eval_learned_grf_models
%% Initialize workspace
clear;
close all;
clc;

%addpath('../') % adds foot_opt/code/planar to MATLABPATH
%addpath('../learnedGRFModels') % adds foot_opt/code/planar/learnedGRFModels to MATLABPATH
%init_env();

%% Import learned GRF models
grfModel1 = groundReactionModel1;

%% Parametertic Exploration 1:
% Vary the foot tilt ("beta"), angle of intrusion ("gamma"), and depth
% ("depth") and plot the GRF components ("grf_x, grf_y") that result at
% each (beta, gamma, depth) point:

betas = linspace(-pi/2,pi/2,11);
gammas = linspace(-pi/2,pi/2,11);
depths = linspace(0,0.12,200);

% preallocate arrays for grf1 (corresponds to grfModel1):
grf1_x = zeros(11,11,11);
grf1_y = zeros(11,11,11);

dx_dt = -1;
dy_dt = 1;
dtheta_dt = -1;

for k = 1:numel(depths)
    for i = 1:numel(betas)
        for j = 1:numel(gammas)
            [grf1_x(k,i,j),grf1_y(k,i,j)] = ...
                grfModel1.computeGRF(gammas(j),betas(i),depths(k));
            %[grf2_x(k,i,j),grf2_y(k,i,j),grm2_z(i,j)] = ...
            %   grfModel2.computeGRF(gammas(j),betas(i),depths(k),...
            %   dx_dt, dy_dt, dtheta_dt);
        end
    end
end

% animate the GRF predictions as depth increases:
betas_for_surf_plot = ones(11,1)*betas;
gammas_for_surf_plot = transpose(ones(11,1)*gammas);

vels_str = sprintf('$\\dot{x} = $ %0.5g, $\\dot{y} = $ %0.5g, $\\dot{\\theta} = $ %0.5g.',dx_dt,dy_dt,dtheta_dt);

animfig = figure('Renderer', 'painters', 'Position', [10 10 1200 800]);
v = VideoWriter('animate_both_models');
open(v);
for k = 1:numel(depths)
    clf;
    
    subplot(2,2,1)
    surf(betas_for_surf_plot,gammas_for_surf_plot,...
        reshape(grf1_x(k,:,:),numel(betas),numel(gammas)));
    zlim([-10.5, 50.5])
    xlabel('$\beta$ [rad]')
    ylabel('$\gamma$ [rad]')
    zlabel('$F_x(\beta,\gamma,y_f)$ [N]')
    title('GRF model 1, Fx')

    subplot(2,2,2)
    surf(betas_for_surf_plot,gammas_for_surf_plot,...
        reshape(grf1_y(k,:,:),numel(betas),numel(gammas)));
    zlim([-10.5, 50.5])
    xlabel('$\beta$ [rad]')
    ylabel('$\gamma$ [rad]')
    zlabel('$F_y(\beta,\gamma,y_f)$ [N]')
    title('GRF model 1, Fy')
    
    sgtitle(['Foot depth: $y_f =$',num2str(depths(k),3),' [m]'])
    
    drawnow()
    
    M(k) = getframe(animfig);
    writeVideo(v,M(k));
end
close(v);
movie(gcf,M);

end
