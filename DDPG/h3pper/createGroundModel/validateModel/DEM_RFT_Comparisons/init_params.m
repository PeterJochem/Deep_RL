%% init_params
% Description:
%   Initializes parameters and returns them packed in a struct "params"
% Inputs:
%   none
% Outputs:
%   "params" - a multi-level struct composed of ground parameters (eg. RFT
%       coefficients), foot parameters (mass, inertia, length, etc),
%       simulation parameters, and more.

function params = init_params
    params.geom.foot_radius = 0.05; % distance from foot CoM [m]
    params.geom.foot_height = 0.005; % [m]
    params.geom.foot_area = (2*params.geom.foot_radius)^2;
    
    params.foot_mass = 0.25;   % [kg]
%     params.foot_moment_of_inertia = 1;    % [kg m^2], foot = slender rod
    params.foot_moment_of_inertia = params.foot_mass*...
        (params.geom.foot_radius^2)/3;    % [kg m^2], foot = slender rod
    params.grav = 9.81;     % [m/s^2]
    
    % text strings for each element of the foot state:
    params.state_strings  = {'$x_f$';
                             '$\dot{x}_f$';
                             '$y_f$';'$\dot{y}_f$';
                             '$\theta_f$';
                             '$\dot{\theta}_f$'};
    
    % initial state:
    params.init_state = [0;
                         0.0;
                         0.0;
                         -1;
                         0;
                         -0];
    
    % initial wrench (control input):
    params.init_wrench = [0;
                          0;
                          0.0];
    
    % "generic coefficients" from Table S2 of Li et al. supplementary
    % material:
    params.poppy_seeds_LP.A00 = 0.051; % 0.206
    params.poppy_seeds_LP.A10 = 0.047; % 0.169
    params.poppy_seeds_LP.B11 = 0.053; % 0.212
    params.poppy_seeds_LP.B01 = 0.083; % 0.358
    params.poppy_seeds_LP.Bn1 = 0.020; % 0.055, B(-1,1) in supp. matl. of Li et al.
    params.poppy_seeds_LP.C11 = -0.026;% -0.124
    params.poppy_seeds_LP.C01 = 0.057; % 0.253
    params.poppy_seeds_LP.Cn1 = 0.0; % 0.007, B(-1,1) in supp. matl. of Li et al.
    params.poppy_seeds_LP.D10 = 0.025; % 0.088
    params.poppy_seeds_LP.zeta = 1;    % 1, scaling factor
    
    params.glass_sphere_3mm_CP.A00 = 0.045;
    params.glass_sphere_3mm_CP.A10 = 0.031;
    params.glass_sphere_3mm_CP.B11 = 0.046;
    params.glass_sphere_3mm_CP.B01 = 0.084;
    params.glass_sphere_3mm_CP.Bn1 = 0.012;
    params.glass_sphere_3mm_CP.C11 = -0.124;
    params.glass_sphere_3mm_CP.C01 = 0.060;
    params.glass_sphere_3mm_CP.Cn1 = 0.000;
    params.glass_sphere_3mm_CP.D10 = 0.015;
    params.glass_sphere_3mm_CP.zeta = 1; % 0.214
    
    % Fitted DEM coefficients from DEM simulations (using Chrono):
    params.DEM.A00 = 0.04733;
    params.DEM.A10 = 0.04509;
    params.DEM.B11 = 0.06563;
    params.DEM.B01 = 0.08805;
    params.DEM.Bn1 = 0.01535;
    params.DEM.C11 = -0.04713;
    params.DEM.C01 = 0.05574;
    params.DEM.Cn1 = 0.01309;
    params.DEM.D10 = 0.03164;
    params.DEM.zeta = 1;
    
    params.gnd = params.poppy_seeds_LP; % poppy seeds are default terrain
    
    % damping constants, added to improve optim-to-sim transfer:
    params.gnd.damping.x = 0.00;
    params.gnd.damping.y = 0.00;
    params.gnd.damping.theta = 0.00;
    
    params.sim.N_timesteps = 201;
    params.sim.grf_Npts = 11;
    params.sim.stance = 'yielding'; % 'yielding' or 'static'
    params.sim.stop_vel_threshold = 1e-6;
    params.sim.start_horz_force_threshold = 1e-6;
	params.sim.start_vert_force_threshold = 1e-6;
    params.sim.start_rot_force_threshold = 1e-6;
    
    params.sim.type = 'opt'; % 'opt' or 'val'
end