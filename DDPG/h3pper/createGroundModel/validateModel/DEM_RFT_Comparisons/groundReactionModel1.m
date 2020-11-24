
classdef groundReactionModel1
   properties
      % Change this line to be the path to the <name>.h5 file
      network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/model50.h5');
   end
   methods
      
       function [grf_x, grf_z, torque] = computeGRF(obj, gamma, beta, depth, vel_x, vel_z, theta_dt)
         
         depth_norm = (depth - (-0.029066263126838153))/(0.08 * 0.0802109606652637);
         vel_x_norm = (vel_x - (-0.13453838217498656))/(0.08 * 0.20105134664999805);
         vel_z_norm = (vel_z - (-0.1925198440300772))/(0.08 * 0.19517672593222468);
         
         prediction = obj.network.predict([gamma, beta, depth, vel_x, vel_z]);
         % prediction = obj.network.predict([gamma, beta, depth, vel_x, vel_z]);
         
         %grf_x = (prediction(1) * 3.372860633831931) + (-0.13453838217498656); 
         %grf_z = (prediction(2) * 7.805342216418955) + (-0.1925198440300772);
         %torque = (prediction(3) * 0.024827584625318085) + (-0.1925198440300772);
         
         grf_x = prediction(1); 
         grf_z = prediction(2);
         torque = -prediction(3);
        
         % /10.0; % / depth; %/ depth; %* 0.05 * 0.05 * depth * 1000000;
         %grf_z = prediction(2); % /10.0; % / depth; %* 0.05 * 0.05 * depth * 1000000;
         %grf_x = 0.0; % prediction(1) * 25; % (depth * 100); %* 10.0) * (5 * 5); %/ depth; %* 0.05 * 0.05 * depth * 1000000;
         %grf_z = prediction(2) * 25; % * (depth * 100); % * 10.0) * (5 * 5);
         %torque = 0.0;
         %torque = prediction(3); %/10.0;
      end
   end
end
