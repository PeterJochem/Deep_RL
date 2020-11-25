classdef groundReactionModel2
   properties
      % Change this line to be the path to the <name>.h5 file
      network = importKerasNetwork('/home/peterjochem/Desktop/model50.h5');
   end
   methods
      
       function [grf_x, grf_z, torque] = computeGRF(obj, gamma, beta, depth, vel_x, vel_z, theta_dt)
         
         depth_mean = -0.029066263126838153;  
         depth_std_dev = 0.0802109606652637;
         velocity_x_mean = -0.029066263126838153;
         velocity_x_std_dev = 0.20105134664999805;
         velocity_z_mean = -0.1925198440300772;
         velocity_z_std_dev = 0.19517672593222468;
         
         depth_norm = (depth - depth_mean)/(0.08 * depth_std_dev);
         vel_x_norm = (vel_x - velocity_x_mean)/(0.08 * velocity_x_std_dev);
         vel_z_norm = (vel_z - velocity_z_mean)/(0.08 * velocity_z_std_dev);
         
         prediction = obj.network.predict([gamma, beta, depth, vel_x, vel_z]);
         
         grf_x = prediction(1); 
         grf_z = prediction(2);
         torque = -prediction(3);
      end
   end
end
