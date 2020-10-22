
classdef groundReactionModel2
   properties
      % Change this line to be the path to the <name>.h5 file
      network = importKerasNetwork('model2.h5')
   end
   methods
       function [grf_x, grf_z, torque] = computeGRF(obj, gamma, beta, depth, velocity_x, velocity_z, theta_dt)
         prediction = obj.network.predict([gamma, beta, depth, velocity_x, velocity_z, theta_dt])
         grf_x = prediction(1)
         grf_z = prediction(2)
         torque = prediction(3)
      end
   end
end