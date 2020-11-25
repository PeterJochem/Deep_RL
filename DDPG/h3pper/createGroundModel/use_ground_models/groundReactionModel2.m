classdef groundReactionModel2
   properties
      % Change this line to be the path to the <name>.h5 file
      %network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/model2.h5')
      network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/model2.h5')
      % network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/visualizeData/validateChrono/model.h5');
   end
   methods
       function [grf_x, grf_z, torque] = computeGRF(obj, gamma, beta, depth, vel_x, vel_z)
         prediction = obj.network.predict([gamma, beta, depth, vel_x, vel_z]);
         grf_x = prediction(1);
         grf_z = prediction(2);
         torque = prediction(3)
      end
   end
end
