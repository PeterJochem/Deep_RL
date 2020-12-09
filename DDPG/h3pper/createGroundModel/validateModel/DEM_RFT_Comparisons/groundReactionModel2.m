
classdef groundReactionModel2
   properties
      % Change this line to be the path to the <name>.h5 file
      %network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/visualizeData/validateChrono/mlab_data/model2.h5')
      %network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/model2.h5')
      
       network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/validateModel/DEM_RFT_Comparisons/model.h5')
       %network = importKerasNetwork('/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/validateModel/chen_li_paper_comparisons/model6.h5');
   
   end
   methods
       function [grf_x, grf_z, torque] = computeGRF(obj, gamma, beta, depth, velocity_x, velocity_z, theta_dt)
         prediction = obj.network.predict([gamma, beta, depth, velocity_x, velocity_z]);
         %gamma_deg = (gamma * 180.0)/3.14;
         %beta_deg = (beta * 180.0)/3.14;
         %prediction = obj.network.predict([gamma, beta, depth]);
         grf_x = prediction(1);
         grf_z = prediction(2);
         %torque = 0.0; % prediction(3);
         torque = prediction(3);
      end
   end
end