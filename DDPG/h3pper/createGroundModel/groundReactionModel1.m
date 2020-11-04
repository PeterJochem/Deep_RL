
classdef groundReactionModel1
   properties
      % Change this line to be the path to the <name>.h5 file
      % network = importKerasNetwork('model1.h5')
      network = importKerasNetwork('/home/peter/Desktop/HoppingRobot_Fall/src/dset2Work/model.h5')
   end
   methods
       function [grf_x, grf_z] = computeGRF(obj, gamma, beta, depth)
         prediction = obj.network.predict([gamma, beta, depth])
         grf_x = prediction(1)
         grf_z = prediction(2)
      end
   end
end
