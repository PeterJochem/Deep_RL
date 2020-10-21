# URDF Folder
This has the hopping robot's URDF

# neural_networks 
This has the neural network mapping the robot's state to the ground reaction force. In order for Matlab to be able to read in and convert your neural network, you must save your model as a *.h5 file. The syntax is ```myModel.save(fileName.h5)```. If you do ```myModel.save(fileName)``` the model gets saved as an entire directory of smaller files. Matlab requires the .h5 file format. 
