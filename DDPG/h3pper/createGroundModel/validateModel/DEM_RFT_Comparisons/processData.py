import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import statistics
import warnings
import random
import keras 
import time

"""Wrapper class for a training instance. Makes it easier to wrangle data later"""
class trainInstance:

    def __init__(self, inputVector, label):

        self.inputVector = inputVector
        self.label = label

"""Set of data and methods for creating a dataset which can be processed by Tensorflow or Keras"""
class dataSet:
    
    def __init__(self, dataFilePath):
        
        dataFile = open(dataFilePath, "r")
        self.allData = []
        
        self.minDepth = 1000000 
        self.maxDepth = -1000000
            
        all_grf_x = []
        all_grf_z = []
            
        all_depth = []
        
        all_velocity_x = []
        all_velocity_z = []
        
        all_torque = []
        
        for dataItem in zip(dataFile):
            x = dataItem[0].split(",")
            
            grf_x = float(x[16]) #* 10.0
            grf_z = float(x[18]) #* 10.0
            velocity_x = float(x[7]) / 100.0 # Convert to meters
            velocity_z = float(x[9]) / 100.0
            torque_y = float(x[14])
            depth = float(x[3]) / 100.0

            all_grf_x.append(grf_x)
            all_grf_z.append(grf_z)
            
            all_depth.append(depth)
            
            all_velocity_x.append(velocity_x)
            all_velocity_z.append(velocity_z)

            all_torque.append(torque_y)
        
            
        std_dev_grf_x = statistics.pstdev(all_grf_x)     
        std_dev_grf_z = statistics.pstdev(all_grf_z)
        std_dev_depth = statistics.pstdev(all_depth)
        std_dev_vel_x = statistics.pstdev(all_velocity_x)
        std_dev_vel_z = statistics.pstdev(all_velocity_z)
        std_dev_torque = statistics.pstdev(all_torque)
        
        avg_grf_x = statistics.mean(all_grf_x)
        avg_grf_z = statistics.mean(all_grf_z)
        avg_depth = statistics.mean(all_depth)
        avg_vel_x = statistics.mean(all_velocity_x)
        avg_vel_z = statistics.mean(all_velocity_z)
        avg_torque = statistics.mean(all_torque)
        
        min_grf_x = min(all_grf_x)
        min_grf_z = min(all_grf_z)
        min_depth = min(all_depth)
        min_vel_x = min(all_velocity_x)
        min_vel_z = min(all_velocity_z)
        min_torque = min(all_torque)
        
        print("avg_grf_x: " + str(avg_grf_x))
        print("avg_grf_z: " + str(avg_grf_z))
        print("avg_depth: " + str(avg_depth))
        print("avg_vel_x: " + str(avg_vel_x))
        print("avg_vel_z: " + str(avg_vel_z))
        print("avg_torque: " + str(avg_torque))
        
        print("std_dev_grf_x " + str(std_dev_grf_x))
        print("std_dev_grf_z " + str(std_dev_grf_z))
        print("std_dev_depth " + str(std_dev_depth))
        print("std_dev_vel_x " + str(std_dev_vel_x))
        print("std_dev_vel_z " + str(std_dev_vel_z))
        print("std_dev_torque " + str(std_dev_torque))
              
        dataFile = open(dataFilePath, "r")
        for dataItem in zip(dataFile):
             
            x = dataItem[0].split(",") # [Gamma, Beta, Depth, Position_x, Position_z, Velocity x, Velocity y, Velocity Z, GRF X, GRF Y, GRF Z] - the csv file format
            
            # Convert everything to SI units
            gamma = float(x[1])  # The angles are in radians
            beta = float(x[2]) 
            depth = float(x[3]) / 100.0

            grf_x = float(x[16]) 
            grf_z = float(x[18])  

            velocity_x = float(x[7]) / 100.0 # Convert to meters 
            velocity_z = float(x[9]) / 100.0
            
            theta_dt = float(x[11]) 
            torque = float(x[14]) 
            
            # Normalize the data
            norm_depth = (depth - avg_depth)/(0.08 * std_dev_depth)
            norm_vel_x = (velocity_x - avg_vel_x)/(0.08 * std_dev_vel_x)
            norm_vel_z = (velocity_z - avg_vel_z) /(0.08 * std_dev_vel_z)

            norm_grf_x = (grf_x - avg_grf_x)/std_dev_grf_x
            norm_grf_z = (grf_z - avg_grf_z)/std_dev_grf_z            
            norm_torque = (torque - avg_torque)/std_dev_torque     
            
             
            #newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z, theta_dt], [grf_x, grf_z, torque_y])
            newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z], [grf_x, grf_z, torque])
            
            #if (depth < 0.011 and depth > -0.011):
            #if (abs(grf_z) > 0.01):
            if (True):
                self.allData.append(newTrainInstance)
            
                # Record the min and max depth for debugging and log it to console
                if (depth < self.minDepth):
                    self.minDepth = depth
                if (depth > self.maxDepth):
                    self.maxDepth = depth
    
        self.logStats()
        
    def logStats(self):
        print("The dataset was created succesfully")
        numInstances = len(self.allData)
        print("The dataset has " + f"{numInstances:,}" + " training instances") 


    """ Tensorflow reformatting. Split the lists of data 
    into their input and outputs """
    def reformat(self, trainSet, testSet):
            
        train_inputVectors = []
        train_labels = []
        for i in range(len(trainSet)):
            train_inputVectors.append(trainSet[i].inputVector)
            train_labels.append(trainSet[i].label)
        
        test_inputVectors = []
        test_labels = []
        for i in range(len(testSet)):
            test_inputVectors.append(testSet[i].inputVector)
            test_labels.append(testSet[i].label)

        return train_inputVectors, train_labels, test_inputVectors, test_labels


"""Wrapper class for a Keras/Tensorflow neural network"""
class NeuralNetwork:

    def __init__(self, dataset):
            
        self.dataset = dataset
        self.inputShape = len(dataset.allData[0].inputVector)
        self.outputShape = len(dataset.allData[0].label)
        
        # Split data set into train and test
        np.random.shuffle(self.dataset.allData)
        self.trainSet = self.dataset.allData[:int(0.80 * len(self.dataset.allData))]
        self.testSet = self.dataset.allData[int(0.80 * len(self.dataset.allData)):]

        # Reformat the data to make it easier to wrangle
        self.train_inputVectors, self.train_labels, self.test_inputVectors, self.test_labels = dataset.reformat(self.trainSet, self.testSet)
        
        # Hyper parameters 
        self.learningRate = 0.001

    def defineGraph_keras(self):
            
        self.useKeras = True 
        
        self.network = keras.Sequential([
            keras.layers.Dense(200, input_dim = self.inputShape),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(120),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(60),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(110),
            keras.layers.Activation(keras.activations.relu),
            #keras.layers.Dense(75),
            #keras.layers.Activation(keras.activations.relu),
            #keras.layers.Dense(50),
            #keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(20),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(self.outputShape)
        ])
           
        # Set more of the model's parameters
        self.optimizer = keras.optimizers.Adam(learning_rate = self.learningRate)
        self.network.compile(loss='mse', optimizer = self.optimizer, metrics=['mae']) 
       
    
    """ This takes a learned model and makes the same graphs in the original Chen Li paper. Useful for comparing
    our model to the experimental data in the Chen Li paper.
    F_Z - List of ground reaction forces in Z-direction 
    F_X - List of ground reaction forces in X-direction
    Model must map [gamma, beta] -> [ground_reaction_force_x/depth, ground_reaction_force_z/depth]"""
    def createPlots(self, F_Z, F_X, useDefaultMap = False, customLimits = False):
            
        if (customLimits == True):
            plt.rcParams['image.cmap'] = 'jet' # Similiar color scale as Juntao's data visualizations
        
        fig = plt.figure(figsize = (4, 4))
        columns = 2
        rows = 1
        ax1 = fig.add_subplot(rows, columns, 1)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        plt.imshow(F_Z)
        ax1.set_ylabel('β')
        ax1.set_xlabel('γ')
        ax1.title.set_text('α_Z')

        if (customLimits == True):
            plt.clim(-0.25, 0.25); # This conrols the scale of the color map - I set it to better match Junatos visualization
        
        plt.colorbar()

        ax2 = fig.add_subplot(rows, columns, 2)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        plt.imshow(F_X)
        ax2.set_ylabel('β')
        ax2.set_xlabel('γ')
        ax2.title.set_text('α_X')
        
        if (customLimits == True):
            plt.clim(-0.1, 0.1); # This conrols the scale of the color map - I set it to better match Junatos visualization

        plt.colorbar()
        plt.show()
       

    def train_keras(self, epochs=100):
        
        sess_batch_size = len(self.train_inputVectors) # Use this for gradient descent 
        # sess_batch_size = 32 # Use this for stochastic gradient descent

        self.network.fit([self.train_inputVectors], [self.train_labels], batch_size = sess_batch_size, epochs = epochs)
        self.network.save('model.h5')

def main():
   
    # Create a datset object with all our data
    dataFile = "/home/peterjochem/Desktop/Deep_RL/DDPG/h3pper/createGroundModel/datasets/dset3/compiledSet.csv"   
      
    myDataSet = dataSet(dataFile)
    myNetwork = NeuralNetwork(myDataSet)
    myNetwork.defineGraph_keras()
    myNetwork.train_keras(100)

if __name__ == "__main__":
    main()

