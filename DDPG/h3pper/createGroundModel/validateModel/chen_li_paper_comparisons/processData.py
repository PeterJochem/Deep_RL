import time
import keras
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

"""Wrapper class for a training instance. Makes it easier to wrangle data later"""
class trainInstance:

    def __init__(self, inputVector, label):

        self.inputVector = inputVector
        self.label = label

"""A set of data and methods for creating a dataset which can be 
   processed by Tensorflow or Keras"""
class dataSet:
    
    def __init__(self, dataFilePath):
        
        dataFile = open(dataFilePath, "r")
        self.allData = []
        
        self.minDepth = float("inf")
        self.maxDepth = -float("inf")

        for dataItem in zip(dataFile):
             
            x = dataItem[0].split(",") # [Gamma (degrees), beta (degrees), depth, X-stress (N/cm^2), Z-stress (N/cm^2)]               
            gamma = float(x[0]) / 90.0 # * (2 * np.pi/360.0) 
            beta = float(x[1]) / 90.0  # * (2 * np.pi/360.0)
            depth = float(x[2])

            stress_x = float(x[3])
            stress_z = float(x[4])

            #newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z, theta_dt], [grf_x, grf_z, torque_y])
            #newTrainInstance = trainInstance([gamma, beta, depth], [grf_x, grf_z, torque_y]) 
            newTrainInstance = trainInstance([gamma, beta], [stress_x/depth, stress_z/depth])
            #newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z, theta_dt], [grf_z])

            #if (abs(grf_z) > 0.0001):
            #if (depth > 0.000001):
            if (True):
                self.allData.append(newTrainInstance)
            
                # Record the min and max depth for debugging and log it to console
                if (depth < self.minDepth):
                    self.minDepth = depth
                if (depth > self.maxDepth):
                    self.maxDepth = depth
        
        self.logStats()
        
    def logStats(self):
        
        numInstances = len(self.allData)
        print("\n\nThe dataset was created succesfully")
        print("The dataset has " + f"{numInstances:,}" + " training instances")
        print("The min depth is " + str(self.minDepth))
        print("The max depth is " + str(self.maxDepth))
        print("\n")

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
        self.epochs = 100 # Default value
        self.learning_rate = 0.01
        self.batchSize = 32
        
    def defineGraph_keras(self):
            
        self.network = keras.Sequential([
            keras.layers.Dense(300, input_dim = self.inputShape),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(200),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(100),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(50),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(14),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(self.outputShape),
        ])

        #self.network = keras.Sequential([
        #    keras.layers.Dense(20, input_dim = 2),
        #    keras.layers.Activation(keras.activations.relu),
        #    keras.layers.Dense(28),
        #    keras.layers.Activation(keras.activations.relu),
        #    keras.layers.Dense(14),
        #    keras.layers.Activation(keras.activations.relu),
        #    keras.layers.Dense(2)
        #])

        # Set more of the model's parameters
        self.optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.network.compile(loss='mse', optimizer = self.optimizer, metrics=['mse']) # use mae

    
    def createPlots(self, F_Z, F_X, useDefaultMap = False, customLimits = False):
            
        if (customLimits == True):
            plt.rcParams['image.cmap'] = 'jet' # Similiar color scale as Juntao's and Chen Li's data visualizations
        
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
            plt.clim(-0.25, 0.25); # This conrols the scale of the color map 
        
        plt.colorbar()
        ax2 = fig.add_subplot(rows, columns, 2)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        plt.imshow(F_X)
        ax2.set_ylabel('β')
        ax2.set_xlabel('γ')
        ax2.title.set_text('α_X')
        
        if (customLimits == True):
            plt.clim(-0.1, 0.1); # This conrols the scale of the color map

        plt.colorbar()
        plt.show()

    """Train the neural network on the dataset. 
       epochs: the number of passes over the entire dataset to make """ 
    def train_keras(self, epochs):
            
        epochs_default = 200
        sess_batch_size = len(self.train_inputVectors) # Use this for gradient descent 
        #sess_batch_size = 32 # Use this for stochastic gradient descent

        self.network.fit([self.train_inputVectors], [self.train_labels], batch_size = sess_batch_size, epochs = epochs)
        self.network.save('model.h5')
         

    """ Use the network's predictions to make graphs similiar to the stress/depth
        experimental data graphs. This shows us, in a human readable way, if the learned
        mapping from foot state to stresses is good. """
    def visualizeMapping(self):
        
        angle_resolution = 20
        angle_increment = 180.0 / angle_resolution # Remember we are using DEGREES

        F_X = np.zeros((angle_resolution, angle_resolution))
        F_Z = np.zeros((angle_resolution, angle_resolution))

        for i in range(angle_resolution):
            for j in range(angle_resolution):

                gamma = (-1 * (180.0 / 2.0) + (j * angle_increment)) / 90.0            
                beta = ((180.0 / 2.0) - (i * angle_increment)) / 90.0                

                prediction = self.network.predict([[gamma, beta]])

                F_X[i][j] = prediction[0][0]
                F_Z[i][j] = prediction[0][1]

        self.createPlots(F_Z, F_X, False, True)

def main():
   
    # Create a datset object with all our data
    dataFile = "dataset/compiledSet.csv"
    myDataSet = dataSet(dataFile)
    
    myNetwork = NeuralNetwork(myDataSet)
    myNetwork.defineGraph_keras()
    myNetwork.train_keras(300)
    myNetwork.visualizeMapping()

if __name__ == "__main__":
    main()
