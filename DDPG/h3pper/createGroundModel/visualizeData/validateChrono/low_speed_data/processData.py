import numpy as np
import time
import random
import matplotlib.pyplot as plt
import warnings
import keras 

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
        self.minZ = 1000000
        self.maxZ = -1000000

        numInRange = 0

        for dataItem in zip(dataFile):
             
            x = dataItem[0].split(",") # [Gamma (degrees), beta (degrees), depth, x-stress (N), z-stress (N)]            
            
            # Do input normalization?

            # Convert everything to SI units
            gamma = float(x[0]) #* (2 * np.pi/360.0) #* 10.0 # The angles are in radians
            beta = float(x[1]) #* (2 * np.pi/360.0) #* 10.0
            
            depth = float(x[2]) #* 10.0

            grf_x = float(x[3]) #* 25 #* 100.0
            grf_z = float(x[4]) #* 25 #* 100.0

            #print(str(depth) + ", " + str(grf_x) + ", " + str(grf_z) )

            #newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z, theta_dt], [grf_x, grf_z, torque_y])
            #newTrainInstance = trainInstance([gamma, beta, depth], [grf_x, grf_z, torque_y]) 
            #newTrainInstance = trainInstance([gamma, beta, depth], [grf_x, grf_z])
            newTrainInstance = trainInstance([gamma, beta], [grf_x/depth, grf_z/depth])
            #newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z, theta_dt], [grf_z])

            if (abs(grf_z) > 0.0001):
                #if (depth > 0.000001):
                #if (True):
                self.allData.append(newTrainInstance)
            
                # Record the min and max depth for debugging and log it to console
                if (depth < self.minDepth):
                    self.minDepth = depth
                if (depth > self.maxDepth):
                    self.maxDepth = depth
            
                if (grf_z < self.minZ):
                    self.minZ = grf_z
                if (grf_z > self.maxZ):
                    self.maxZ = grf_z



            #print(depth)
            if (depth > 0.0046 and depth < 0.0611):
                numInRange = numInRange + 1
                


        print("The number of items in the range is " + str(numInRange))
        self.logStats()
        
    def logStats(self):
        print("\n\nThe dataset was created succesfully")
        numInstances = len(self.allData)
        print("The dataset has " + f"{numInstances:,}" + " training instances")
        print("The min depth is " + str(self.minDepth) )
        print("The max depth is " + str(self.maxDepth) )
        print("The min grf z is " + str(self.minZ) )
        print("The max grf z is " + str(self.maxZ) )

        print("\n\n\n\n")


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


"""Wrapper class for a Tensorflow neural network"""
class NeuralNetwork:

    def __init__(self, dataset):
            
        self.useKeras = False
        self.useTf = False

        self.dataset = dataset
        self.inputShape = len(dataset.allData[0].inputVector)
        self.outputShape = len(dataset.allData[0].label)
        
        # Split data set into train and test
        np.random.shuffle(self.dataset.allData)
        self.trainSet = self.dataset.allData[ :int(0.80 * len(self.dataset.allData))]
        self.testSet = self.dataset.allData[int(0.80 * len(self.dataset.allData)): ]

        # Reformat the data to make it easier to wrangle
        self.train_inputVectors, self.train_labels, self.test_inputVectors, self.test_labels = dataset.reformat(self.trainSet, self.testSet)
        
        # Setup the saving of the network
        #saver = tf.train.Saver()
        
        # Hyper parameters 
        self.epochs = 50 # 120
        #self.batchSize = 
        #self.learningRate = 
        #
        # numLayers ...
        # FIX ME
    
    def defineGraph_keras(self):
            
        self.useKeras = True 
        
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

        # Set more of the model's parameters
        self.optimizer = keras.optimizers.Adam(learning_rate = 0.001)
        
        # use mae
        self.network.compile(loss='mae', optimizer = self.optimizer, metrics=['mse'])

    
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

        
    def train_keras(self, epochs):
        #batch_size = 1000
        #self.network.fit([self.train_inputVectors], [self.train_labels], batch_size = len(self.train_inputVectors), epochs = epochs)
        self.network.fit([self.train_inputVectors], [self.train_labels], batch_size = 32, epochs = epochs)

        self.network.save('model.h5')
         
        angle_resolution = 20
        angle_increment = 180.0 / angle_resolution # Remember we are using DEGREES

        F_X = np.zeros((angle_resolution, angle_resolution))
        F_Z = np.zeros((angle_resolution, angle_resolution))

        for i in range(angle_resolution):
            for j in range(angle_resolution):

                gamma = -1 * (180.0 / 2.0) + (j * angle_increment)
                beta = (180.0 / 2.0) - (i * angle_increment)

                prediction = self.network.predict([[gamma, beta]])

                F_X[i][j] = prediction[0][0]
                F_Z[i][j] = prediction[0][1]

        self.createPlots(F_Z, F_X, False, True)

def main():
   
    # Create a datset object with all our data
    #dataFile = "A1.csv"   
    dataFile = "compiledSet.csv"
    myDataSet = dataSet(dataFile)
    
    myNetwork = NeuralNetwork(myDataSet)
    myNetwork.defineGraph_keras()
    myNetwork.train_keras(100)
    #createPlots()

if __name__ == "__main__":
    main()

