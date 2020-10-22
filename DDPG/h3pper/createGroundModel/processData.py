import numpy as np
import time
import random
import matplotlib.pyplot as plt
import warnings
import keras 

"""
warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    import tensorflow as tf
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""

"""Wrapper class for a training instance. Makes it easier to wrangle data later"""
class trainInstance:

    def __init__(self, inputVector, label):

        self.inputVector = inputVector # [gamma, beta, velocity_x, velocity_y, velocity_z]
        self.label = label # [x-stress/depth, y-stress/depth]

"""Set of data and methods for creating a dataset which can be processed by Tensorflow or Keras"""
class dataSet:
    
    def __init__(self, dataFilePath):
        
        dataFile = open(dataFilePath, "r")
        self.allData = []
        
        self.minDepth = 1000000 
        self.maxDepth = -1000000

        for dataItem in zip(dataFile):
             
            x = dataItem[0].split(",") # [Gamma, Beta, Depth, Position_x, Position_z, Velocity x, Velocity y, Velocity Z, GRF X, GRF Y, GRF Z] - the csv file format
            
            # intrude.cpp
            """
             t << ", " << gamma << ", " << beta << ", " << depth << ", " << position_x << ", " << position_y << ", " << position_z << ", " << x_dt << ", " << y_dt << ", " << z_dt << ", " << plate_ang_vel[0] << ", " << plate_ang_vel[1] << ", " << plate_ang_vel[2] << ", " << torque_x << ", " << torque_y << ", " << torque_z << ", " << grf_x << ", " << grf_y << ", " << grf_z << "\n";
            """
            
            # Convert everything to SI units
            gamma = float(x[1])  # / 3.14 # The angles are in radians
            beta = float(x[2]) # / 3.14
            depth = float(x[3])/100.0 # Convert cm to m - Chen Li uses cm^3
            
            
            grf_x = float(x[16]) 
            grf_z = float(x[18]) 
            
            velocity_x = float(x[7])/100.0 # Convert to meters 
            velocity_z = float(x[9])/100.0
            theta_dt = float(x[11]) 

            torque_y = float(x[14])

            newTrainInstance = trainInstance([gamma, beta, depth, velocity_x, velocity_z, theta_dt], [grf_x, grf_z, torque_y])
            #newTrainInstance = trainInstance([gamma, beta, depth], [grf_x, grf_z]) 

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
    
    
    def defineGraph_tf(self):
        
        self.useTf = True

        self.x = tf.placeholder(tf.float32, shape=(None, self.inputShape), name = 'x')
        self.y = tf.placeholder(tf.float32, shape=(None, self.outputShape), name = 'y')
        
        # (0, 0.1 and 120 epochs was best so far
        self.W1 = tf.Variable(tf.random_normal([self.inputShape, 28], 0.0, 0.1), name = 'W1')
        self.b1 = tf.Variable(tf.zeros([28]), name = 'b1')

        self.W2 = tf.Variable(tf.random_normal([28, 14], 0.0, 0.1), name = 'W2')
        self.b2 = tf.Variable(tf.zeros([14]), name = 'b2')

        self.W3 = tf.Variable(tf.random_normal([14, self.outputShape], 0.0, 0.1), name = 'W3')
        self.b3 = tf.Variable(tf.zeros([self.outputShape]), name = 'b3')

        #self.W4 = tf.Variable(tf.random_normal([12, 5], stddev = 0.03), name = 'W4')
        #self.b4 = tf.Variable(tf.random_normal([5]), name = 'b4')

        #self.W5 = tf.Variable(tf.random_normal([5, self.outputShape], stddev = 0.03), name = 'W5')
        #self.b5 = tf.Variable(tf.random_normal([self.outputShape]), name = 'b5')

        self.hidden_out1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.W1), self.b1))
        self.hidden_out2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_out1, self.W2), self.b2))
        #self.hidden_out3 = tf.nn.relu(tf.add(tf.matmul(self.hidden_out2, self.W3), self.b3))
        #self.hidden_out4 = tf.nn.relu(tf.add(tf.matmul(self.hidden_out3, self.W4), self.b4))

        self.y_pred = tf.add(tf.matmul(self.hidden_out2, self.W3), self.b3)

        #self.loss = tf.nn.l2_loss(self.y - self.y_pred) # / (len(myDataSet.allData))
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_pred))
        
        # Derivatives of the graph
        self.d_loss_dx = tf.gradients(self.loss, self.x)[0]  
        self.optimizer = tf.train.AdamOptimizer(0.1)
        self.train_op = self.optimizer.minimize(self.loss)


    def defineGraph_keras(self):
            
        self.useKeras = True 
        self.network = keras.Sequential([
            keras.layers.Dense(100, input_dim = self.inputShape),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(200),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(100),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(50),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(14),
            keras.layers.Activation(keras.activations.relu),
            keras.layers.Dense(self.outputShape)
        ])

        # Set more of the model's parameters
        self.optimizer = keras.optimizers.Adam(learning_rate = 0.01)
        
        # FIX ME - try removing the metrics - redundant
        self.network.compile(loss='mse', optimizer = self.optimizer, metrics=['mse']) 


    """Train the neural network using Tensorflow directly"""
    def train_tf(self):
        
        with tf.Session() as sess:
                
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
        
            #save_path = saver.save(sess, "../myNetworks/myNet.ckpt")
            
            feed_dict_train = {self.x: self.train_inputVectors, self.y: self.train_labels}
            feed_dict_test = {self.x: self.test_inputVectors, self.y: self.test_labels}

            loss_train_array = np.zeros(self.epochs)
            loss_test_array = np.zeros(self.epochs)  

            for i in range(self.epochs):
                sess.run(self.train_op, feed_dict_train)
                loss_train_array[i] = self.loss.eval(feed_dict_train) 
                loss_test_array[i] = self.loss.eval(feed_dict_test)

            # Feed in a real pair of data we trained on
            # grad_numerical = sess.run(self.d_loss_dx, {self.x: [self.test_inputVectors[0] ], self.y: [self.test_labels[0] ] } )  
            prediction = sess.run(self.y_pred, {self.x: [self.test_inputVectors[0]], self.y: [self.test_labels[0]]})
            
            # FIX ME - create a method to do this
            plt.plot(np.linspace(0, self.epochs, self.epochs), loss_test_array, color = "blue")
            plt.title('Loss Function vs Epoch - Hopping Foot')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()

            angle_resolution = 20
            angle_increment = 180.0 / angle_resolution # Remember we are using DEGREES
            
            F_X = np.zeros((angle_resolution, angle_resolution)) 
            F_Z = np.zeros((angle_resolution, angle_resolution))

            for i in range(angle_resolution):
                for j in range(angle_resolution):
                    
                    gamma = -1 * (180.0 / 2.0) + (j * angle_increment)
                    beta = (180.0 / 2.0) - (i * angle_increment)

                    # The label (self.y) is irrelevant to predicting the y here, after training
                    # FIX ME - try removing the y
                    prediction = sess.run(self.y_pred, {self.x: [[gamma, beta]], self.y: [[1.0, 1.0]]})
                        
                    F_X[i][j] = prediction[0][0]
                    F_Z[i][j] = prediction[0][1]
                    
            self.createPlots(F_Z, F_X) 
        
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
        
        self.network.fit([self.train_inputVectors], [self.train_labels], batch_size = len(self.train_inputVectors), epochs = epochs)
            
        angle_resolution = 40
        angle_increment = 180.0 / angle_resolution # Remember we are using DEGREES

        F_X = np.zeros((angle_resolution, angle_resolution))
        F_Z = np.zeros((angle_resolution, angle_resolution))
        
        self.network.save('model.h5')


def main():
   
    # Create a datset object with all our data
    #dataFile = "../../dataSets/dset3/allData/compiledSet.csv"
    
    dataFile = "datasets/dset3/intrude_dset.csv"
    myDataSet = dataSet(dataFile)
    myNetwork = NeuralNetwork(myDataSet)

    #myNetwork.defineGraph_tf()
    #myNetwork.train_tf()
    myNetwork.defineGraph_keras()
    myNetwork.train_keras(300)

if __name__ == "__main__":
    main()

