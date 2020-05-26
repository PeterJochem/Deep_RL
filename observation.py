import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# This class describes an observation seen by the agent  
class observation:
   
    def __init__(self, pixels):
        
        # Size of the image after downsampling
        self.downSizedLength = 64
        self.downSizedWidth = 64

        self.pixels = pixels
        self.pixels = self.processImage()
        
        # self.viewImage()
    
    # This method takes a 3-D array and converts
    # it to a format so that a human can view it
    # Input: The 3-D array that is the state
    # Returns: Void
    def viewImage(self):
        plt.imshow(self.pixels)
        plt.show()

    # Convert the image to grayscale and rescale it to 86x86
    # Input: The original 210 x 160 x 3 image
    # Returns: The 86 x 86 x 1 image
    def processImage(self):

        length = len(self.pixels)
        width = len(self.pixels[0])

        # This will store the new image
        newArray = np.zeros((length, width))

        # Convert the image to grayscale
        for i in range(len(self.pixels)):
            for j in range(len(self.pixels[i])):

                average = (self.pixels[i][j][0] + self.pixels[i][j][1] + self.pixels[i][j][2]) / 3.0
                newArray[i][j] = average

        # Resize the image
        newArray = cv2.resize(newArray, dsize = (self.downSizedLength, self.downSizedWidth), interpolation = cv2.INTER_CUBIC)

        return newArray

