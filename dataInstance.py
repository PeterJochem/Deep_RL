import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# This class describes a data instance
class dataInstance():

    # Constructor
    def __init__(self, obserT1, obserT2, obserT3, obserT4):
         
        # Avoid copying each image
        # Store "pointers" to images instead
        self.obserT1 = obserT1
        self.obserT2 = obserT2
        self.obserT3 = obserT3
        self.obserT4 = obserT4
            

    # This will display all 4 images in a matplotlib display
    def viewAllImages(self):        
        w = 10
        h = 10
        fig = plt.figure(figsize = (10, 10))
        columns = 4
        rows = 1
        
        fig.suptitle('Data Instance', fontsize = 16)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.obserT4.pixels)
        
        fig.add_subplot(rows, columns, 2)
        plt.imshow(self.obserT3.pixels)
    
        fig.add_subplot(rows, columns, 3)
        plt.imshow(self.obserT2.pixels)

        fig.add_subplot(rows, columns, 4)
        plt.imshow(self.obserT1.pixels)

        plt.show() 

