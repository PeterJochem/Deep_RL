# Description
The dataset's csv files are formatted in the following way: [Gamma (degrees), beta (degrees), depth (cm), X-stress (N/cm^2), Z-stress (N/cm^2)] <br />

Juntao replicated Chen Li's physical experiment in a DEM simulation. He used the simulation to generate a dataset. One execution of Juntao's code had a fixed gamma value and varied the beta value. So a single run of the executable had a few iterations of the following: the granular material settles, the foot resets to a home postion, and the foot intrudes into the granular material. On each iteration of the previous cycle, a new beta value was used. Generating a dataset over the space we care about requires editing the program to have a diffrent gamma values, recompiling, and re-running. I later automated the above steps so that we can run one executable and it generates data over the entire space of parameters we care about. The dataset I used to create the graphs is from Juntao's simulations. On each run of his executable, he created a new csv file to record gamma, beta, depth, and stresses. I wrote a simple script to take each of the csv files (1 for each run of the exectuable) and combine them into one big csv file. This utility script is called ```createOneDataSet.py```. In it, you simply need to specify the names of the csv files you want to combine and it creates one big csv file called compiledSet.csv. This is then used as the dataset for the neural network to learn from. <br /> <br />


The dataset for generating the plots of the model's predictions as we increase the depth is slightly diffrent. This data is available [here](../../DEM_RFT_Comparisons/dataset/neural_net_data). This is the dataset I have been using to train the neural networks in the ../../DEM_RFT_Comparisons code. 


  
