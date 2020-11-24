""" Describe this file """


def addFile(fileName, outputFile): 
    newFile = open(fileName, "r")

    for line in newFile:
        outputFile.write(line)


"""Open the file for writing. The file is created if it does not exist.
The handle is positioned at the end of the file. The data being written
will be inserted at the end, after the existing data. """

dataDirectory = "low_speed_data/"
outputFile = open(str(dataDirectory) + "compiledSet.csv", 'a')  

# Could give it a directory that is only .csv files rather than listing the files individually
listOfFiles = ["A1.csv", "A2.csv", "A3.csv", "B1.csv", "B2.csv", "B3.csv", 
               "C1.csv", "C2.csv", "C3.csv", "D1.csv", "D2.csv", "D3.csv",
               "E1.csv", "E2.csv", "E3.csv", "F1.csv", "F2.csv", "F3.csv",
               "G1.csv", "G2.csv", "G3.csv", "H1.csv", "H2.csv", "H3.csv", 
               "I1.csv", "I2.csv", "I3.csv", "J1.csv", "J2.csv", "J3.csv",
               "K1.csv", "K2.csv", "K3.csv", "L1.csv", "L2.csv", "L3.csv",
               "M1.csv", "M2.csv", "M3.csv"]       

#listOfFiles = ["A1.csv", "B1.csv", "C1.csv", "D1.csv", "E1.csv", "F1.csv", 
#               "G1.csv", "H1.csv", "I1.csv", "J1.csv", "K1.csv", "L1.csv",
#               "M1.csv"]

for item in listOfFiles:
    print("Adding " + str(item))
    nextFileName = dataDirectory + str(item)
    addFile(nextFileName, outputFile)


