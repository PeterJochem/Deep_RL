""" This is a utility file for taking a list of csv files and combinig them into one """

def addFile(fileName, outputFile): 
    newFile = open(fileName, "r")

    for line in newFile:
        outputFile.write(line)


"""Open the file for writing. The file is created if it does not exist.
The handle is positioned at the end of the file. The data being written
will be inserted at the end, after the existing data. """
dataDirectory = "dataset/"
outputFile = open(str(dataDirectory) + "compiledSet.csv", 'a')  

listOfFiles = ["intrude/data.csv", "extract/data.csv"]

for item in listOfFiles:
    print("Adding " + str(item))
    nextFileName = dataDirectory + str(item)
    addFile(nextFileName, outputFile)


