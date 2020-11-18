""" Describe this file """


def addFile(fileName, outputFile): 
    newFile = open(fileName, "r")

    for line in newFile:
        outputFile.write(line)


"""Open the file for writing. The file is created if it does not exist.
The handle is positioned at the end of the file. The data being written
will be inserted at the end, after the existing data. """

dataDirectory = "datasets/dset3/"
outputFile = open(str(dataDirectory) + "compiledSet.csv", 'a')  

# Could give it a directory that is only .csv files rather than listing the files individually
listOfFiles = ["extract.csv", "intrude.csv"]        

for item in listOfFiles:
    print("Adding " + str(item))
    nextFileName = dataDirectory + str(item)
    addFile(nextFileName, outputFile)


