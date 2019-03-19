import os

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def main(dirName):

#    dirName = '/home/binuri/work_Msc/IR_Assignment3/20_newsgroup';
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    
    # Print the files
#    for elem in listOfFiles:
#        print(elem)
# 
#    print ("****************")
    
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
 

        
    # Print the files    
#    for elem in listOfFiles:
#        i = i+1
#        print(elem)    
#    print(i)           
   
    return (listOfFiles)

     
if __name__ == '__main__':
#    dirName = '/home/binuri/work_Msc/IR_Assignment3/20_newsgroupSample';
    listOfFiles = main(dirName)
    print(listOfFiles)
    
    
# your corpus
#import fetchData
#dirName = '/home/binuri/work_Msc/IR_Assignment3/20_newsgroup';
#listOfFiles = fetchData.main(dirName)
#fileList = list()
#for f in listOfFiles:
#    fileList.append(codecs.open(f, "r", encoding="ISO-8859-1").read())
