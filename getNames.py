################################################################################################################3#
# Author: Tomás Artiach Hortelano                                                                                #
# description: This script writes a file with the image names of the data in the dataset                         #
##################################################################################################################

import os
#path of the dataset
mainPath = "../dataset"
#number of classes of the dataset
regiones = 15
#name of the output file
fileName = "_"+str(regiones)+"_names.txt"
#Max number of images desired for the train set
maxElements = 3000
#different sets of data
paths = ("train", "validation", "test.txt")
iteracion = 0
counter = 0


for path in paths:
    numElements = 0
    if path == "validation":
        maxElements = maxElements*0.2
        print("validation")
    elif path == "test":
        maxElements = maxElements*0.1
    names = []
    for region in range(regiones):
        print(numElements)
        numElements = 0
        print(region)
        imageList = os.listdir(os.path.join(mainPath, path, str(region), "Face"))
        for image in imageList:
            if numElements < maxElements:
                if image == ".DS_Store":
                    pass
                else:
                    names.append(str(region) + "_" + image)
                    numElements+=1

    print(len(names))
    name = path + fileName
    with open(name, 'w') as f:
        for item in names:
            f.write("%s\n" % item)

