################################################################################################################3#
# Author: Tomás Artiach Hortelano                                                                                #
# description: Prepare the data from the GazeCapture dataset                                                     #
##################################################################################################################

import shutil
import os
import numpy as np
from numpy import savetxt
import json

gridSize = (25, 25)

#max size of the images
dimFace = (224, 224)
dimEyes = (64, 64)

#number of classes(options: 6, 15)
regions = 15

#path of the data extracted of the GazeCapture dataset
dataPath = "../data"
if regions == 15:
    destinationPath = "./dataset"
elif regions == 6:
    destinationPAth = "./dataset6Regions"

#create directory to store data
paths = ('train', 'test', 'validation')
for path in paths:
    os.mkdir(os.path.join(destinationPath,path))
    for i in range(regions):
        os.mkdir(os.path.join(destinationPath, path, str(i), "Face"))
        os.mkdir(os.path.join(destinationPath, path, str(i), "FaceGrid"))
        os.mkdir(os.path.join(destinationPath, path, str(i), "LeftEye"))
        os.mkdir(os.path.join(destinationPath, path, str(i), "RightEye"))


def discretize(x, y):
    if regions == 15:
        columns = [-1.21, 1.22]
        rows = [-2.116, -4.232, -6.38, -8.464]

    elif regions == 6:
        columns = [0]
        rows = [-4.232, -8.464]

    c = np.digitize(x, columns)
    r = np.digitize(y, rows)
    if regions == 15:
        region = r * 3 + c
    elif regions == 6:
        region = r * 2 +c

    return region


def createGrid(x, y, w, h):
    gridArray = np.zeros(gridSize)
    if (x + w >= 25):
        e = 25
    else:
        e = x + w
    for i in range(x, e):
        if (y + h >= 25):
            q = 25
        else:
            q = y + h
        for j in range(y, q):
            gridArray[i][j] = 1
    gridArray[y:y + h]

    return gridArray


def prepareData(dataPath, destinationPath):
    for subject in os.listdir(dataPath):
        if subject == ".DS_Store":
            pass
        else:
            subjectPath = os.path.join(dataPath, subject)
            dotInfoJson = open(os.path.join(subjectPath, "dotInfo.json"), "r")
            faceGridJson = open(os.path.join(subjectPath, "faceGrid.json"), "r")
            infoJson = open(os.path.join(subjectPath, "info.json"), "r")
            faceGrid = json.load(faceGridJson)
            dotInfo = json.load(dotInfoJson)
            info = json.load(infoJson)
            xCam = dotInfo["XCam"]
            yCam = dotInfo["YCam"]
            xGrid = faceGrid["X"]
            yGrid = faceGrid["Y"]
            wGrid = faceGrid["W"]
            hGrid = faceGrid["H"]
            dataset = info["Dataset"]
            if dataset == "val":
                dataset = "validation"
            for frame in os.listdir(os.path.join(subjectPath, "appleFace")):
                if frame == ".DS_Store":
                    pass
                else:
                    frameNumber = int(frame.split(".")[0])

                    region = discretize(xCam[frameNumber], yCam[frameNumber])
                    Grid = createGrid(xGrid[frameNumber], yGrid[frameNumber], wGrid[frameNumber], hGrid[frameNumber])
                    savePath = os.path.join(destinationPath, dataset, str(region))
                    # Face
                    shutil.copy(os.path.join(subjectPath, "appleFace", frame), os.path.join(savePath, "Face"))
                    os.rename(os.path.join(savePath, "Face", frame),
                              os.path.join(savePath, "Face", subject + "_" + frame))
                    # Left eye
                    shutil.copy(os.path.join(subjectPath, "appleLeftEye", frame), os.path.join(savePath, "LeftEye"))
                    os.rename(os.path.join(savePath, "LeftEye", frame),
                              os.path.join(savePath, "LeftEye", subject + "_" + frame))
                    # Right eye
                    shutil.copy(os.path.join(subjectPath, "appleRightEye", frame), os.path.join(savePath, "RightEye"))
                    os.rename(os.path.join(savePath, "RightEye", frame),
                              os.path.join(savePath, "RightEye", subject + "_" + frame))
                    # Face grid
                    savetxt(os.path.join(savePath, "FaceGrid", subject + "_" + str(frameNumber)), Grid, delimiter=',')


prepareData(dataPath, destinationPath)





