################################################################################################################3#
# Author: Tomás Artiach Hortelano                                                                                #
# description: This script contains functions that load batches of data. Used by the data generators in Model.py #
##################################################################################################################

import numpy as np
from random import randint
import cv2
from os.path import join
import os
import gc
from tensorflow.keras.utils import to_categorical

def loadNames(path):
    existingNames = []
    f = open(path, "r")
    name = f.read().splitlines()
    existingNames.append(name)

    return existingNames

def loadRandomBatch(names, path, faceShape, eyeShape, batchSize, architectureType, regions):

    if architectureType == "faceOnly":
        # data structures for batches
        face_batch = np.zeros(shape=(batchSize, faceShape[0], faceShape[1], faceShape[2]), dtype=np.float32)
        y_batch = np.zeros((batchSize, 1), dtype=np.float32)
        dim = (faceShape[0], faceShape[1])
        i = 0

        while i < batchSize:

            # get random number for random frame
            randomIndex = randint(0, len(names)-1)
            randomName = names[randomIndex]
            region = int(randomName.split('_')[0])
            frameName = randomName.split('_', 1)[1]
            #get image and save it in corresponding array
            #face
            facePath = join(path, str(region), 'Face', frameName)
            if os.path.exists(facePath) and os.path.getsize(facePath) > 0:
                face = cv2.imread(facePath)
                face = cv2.normalize(face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_32F)
                face = cv2.resize(face, dim, interpolation = cv2.INTER_AREA)

                # save y
                y_batch[i] = region
                #save to the batch
                face_batch[i] = face
                i += 1

        y_batch = to_categorical(y_batch, regions)
        return [face_batch], y_batch

    elif architectureType == "eyesOnly":
        #data structures for batches
        left_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        right_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        y_batch = np.zeros((batchSize, 1), dtype=np.float32)

        dimEyes = (eyeShape[0], eyeShape[1])
        i = 0

        while i < batchSize:

            # get random number for random frame
            randomIndex = randint(0, len(names)-1)
            randomName = names[randomIndex]
            region = int(randomName.split('_')[0])
            frameName = randomName.split('_', 1)[1]
            #get image and save it in corresponding array
            #leftEye
            leftEyePath = join(path, str(region), 'LeftEye', frameName)
            if os.path.exists(leftEyePath) and os.path.getsize(leftEyePath) > 0:
                leftEye = cv2.imread(leftEyePath)
                leftEye = cv2.normalize(leftEye, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
                leftEye = cv2.resize(leftEye, dimEyes, interpolation = cv2.INTER_AREA)
                #rightEye
                rightEyePath = join(path, str(region), 'RightEye',  frameName )
                if os.path.exists(rightEyePath) and os.path.getsize(rightEyePath) > 0:
                    rightEye = cv2.imread(rightEyePath)
                    rightEye = cv2.normalize(rightEye, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_32F)
                    rightEye = cv2.resize(rightEye, dimEyes, interpolation = cv2.INTER_AREA)
                    # save y
                    y_batch[i] = region
                    #save to the batch
                    left_eye_batch[i] = leftEye
                    right_eye_batch[i] = rightEye
                    i += 1
        y_batch = to_categorical(y_batch, regions)
        return [left_eye_batch, right_eye_batch], y_batch

    elif architectureType == "full":
        # data structures for batches
        face_batch = np.zeros(shape=(batchSize, faceShape[0], faceShape[1], faceShape[2]), dtype=np.float32)
        right_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        left_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        face_grid_batch = np.zeros(shape=(batchSize, 25, 25), dtype=np.float32)
        y_batch = np.zeros((batchSize, 1), dtype=np.float32)

        dim = (faceShape[0], faceShape[1])
        dimEyes = (eyeShape[0],eyeShape[1])
        i = 0

        while i < batchSize:

            # get random number for random frame
            randomIndex = randint(0, len(names)-1)
            randomName = names[randomIndex]
            region = int(randomName.split('_')[0])
            frameName = randomName.split('_', 1)[1]
            #get image and save it in corresponding array
            #face
            facePath = join(path, str(region), 'Face', frameName)
            if os.path.exists(facePath) and os.path.getsize(facePath) > 0:
                face = cv2.imread(facePath)
                face = cv2.normalize(face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_32F)
                face = cv2.resize(face, dim, interpolation = cv2.INTER_AREA)
                #leftEye
                leftEyePath = join(path, str(region), 'LeftEye', frameName)
                if os.path.exists(leftEyePath) and os.path.getsize(leftEyePath) > 0:
                    leftEye = cv2.imread(leftEyePath)
                    leftEye = cv2.normalize(leftEye, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_32F)
                    leftEye = cv2.resize(leftEye, dimEyes, interpolation = cv2.INTER_AREA)
                    #rightEye
                    rightEyePath = join(path, str(region), 'RightEye',  frameName )
                    if os.path.exists(rightEyePath) and os.path.getsize(rightEyePath) > 0:
                        rightEye = cv2.imread(rightEyePath)
                        rightEye = cv2.normalize(rightEye, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_32F)
                        rightEye = cv2.resize(rightEye, dimEyes, interpolation = cv2.INTER_AREA)
                        #faceGrid
                        faceGridName = frameName.split(".")[0]
                        tmp = faceGridName.split("_")
                        faceGridName = tmp[0] + "_" + str(int(tmp[1]))
                        faceGridPath = join(path, str(region), 'FaceGrid', faceGridName)
                        faceGrid = np.loadtxt(faceGridPath, delimiter = ',')
                        # save y
                        y_batch[i] = region
                        #save to the batch
                        face_batch[i] = face
                        left_eye_batch[i] = leftEye
                        right_eye_batch[i] = rightEye
                        face_grid_batch[i] = faceGrid
                        i += 1
        y_batch = to_categorical(y_batch, regions)
        return [face_batch, left_eye_batch, right_eye_batch, face_grid_batch], y_batch


def getTestBatch(names, datasetPath, faceShape, eyeShape, batchSize, callCounter, architectureType, regions = 6):

    if architectureType == "faceonly":
        face_batch = np.zeros(shape=(batchSize, faceShape[0], faceShape[1], faceShape[2]), dtype=np.float32)
        y_batch = np.zeros((batchSize, 1), dtype=np.float32)

        dimFace = (faceShape[0], faceShape[1])
        j = 0

        while j < batchSize:

            region = int(names[j+callCounter*batchSize].split('_')[0])
            frameName = names[j+callCounter*batchSize].split('_', 1)[1]
            # get image and save it in corresponding array
            # face
            facePath = os.path.join(datasetPath, str(region), "Face", frameName)
            if os.path.exists(facePath) and os.path.getsize(facePath) > 0:
                face = cv2.imread(facePath)
                face = cv2.resize(face, dimFace, interpolation=cv2.INTER_AREA)

                # save y
                y_batch[j] = region
                # save to the batch
                face_batch[j] = face

                j+=1

        y_batch = to_categorical(y_batch, regions)
        return [face_batch], y_batch

    elif architectureType == "eyesOnly":
        left_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        right_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        y_batch = np.zeros((batchSize, 1), dtype=np.float32)

        dimEyes = (eyeShape[0], eyeShape[1])
        j = 0

        while j < batchSize:

            region = int(names[j+callCounter*batchSize].split('_')[0])
            frameName = names[j+callCounter*batchSize].split('_', 1)[1]
            # get image and save it in corresponding array
            # leftEye
            leftEyePath = join(datasetPath,  str(region), "LeftEye", frameName)
            if os.path.exists(leftEyePath) and os.path.getsize(leftEyePath) > 0:
                leftEye = cv2.imread(leftEyePath)
                leftEye = cv2.resize(leftEye, dimEyes, interpolation=cv2.INTER_AREA)
                # rightEye
                rightEyePath = join(datasetPath , str(region) , "RightEye" , frameName)
                if os.path.exists(rightEyePath) and os.path.getsize(rightEyePath) > 0:
                    rightEye = cv2.imread(rightEyePath)
                    rightEye = cv2.resize(rightEye, dimEyes, interpolation=cv2.INTER_AREA)
                    # save y
                    y_batch[j] = region
                    # save to the batch
                    left_eye_batch[j] = leftEye
                    right_eye_batch[j] = rightEye
                    j+=1

        y_batch = to_categorical(y_batch, regions)
        return [left_eye_batch, right_eye_batch], y_batch

    elif architectureType == "full":
        face_batch = np.zeros(shape=(batchSize, faceShape[0], faceShape[1], faceShape[2]), dtype=np.float32)
        right_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        left_eye_batch = np.zeros(shape=(batchSize, eyeShape[0], eyeShape[1], eyeShape[2]), dtype=np.float32)
        face_grid_batch = np.zeros(shape=(batchSize, 25, 25), dtype=np.float32)
        y_batch = np.zeros((batchSize, 1), dtype=np.float32)

        dimFace = (faceShape[0], faceShape[1])
        dimEyes = (eyeShape[0], eyeShape[1])
        j = 0

        while j < batchSize:

            region = int(names[j+callCounter*batchSize].split('_')[0])
            frameName = names[j+callCounter*batchSize].split('_', 1)[1]
            # get image and save it in corresponding array
            # face
            facePath = os.path.join(datasetPath, str(region), "Face", frameName)
            if os.path.exists(facePath) and os.path.getsize(facePath) > 0:
                face = cv2.imread(facePath)
                face = cv2.resize(face, dimFace, interpolation=cv2.INTER_AREA)
                # leftEye
                leftEyePath = join(datasetPath,  str(region), "LeftEye", frameName)
                if os.path.exists(leftEyePath) and os.path.getsize(leftEyePath) > 0:
                    leftEye = cv2.imread(leftEyePath)
                    leftEye = cv2.resize(leftEye, dimEyes, interpolation=cv2.INTER_AREA)
                    # rightEye
                    rightEyePath = join(datasetPath , str(region) , "RightEye" , frameName)
                    if os.path.exists(rightEyePath) and os.path.getsize(rightEyePath) > 0:
                        rightEye = cv2.imread(rightEyePath)
                        rightEye = cv2.resize(rightEye, dimEyes, interpolation=cv2.INTER_AREA)
                        #FaceGrid
                        faceGridName = frameName.split(".")[0]
                        tmp = faceGridName.split("_")
                        faceGridName = tmp[0] + "_" + str(int(tmp[1]))
                        faceGridPath = join(datasetPath, str(region), "FaceGrid", faceGridName)
                        faceGrid = np.loadtxt(faceGridPath, delimiter=',')

                        # save y
                        y_batch[j] = region
                        # save to the batch
                        face_batch[j] = face
                        left_eye_batch[j] = leftEye
                        right_eye_batch[j] = rightEye
                        face_grid_batch[j] = faceGrid

                        j+=1

        y_batch = to_categorical(y_batch, regions)
        return [face_batch, left_eye_batch, right_eye_batch, face_grid_batch], y_batch