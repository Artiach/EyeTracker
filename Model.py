################################################################################################################3#
# Author: Tomás Artiach Hortelano                                                                                #
# description: This script contains the model architecture as well as the training of such model                 #
##################################################################################################################

from os import mkdir, path
from sys import exit
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from LoadData import loadRandomBatch, loadNames, getTestBatch

#Global constants
activation = 'relu'
#optimizer = keras.optimizers.SGD()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.categorical_crossentropy

#output units of the model(options: 6, 15)
regions = 15
if regions is not 6 or 15:
    exit("output units not supported. Terminating program")

#Specify the architecture type here(options: full, eyesOnly, faceOnly)
architectureType = "full"
if architectureType is not "full" or "eyesOnly" or "faceOnly":
    exit("Architecture type not supported. Terminating program")

#use appropriate dataset
if regions == 6:
    datasetPath = './dataset6Regions'
    trainNamesPath = 'names/train_6_names.txt'
    validationNamesPath = 'names/validation_6_names.txt'
    testNamesPath = 'names/test_6_names.txt'
elif regions == 15:
    datasetPath = './dataset'
    trainNamesPath = 'train_15_names.txt'
    validationNamesPath = 'validation_15_names.txt'
    testNamesPath = 'test_15_names.txt'

trainNames = loadNames(trainNamesPath)[0]
valNames = loadNames(validationNamesPath)[0]
testNames = loadNames(testNamesPath)[0]

#Input shapes
faceShape = (64, 64, 3)
eyeShape = (64,64, 3)

classes = []
for i in range(regions):
    classes.append(str(i))

#name of the folder to store the results. Store different tests made
FolderName = "prueba1"
mkdir(FolderName)
mkdir(path.join(FolderName, "h5"))
mkdir(path.join(FolderName, "plots"))
mkdir(path.join(FolderName, "savedModel"))

#Section wih different models. Feel free to implement your own architectures

def getEyeModel(eyeShape):
    # Eye tensor
    eye_input = keras.layers.Input(shape=eyeShape)

    # Eyes architecture
    # 1st layer
    eyes = keras.layers.Conv2D(64, (7, 7), name='CE1', activation=activation)(eye_input)
    eyes = keras.layers.MaxPool2D(pool_size=(2, 2))(eyes)
    # 2nd layer
    eyes = keras.layers.Conv2D(128, (5, 5), name='CE2', activation=activation)(eyes)
    eyes = keras.layers.MaxPool2D(pool_size=(2, 2))(eyes)
    # 3rd layer
    eyes = keras.layers.Conv2D(256, (3, 3), name='CE3', activation=activation)(eyes)
    eyes = keras.layers.MaxPool2D(pool_size=(2, 2))(eyes)
    # out layer
    out_eyes = keras.layers.Conv2D(64, (1, 1), name='CE4', activation=activation)(eyes)

    eye_model = keras.models.Model(inputs=eye_input, outputs=out_eyes)

    return eye_model


def getFaceModel(faceShape):
    # Face tensor
    face_input = keras.layers.Input(shape=faceShape)

    # Face architecture
    # 1st layer
    face = keras.layers.Conv2D(64, (11, 11), name='CF1', activation=activation)(face_input)
    face = keras.layers.MaxPool2D(pool_size=(2, 2))(face)
    # 2nd layer
    face = keras.layers.Conv2D(128, (7, 7), name='CF2', activation=activation)(face)
    face = keras.layers.MaxPool2D(pool_size=(2, 2))(face)
    # 3rd layer
    face = keras.layers.Conv2D(256, (5, 5), name='CF3', activation=activation)(face)
    face = keras.layers.MaxPool2D(pool_size=(2, 2))(face)
    # OUT layer
    out_face = keras.layers.Conv2D(64, (3, 3), name='CF4', activation=activation)(face)
    face_model = keras.models.Model(inputs=face_input, outputs=out_face)

    return face_model


def getModel(faceShape, eyeShape):

    #faceGrid
    faceGridInput = keras.layers.Input(shape = (1, 25,25))
    #Face input
    faceInput = keras.layers.Input(shape=faceShape)
    #Face model
    faceModel = getFaceModel(faceShape)
    #faceNet
    faceNet = faceModel(faceInput)
    # define the two eyes inputs
    rightEyeInput = keras.layers.Input(eyeShape)
    leftEyeInput = keras.layers.Input(eyeShape)
    # eye models
    leftEyeModel = getEyeModel(eyeShape)
    rightEyeModel = getEyeModel(eyeShape)
    # eye nets
    leftEyeNet = leftEyeModel(leftEyeInput)
    rightEyeNet = rightEyeModel(rightEyeInput)
    #dense layers for eyes
    e = keras.layers.concatenate([leftEyeNet, rightEyeNet])
    e = keras.layers.Flatten()(e)
    fc_e1 = keras.layers.Dense(128, activation = activation)(e)
    #dense layers for face
    f = keras.layers.Flatten()(faceNet)
    fc_f1 = keras.layers.Dense(128, activation = activation)(f)
    fc_f2 = keras.layers.Dense(64, activation = activation)(fc_f1)
    #dense layers for face grid
    fg = keras.layers.Flatten()(faceGridInput)
    fc_fg1 = keras.layers.Dense(256, activation=activation)(fg)
    fc_fg2 = keras.layers.Dense(128, activation=activation)(fc_fg1)
    #final dense layers
    h = keras.layers.concatenate([fc_e1, fc_f2, fc_fg2])
    #h = keras.layers.concatenate([fc_e1, fc_f2])
    fc1 = keras.layers.Dense(128, activation=activation)(h)
    fc1 = keras.layers.Dense(64, activation=activation)(fc1)
    fc1 = keras.layers.Dense(32, activation=activation)(fc1)
    fc2 = keras.layers.Dense(regions, activation='softmax')(fc1)
    #Final model
    eyeTrackerModel = keras.models.Model(inputs = [faceInput, leftEyeInput,rightEyeInput, faceGridInput],
                                         outputs = fc2)
    #compile the model
    eyeTrackerModel.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    eyeTrackerModel.summary()
    print('model compiled succesfully!')

    return eyeTrackerModel


def getModelEyesOnly(eyeShape):
    # define the two eyes inputs
    rightEyeInput = keras.layers.Input(eyeShape)
    leftEyeInput = keras.layers.Input(eyeShape)

    # eye models
    leftEyeModel = getEyeModel(eyeShape)
    rightEyeModel = getEyeModel(eyeShape)

    # eye nets
    leftEyeNet = leftEyeModel(leftEyeInput)
    rightEyeNet = rightEyeModel(rightEyeInput)

    # dense layers for eyes
    e = keras.layers.concatenate([leftEyeNet, rightEyeNet])
    e = keras.layers.Flatten()(e)
    fc_e1 = keras.layers.Dense(128, activation=activation)(e)
    fc1 = keras.layers.Dense(64, activation=activation)(fc_e1)
    fc1 = keras.layers.Dense(32, activation=activation)(fc1)
    fc2 = keras.layers.Dense(regions, activation='softmax')(fc1)

    #Final model
    eyeTrackerModel = keras.models.Model(inputs=[leftEyeInput, rightEyeInput],
                                                                 outputs = fc2)
    #compile the model
    eyeTrackerModel.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    eyeTrackerModel.summary()
    print('model compiled succesfully!')

    return eyeTrackerModel


def getModelFaceOnly(faceShape):
    # Face input
    faceInput = keras.layers.Input(shape=faceShape)
    # Face model
    faceModel = getFaceModel(faceShape)
    # faceNet
    faceNet = faceModel(faceInput)

    # dense layers for face
    f = keras.layers.Flatten()(faceNet)
    fc_f1 = keras.layers.Dense(128, activation=activation)(f)
    fc_f2 = keras.layers.Dense(64, activation=activation)(fc_f1)

    fc1 = keras.layers.Dense(64, activation=activation)(fc_f2)
    fc1 = keras.layers.Dense(32, activation=activation)(fc1)
    fc2 = keras.layers.Dense(regions, activation='softmax')(fc1)

    #Final model
    eyeTrackerModel = keras.models.Model(inputs=[faceInput],
                                        outputs = fc2)
    #compile the model
    eyeTrackerModel.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    eyeTrackerModel.summary()
    print('model compiled succesfully!')

    return eyeTrackerModel


#Generators

def train_generator(names, datasetPath, batch_size):
    while True:
        x, y = loadRandomBatch(names, datasetPath, faceShape, eyeShape, batch_size, architectureType, regions)
        yield x, y


def test_generator(names, datasetPath, batch_size):
    callCounter = 0
    while callCounter < (len(names)/batch_size - 1):
        x, y = getTestBatch(names, datasetPath, faceShape, eyeShape,batch_size, architectureType, callCounter)
        callCounter += 1
        yield x, y


def validation_generator(names, datasetPath, batch_size):
    while True:
        #funcion que devuelva un np array compuesto de las entradas y de las regiones. Esta funcion tiene que imagenes random del dataset
        x, y = loadRandomBatch(names, datasetPath, faceShape, eyeShape, batch_size, architectureType, regions)
        yield x, y



def trainModel(eyeTracker, batchSize = 64, epochs = 20):

    history = eyeTracker.fit(train_generator(trainNames, (datasetPath+'/train'), batchSize),
                             steps_per_epoch = (len(trainNames)) / batchSize,
                             epochs = epochs,
                             verbose = 1,
                             validation_data = validation_generator(valNames, (datasetPath+'/validation'), batchSize),
                             validation_steps = (len(valNames)) / batchSize)

    eyeTracker.save_weights(FolderName+"/h5/weights.h5")

    return history


def evaluateModel(eyeTracker, batchSize = 64):

    eyeTracker.evaluate(test_generator(testNames, datasetPath+"/test", batchSize))


def loadPredictSamples(path,testNames, faceShape, eyeShape, batchSize):

    x, y_true = getTestBatch(testNames, path, faceShape, eyeShape, batchSize)

    return x, y_true


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plotAccuracy(history):
    print(history.history['accuracy'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(FolderName + "/plots/Accuracy.png")
    plt.show()

def plotLoss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(FolderName+"/plots/Loss.png")
    plt.show()

def getConfusionMatrix():
    testNamesPath = 'test_names.txt'
    names = loadNames(testNamesPath)[0]
    for i in range(len(names)):
        x, y_true = loadPredictSamples("../dataset/test", names, faceShape, eyeShape, 32)
        y_pred = eyeTracker.predict_on_batch(x)
        oneHot_yPred = np.zeros((32))

    i = 0
    for prediction in y_pred:
        max = np.argmax(prediction)
        oneHot_yPred[i] = max
        i += 1
    y_true = np.argmax(y_true, axis=1)
    # plot confussion matrix
    cnf_matrix = confusion_matrix(y_true, oneHot_yPred)
    print(cnf_matrix)

    return confusion_matrix

#Uncomment in order to run Tensorboard
#root_logdir = os.path.join(os.curdir, "logs/fit")
#def get_run_logdir():
#    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#    return os.path.join(root_logdir, run_id)

#run_logdir = get_run_logdir()
#tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


#Compile, train and save model and plot results
if architectureType == "faceOnly":
    eyeTracker = getModelFaceOnly(faceShape)
elif architectureType == "full":
    eyeTracker = getModel(faceShape, eyeShape)
elif architectureType == "eyesOnly":
    eyeTracker = getModelEyesOnly(eyeShape)

#If previous training was done on this model, load weights and continue training
if path.exists(FolderName+"/h5/weights.h5"):
    eyeTracker.load_weights(FolderName+"/h5/weights.h5")

history = trainModel(eyeTracker, faceShape, eyeShape, FolderName)
evaluateModel(eyeTracker)
eyeTracker.save(FolderName+"/savedModel")
plotAccuracy(history)
plotLoss(history)
#get and plot confusion matrix
cnf_matrix = getConfusionMatrix()
plot_confusion_matrix(cnf_matrix, classes)

