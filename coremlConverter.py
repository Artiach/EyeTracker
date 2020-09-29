################################################################################################################3#
# Author: Tomás Artiach Hortelano                                                                                #
# description: This script converts the keras model compiled into a model capable of running in an iOS device    #
##################################################################################################################

from tensorflow.keras.models import load_model
import coremltools

#number of classes of the model
regions = 6
#create the labels list
classLabels = []
for i in range(regions):
    classLabels.append(str(i))
#name of the folder where the model was saved
modelPath = "prueba2"
#name of the .mlmodel (output file)
outputName = "coremlModel/eyeTracker2.mlmodel"

#load model
print("[INFO] loading the model")
model = load_model(modelPath+"/savedModel/")

#convert the model
print("[INFO] converting model")
coreml_model = coremltools.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=classLabels,
	is_bgr=True)

# save the model to disk
output = outputName
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)

