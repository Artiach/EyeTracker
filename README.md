# EyeTracker
Final thesis of my Bachelor degree

Prepare the Data

Download the dataset from the Gazecapture project on this link \newline https://gazecapture.csail.mit.edu/download.php . It is a very large file, around 300GB, and it will take a long time to download depending on your connection. Once, the data is downloaded decompress the main zip file and all of the zip files inside it, which correspond to each of the subjects. If you have bash you can execute this command:  for a in `ls -1 *.tar.gz`; do gzip -dc \$a | tar xf -; done
It will automatically decompress all of the zip files in the current directory. Once the files are decompressed execute the python script prepareDataset.py. This is a script made by \cite{krafka2016eye} that takes two arguments: the dataset path with the extracted subject folders and the destination folder. This script takes all of the valid frames for each subject and copies a crop of the face, right eye and left eye into the corresponding folder, appleFace, appleRightEye or appleLeftEye in each subject's directory. After this we also need to keep the following files for later processing: dotInfo.json, faceGrid.json and info.json. The script copyJsonFiles.py will do it for you and takes the same two arguments as the previous script. 

Now the data is pre-processed and it is necessary to create the final directory where to store the dataset and separate it into classes. For this task open the script dataPreparation.py. There are a few variables that need to be set before executing this script. The first is the regions variable, by default it is set to 15 but it is also possible to set it to 6. This is going to divide the dataset into either 6 or 15 classes, dividing the output prediction space. Then, you need to set the variable dataPath with the path to the pre-processed data. The data is now correctly stored and ready to be fed into the model.

Before building the model there is one last script you need to execute. This is the getNames.py script. Open it, and set the mainPath, the regions and the maxElements variables. This will generate three .txt files with the names of the images. This is later used to feed the data to the model.

Build and train the model

In order to build the model open the script named model.py. In this script there are three available models. One that has the full architecture and takes as inputs the face, eyes and faceGrid and two more that only include the eyes inputs and the face input respectively. In order to build each of the architectures set the variable architectureType. Also set the regions variable if the dataset has 6 or 15 classes. You can build several models, train them and save different results as you like. You can use the variable folderName to store the different models and and tests you do.

Pre-trained model

You can also use pre-trained models that you will find in the folder pretrainedModels. There are four different pre-trained models which are the ones described in the results chapter. You can use these models to directly compile them to an iOS model or you can continue training them.

Compile the model

In order to compile the model to a .mlmodel open the script coremlConverter.py, set the appropriate variables that are described and run the script. This will generate a .mlmodel file that you can import into an XCode project and it will automatically generate the necessary classes.

iOS app

Under the folder EyeTrackerApp you will find the XCode project that contains the design of the GUI and the main logic. In order to deploy it to an iOS device you will need an Apple developer account as well as an iOS device and a computer running macOS.
