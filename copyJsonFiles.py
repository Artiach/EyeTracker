import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='copy necessary json files')
parser.add_argument('--dataset_path', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default=None, help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
args = parser.parse_args()

#path of the original decompressed data
path = args.dataset_path
#destination path after executing the prepareDataset.py script
destination_path = args.output_path

for subject in os.listdir(path):

    shutil.copy(os.path.join(path, subject, "info.json"), os.path.join(destination_path, subject))
    shutil.copy(os.path.join(path, subject, "dotInfo.json"), os.path.join(destination_path, subject))
    shutil.copy(os.path.join(path, subject, "faceGrid.json"), os.path.join(destination_path, subject))
