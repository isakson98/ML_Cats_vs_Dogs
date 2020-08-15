# preprocessing this dataset using Keras ImageDataGenerator and flow_from_directory() API
# folders of test and train with subfolders of classifications in each
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# create directories
dataset_home = 'preprocessed_dogs-vs-cats_dataset/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    #create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        makedirs(newdir)

#generate random num
seed(1)
# ratio of splitting train and test dataset
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'dogs-vs-cats_dataset/train/train/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, dst)
