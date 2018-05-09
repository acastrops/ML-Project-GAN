# coding: utf-8

'''
This script creates 3-channel gray images from FER 2013 dataset.
It has been done so that the CNNs designed for RGB images can 
be used without modifying the input shape. 

This script requires two command line parameters:
1. The path to the CSV file
2. The output directory

It generates the images and saves them in three directories inside 
the output directory - Training, PublicTest, and PrivateTest. 
These are the three original splits in the dataset. 

Run by:
mkdir FerImages
python3 covertFer2013tojpeg.py -f fer2013.csv -o FerImages
'''


import os
import csv
import argparse
import numpy as np 
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-o', '--output', required=True, help="path of the output directory")

args = parser.parse_args()

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1
with open(args.file, 'rt') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    headers = next(datareader, None)
    print(headers) 
    i = 0
    for row in datareader:
          
        emotion = row[0]
        pixels = row[1].split()
        usage = row[2]

        pixels_array = np.asarray(pixels).astype(np.float)
        image = pixels_array.reshape(w, h)
        stacked_image = np.dstack((image,) * 3)

        image_folder = os.path.join(args.output, usage, emotion)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file =  os.path.join(image_folder , str(id) + '.jpg')
        scipy.misc.imsave(image_file, stacked_image)
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))