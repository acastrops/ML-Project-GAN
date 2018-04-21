# coding: utf-8

'''
This script creates a 3D numpy array 

This script requires two command line parameters:
1. The path to the CSV file
2. The output directory

Run with:
python3 converttonumpyarray.py -f fer2013.csv
'''


import os
import csv
import argparse
import numpy as np 
import scipy.misc
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
#parser.add_argument('-o', '--output', required=True, help="path of the output directory")

args = parser.parse_args()

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 0
list3d = [0]*35887
with open(args.file, 'rt') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    headers = next(datareader, None)
    print(headers) 
    for row in datareader:

        emotion = row[0]
        pixels = row[1].split()
        usage = row[2]

        pixels_array = np.asarray(pixels).astype(np.float)
        image = pixels_array.reshape(w, h)
        
        list3d[id] = image
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))
array3d = np.dstack(list3d)
print(array3d)