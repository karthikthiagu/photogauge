# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:42:02 2017

This script reads in images in the folder SOURCE, performs preprocessing
and writes them to the folder LOC.
@author: admin
"""
import os
import cv2
#Coordinates to crop image
Y_START = 1495
Y_END = 2236
X_START = 1260
X_END = 1880

#Y_START = 1550
#Y_END = 2200
#X_START = 1250
#X_END = 1800

SOURCE = './data/raw'
LOC = './data/processed'

 

def run(SOURCE, DST, MASK):
    for fileName in os.listdir(SOURCE):
        im = cv2.imread(os.path.join(SOURCE, fileName), 0) # read image as grayscale
        img2_fg = cv2.bitwise_and(im,im,mask = MASK)
        img2_fg = img2_fg[Y_START:Y_END,X_START:X_END]
        outname = os.path.join(DST, fileName)
        cv2.imwrite(outname,img2_fg)
        
if __name__=="__main__":
    right_mask = cv2.imread('./data/masks/right_flank_mask.jpg', 0) # read the relevant mask as grayscale image
    right_mask = cv2.resize(right_mask, dsize=(3036,4048))
    left_mask = cv2.imread('./data/masks/left_flank_mask.jpg', 0) # read the relevant mask as grayscale image
    left_mask = cv2.resize(left_mask, dsize=(3036,4048))

    run('./data/raw/right_good', './data/processed/right_good', right_mask)
    print 'written right_good'
    run('./data/raw/right_bad', './data/processed/right_bad', right_mask)
    print 'written right_bad'
    run('./data/raw/left_good', './data/processed/left_good', left_mask)
    print 'written left_good'
    run('./data/raw/left_bad', './data/processed/left_bad', left_mask)
    print 'written left_bad'
        
