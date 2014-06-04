# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:13:04 2014

@author: chuong
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
import getopt, sys, os
import cv2
import cv2yml
import glob
from scipy import optimize
from timestream.parse import ts_iter_images
from multiprocessing import Pool

def getRectangleParamters(Rect):
    tl = np.asarray(Rect[0])
    bl = np.asarray(Rect[1])
    br = np.asarray(Rect[2])
    tr = np.asarray(Rect[3])
    
    # paramters of fitted Rectangle
    Centre = (tl + bl + br + tr)/4.0
    Width  = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))/2.0
    Height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))/2.0
    Angle = (np.arctan2(-(tr[1] - tl[1]), tr[0] - tl[0]) + \
             np.arctan2(-(br[1] - bl[1]), br[0] - bl[0]) + \
             np.arctan2(  bl[0] - tl[0] , bl[1] - tl[1]) + \
             np.arctan2(  br[0] - tr[0] , br[1] - tr[1]))/4
    return Centre, Width, Height, Angle

def createMap(Centre, Width, Height, Angle):
    MapX, MapY = np.meshgrid(np.arange(Width), np.arange(Height))
    MapX = MapX - Width/2.0
    MapY = MapY - Height/2.0
    MapX2 =  MapX*np.cos(Angle) + MapY*np.sin(Angle) + Centre[0]
    MapY2 = -MapX*np.sin(Angle) + MapY*np.cos(Angle) + Centre[1]
    return MapX2.astype(np.float32), MapY2.astype(np.float32)

def rotateImage(Image, RotationAngle = 0.0):
    Image_ = Image
    if RotationAngle%90.0 == 0:
        k = RotationAngle//90.0
        Image_ = np.rot90(np.rot90(Image_), k)
    elif RotationAngle != 0:
        center=tuple(np.array(Image_.shape[0:2])/2)
        rot_mat = cv2.getRotationMatrix2D(center, RotationAngle,1.0)
        Image_ = cv2.warpAffine(Image_, rot_mat, Image_.shape[0:2],flags=cv2.INTER_LINEAR)
    return Image_


def findColorbarPyramid(Image, Colorbar, RotationAngle = None, NearCenter = True, NoLevels = 3, FinalLevel = 0):
    for i in range(NoLevels):
        if i == 0:
            PyramidImages = [Image]
            PyramidColorbars = [Colorbar]
        else:
            PyramidImages.append(cv2.pyrDown(PyramidImages[i-1]))
            PyramidColorbars.append(cv2.pyrDown(PyramidColorbars[i-1]))

    for i in range(NoLevels-1, -1, -1):
        if i == NoLevels-1:
            corrMap = cv2.matchTemplate(PyramidImages[i], PyramidColorbars[i], cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(corrMap)
            if RotationAngle == None:
                corrMap180 = cv2.matchTemplate(np.rot90(PyramidImages[i],2), PyramidColorbars[i], cv2.TM_CCOEFF_NORMED)
                _, maxVal180, _, maxLoc180 = cv2.minMaxLoc(corrMap180)
                Radius = np.sqrt((maxLoc[0]-PyramidImages[i].shape[1]/2)**2 + (maxLoc[1]-PyramidImages[i].shape[0]/2)**2)
                Radius180 = np.sqrt((maxLoc180[0]-PyramidImages[i].shape[1]/2)**2 + (maxLoc180[1]-PyramidImages[i].shape[0]/2)**2)
                if Radius/Radius180 > 0.9:
                    print('Cannot find a colorbar')
                    return None, None, None
                if (NearCenter and  maxVal/Radius < maxVal180/Radius180) or \
                   ((not NearCenter) and maxVal < maxVal180):
                    PyramidImages = [np.rot90(Img,2) for Img in PyramidImages]
                    maxVal, maxVal180 = maxVal180, maxVal
                    maxLoc, maxLoc180 = maxLoc180, maxLoc
                    corrMap, corrMap180 = corrMap180, corrMap
                    RotationAngle = 180
                else:
                    RotationAngle = 0
            # recalculate max position in image space
            maxLocImage = (maxLoc[0] + PyramidColorbars[i].shape[1]//2, 
                           maxLoc[1] + PyramidColorbars[i].shape[0]//2)
            # rescale to location in level-0 image
            maxLocImage = (maxLocImage[0]*2**i, maxLocImage[1]*2**i)
        else:
            maxLocEst = (maxLocImage[0]//2**i, maxLocImage[1]//2**i)
            searchRange = (6,6)
                
            CroppedHalfWidth  = PyramidColorbars[i].shape[1]//2 + searchRange[0]
            CroppedHalfHeight = PyramidColorbars[i].shape[0]//2 + searchRange[1]
            CropedImage = PyramidImages[i][maxLocEst[1]-CroppedHalfHeight:maxLocEst[1]+CroppedHalfHeight,\
                                           maxLocEst[0]-CroppedHalfWidth: maxLocEst[0]+CroppedHalfWidth ,:]
#            plt.figure()
#            plt.imshow(CropedImage)

            corrMap = cv2.matchTemplate(CropedImage, PyramidColorbars[i], cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(corrMap)
            
            # recalculate max position in cropped image space
            maxLocImageCropped = (maxLoc[0] + PyramidColorbars[i].shape[1]//2, 
                                  maxLoc[1] + PyramidColorbars[i].shape[0]//2)
            # recalculate max position in full image space
            maxLocImage = (maxLocEst[0]-CroppedHalfWidth  + maxLocImageCropped[0], \
                           maxLocEst[1]-CroppedHalfHeight + maxLocImageCropped[1])
            # rescale to location in level-0 image
            maxLocImage = (maxLocImage[0]*2**i, maxLocImage[1]*2**i)
        print('maxVal, maxLocImage, RotationAngle =', maxVal, maxLocImage, RotationAngle)
        if i ==  FinalLevel:
            # Skip early to save time
            break
        
#        plt.figure()
#        plt.imshow(corrMap)
#        plt.hold(True)
#        plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
#        plt.title('maxVal = %f' %maxVal)
#        
#        plt.figure()
#        plt.imshow(PyramidImages[i])
#        plt.hold(True)
#        plt.plot([maxLocImage[0]//2**i], [maxLocImage[1]//2**i], 'o')
#        plt.title('Level = %d, RotationAngle = %f' %(i, RotationAngle))
#        plt.show()
        
    return maxVal, maxLocImage, RotationAngle
    

RectData = cv2yml.yml2dic('/home/chuong/Data/ColorbarPositions/ColorbarRectangle.yml')
RotationAngle = RectData['RotationAngle']
Rect = RectData['Colorbar'].tolist()
print('Rect =', Rect)
Centre, Width, Height, Angle = getRectangleParamters(Rect)
print(Centre, Width, Height, Angle)
ColCardMapX, ColCardMapY = createMap(Centre, Width, Height, Angle)

P24ColorCard = cv2.imread('/home/chuong/Data/ColorbarPositions/CameraTrax_24ColorCard_2x3in.png')[:,:,::-1] # read and convert to R-G-B image
SquareSize = int(P24ColorCard.shape[0]/4)
HalfSquareSize = int(SquareSize/2)

P24ColorCardCaptured = cv2.imread('/home/chuong/Data/ColorbarPositions/CameraTrax_24ColorCard_2x3inCaptured.png')[:,:,::-1] # read and convert to R-G-B image
SquareSizeCaptured = int(P24ColorCardCaptured.shape[0]/4)
HalfSquareSizeCaptured = int(SquareSizeCaptured/2)

#img_iter = ts_iter_images('/home/chuong/Data/ColorbarPositions')
img_iter = ts_iter_images('/home/chuong/Data/BVZ0012-GC02L-CN650D-Cam01') 
for ImageFile in img_iter:
    Image = cv2.imread(ImageFile)
    print(ImageFile)
    if Image.shape[0] > Image.shape[1]:
        RotationAngle = 90
        Image = rotateImage(Image, RotationAngle)
#    maxVal, maxLoc = findColorbarPyramid(Image, P24ColorCardCaptured)
    maxVal, maxLoc, RotationAngle2 = findColorbarPyramid(Image, P24ColorCardCaptured, NoLevels = 4, FinalLevel = 1)
    if maxVal == None:
        continue
    RotationAngle = RotationAngle + RotationAngle2
#    if maxVal < maxVal180:
#        RotationAngle = RotationAngle + 180
#        maxVal, maxLoc, maxVal180, maxLoc180 = maxVal180, maxLoc180, maxVal, maxLoc
#    print('Selected maxVal, maxLoc =', maxVal, maxLoc)
    
#    plt.figure()
#    plt.imshow(Image)
#    plt.figure()
#    plt.imshow(corrMap)
#    plt.show()