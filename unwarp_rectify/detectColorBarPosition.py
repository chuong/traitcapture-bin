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

def matchTemplate(Image, Template, SearchTopLeftCorner, SearchBottomRightCorner):
    CropedImage = Image[SearchTopLeftCorner[1]:SearchBottomRightCorner[1], SearchTopLeftCorner[0]:SearchBottomRightCorner[0]]
    corrMap = cv2.matchTemplate(CropedImage.astype(np.uint8), Template.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(corrMap)
    # recalculate max position in cropped image space
    matchedLocImageCropped = (maxLoc[0] + Template.shape[1]//2, 
                              maxLoc[1] + Template.shape[0]//2)
    # recalculate max position in full image space
    matchedLocImage = (matchedLocImageCropped[0] + SearchTopLeftCorner[0], \
                       matchedLocImageCropped[1] + SearchTopLeftCorner[1])
#    plt.figure()
#    plt.imshow(corrMap)
#    plt.hold(True)
#    plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
#    plt.figure()
#    plt.imshow(CropedImage)
#    plt.hold(True)
#    plt.plot([matchedLocImageCropped[0]], [matchedLocImageCropped[1]], 'o')
#    plt.figure()
#    plt.imshow(Image)
#    plt.hold(True)
#    plt.plot([matchedLocImage[0]], [matchedLocImage[1]], 'o')
#    plt.show()

    return matchedLocImage, maxVal, maxLoc, corrMap
    
def findColorbarPyramid(Image, Colorbar, RotationAngle = None, SearchRange = 0.5, NoLevels = 5, FinalLevel = 1):
    for i in range(NoLevels):
        if i == 0:
            PyramidImages = [Image]
            PyramidColorbars = [Colorbar]
        else:
            PyramidImages.append(cv2.pyrDown(PyramidImages[i-1]))
            PyramidColorbars.append(cv2.pyrDown(PyramidColorbars[i-1]))

    for i in range(NoLevels-1, -1, -1):
        if i == NoLevels-1:
            maxLocEst = [PyramidImages[i].shape[1]//2, PyramidImages[i].shape[0]//2] # image center
            if SearchRange > 0 and SearchRange <= 1.0:
                CroppedHalfWidth  = SearchRange*PyramidImages[i].shape[1]//2 
                CroppedHalfHeight = SearchRange*PyramidImages[i].shape[0]//2
            else:
                CroppedHalfWidth  = PyramidImages[i].shape[1]//2 
                CroppedHalfHeight = PyramidImages[i].shape[0]//2
            SearchTopLeftCorner     = [maxLocEst[0]-CroppedHalfWidth, maxLocEst[1]-CroppedHalfHeight]
            SearchBottomRightCorner = [maxLocEst[0]+CroppedHalfWidth, maxLocEst[1]+CroppedHalfHeight]

            matchedLocImage, maxVal, maxLoc, corrMap = matchTemplate(PyramidImages[i], PyramidColorbars[i], SearchTopLeftCorner, SearchBottomRightCorner)

            if RotationAngle == None:
                matchedLocImage180, maxVal180, maxLoc180, corrMap180 = matchTemplate(np.rot90(PyramidImages[i],2).astype(np.uint8), PyramidColorbars[i], SearchTopLeftCorner, SearchBottomRightCorner)
                print('maxVal, maxVal180', maxVal, maxVal180)
                if maxVal < 0.3 and maxVal180 < 0.3:
                    # similar distance: very likely cannot find colorbar
                    print('#### Cannot find a colorbar ####')
#                    return None, None, None
                if maxVal < maxVal180:
                    PyramidImages = [np.rot90(Img,2) for Img in PyramidImages]
                    matchedLocImage, matchedLocImage180 = matchedLocImage180, matchedLocImage
                    maxVal, maxVal180 = maxVal180, maxVal
                    maxLoc, maxLoc180 = maxLoc180, maxLoc
                    corrMap, corrMap180 = corrMap180, corrMap
                    RotationAngle = 180
                else:
                    RotationAngle = 0
            # rescale to location in level-0 image
            matchedLocImage0 = (matchedLocImage[0]*2**i, matchedLocImage[1]*2**i)
        else:
            maxLocEst = (matchedLocImage0[0]//2**i, matchedLocImage0[1]//2**i)
            searchRange = (6,6)
                
            CroppedHalfWidth  = PyramidColorbars[i].shape[1]//2 + searchRange[1]//2
            CroppedHalfHeight = PyramidColorbars[i].shape[0]//2 + searchRange[0]//2
            SearchTopLeftCorner = [maxLocEst[0]-CroppedHalfWidth, maxLocEst[1]-CroppedHalfHeight]
            SearchBottomRightCorner = [maxLocEst[0]+CroppedHalfWidth, maxLocEst[1]+CroppedHalfHeight]
            
            matchedLocImage, maxVal, maxLoc, corrMap = matchTemplate(PyramidImages[i], PyramidColorbars[i], SearchTopLeftCorner, SearchBottomRightCorner)
            # rescale to location in level-0 image
            matchedLocImage0 = (matchedLocImage[0]*2**i, matchedLocImage[1]*2**i)

        plt.figure()
        plt.imshow(corrMap)
        plt.hold(True)
        plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
        plt.title('maxVal = %f' %maxVal)
        
        plt.figure()
        plt.imshow(PyramidImages[i])
        plt.hold(True)
        plt.plot([matchedLocImage[0]], [matchedLocImage[1]], 'o')
        plt.title('Level = %d, RotationAngle = %f' %(i, RotationAngle))
        plt.show()

        if i ==  FinalLevel:
            # Skip early to save time
            break
        
    print('maxVal, maxLocImage, RotationAngle =', maxVal, matchedLocImage0, RotationAngle)
    return maxVal, matchedLocImage0, RotationAngle
    

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

img_iter = ts_iter_images('/home/chuong/Data/ColorbarPositions')
#img_iter = ts_iter_images('/home/chuong/Data/BVZ0012-GC02L-CN650D-Cam01') 
for ImageFile in img_iter:
    Image = cv2.imread(ImageFile)[:,:,::-1]
    print(ImageFile)
    if Image.shape[0] > Image.shape[1]:
        RotationAngle = 90
        Image = rotateImage(Image, RotationAngle)
#    maxVal, maxLoc = findColorbarPyramid(Image, P24ColorCardCaptured)
    maxVal, maxLoc, RotationAngle2 = findColorbarPyramid(Image, P24ColorCardCaptured, NoLevels = 5, FinalLevel = 3)
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