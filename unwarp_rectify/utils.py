# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:45:22 2014

@author: chuong
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import cv2yml
import cv2

def getRectCornersFrom2Points(Image, Points, AspectRatio, Rounded = False):
#    print('Points =', Points)
    Length = np.sqrt((Points[0][0] - Points[1][0])**2 + \
                     (Points[0][1] - Points[1][1])**2)
    Height = Length/np.sqrt(1+AspectRatio**2)
    Width = Height*AspectRatio
    Centre = np.asarray([Points[0][0] + Points[1][0], Points[0][1] + Points[1][1]])/2.0
    Angle = np.arctan2(Height, Width) - \
            np.arctan2(Points[1][1] - Points[0][1], Points[1][0] - Points[0][0])
    InitRect = createRectangle(Centre, Width, Height, Angle)
    CornerTypes = ['topleft', 'bottomleft', 'bottomright', 'topright']
    Rect = []
    for Corner, Type in zip(InitRect, CornerTypes):
        if not Rounded:
            Corner = findCorner(Image, Corner, Type)
        else:
            Corner = findRoundedCorner(Image, Corner, Type)
        Rect.append(Corner)
    return Rect

def createRectangle(Centre, Width, Height, Angle):
    tl2 = np.asarray([-Width, -Height])/2.0
    bl2 = np.asarray([-Width,  Height])/2.0
    br2 = np.asarray([ Width,  Height])/2.0
    tr2 = np.asarray([ Width, -Height])/2.0
    RectFit = [tl2, bl2, br2, tr2]
    for i in range(len(RectFit)):
        # rotate around center
        xrot =  RectFit[i][0]*np.cos(Angle) + RectFit[i][1]*np.sin(Angle)
        yrot = -RectFit[i][0]*np.sin(Angle) + RectFit[i][1]*np.cos(Angle)
        RectFit[i][0], RectFit[i][1] = (xrot+Centre[0]), (yrot+Centre[1])
    return RectFit

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

def findCorner(Image, Corner, CornerType = 'topleft', WindowSize = 100, Threshold = 50):
    x, y = Corner
    HWindowSize = int(WindowSize/2)
    window = Image[y-HWindowSize:y+HWindowSize+1, x-HWindowSize:x+HWindowSize+1,:].astype(np.float)
#        cv2.imwrite('/home/chuong/Data/GC03L-temp/corrected/'+CornerType+'.jpg', window)
    foundLeftEdgeX = False
    foundRightEdgeX = False
    foundTopEdgeY = False
    foundBottomEdgeY = False
    for i in range(HWindowSize+1):
        diff0 = np.sum(np.abs(window[HWindowSize, HWindowSize-i,:] - window[HWindowSize, HWindowSize,:]))
        diff1 = np.sum(np.abs(window[HWindowSize, HWindowSize+i,:] - window[HWindowSize, HWindowSize,:]))
        diff2 = np.sum(np.abs(window[HWindowSize-i, HWindowSize,:] - window[HWindowSize, HWindowSize,:]))
        diff3 = np.sum(np.abs(window[HWindowSize+i, HWindowSize,:] - window[HWindowSize, HWindowSize,:]))
        if diff0 > Threshold and not foundLeftEdgeX:
            xLeftNew = x-i
            foundLeftEdgeX = True
        elif diff1 > Threshold and not foundRightEdgeX:
            xRightNew = x+i
            foundRightEdgeX = True
        if diff2 > Threshold and not foundTopEdgeY:
            yTopNew = y-i
            foundTopEdgeY = True
        elif diff3 > Threshold and not foundBottomEdgeY:
            yBottomNew = y+i
            foundBottomEdgeY = True
            
    if CornerType.lower() == 'topleft' and foundLeftEdgeX and foundTopEdgeY:
        return [xLeftNew, yTopNew]
    elif CornerType.lower() == 'bottomleft' and foundLeftEdgeX and foundBottomEdgeY:
        return [xLeftNew, yBottomNew]
    elif CornerType.lower() == 'bottomright' and foundRightEdgeX and foundBottomEdgeY:
        return [xRightNew, yBottomNew]
    elif CornerType.lower() == 'topright' and foundRightEdgeX and foundTopEdgeY:
        return [xRightNew, yTopNew]
    else:
        print('Cannot detect corner ' + CornerType)
        return [x, y]

def findRoundedCorner(Image, Corner, CornerType = 'topleft', WindowSize = 100, Threshold = 50):
    #TODO: add search for rounded corner with better accuracy
    return Corner

def correctPointOrder(Rect, tolerance = 40):
    # find minimum values of x and y
    minX = 10e6
    minY = 10e6
    for i in range(len(Rect[0])):
        if minX > Rect[i][0]:
            minX = Rect[i][0]
        if minY > Rect[i][1]:
            minY = Rect[i][1]
    #separate left and right
    topLeft, bottomLeft, topRight, bottomRight = [], [], [], []
    for i in range(len(Rect[0])):
        if abs(minX - Rect[0][i]) < tolerance:
            if abs(minY - Rect[i][1]) < tolerance:
                topLeft = [Rect[i][0], Rect[i][1]]
            else:
                bottomLeft = [Rect[i][0], Rect[i][1]]
        else:
            if abs(minY - Rect[i][1]) < tolerance:
                topRight = [Rect[i][0], Rect[i][1]]
            else:
                bottomRight = [Rect[i][0], Rect[i][1]]
    if len(topLeft)*len(bottomLeft)*len(topRight)*len(bottomRight) == 0:
        print('Cannot find corRect corner order. Change tolerance value.')
        return Rect
    else:
        Rect = [topLeft, bottomLeft, bottomRight, topRight]
        return Rect

def getMedianRectSize(RectList):
    WidthList = []
    HeightList = []
    for Rect in RectList:
        Centre, Width, Height, Angle = getRectangleParamters(Rect)
        WidthList.append(Width)
        HeightList.append(Height)
    MedianWidth = int(sorted(WidthList)[int(len(RectList)/2)])
    MedianHeight = int(sorted(HeightList)[int(len(RectList)/2)])
    return MedianWidth, MedianHeight

def rectifyRectImages(Image, RectList, MedianSize):
    Width, Height = MedianSize
    RectifiedCorners = np.float32([[0,0], [0,Height], [Width,Height], [Width,0]])
    RectifiedTrayImages = []
    for Rect in RectList:
        Corners = np.float32(Rect)
        M = cv2.getPerspectiveTransform(Corners, RectifiedCorners)
        RectifiedTrayImage = cv2.warpPerspective(Image, M,(Width, Height))
        RectifiedTrayImages.append(RectifiedTrayImage)
    return RectifiedTrayImages

def readCalibration(CalibFile):
    parameters = cv2yml.yml2dic(CalibFile)
    SquareSize = parameters['square_size']
    ImageWidth = parameters['image_width']
    ImageHeight = parameters['image_height']
    ImageSize = (ImageWidth, ImageHeight)
    CameraMatrix = parameters['camera_matrix']
    DistCoefs = parameters['distortion_coefficients']
    RVecs = parameters['RVecs']
    TVecs = parameters['TVecs']
    return ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs
