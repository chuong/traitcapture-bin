# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:45:22 2014

@author: chuong
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import cv2yml
import cv2
from scipy import optimize

#RED GRN BLU
CameraTrax_24ColorCard = \
 [[ 115., 196.,  91.,  94., 129.,  98., 223.,  58., 194.,  93., 162., 229., \
    49.,  77., 173., 241., 190.,   0., 242., 203., 162., 120.,  84.,  50.], \
 [  83., 147., 122., 108., 128., 190., 124.,  92.,  82.,  60., 190., 158., \
    66., 153.,  57., 201.,  85., 135., 243., 203., 163., 120.,  84.,  50.], \
 [  68., 127., 155.,  66., 176., 168.,  47., 174.,  96., 103.,  62.,  41., \
   147.,  71.,  60.,  25., 150., 166., 245., 204., 162., 120.,  84.,  52.]]
CameraTrax_24ColorCard180deg = \
 [[  50.,  84., 120., 162., 203., 242.,   0., 190., 241., 173.,  77.,  49., \
   229., 162.,  93., 194.,  58., 223.,  98., 129.,  94.,  91., 196., 115.], \
 [  50.,  84., 120., 163., 203., 243., 135.,  85., 201.,  57., 153.,  66., \
   158., 190.,  60.,  82.,  92., 124., 190., 128., 108., 122., 147.,  83.], \
 [  52.,  84., 120., 162., 204., 245., 166., 150.,  25.,  60.,  71., 147., \
    41.,  62., 103.,  96., 174.,  47., 168., 176.,  66., 155., 127.,  68.]]

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

def readGeometries(GeometryFile):
    parameters = cv2yml.yml2dic(GeometryFile)
    rotationAngle = parameters['rotationAngle']
    distortionCorrected = bool(parameters['distortionCorrected'])
    colorcardList = parameters['colorcardList'].tolist()
    colorcardList2 = []
    for i in range(0,len(colorcardList),4):
        colorcardList2.append([colorcardList[i], colorcardList[i+1], \
                               colorcardList[i+2], colorcardList[i+3]])
    trayList = parameters['trayList'].tolist()
    trayList2 = []
    for i in range(0,len(trayList),4):
        trayList2.append([trayList[i], trayList[i+1], \
                          trayList[i+2], trayList[i+3]])
    potList = parameters['potList'].tolist()
    potList2 = []
    for i in range(0,len(potList),4):
        potList2.append([potList[i], potList[i+1], \
                         potList[i+2], potList[i+3]])
    return rotationAngle, distortionCorrected, colorcardList2, trayList2, potList2

def createMap(Centre, Width, Height, Angle):
    MapX, MapY = np.meshgrid(np.arange(Width), np.arange(Height))
    MapX = MapX - Width/2.0
    MapY = MapY - Height/2.0
    MapX2 =  MapX*np.cos(Angle) + MapY*np.sin(Angle) + Centre[0]
    MapY2 = -MapX*np.sin(Angle) + MapY*np.cos(Angle) + Centre[1]
    return MapX2.astype(np.float32), MapY2.astype(np.float32)
    
def getColorcardColors(ColorCardCaptured, GridSize):
    GridCols, GridRows = GridSize
    Captured_Colors = np.zeros([3,GridRows*GridCols])
    STD_Colors = np.zeros([GridRows*GridCols])
    SquareSize2 = int(ColorCardCaptured.shape[0]/GridRows)
    HalfSquareSize2 = int(SquareSize2/2)
    for i in range(GridRows*GridCols):
        Row = i//GridCols
        Col = i - Row*GridCols
        rr = Row*SquareSize2 + HalfSquareSize2
        cc = Col*SquareSize2 + HalfSquareSize2
        Captured_R = ColorCardCaptured[rr-10:rr+10, cc-10:cc+10, 0].astype(np.float)
        Captured_G = ColorCardCaptured[rr-10:rr+10, cc-10:cc+10, 1].astype(np.float)
        Captured_B = ColorCardCaptured[rr-10:rr+10, cc-10:cc+10, 2].astype(np.float)
        STD_Colors[i] = np.std(Captured_R) + np.std(Captured_G) + np.std(Captured_B)
        Captured_R = np.sum(Captured_R)/Captured_R.size
        Captured_G = np.sum(Captured_G)/Captured_G.size
        Captured_B = np.sum(Captured_B)/Captured_B.size
        Captured_Colors[0,i] = Captured_R
        Captured_Colors[1,i] = Captured_G
        Captured_Colors[2,i] = Captured_B
    return Captured_Colors, STD_Colors

# Using modified Gamma Correction Algorithm by
# Constantinou2013 - A comparison of color correction algorithms for endoscopic cameras
def getColorMatchingError(Arg, Colors, Captured_Colors):
    ColorMatrix = Arg[:9].reshape([3,3])
    ColorConstant = Arg[9:12]
    ColorGamma = Arg[12:15]
    ErrorList = []
    for Color, Captured_Color in zip(Colors, Captured_Colors):
        Color2 = np.dot(ColorMatrix, Captured_Color) + ColorConstant
        Color3 = 255.0 * np.power(Color2/255.0, ColorGamma)
        Error = np.linalg.norm(Color - Color3)
        ErrorList.append(Error)
    return ErrorList
    
def correctColor(Image, ColorMatrix, ColorConstant, ColorGamma):
    ImageCorrected = np.zeros_like(Image)
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            Captured_Color = Image[i,j,:].reshape([3])
            Color2 = np.dot(ColorMatrix, Captured_Color) + ColorConstant
            Color3 = 255.0 * np.power(Color2/255.0, ColorGamma)
            ImageCorrected[i,j,:] = np.uint8(Color3)
    return ImageCorrected

# Using modified Gamma Correction Algorithm by
# Constantinou2013 - A comparison of color correction algorithms for endoscopic cameras
def getColorMatchingErrorVectorised(Arg, Colors, Captured_Colors):
    ColorMatrix = Arg[:9].reshape([3,3])
    ColorConstant = Arg[9:12].reshape([3,1])
    ColorGamma = Arg[12:15]
    
    TempRGB = np.dot(ColorMatrix, Captured_Colors) + ColorConstant
    Corrected_Colors = np.zeros_like(TempRGB)
    Corrected_Colors[0,:] = 255.0*np.power(TempRGB[0,:]/255.0, ColorGamma[0])
    Corrected_Colors[1,:] = 255.0*np.power(TempRGB[1,:]/255.0, ColorGamma[1])
    Corrected_Colors[2,:] = 255.0*np.power(TempRGB[2,:]/255.0, ColorGamma[2])
    
    Diff = Colors - Corrected_Colors
    ErrorList = np.sqrt(np.sum(Diff*Diff, axis= 0)).tolist()
    return ErrorList
    
def estimateColorParameters(TrueColors, ActualColors):
    # estimate color-correction parameters
    colorMatrix = np.eye(3)
    colorConstant = np.zeros([3,1])
    colorGamma = np.ones([3,1])
 
    Arg2 = np.zeros([9 + 3 + 3])
    Arg2[:9] = colorMatrix.reshape([9])
    Arg2[9:12] = colorConstant.reshape([3])
    Arg2[12:15] = colorGamma.reshape([3])
    
    ArgRefined, _ = optimize.leastsq(getColorMatchingErrorVectorised, \
            Arg2, args=(TrueColors, ActualColors), maxfev=10000)

    colorMatrix = ArgRefined[:9].reshape([3,3])
    colorConstant = ArgRefined[9:12].reshape([3,1])
    colorGamma = ArgRefined[12:15]
    return colorMatrix, colorConstant, colorGamma

def correctColorVectorised(Image, ColorMatrix, ColorConstant, ColorGamma):
    Width, Height = Image.shape[1::-1]
    CapturedR = Image[:,:,0].reshape([1,Width*Height])
    CapturedG = Image[:,:,1].reshape([1,Width*Height])
    CapturedB = Image[:,:,2].reshape([1,Width*Height])
    CapturedRGB = np.concatenate((CapturedR, CapturedG, CapturedB), axis=0)
    
    TempRGB = np.dot(ColorMatrix, CapturedRGB) + ColorConstant
    CorrectedRGB = np.zeros_like(TempRGB)
    CorrectedRGB[0,:] = 255.0*np.power(TempRGB[0,:]/255.0, ColorGamma[0])
    CorrectedRGB[1,:] = 255.0*np.power(TempRGB[1,:]/255.0, ColorGamma[1])
    CorrectedRGB[2,:] = 255.0*np.power(TempRGB[2,:]/255.0, ColorGamma[2])
    
    CorrectedR = CorrectedRGB[0,:].reshape([Height, Width])
    CorrectedG = CorrectedRGB[1,:].reshape([Height, Width])
    CorrectedB = CorrectedRGB[2,:].reshape([Height, Width])
    ImageCorrected = np.zeros_like(Image)
    ImageCorrected[:,:,0] = CorrectedR
    ImageCorrected[:,:,1] = CorrectedG
    ImageCorrected[:,:,2] = CorrectedB
    return ImageCorrected

def rotateImage(Image, RotationAngle = 0.0):
    Image_ = Image
    if RotationAngle%90.0 == 0:
        k = RotationAngle//90.0
        Image_ = np.rot90(Image_, k)
    elif RotationAngle != 0:
        center=tuple(np.array(Image_.shape[0:2])/2)
        rot_mat = cv2.getRotationMatrix2D(center, RotationAngle,1.0)
        Image_ = cv2.warpAffine(Image_, rot_mat, Image_.shape[0:2],flags=cv2.INTER_LINEAR)
    return Image_
    
def matchTemplateLocation(Image, Template, EstimatedLocation, SearchRange = [0.5, 0.5], RangeInImage = True):
    if RangeInImage: # use image size
        Width = Image.shape[1]
        Height = Image.shape[0]
    else: # use template size
        Width = Template.shape[1]
        Height = Template.shape[0]
        
    if SearchRange == None: # search throughout the whole images
        CroppedHalfWidth  = Width//2 
        CroppedHalfHeight = Height//2
    elif SearchRange[0] <= 1.0 and SearchRange[1] <= 1.0: # in fraction values
        CroppedHalfWidth  = (Template.shape[1]+SearchRange[0]*Width)//2 
        CroppedHalfHeight = (Template.shape[0]+SearchRange[1]*Height)//2
    else: # in pixels values
        CroppedHalfWidth  = (Template.shape[1]+SearchRange[0])//2 
        CroppedHalfHeight = (Template.shape[0]+SearchRange[1])//2
        
    if CroppedHalfWidth > Image.shape[1]//2-1:
        CroppedHalfWidth = Image.shape[1]//2-1
    if CroppedHalfHeight > Image.shape[0]//2-1:
        CroppedHalfHeight = Image.shape[0]//2-1

    SearchTopLeftCorner     = [EstimatedLocation[0]-CroppedHalfWidth, EstimatedLocation[1]-CroppedHalfHeight]
    SearchBottomRightCorner = [EstimatedLocation[0]+CroppedHalfWidth, EstimatedLocation[1]+CroppedHalfHeight]

    return matchTemplateWindow(Image, Template, SearchTopLeftCorner, SearchBottomRightCorner)

def matchTemplateWindow(Image, Template, SearchTopLeftCorner, SearchBottomRightCorner):
    CropedImage = Image[SearchTopLeftCorner[1]:SearchBottomRightCorner[1], SearchTopLeftCorner[0]:SearchBottomRightCorner[0]]
    corrMap = cv2.matchTemplate(CropedImage.astype(np.uint8), Template.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(corrMap)
    # recalculate max position in cropped image space
    matchedLocImageCropped = (maxLoc[0] + Template.shape[1]//2, 
                              maxLoc[1] + Template.shape[0]//2)
    # recalculate max position in full image space
    matchedLocImage = (matchedLocImageCropped[0] + SearchTopLeftCorner[0], \
                       matchedLocImageCropped[1] + SearchTopLeftCorner[1])
#    if isShow:
#        plt.figure()
#        plt.imshow(Template)
#        plt.figure()
#        plt.imshow(corrMap)
#        plt.hold(True)
#        plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
#        plt.figure()
#        plt.imshow(CropedImage)
#        plt.hold(True)
#        plt.plot([matchedLocImageCropped[0]], [matchedLocImageCropped[1]], 'o')
#        plt.figure()
#        plt.imshow(Image)
#        plt.hold(True)
#        plt.plot([matchedLocImage[0]], [matchedLocImage[1]], 'o')
#        plt.show()

    return matchedLocImage, maxVal, maxLoc, corrMap
    
def createImagePyramid(Image, NoLevels = 5):
    for i in range(NoLevels):
        if i == 0:
            PyramidImages = [Image.astype(np.uint8)]
        else:
            PyramidImages.append(cv2.pyrDown(PyramidImages[i-1]).astype(np.uint8))
    return PyramidImages

def matchTemplatePyramid(PyramidImages, PyramidTemplates, RotationAngle = None, \
        EstimatedLocation = None, SearchRange = None, NoLevels = 4, FinalLevel = 1):
    for i in range(NoLevels-1, -1, -1):
        if i == NoLevels-1:
            if EstimatedLocation == None:
                maxLocEst = [PyramidImages[i].shape[1]//2, PyramidImages[i].shape[0]//2] # image center
            else:
                maxLocEst = [EstimatedLocation[0]//2**i, EstimatedLocation[1]//2**i] # scale position to the pyramid level

            if SearchRange[0] > 1.0 and SearchRange[1] > 1.0:
                SearchRange2 = [SearchRange[0]//2**i, SearchRange[1]//2**i]
            else:
                SearchRange2 = SearchRange

            matchedLocImage, maxVal, maxLoc, corrMap = matchTemplateLocation(PyramidImages[i], PyramidTemplates[i], maxLocEst, SearchRange = SearchRange2)
            if RotationAngle == None:
                matchedLocImage180, maxVal180, maxLoc180, corrMap180 = matchTemplateLocation(np.rot90(PyramidImages[i],2).astype(np.uint8), PyramidTemplates[i], maxLocEst, SearchRange)
                if maxVal < 0.3 and maxVal180 < 0.3:
                    print('#### Warning: low matching score ####')
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
            searchRange = [6,6]
                            
            matchedLocImage, maxVal, maxLoc, corrMap = matchTemplateLocation(PyramidImages[i], PyramidTemplates[i], maxLocEst, searchRange)
            # rescale to location in level-0 image
            matchedLocImage0 = (matchedLocImage[0]*2**i, matchedLocImage[1]*2**i)

#        plt.figure()
#        plt.imshow(PyramidTemplates[i])     
#        
#        plt.figure()
#        plt.imshow(corrMap)
#        plt.hold(True)
#        plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
#        plt.title('maxVal = %f' %maxVal)
#        
#        plt.figure()
#        plt.imshow(PyramidImages[i])
#        plt.hold(True)
#        plt.plot([matchedLocImage[0]], [matchedLocImage[1]], 'o')
#        plt.plot([maxLocEst[0]], [maxLocEst[1]], 'x')
#        plt.title('Level = %d, RotationAngle = %f' %(i, RotationAngle))
#        plt.show()

        if i ==  FinalLevel:
            # Skip early to save time
            break
        
    print('maxVal, maxLocImage, RotationAngle =', maxVal, matchedLocImage0, RotationAngle)
    return maxVal, matchedLocImage0, RotationAngle
