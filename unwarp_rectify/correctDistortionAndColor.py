# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:17:53 2014

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

global isShow
isShow= False
    
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
#        plt.imshow(corrMap)
#        plt.hold(True)
#        plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
#        plt.title('maxVal = %f' %maxVal)
#        
#        plt.figure()
#        plt.imshow(PyramidImages[i])
#        plt.hold(True)
#        plt.plot([matchedLocImage[0]], [matchedLocImage[1]], 'o')
#        plt.title('Level = %d, RotationAngle = %f' %(i, RotationAngle))
#        plt.show()

        if i ==  FinalLevel:
            # Skip early to save time
            break
        
    print('maxVal, maxLocImage, RotationAngle =', maxVal, matchedLocImage0, RotationAngle)
    return maxVal, matchedLocImage0, RotationAngle

def correctDistortionAndColor(Arg):
    ImageFile_, UndistMapX, UndistMapY, P24ColorCardCaptured_PyramidImages, Colors, Tray_PyramidImagesList, Pot_PyramidImages, OutputFile = Arg
    Image = cv2.imread(ImageFile_)[:,:,::-1] # read and convert to R-G-B image
    
    if UndistMapX != None:
        Image = cv2.remap(Image.astype(np.uint8), UndistMapX, UndistMapY, cv2.INTER_CUBIC)

    RotationAngle = 0
    if Image.shape[0] > Image.shape[1]:
        RotationAngle = 90
        Image = rotateImage(Image, RotationAngle)
    PyramidImages = createImagePyramid(Image)
    ColorCardScore, ColorCardLoc, ColorCardAngle = matchTemplatePyramid(PyramidImages, P24ColorCardCaptured_PyramidImages, SearchRange = [0.5, 0.5])
    if ColorCardScore > 0.3:
        Image = rotateImage(Image, ColorCardAngle)
        ColorCardCaptured = Image[ColorCardLoc[1]-P24ColorCardCaptured_PyramidImages[0].shape[0]//2:ColorCardLoc[1]+P24ColorCardCaptured_PyramidImages[0].shape[0]//2, \
                                  ColorCardLoc[0]-P24ColorCardCaptured_PyramidImages[0].shape[1]//2:ColorCardLoc[0]+P24ColorCardCaptured_PyramidImages[0].shape[1]//2]
        
        Captured_Colors = np.zeros([3,24])
        STD_Colors = np.zeros([24])
        SquareSize2 = int(ColorCardCaptured.shape[0]/4)
        HalfSquareSize2 = int(SquareSize2/2)
        for i in range(24):
            Row = int(i/6)
            Col = i - Row*6
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
    
        # initial values
        ColorMatrix = np.eye(3)
        ColorConstant = np.zeros([3,1])
        ColorGamma = np.ones([3,1])
     
        Arg2 = np.zeros([9 + 3 + 3])
        Arg2[:9] = ColorMatrix.reshape([9])
        Arg2[9:12] = ColorConstant.reshape([3])
        Arg2[12:15] = ColorGamma.reshape([3])
        
        ArgRefined, _ = optimize.leastsq(getColorMatchingErrorVectorised, Arg2, args=(Colors, Captured_Colors), maxfev=10000)
        
        ErrrorList = getColorMatchingErrorVectorised(ArgRefined, Colors, Captured_Colors)
        ColorCorrectionError = np.sum(np.asarray(ErrrorList))
        
        ColorMatrix = ArgRefined[:9].reshape([3,3])
        ColorConstant = ArgRefined[9:12].reshape([3,1])
        ColorGamma = ArgRefined[12:15]
        
        ImageCorrected = correctColorVectorised(Image.astype(np.float), ColorMatrix, ColorConstant, ColorGamma)
        ImageCorrected[np.where(ImageCorrected < 0)] = 0
        ImageCorrected[np.where(ImageCorrected > 255)] = 255
        Corrected_PyramidImages = createImagePyramid(ImageCorrected)
        
        TrayLocs = []
        PotLocs2 = []
        PotLocs2_ = []
        PotIndex = 0
        for Tray_PyramidImages in Tray_PyramidImagesList:
            TrayScore, TrayLoc, TrayAngle = matchTemplatePyramid(Corrected_PyramidImages, Tray_PyramidImages, RotationAngle = 0, SearchRange = [1.0, 1.0])
            TrayLocs.append(TrayLoc)

            StepX = Tray_PyramidImages[0].shape[1]//4
            StepY = Tray_PyramidImages[0].shape[0]//5
            StartX = TrayLoc[0] - Tray_PyramidImages[0].shape[1]//2 + StepX//2
            StartY = TrayLoc[1] + Tray_PyramidImages[0].shape[0]//2 - StepY//2
            SearchRange = [Pot_PyramidImages[0].shape[1]//6, Pot_PyramidImages[0].shape[0]//6]
            print('SearchRange=', SearchRange)
            PotLocs = []
            PotLocs_ = []
            for k in range(4):
                for l in range(5):
#                    if PotIndex == 70:
#                        global isShow
#                        isShow = True
#                    else:
#                        global isShow
#                        isShow = False
                        
                    EstimateLoc = [StartX + StepX*k, StartY - StepY*l]
                    PotScore, PotLoc, PotAngle = matchTemplatePyramid(Corrected_PyramidImages, \
                        Pot_PyramidImages, RotationAngle = 0, \
                        EstimatedLocation = EstimateLoc, NoLevels = 3, SearchRange = SearchRange)
                    PotLocs.append(PotLoc)
                    PotLocs_.append(EstimateLoc)
                    PotIndex = PotIndex + 1
            PotLocs2.append(PotLocs)
            PotLocs2_.append(PotLocs_)

        plt.figure()
        plt.imshow(Pot_PyramidImages[0])
        plt.figure()
        plt.imshow(Corrected_PyramidImages[0])
        plt.hold(True)
        plt.plot([ColorCardLoc[0]], [ColorCardLoc[1]], 'ys')
        plt.text(ColorCardLoc[0]-30, ColorCardLoc[1]-15, 'ColorCard', color='yellow')
        PotIndex = 0
        for i,Loc in enumerate(TrayLocs):
            plt.plot([Loc[0]], [Loc[1]], 'bo')
            plt.text(Loc[0], Loc[1]-15, 'T'+str(i+1), color='blue', fontsize=20)
            for PotLoc,PotLoc_ in zip(PotLocs2[i], PotLocs2_[i]):
                plt.plot([PotLoc[0]], [PotLoc[1]], 'ro')
                plt.text(PotLoc[0], PotLoc[1]-15, str(PotIndex+1), color='red')  
                plt.plot([PotLoc_[0]], [PotLoc_[1]], 'rx')
                PotIndex = PotIndex + 1
                
        plt.title(os.path.basename(ImageFile_))
        plt.show()
            
    else:
        print('Skip color correction of', ImageFile_)
        ImageCorrected = Image
        ColorCardLoc = [-1.0, -1.0]
        ColorCorrectionError = -1.0
        TrayLocs = []
        
#    OutputPath = os.path.dirname(OutputFile)
#    if not os.path.exists(OutputPath):
#        print('Make', OutputPath)
#        os.makedirs(OutputPath)
#    cv2.imwrite(OutputFile, np.uint8(ImageCorrected[:,:,2::-1])) # convert to B-G-R image and save
#    print('Saved', OutputFile)
        
    return ColorCardScore, ColorCardLoc, ColorCorrectionError, TrayLocs


def main(argv):
    HelpString = 'correctDistortionAndColor.py -i <image file> ' + \
                    '-f <root image folder> '+ \
                    '-o <output file>\n' + \
                 'Example:\n' + \
                 "$ ./correctDistortionAndColor.py -f /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-orig/ -k /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/calib_param_700Dcam.yml -g /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3in.png -c /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3inCaptured.png -o /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/ -j 16"
    try:
        opts, args = getopt.getopt(argv,"hi:f:c:k:b:g:t:o:j:",\
            ["ifile=","ifolder=","--configfolder","calibfile=","captured-colorcard=",\
             "groundtruth-colorcard=","--tray-image-pattern","ofolder=","jobs="])
    except getopt.GetoptError:
        print(HelpString)
        sys.exit(2)
    if len(opts) == 0:
        print(HelpString)
        sys.exit()

    ImageFile = ''
    InputRootFolder = ''
    OutputFolder = ''
    ColorCardTrueFile = 'CameraTrax_24ColorCard_2x3in.png'
    ColorCardCapturedFile = 'CameraTrax_24ColorCard_2x3inCaptured.png'
    TrayCapturedFile = 'Tray%02d.png'
    PotCapturedFile = 'PotCaptured2.png' #'PotCaptured.png' #'PotCaptured4.png' #'PotCaptured3.png' #
    CalibFile = 'Canon700D_18mm_CalibParam.yml'
    ConfigFolder = ''
    NoJobs = 1
    for opt, arg in opts:
        if opt == '-h':
            print(HelpString)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ImageFile = arg
        elif opt in ("-f", "--ifolder"):
            InputRootFolder = arg
        elif opt in ("-c", "--configfolder"):
            ConfigFolder = arg
        elif opt in ("-b", "--captured-colorcard"):
            ColorCardCapturedFile = arg
        elif opt in ("-g", "--groundtruth-colorcard"):
            ColorCardTrueFile = argrot90
        elif opt in ("-t", "--tray-image-pattern"):
            TrayCapturedFile = arg
        elif opt in ("-k", "--calibfile"):
            CalibFile = arg
        elif opt in ("-o", "--ofolder"):
            OutputFolder = arg
        elif opt in ("-j", "--jobs"):
            NoJobs = int(arg)

    if len(ConfigFolder) > 0:
        ColorCardTrueFile     = os.path.join(ConfigFolder, os.path.basename(ColorCardTrueFile))
        ColorCardCapturedFile = os.path.join(ConfigFolder, os.path.basename(ColorCardCapturedFile))
        TrayCapturedFile      = os.path.join(ConfigFolder, os.path.basename(TrayCapturedFile))
        PotCapturedFile       = os.path.join(ConfigFolder, os.path.basename(PotCapturedFile))
        CalibFile              = os.path.join(ConfigFolder, os.path.basename(CalibFile))

    if len(OutputFolder) > 0 and not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder) 
    
    if len(CalibFile) > 0:
        ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs = readCalibration(CalibFile)
        print('CameraMatrix =', CameraMatrix)
        print('DistCoefs =', DistCoefs)
        UndistMapX, UndistMapY = cv2.initUndistortRectifyMap(CameraMatrix, DistCoefs, \
            None, CameraMatrix, ImageSize, cv2.CV_32FC1)    
            
    P24ColorCardTrueImage = cv2.imread(ColorCardTrueFile)[:,:,::-1] # read and convert to R-G-B image
    SquareSize = int(P24ColorCardTrueImage.shape[0]/4)
    HalfSquareSize = int(SquareSize/2)
    
    P24ColorCardCapturedImage = cv2.imread(ColorCardCapturedFile)[:,:,::-1] # read and convert to R-G-B image
    P24ColorCardCaptured_PyramidImages = createImagePyramid(P24ColorCardCapturedImage)

    Tray_PyramidImagesList = []
    for i in range(8):
        TrayFilename = TrayCapturedFile %(i+1)
        TrayImage = cv2.imread(TrayFilename)[:,:,::-1]
        if TrayImage == None:
            print('Unable to read', TrayFilename)
        Tray_PyramidImages = createImagePyramid(TrayImage)
        Tray_PyramidImagesList.append(Tray_PyramidImages)

    PotCapturedImage = cv2.imread(PotCapturedFile)[:,:,::-1] # read and convert to R-G-B image
    if PotCapturedImage == None:
        print('Unable to read', TrayFilename)
    Pot_PyramidImages = createImagePyramid(PotCapturedImage)

    # collect 24 colours from the captured color card:
    Colors = np.zeros([3,24])
    for i in range(24):
        Row = int(i/6)
        Col = i - Row*6
        rr = Row*SquareSize + HalfSquareSize
        cc = Col*SquareSize + HalfSquareSize
        Colors[0,i] = P24ColorCardTrueImage[rr,cc,0]
        Colors[1,i] = P24ColorCardTrueImage[rr,cc,1]
        Colors[2,i] = P24ColorCardTrueImage[rr,cc,2]
    print('Colors = \n', Colors)
    
    if len(ImageFile):
        img_iter = [sorted(glob.glob(ImageFile))]
    elif len(InputRootFolder):
        img_iter = ts_iter_images(InputRootFolder)
    else:
        print('Need imput image for image folder')
        return
    
    ArgList = []
    for i,ImageFile_ in enumerate(img_iter):
        if len(OutputFolder) > 0 and len(InputRootFolder) == 0:
            OutputFile = os.path.join(OutputFolder, os.path.basename(ImageFile_))
        elif len(OutputFolder) > 0 and len(InputRootFolder) > 0:
            ImagePath = os.path.dirname(ImageFile_)
            ImageName = os.path.basename(ImageFile_)
            OutputPath = os.path.join(OutputFolder, ImagePath[len(InputRootFolder):])
            OutputFile = os.path.join(OutputPath, ImageName)
        ArgList.append([ImageFile_, UndistMapX, UndistMapY, P24ColorCardCaptured_PyramidImages, Colors, Tray_PyramidImagesList, Pot_PyramidImages, OutputFile])
#        if i == 50:
#            break
    Process = Pool(processes = NoJobs)
    import time
    time1 = time.time()
    
#    Results = Process.map(correctDistortionAndColor, ArgList)
    for Arg in ArgList:
        correctDistortionAndColor(Arg)
    
    time2 = time.time()
    InfoFile = os.path.join(OutputFolder, 'ColorCorrectionInfo.txt')
    with open(InfoFile, 'w') as myfile:
        myfile.write('It took %0.3f seconds to process %d files using %d processes\n' % (time2-time1, len(Results), NoJobs))
        myfile.write('ImageFileName; MatchingScore; ColorPosition-X(-1.0 for undetected colorbar); ColorbarPosition-Y(-1.0 for undetected colorbar); CorrectionError(-1.0 for undetected colorbar)\n')
        for Result,Arg in zip(Results, ArgList):
            myfile.write('%s; %f; %d; %d; %f\n' %(Arg[0], Result[0], Result[1][0], Result[1][1], Result[2]) )
    print('Finished. Saved color correction info to', InfoFile)
if __name__ == "__main__":
   main(sys.argv[1:])


