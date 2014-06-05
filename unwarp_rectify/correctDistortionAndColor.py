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
    ImageFile_, UndistMapX, UndistMapY, P24ColorCardCaptured, Colors, OutputFile = Arg
    Image = cv2.imread(ImageFile_)[:,:,::-1] # read and convert to R-G-B image
    
    if UndistMapX != None:
        Image = cv2.remap(Image.astype(np.uint8), UndistMapX, UndistMapY, cv2.INTER_CUBIC)

    RotationAngle = 0
    if Image.shape[0] > Image.shape[1]:
        RotationAngle = 90
        Image = rotateImage(Image, RotationAngle)

    maxVal, matchedLoc, RotationAngle2 = findColorbarPyramid(Image.astype(np.uint8), P24ColorCardCaptured.astype(np.uint8))
    if maxVal > 0.3:
        Image = rotateImage(Image, RotationAngle2)
        ColorCardCaptured = Image[matchedLoc[1]-P24ColorCardCaptured.shape[0]//2:matchedLoc[1]+P24ColorCardCaptured.shape[0]//2, \
                                  matchedLoc[0]-P24ColorCardCaptured.shape[1]//2:matchedLoc[0]+P24ColorCardCaptured.shape[1]//2]
        
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
        CorrectionError = np.sum(np.asarray(ErrrorList))
        
        ColorMatrix = ArgRefined[:9].reshape([3,3])
        ColorConstant = ArgRefined[9:12].reshape([3,1])
        ColorGamma = ArgRefined[12:15]
        
        ImageCorrected = correctColorVectorised(Image.astype(np.float), ColorMatrix, ColorConstant, ColorGamma)
        ImageCorrected[np.where(ImageCorrected < 0)] = 0
        ImageCorrected[np.where(ImageCorrected > 255)] = 255
    else:
        print('Skip color correction of', ImageFile_)
        ImageCorrected = Image
        matchedLoc = [-1.0, -1.0]
        CorrectionError = -1.0
        
    OutputPath = os.path.dirname(OutputFile)
    if not os.path.exists(OutputPath):
        print('Make', OutputPath)
        os.makedirs(OutputPath)
    cv2.imwrite(OutputFile, np.uint8(ImageCorrected[:,:,2::-1])) # convert to B-G-R image and save
    print('Saved', OutputFile)
    return maxVal, matchedLoc, CorrectionError


def main(argv):
    HelpString = 'correctDistortionAndColor.py -i <image file> ' + \
                    '-f <root image folder> '+ \
                    '-o <output file>\n' + \
                 'Example:\n' + \
                 "$ ./correctDistortionAndColor.py -f /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-orig/ -k /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/calib_param_700Dcam.yml -g /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3in.png -c /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3inCaptured.png -o /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/ -j 16"
    try:
        opts, args = getopt.getopt(argv,"hi:f:k:c:g:o:j:",\
            ["ifile=","ifolder=","calibfile=","captured-colorcard=","groundtruth-colorcard=","ofolder=","jobs="])
    except getopt.GetoptError:
        print(HelpString)
        sys.exit(2)
    if len(opts) == 0:
        print(HelpString)
        sys.exit()

    ImageFile = ''
    InputRootFolder = ''
    OutputFolder = ''
    ColorCardCapturedImage = 'CameraTrax_24ColorCard_2x3in.png'
    ColorCardTrueImage = 'CameraTrax_24ColorCard_2x3inCaptured.png'
    CalibFile = ''
    NoJobs = 1
    for opt, arg in opts:
        if opt == '-h':
            print(HelpString)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ImageFile = arg
        elif opt in ("-f", "--ifolder"):
            InputRootFolder = arg
        elif opt in ("-c", "--captured-colorcard"):
            ColorCardCapturedImage = arg
        elif opt in ("-g", "--groundtruth-colorcard"):
            ColorCardTrueImage = arg
        elif opt in ("-k", "--calibfile"):
            CalibFile = arg
        elif opt in ("-o", "--ofolder"):
            OutputFolder = arg
        elif opt in ("-j", "--jobs"):
            NoJobs = int(arg)

    if len(OutputFolder) > 0 and not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder) 
    
    if len(CalibFile) > 0:
        ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs = readCalibration(CalibFile)
        print('CameraMatrix =', CameraMatrix)
        print('DistCoefs =', DistCoefs)
        UndistMapX, UndistMapY = cv2.initUndistortRectifyMap(CameraMatrix, DistCoefs, \
            None, CameraMatrix, ImageSize, cv2.CV_32FC1)    
            
    P24ColorCardTrue = cv2.imread(ColorCardTrueImage)[:,:,::-1] # read and convert to R-G-B image
    SquareSize = int(P24ColorCardTrue.shape[0]/4)
    HalfSquareSize = int(SquareSize/2)
    
    P24ColorCardCaptured = cv2.imread(ColorCardCapturedImage)[:,:,::-1] # read and convert to R-G-B image

    # collect 24 colours from the captured color card:
    Colors = np.zeros([3,24])
    for i in range(24):
        Row = int(i/6)
        Col = i - Row*6
        rr = Row*SquareSize + HalfSquareSize
        cc = Col*SquareSize + HalfSquareSize
        Colors[0,i] = P24ColorCardTrue[rr,cc,0]
        Colors[1,i] = P24ColorCardTrue[rr,cc,1]
        Colors[2,i] = P24ColorCardTrue[rr,cc,2]
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
        ArgList.append([ImageFile_, UndistMapX, UndistMapY, P24ColorCardCaptured, Colors, OutputFile])
#        if i == 50:
#            break
    Process = Pool(processes = NoJobs)
    import time
    time1 = time.time()
    Results = Process.map(correctDistortionAndColor, ArgList)
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


