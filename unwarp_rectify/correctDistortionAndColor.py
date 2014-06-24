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
import glob
from scipy import optimize
from timestream.parse import ts_iter_images
from multiprocessing import Pool
import utils
import json

global isShow
isShow= False
    

def correctDistortionAndColor(Arg):
    ImageFile_, RotationAngle, UndistMapX, UndistMapY, P24ColorCardCaptured_PyramidImages, colorcardPosition, Colors, Tray_PyramidImagesList, trayPositions, Pot_PyramidImages, OutputFile = Arg
    Image = cv2.imread(ImageFile_)[:,:,::-1] # read and convert to R-G-B image
    print(ImageFile_)
    if Image == None:
        print('Cannot read file')
        return 

    Image = utils.rotateImage(Image, RotationAngle)
    if UndistMapX != None:
        Image = cv2.remap(Image.astype(np.uint8), UndistMapX, UndistMapY, cv2.INTER_CUBIC)

    # set fallback settings
    ImageCorrected = Image
    ColorCardScore = None
    ColorCardLoc = None
    ColorCorrectionError = None
    TrayLocs = None
    PotLocs = None

    meanIntensity = np.mean(Image.astype(np.float))
    if meanIntensity < 10:
        print('meanIntensity = ', meanIntensity)
        print('Image is too dark to process')
        plt.figure()
        plt.imshow(Image)
        plt.show()
        return meanIntensity, ColorCardScore, ColorCardLoc, ColorCorrectionError, TrayLocs, PotLocs

    PyramidImages = utils.createImagePyramid(Image)
    SearchRange = [P24ColorCardCaptured_PyramidImages[0].shape[1]//2, P24ColorCardCaptured_PyramidImages[0].shape[0]//2]
    ColorCardScore, ColorCardLoc, ColorCardAngle = utils.matchTemplatePyramid(PyramidImages, P24ColorCardCaptured_PyramidImages, 0, EstimatedLocation = colorcardPosition, SearchRange = SearchRange)
    if ColorCardScore > 0.3:
        Image = utils.rotateImage(Image, ColorCardAngle)
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
        
        ArgRefined, _ = optimize.leastsq(utils.getColorMatchingErrorVectorised, Arg2, args=(Colors, Captured_Colors), maxfev=10000)
        
        ErrrorList = utils.getColorMatchingErrorVectorised(ArgRefined, Colors, Captured_Colors)
        ColorCorrectionError = np.sum(np.asarray(ErrrorList))
        
        ColorMatrix = ArgRefined[:9].reshape([3,3])
        ColorConstant = ArgRefined[9:12].reshape([3,1])
        ColorGamma = ArgRefined[12:15]
        
        ImageCorrected = utils.correctColorVectorised(Image.astype(np.float), ColorMatrix, ColorConstant, ColorGamma)
        ImageCorrected[np.where(ImageCorrected < 0)] = 0
        ImageCorrected[np.where(ImageCorrected > 255)] = 255
    else:    
        print('Skip color correction of', ImageFile_)
        
    # suppress green information to improve tray and pot detection
    ImageCorrected_NoGreen = np.zeros_like(ImageCorrected)
    ImageCorrected_NoGreen[:,:,0] = ImageCorrected[:,:,0]
    ImageCorrected_NoGreen[:,:,2] = ImageCorrected[:,:,2]
    Corrected_PyramidImages = utils.createImagePyramid(ImageCorrected_NoGreen)
    Corrected_PyramidImages = utils.createImagePyramid(ImageCorrected_NoGreen)

    TrayLocs = []
    PotLocs2 = []
    PotLocs2_ = []
    PotIndex = 0
    for i,Tray_PyramidImages in enumerate(Tray_PyramidImagesList):
        SearchRange = [Tray_PyramidImages[0].shape[1]//6, Tray_PyramidImages[0].shape[0]//6]
        TrayScore, TrayLoc, TrayAngle = utils.matchTemplatePyramid(Corrected_PyramidImages, Tray_PyramidImages, RotationAngle = 0, EstimatedLocation = trayPositions[i], SearchRange = SearchRange)
        if TrayScore < 0.3:
            print('Low tray matching score. Likely tray %d is missing.' %i)
            TrayLocs.append(None)
            continue
        TrayLocs.append(TrayLoc)

        StepX = Tray_PyramidImages[0].shape[1]//4
        StepY = Tray_PyramidImages[0].shape[0]//5
        StartX = TrayLoc[0] - Tray_PyramidImages[0].shape[1]//2 + StepX//2
        StartY = TrayLoc[1] + Tray_PyramidImages[0].shape[0]//2 - StepY//2
        SearchRange = [Pot_PyramidImages[0].shape[1]//6, Pot_PyramidImages[0].shape[0]//6]
#        SearchRange = [32, 32]
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
                PotScore, PotLoc, PotAngle = utils.matchTemplatePyramid(Corrected_PyramidImages, \
                    Pot_PyramidImages, RotationAngle = 0, \
                    EstimatedLocation = EstimateLoc, NoLevels = 3, SearchRange = SearchRange)
                PotLocs.append(PotLoc)
                PotLocs_.append(EstimateLoc)
                PotIndex = PotIndex + 1
        PotLocs2.append(PotLocs)
        PotLocs2_.append(PotLocs_)

#    plt.figure()
#    plt.imshow(Pot_PyramidImages[0])
#    plt.figure()
#    plt.imshow(ImageCorrected.astype(np.uint8))
#    plt.hold(True)
#    plt.plot([ColorCardLoc[0]], [ColorCardLoc[1]], 'ys')
#    plt.text(ColorCardLoc[0]-30, ColorCardLoc[1]-15, 'ColorCard', color='yellow')
#    PotIndex = 0
#    for i,Loc in enumerate(TrayLocs):
#        if Loc == None:
#            continue
#        plt.plot([Loc[0]], [Loc[1]], 'bo')
#        plt.text(Loc[0], Loc[1]-15, 'T'+str(i+1), color='blue', fontsize=20)
#        for PotLoc,PotLoc_ in zip(PotLocs2[i], PotLocs2_[i]):
#            plt.plot([PotLoc[0]], [PotLoc[1]], 'ro')
#            plt.text(PotLoc[0], PotLoc[1]-15, str(PotIndex+1), color='red')  
#            plt.plot([PotLoc_[0]], [PotLoc_[1]], 'rx')
#            PotIndex = PotIndex + 1
#            
#    plt.title(os.path.basename(ImageFile_))
#    plt.show()
            
        
    OutputPath = os.path.dirname(OutputFile)
    if not os.path.exists(OutputPath):
        print('Make', OutputPath)
        os.makedirs(OutputPath)
    cv2.imwrite(OutputFile, np.uint8(ImageCorrected[:,:,2::-1])) # convert to B-G-R image and save
    print('Saved', OutputFile)
        
    return OutputFile, meanIntensity, ColorCardScore, ColorCardLoc, ColorCorrectionError, TrayLocs


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
    InitialGeometryFile = 'ColorcardTrayPotSelections.yml'
    ColorCardTrueFile = 'CameraTrax_24ColorCard_2x3in.png'
#    ColorCardTrueFile = 'CameraTrax_24ColorCard_2x3in180deg.png'
#    ColorCardCapturedFile = 'CameraTrax_24ColorCard_2x3inCaptured.png'
    ColorCardCapturedFile = 'Card_%d.png'
    TrayCapturedFile = 'Tray_%d.png'
    PotCapturedFile = 'PotCaptured2.png' #'PotCaptured.png' #'PotCaptured4.png' #'PotCaptured3.png' #
    CalibFile = 'Canon700D_18mm_CalibParam.yml'
    ConfigFolder = ''
    RotationAngle = None
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
            ColorCardTrueFile = arg
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
        CalibFile             = os.path.join(ConfigFolder, os.path.basename(CalibFile))
        InitialGeometryFile   = os.path.join(ConfigFolder, os.path.basename(InitialGeometryFile))

    if len(OutputFolder) > 0 and not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder) 
    
    if len(CalibFile) > 0:
        ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs = utils.readCalibration(CalibFile)
        print('CameraMatrix =', CameraMatrix)
        print('DistCoefs =', DistCoefs)
        UndistMapX, UndistMapY = cv2.initUndistortRectifyMap(CameraMatrix, DistCoefs, \
            None, CameraMatrix, ImageSize, cv2.CV_32FC1)    
    
    if len(InitialGeometryFile):
        rotationAngle, distortionCorrected, colorcardList, trayList, potList = utils.readGeometries(InitialGeometryFile)
        print('trayList =', trayList)
        RotationAngle = rotationAngle
        colorcardCentre, colorcardWidth, colorcardHeight, colorcardAngle = utils.getRectangleParamters(colorcardList[0])
        colorcardPosition = [int(colorcardCentre[0]), int(colorcardCentre[1])]
        print('colorcardPosition =', colorcardPosition)
        potCentre, potWidth, potHeight, potAngle = utils.getRectangleParamters(potList[0])
        potSize = (int(potWidth), int(potHeight))
        print('potSize =', potSize)
        trayPositions = []
        for tray in trayList:
            trayCentre, trayWidth, trayHeight, trayAngle = utils.getRectangleParamters(tray)
            trayPositions.append([int(trayCentre[0]), int(trayCentre[1])])
        
    P24ColorCardTrueImage = cv2.imread(ColorCardTrueFile)[:,:,::-1] # read and convert to R-G-B image
    SquareSize = int(P24ColorCardTrueImage.shape[0]/4)
    HalfSquareSize = int(SquareSize/2)
    
    P24ColorCardCapturedImage = cv2.imread(ColorCardCapturedFile %0)[:,:,::-1] # read and convert to R-G-B image
    P24ColorCardCaptured_PyramidImages = utils.createImagePyramid(P24ColorCardCapturedImage)

    Tray_PyramidImagesList = []
    for i in range(8):
        TrayFilename = TrayCapturedFile %(i)
        TrayImage = cv2.imread(TrayFilename)
        if TrayImage == None:
            print('Unable to read', TrayFilename)
#            Tray_PyramidImages = None
            continue
        else:
            TrayImage = TrayImage[:,:,::-1]
            Tray_PyramidImages = utils.createImagePyramid(TrayImage)
        Tray_PyramidImagesList.append(Tray_PyramidImages)

    PotCapturedImage = cv2.imread(PotCapturedFile)[:,:,::-1] # read and convert to R-G-B image
    # supress green channel
    PotCapturedImage[:,:,1] = 0
    if PotCapturedImage == None:
        print('Unable to read', TrayFilename)
    scaleFactor = potSize[1]/PotCapturedImage.shape[0]
    print('scaleFactor', scaleFactor)
    PotCapturedImage = cv2.resize(PotCapturedImage, potSize, interpolation = cv2.INTER_CUBIC)
    print('PotCapturedImage.shape')
    Pot_PyramidImages = utils.createImagePyramid(PotCapturedImage)

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
        if i <= 830:
            continue
        if len(OutputFolder) > 0 and len(InputRootFolder) == 0:
            OutputFile = os.path.join(OutputFolder, os.path.basename(ImageFile_))
        elif len(OutputFolder) > 0 and len(InputRootFolder) > 0:
            ImagePath = os.path.dirname(ImageFile_)
            ImageName = os.path.basename(ImageFile_)
            OutputPath = os.path.join(OutputFolder, ImagePath[len(InputRootFolder):])
            OutputFile = os.path.join(OutputPath, ImageName)
        ArgList.append([ImageFile_, RotationAngle, UndistMapX, UndistMapY, P24ColorCardCaptured_PyramidImages, colorcardPosition, Colors, Tray_PyramidImagesList, trayPositions, Pot_PyramidImages, OutputFile])
#        if i == 50:
#            break
    Process = Pool(processes = NoJobs)
    import time
    time1 = time.time()
    
    Results = Process.map(correctDistortionAndColor, ArgList)
#    for Arg in ArgList:
#        correctDistortionAndColor(Arg)
    
    time2 = time.time()
    json.dump(Results, open(os.path.join(OutputFolder, 'Result.json')))
    InfoFile = os.path.join(OutputFolder, 'ColorCorrectionInfo.txt')
    with open(InfoFile, 'w') as myfile:
        myfile.write('It took %0.3f seconds to process %d files using %d processes\n' % (time2-time1, len(Results), NoJobs))
        myfile.write('ImageFileName; MatchingScore; ColorPosition-X(-1.0 for undetected colorbar); ColorbarPosition-Y(-1.0 for undetected colorbar); CorrectionError(-1.0 for undetected colorbar)\n')
    print('Finished. Saved color correction info to', InfoFile)
if __name__ == "__main__":
   main(sys.argv[1:])


