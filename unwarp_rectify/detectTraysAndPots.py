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

    if isShow:
        plt.figure()
        plt.imshow(Template)
        plt.figure()
        plt.imshow(CropedImage)
        plt.show()

    corrMap = cv2.matchTemplate(CropedImage.astype(np.uint8), Template.astype(np.uint8), cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(corrMap)
    # recalculate max position in cropped image space
    matchedLocImageCropped = (maxLoc[0] + Template.shape[1]//2, 
                              maxLoc[1] + Template.shape[0]//2)
    # recalculate max position in full image space
    matchedLocImage = (matchedLocImageCropped[0] + SearchTopLeftCorner[0], \
                       matchedLocImageCropped[1] + SearchTopLeftCorner[1])
    if isShow:
        plt.figure()
        plt.imshow(corrMap)
        plt.hold(True)
        plt.plot([maxLoc[0]], [maxLoc[1]], 'o')
        plt.figure()
        plt.imshow(CropedImage)
        plt.hold(True)
        plt.plot([matchedLocImageCropped[0]], [matchedLocImageCropped[1]], 'o')
        plt.figure()
        plt.imshow(Image)
        plt.hold(True)
        plt.plot([matchedLocImage[0]], [matchedLocImage[1]], 'o')
        plt.show()

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
        
#    print('maxVal, maxLocImage, RotationAngle =', maxVal, matchedLocImage0, RotationAngle)
    return maxVal, matchedLocImage0, RotationAngle

def detectTraysAndPots(Arg):
    ImageFile_, Tray_PyramidImagesList, TrayEstimatedPositions, \
            Pot_PyramidImages, PotEstimatedPositions, OutputFile = Arg
    print('Process ', ImageFile_)
    Image = cv2.imread(ImageFile_)[:,:,::-1] # read and convert to R-G-B image
    PyramidImages = createImagePyramid(Image)
    
    TrayLocs = []
    PotLocs2 = []
    PotLocs2_ = []
    PotIndex = 0
    for i,Tray_PyramidImages in enumerate(Tray_PyramidImagesList):
        if TrayEstimatedPositions != None:
            EstimateLoc = [TrayEstimatedPositions[i][0]*PyramidImages[0].shape[1], 
                           TrayEstimatedPositions[i][1]*PyramidImages[0].shape[0]]
            TrayScore, TrayLoc, TrayAngle = matchTemplatePyramid(PyramidImages, Tray_PyramidImages, \
                RotationAngle = 0, EstimatedLocation = EstimateLoc, \
                SearchRange = [Tray_PyramidImages[0].shape[1]//2, Tray_PyramidImages[0].shape[0]//2])
        else:
            TrayScore, TrayLoc, TrayAngle = matchTemplatePyramid(PyramidImages, Tray_PyramidImages, RotationAngle = 0, SearchRange = [1.0, 1.0])
        TrayLocs.append(TrayLoc)

        PotLocs = []
        PotLocs_ = []
        SearchRange = [Pot_PyramidImages[0].shape[1]//6, Pot_PyramidImages[0].shape[0]//6]
        if PotEstimatedPositions != None:
            for PotEstimatedPosition in PotEstimatedPositions:
                EstimateLoc = [TrayLoc[0] - Tray_PyramidImages[0].shape[1]//2 + PotEstimatedPosition[0]*Tray_PyramidImages[0].shape[1], \
                               TrayLoc[1] - Tray_PyramidImages[0].shape[0]//2 + PotEstimatedPosition[1]*Tray_PyramidImages[0].shape[0]]
                
                PotScore, PotLoc, PotAngle = matchTemplatePyramid(PyramidImages, \
                    Pot_PyramidImages, RotationAngle = 0, \
                    EstimatedLocation = EstimateLoc, NoLevels = 3, SearchRange = SearchRange)
                PotLocs.append(PotLoc)
                PotLocs_.append(EstimateLoc)
        else:
            StepX = Tray_PyramidImages[0].shape[1]//4
            StepY = Tray_PyramidImages[0].shape[0]//5
            StartX = TrayLoc[0] - Tray_PyramidImages[0].shape[1]//2 + StepX//2
            StartY = TrayLoc[1] + Tray_PyramidImages[0].shape[0]//2 - StepY//2
            # assuming 5x4 pots per tray
            for k in range(4):
                for l in range(5):
                    EstimateLoc = [StartX + StepX*k, StartY - StepY*l]
                    PotScore, PotLoc, PotAngle = matchTemplatePyramid(PyramidImages, \
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
#    plt.imshow(PyramidImages[0])
#    plt.hold(True)
#    PotIndex = 0
#    for i,Loc in enumerate(TrayLocs):
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
        
#    OutputPath = os.path.dirname(OutputFile)
#    if not os.path.exists(OutputPath):
#        print('Make', OutputPath)
#        os.makedirs(OutputPath)
#    cv2.imwrite(OutputFile, np.uint8(ImageCorrected[:,:,2::-1])) # convert to B-G-R image and save
#    print('Saved', OutputFile)
    return TrayLocs, PotLocs


def main(argv):
    HelpString = 'correctDistortionAndColor.py -i <image file> ' + \
                    '-f <root image folder> '+ \
                    '-o <output file>\n' + \
                 'Example:\n' + \
                 "$ ./detectTraysAndPots.py -f /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/ -k /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/calib_param_700Dcam.yml -g /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3in.png -c /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3inCaptured.png -o /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/ -j 16"
    try:
        opts, args = getopt.getopt(argv,"hi:f:c:r:p:t:o:j:",\
            ["ifile=","ifolder=","--configfolder","--tray-estimated-positions=",\
             "--pot-estimated-positions=","--tray-image-pattern","ofolder=","jobs="])
    except getopt.GetoptError:
        print(HelpString)
        sys.exit(2)
    if len(opts) == 0:
        print(HelpString)
        sys.exit()

    ImageFile = ''
    InputRootFolder = ''
    OutputFolder = ''
    PotPositionFile = 'PotEstimatedPositions.yml'
    TrayPositionFile = 'TrayEstimatedPositions.yml'
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
        elif opt in ("-r", "--tray-estimated-positions"):
            TrayPositionFile = arg
        elif opt in ("-p", "--pot-estimated-positions"):
            PotPositionFile = arg
        elif opt in ("-t", "--tray-image-pattern"):
            TrayCapturedFile = arg
        elif opt in ("-o", "--ofolder"):
            OutputFolder = arg
        elif opt in ("-j", "--jobs"):
            NoJobs = int(arg)

    if len(ConfigFolder) > 0:
        PotPositionFile  = os.path.join(ConfigFolder, os.path.basename(PotPositionFile))
        TrayPositionFile = os.path.join(ConfigFolder, os.path.basename(TrayPositionFile))
        TrayCapturedFile = os.path.join(ConfigFolder, os.path.basename(TrayCapturedFile))
        PotCapturedFile  = os.path.join(ConfigFolder, os.path.basename(PotCapturedFile))
        CalibFile        = os.path.join(ConfigFolder, os.path.basename(CalibFile))

    if len(OutputFolder) > 0 and not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder) 
    
    try:
        PotEstimatedPositions  = cv2yml.yml2dic(PotPositionFile)['PotEstimatedPositions'].tolist()
        TrayEstimatedPositions = cv2yml.yml2dic(TrayPositionFile)['TrayEstimatedPositions'].tolist()
        print('Found estimated positions of trays and pots.')
    except:
        print('Estimated positions of trays and pots are not provided.\n Try without this information.')
        PotEstimatedPositions = None
        TrayEstimatedPositions = None
        
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
        ArgList.append([ImageFile_, Tray_PyramidImagesList, TrayEstimatedPositions, \
            Pot_PyramidImages, PotEstimatedPositions, OutputFile])
#        if i == 10:
#            break
        
    Process = Pool(processes = NoJobs)
    import time
    time1 = time.time()
    
    Results = Process.map(detectTraysAndPots, ArgList)
#    Results = []
#    for Arg in ArgList:
#        TrayLocs, PotLocs = detectTraysAndPots(Arg)
#        Results.append([TrayLocs, PotLocs])
    
    time2 = time.time()
    InfoFile = os.path.join(OutputFolder, 'TrayPotDetectionInfo.txt')
    with open(InfoFile, 'w') as myfile:
        myfile.write('It took %0.3f seconds to process %d files using %d processes\n' % (time2-time1, len(Results), NoJobs))
        myfile.write('ImageFileName; 5 Tray x-y positions; 20 Pot x-y positions)\n')
        for Result,Arg in zip(Results, ArgList):
            myfile.write('%s; ' %Arg[0])
            for TrayPosition in Result[0]:
                myfile.write('%d; %d ' %(TrayPosition[0], TrayPosition[1]))
            for PotPosition in Result[1]:
                myfile.write('%d; %d ' %(PotPosition[0], PotPosition[1]))
            myfile.write('\n')

    print('Finished. Saved color correction info to', InfoFile)
if __name__ == "__main__":
   main(sys.argv[1:])


