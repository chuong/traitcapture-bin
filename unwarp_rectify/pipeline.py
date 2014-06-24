# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:14:47 2014

@authors: initial skeleton by Joel Granados, updated by Chuong Nguyen
"""
from __future__ import absolute_import, division, print_function


import yaml
import numpy as np
from timestream.parse import ts_iter_images
import cv2
import utils
import matplotlib.pyplot as plt
import os

class PipeComponent ( object ):
    actName = "None"
    argNames = None
 
    runExpects = None
    runReturns = None
 
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
 
    def run(self, ts):
        raise NotImplementedError()

 
class ImageUndistorter ( PipeComponent ):
    actName = "undistort"
    argNames = {"mess": "Apply lens distortion correction"}
 
    runExpects = np.ndarray
    runReturns = np.ndarray
 
    def __init__(self, **kwargs):
        try:
            self.mess = kwargs["mess"]
            self.cameraMatrix = np.asarray(kwargs["cameraMatrix"])
            self.distortCoefs = np.asarray(kwargs["distortCoefs"])
            self.imageSize = tuple(kwargs["imageSize"])
            self.rotationAngle = kwargs["rotationAngle"]
            self.UndistMapX, self.UndistMapY = cv2.initUndistortRectifyMap( \
                self.cameraMatrix, self.distortCoefs, None, self.cameraMatrix, \
                self.imageSize, cv2.CV_32FC1)    
        except KeyError:
            self.mess = "Unable to read all parameters for " + ImageUndistorter.actName
            self.cameraMatrix = None
            self.distortCoefs = None
            self.imageSize = None
            self.rotationAngle = None
            self.UndistMapX, self.UndistMapY = None, None
 
    def run(self, image):
        print(self.mess)
        self.image = utils.rotateImage(image, self.rotationAngle)
        if self.UndistMapX != None and self.UndistMapY != None:
            self.imageUndistorted = cv2.remap(self.image.astype(np.uint8), \
                self.UndistMapX, self.UndistMapY, cv2.INTER_CUBIC)
        return(self.imageUndistorted)

    def show(self):
        plt.figure()
        plt.imshow(self.image)
        plt.title('Original image')
        plt.figure()
        plt.imshow(self.imageUndistorted)
        plt.title('Undistorted image')
        plt.show()
 
 
class ColorCardDetector ( PipeComponent ):
    actName = "colorcarddetect"
    argNames = {"mess": "Detect color card position and color info"}
 
    runExpects = np.ndarray
    runReturns = [np.ndarray, list]
 
    def __init__(self, **kwargs):
        try:
            self.mess = kwargs["mess"]
            self.colorcardTrueColors = kwargs["colorcardTrueColors"]
            self.colorcardFile = kwargs["colorcardFile"]
            self.colorcardPosition = kwargs["colorcardPosition"]
            self.settingPath = kwargs["settingPath"]
        except KeyError:
            self.mess = "Unable to read parameters for " + ColorCardDetector.actName
            self.colorcardTrueColors = None
            self.colorcardFile = None
            self.colorcardPosition = None
            self.settingPath = None
 
    def run(self, image):
        print(self.mess)
        self.colorcardImage = cv2.imread(os.path.join(self.settingPath, self.colorcardFile))[:,:,::-1]

        # create image pyramid for multiscale matching
        self.colorcardPyramid = utils.createImagePyramid(self.colorcardImage)
        self.imagePyramid = utils.createImagePyramid(image)
        SearchRange = [self.colorcardPyramid[0].shape[1], self.colorcardPyramid[0].shape[0]]
        score, loc, angle = utils.matchTemplatePyramid(self.imagePyramid, self.colorcardPyramid, \
            0, EstimatedLocation = self.colorcardPosition, SearchRange = SearchRange)
        if score > 0.3:
            # extract color information
            self.foundCard = image[loc[1]-self.colorcardImage.shape[0]//2:loc[1]+self.colorcardImage.shape[0]//2, \
                                   loc[0]-self.colorcardImage.shape[1]//2:loc[0]+self.colorcardImage.shape[1]//2]
            self.colorcardColors, _ = utils.getColorcardColors(self.foundCard, GridSize = [6, 4])
            self.colorcardParams = utils.estimateColorParameters(self.colorcardTrueColors, self.colorcardColors)
            # for displaying
            self.loc = loc
            self.image = image
        else:
            print('Cannot find color card')
            self.colorcardParams = [None, None, None]
            
        return(image, self.colorcardParams)

    def show(self):
        plt.figure()
        plt.imshow(self.image)
        plt.hold(True)
        plt.plot([self.loc[0]], [self.loc[1]], 'ys')
        plt.text(self.loc[0]-30, self.loc[1]-15, 'ColorCard', color='yellow')
        plt.title('Detected color card')
        plt.figure()
        plt.imshow(self.foundCard)
        plt.title('Detected color card')
        plt.show()
 
 
class ImageColorCorrector ( PipeComponent ):
    actName = "colorcorrect"
    argNames = {"mess": "Correct image color"}
 
    runExpects = [np.ndarray, list]
    runReturns = np.ndarray
 
    def __init__(self, **kwargs):
        try:
            self.mess = kwargs["mess"]
        except KeyError:
            self.mess = "Unable to read parameters for " + ImageColorCorrector.actName
 
    def run(self, inputs):
        print(self.mess)
        image, colorcardParam = inputs
        colorMatrix, colorConstant, colorGamma = colorcardParam
        if colorMatrix != None:
            self.imageCorrected = utils.correctColorVectorised(image.astype(np.float), colorMatrix, colorConstant, colorGamma)
            self.imageCorrected[np.where(self.imageCorrected < 0)] = 0
            self.imageCorrected[np.where(self.imageCorrected > 255)] = 255
            self.imageCorrected = self.imageCorrected.astype(np.uint8)
            self.image = image # for displaying
        else:
            print('Skip color correction')
            self.imageCorrected = image
        
        return(self.imageCorrected)
            
    def show(self):
        plt.figure()
        plt.imshow(self.image)
        plt.title('Image without color correction')
        plt.figure()
        plt.imshow(self.imageCorrected)
        plt.title('Color-corrected image')
        plt.show()


class TrayDetector ( PipeComponent ):
    actName = "traydetect"
    argNames = {"mess": "Detect tray positions"}
 
    runExpects = np.ndarray
    runReturns = [np.ndarray, list, list]
 
    def __init__(self, **kwargs):
        try:
            self.mess = kwargs["mess"]
            self.trayFiles = kwargs["trayFiles"]
            self.trayNumber = kwargs["trayNumber"]
            self.trayPositions = kwargs["trayPositions"]
            self.settingPath = kwargs["settingPath"]
        except KeyError:
            self.mess = "Unable to read parameters for " + TrayDetector.actName
            self.trayFiles = None
            self.trayNumber = None
            self.trayPositions = None
 
    def run(self, image):
        print(self.mess)
        self.image = image        
        temp = np.zeros_like(self.image)
        temp[:,:,:] = image[:,:,:]
        temp[:,:,1] = 0 # suppress green channel
        self.imagePyramid = utils.createImagePyramid(temp)
        self.trayPyramids = []
        for i in range(self.trayNumber):
            trayImage = cv2.imread(os.path.join(self.settingPath, self.trayFiles % i))[:,:,::-1]
            trayImage[:,:,1] = 0 # suppress green channel
            trayPyramid = utils.createImagePyramid(trayImage)
            self.trayPyramids.append(trayPyramid)
            
        self.trayLocs = []
        for i,trayPyramid in enumerate(self.trayPyramids):
            SearchRange = [trayPyramid[0].shape[1]//6, trayPyramid[0].shape[0]//6]
            score, loc, angle = utils.matchTemplatePyramid(self.imagePyramid, trayPyramid, \
                RotationAngle = 0, EstimatedLocation = self.trayPositions[i], SearchRange = SearchRange)
            if score < 0.3:
                print('Low tray matching score. Likely tray %d is missing.' %i)
                self.trayLocs.append(None)
                continue
            self.trayLocs.append(loc)
            
        return(self.image, self.imagePyramid, self.trayLocs)

    def show(self):
        plt.figure()
        plt.imshow(self.image.astype(np.uint8))
        plt.hold(True)
        PotIndex = 0
        for i,Loc in enumerate(self.trayLocs):
            if Loc == None:
                continue
            plt.plot([Loc[0]], [Loc[1]], 'bo')
            PotIndex = PotIndex + 1
        plt.title('Detected trays')
        plt.show()
        

class PotDetector ( PipeComponent ):
    actName = "potdetect"
    argNames = {"mess": "Detect pot position"}
 
    runExpects = [np.ndarray, list, list]
    runReturns = [np.ndarray, list]
 
    def __init__(self, **kwargs):
        try:
            self.mess = kwargs["mess"]
            self.potFile = kwargs["potFile"]
            self.potTemplateFile = kwargs["potTemplateFile"]
            self.potPosition = kwargs["potPosition"]
            self.potSize = kwargs["potSize"]
            self.traySize = kwargs["traySize"]
            self.settingPath = kwargs["settingPath"]
        except KeyError:
            self.mess = "Unable to read parameters for " + PotDetector.actName
            self.potFile = None
            self.potPosition = None
            self.potSize = None
 
    def run(self, inputs):
        print(self.mess)
        self.image, self.imagePyramid, self.trayLocs = inputs
        # read pot template image and scale to the pot size
        potImage = cv2.imread(os.path.join(self.settingPath, self.potFile))[:,:,::-1]
        potTemplateImage = cv2.imread(os.path.join(self.settingPath, self.potTemplateFile))[:,:,::-1]
        potTemplateImage[:,:,1] = 0 # suppress green channel
        potTemplateImage = cv2.resize(potTemplateImage.astype(np.uint8), (potImage.shape[1], potImage.shape[0]))
        self.potPyramid = utils.createImagePyramid(potTemplateImage)
        
        XSteps = self.traySize[0]//self.potSize[0]
        YSteps = self.traySize[1]//self.potSize[1]
        StepX  = self.traySize[0]//XSteps
        StepY  = self.traySize[1]//YSteps

        self.potLocs2 = []
        self.potLocs2_ = []
        for trayLoc in self.trayLocs:
            StartX = trayLoc[0] - self.traySize[0]//2 + StepX//2
            StartY = trayLoc[1] + self.traySize[1]//2 - StepY//2
            SearchRange = [self.potPyramid[0].shape[1]//4, self.potPyramid[0].shape[0]//4]
#            SearchRange = [32, 32]
            print('SearchRange=', SearchRange)
            potLocs = []
            potLocs_ = []
            for k in range(4):
                for l in range(5):
                    estimateLoc = [StartX + StepX*k, StartY - StepY*l]
                    score, loc,angle = utils.matchTemplatePyramid(self.imagePyramid, \
                        self.potPyramid, RotationAngle = 0, \
                        EstimatedLocation = estimateLoc, NoLevels = 3, SearchRange = SearchRange)
                    potLocs.append(loc)
                    potLocs_.append(estimateLoc)
            self.potLocs2.append(potLocs)
            self.potLocs2_.append(potLocs_)

        return(self.image, self.potLocs2)

    def show(self):
        plt.figure()
        plt.imshow(self.image.astype(np.uint8))
        plt.hold(True)
        PotIndex = 0
        for i,Loc in enumerate(self.trayLocs):
            if Loc == None:
                continue
            plt.plot([Loc[0]], [Loc[1]], 'bo')
            plt.text(Loc[0], Loc[1]-15, 'T'+str(i+1), color='blue', fontsize=20)
            for PotLoc,PotLoc_ in zip(self.potLocs2[i], self.potLocs2_[i]):
                plt.plot([PotLoc[0]], [PotLoc[1]], 'ro')
                plt.text(PotLoc[0], PotLoc[1]-15, str(PotIndex+1), color='red')  
                plt.plot([PotLoc_[0]], [PotLoc_[1]], 'rx')
                PotIndex = PotIndex + 1
        plt.title('Detected trays and pots')                
        plt.show()
            

class PlantExtractor ( PipeComponent ):
    actName = "plantextract"
    argNames = {"mess": "Extract plant biometrics"}
 
    runExpects = [np.ndarray, list]
    runReturns = [np.ndarray, list]
 
    def __init__(self, **kwargs):
        try:
            self.mess = kwargs["mess"]
        except KeyError:
            self.mess = "Unable to read parameters for " + PlantExtractor.actName
 
    def run(self, inputs):
        print(self.mess)
        image, potPositions = inputs
        print("Image size =", image.shape)
        plantBiometrics = []
        return(image, plantBiometrics)
 
    def show(self):
        pass
    
class ImagePipeline ( object ):
    complist = { ImageUndistorter.actName:      ImageUndistorter,
                 ColorCardDetector.actName:     ColorCardDetector, \
                 ImageColorCorrector.actName:   ImageColorCorrector, \
                 TrayDetector.actName:          TrayDetector, \
                 PotDetector.actName:           PotDetector, \
                 PlantExtractor.actName:         PlantExtractor \
               }
 
    def __init__(self, defFilePath):
        f = file(defFilePath)
        yamlStruct = yaml.load(f)
        f.close()
 
        self.pipeline = []
 
        # First elements needs to expect ndarray
        elem = yamlStruct.pop(0)
        if ( ImagePipeline.complist[elem[0]].runExpects is not np.ndarray ):
            raise ValueError("First pipe element should expect ndarray")
        self.pipeline.append( ImagePipeline.complist[elem[0]](**elem[1]) )
 
        # Add elements checking for dependencies
        for elem in yamlStruct:
            elem[1]['settingPath'] = os.path.dirname(defFilePath)
            if ( ImagePipeline.complist[elem[0]].runExpects !=
                    self.pipeline[-1].__class__.runReturns ):
                raise ValueError("Dependancy issue in pipeline")
 
            self.pipeline.append( ImagePipeline.complist[elem[0]](**elem[1]) )
 
    def process(self, ts):
        image = ts.getCurrentImage()        
        
        # First elem with inpupt image
        res = image 
        # Rest elems with previous results
        for elem in self.pipeline:
            res = elem.run(res)
            elem.show()
 
        return (res)
 
    def printCompList(self):
        print(ImagePipeline.complist)
 
 
class DummyTS(object):
    def __init__(self, rootPath):
        self.img_iter = ts_iter_images(rootPath)
        self.counter = -1
 
    def getCurrentImage(self):
        for i in range(750):
            self.img_iter.next()
        self.currentImage = cv2.imread(self.img_iter.next())[:,:,::-1]
        self.counter = self.counter + 1
        return self.currentImage
 
    def getFullImageList(self):
        # go to the start
        self.img_iter.seek(0)
        imageList = list(self.img_iter)
        # go back to current position
        self.img_iter.seek(self.counter)
        return imageList
 
if __name__ == "__main__":
 
    ts = DummyTS('/mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0036/BVZ0036-GC02L~fullres-orig/')
    ip = ImagePipeline("/home/chuong/Workspace/traitcapture-bin/unwarp_rectify/data/pipeline.yml")
 
    ip.process(ts)