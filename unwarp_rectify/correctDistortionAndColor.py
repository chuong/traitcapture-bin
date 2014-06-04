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
        Image_ = np.rot90(np.rot90(Image_), k)
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
    
def correctDistortionAndColor(Arg):
    ImageFile_, UndistMapX, UndistMapY, RotationAngle, ColCardMapX, ColCardMapY, Colors, OutputFile = Arg
    Image = cv2.imread(ImageFile_)[:,:,::-1] # read and convert to R-G-B image
    
    if UndistMapX != None:
        Image = cv2.remap(Image.astype(np.uint8), UndistMapX, UndistMapY, cv2.INTER_CUBIC)

    Image = rotateImage(Image, RotationAngle)
    RectifiedColorCard = cv2.remap(Image.astype(np.uint8), ColCardMapX, ColCardMapY, cv2.INTER_CUBIC)
    
    Captured_Colors = np.zeros([3,24])
    SquareSize2 = int(RectifiedColorCard.shape[0]/4)
    HalfSquareSize2 = int(SquareSize2/2)
    for i in range(24):
        Row = int(i/6)
        Col = i - Row*6
        rr = Row*SquareSize2 + HalfSquareSize2
        cc = Col*SquareSize2 + HalfSquareSize2
        Captured_R = RectifiedColorCard[rr-10:rr+10, cc-10:cc+10, 0].astype(np.float)
        Captured_G = RectifiedColorCard[rr-10:rr+10, cc-10:cc+10, 1].astype(np.float)
        Captured_B = RectifiedColorCard[rr-10:rr+10, cc-10:cc+10, 2].astype(np.float)
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
    
    ColorMatrix = ArgRefined[:9].reshape([3,3])
    ColorConstant = ArgRefined[9:12].reshape([3,1])
    ColorGamma = ArgRefined[12:15]
    
    ImageCorrected = correctColorVectorised(Image.astype(np.float), ColorMatrix, ColorConstant, ColorGamma)
    ImageCorrected[np.where(ImageCorrected < 0)] = 0
    ImageCorrected[np.where(ImageCorrected > 255)] = 255

    OutputPath = os.path.dirname(OutputFile)
    if not os.path.exists(OutputPath):
        print('Make', OutputPath)
        os.makedirs(OutputPath)
    cv2.imwrite(OutputFile, np.uint8(ImageCorrected[:,:,2::-1])) # convert to B-G-R image and save
    print('Saved', OutputFile)


def main(argv):
    HelpString = 'correctDistortionAndColor.py -i <image file> ' + \
                    '-f <root image folder> '+ \
                    '-o <output file>\n' + \
                 'Example:\n' + \
                 "$ ./correctDistortionAndColor.py -i /home/chuong/Data/GC03L-temp/corrected/IMG_6425.JPG -c /home/chuong/Data/GC03L-temp/corrected/CameraTrax_24ColorCard_2x3in.png -r \n" +\
                 "$ ./correctDistortionAndColor.py -f /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-orig/ -k /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/calib_param_700Dcam.yml -c /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/CameraTrax_24ColorCard_2x3in.png -r /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/ColorbarRectangle.yml -o /mnt/phenocam/a_data/TimeStreams/BorevitzTest/BVZ0018/BVZ0018-GC04L~fullres-corr/ -j 12"
    try:
        opts, args = getopt.getopt(argv,"hi:f:k:r:c:o:j:",\
            ["ifile=","ifolder=","calibfile=","colorrectfile=","colorcard=","ofolder=","jobs="])
    except getopt.GetoptError:
        print(HelpString)
        sys.exit(2)
    if len(opts) == 0:
        print(HelpString)
        sys.exit()

    ImageFile = ''
    InputRootFolder = ''
    OutputFolder = ''
    ColorCardImage = 'CameraTrax_24ColorCard_2x3in.png'
    ColorCardRectangle = 'ColorbarRectangle.yml'
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
        elif opt in ("-r", "--colorrectfile"):
            ColorCardRectangle = arg
        elif opt in ("-c", "--colorcard"):
            ColorCardImage = arg
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
            
    RectData = cv2yml.yml2dic(os.path.join(OutputFolder, ColorCardRectangle))
    RotationAngle = RectData['RotationAngle']
    Rect = RectData['Colorbar'].tolist()
    print('Rect =', Rect)
    Centre, Width, Height, Angle = getRectangleParamters(Rect)
    print(Centre, Width, Height, Angle)
    ColCardMapX, ColCardMapY = createMap(Centre, Width, Height, Angle)

    P24ColorCard = cv2.imread(ColorCardImage)[:,:,::-1] # read and convert to R-G-B image
    SquareSize = int(P24ColorCard.shape[0]/4)
    HalfSquareSize = int(SquareSize/2)
    
    # collect 24 colours from the captured color card:
    Colors = np.zeros([3,24])
    for i in range(24):
        Row = int(i/6)
        Col = i - Row*6
        rr = Row*SquareSize + HalfSquareSize
        cc = Col*SquareSize + HalfSquareSize
        Colors[0,i] = P24ColorCard[rr,cc,0]
        Colors[1,i] = P24ColorCard[rr,cc,1]
        Colors[2,i] = P24ColorCard[rr,cc,2]
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
        ArgList.append([ImageFile_, UndistMapX, UndistMapY, RotationAngle, ColCardMapX, ColCardMapY, Colors, OutputFile])
   
    Process = Pool(processes = NoJobs)
    Process.map(correctDistortionAndColor, ArgList)
          
if __name__ == "__main__":
   main(sys.argv[1:])


