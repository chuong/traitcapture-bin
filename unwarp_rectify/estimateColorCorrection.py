# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:17:53 2014

@author: chuong
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
import getopt, sys, os, datetime
import cv2
import cv2yml
import glob
from scipy import optimize

class RectangleBuilder:
    def __init__(self, Rectangle, AspectRatio, Image):
        self.Rectangle = Rectangle
        self.AspectRatio = AspectRatio
        self.Image = Image
        self.RectList = []
        self.cid = Rectangle.figure.canvas.mpl_connect('button_press_event', self.onMouseClicked)
        self.cid = Rectangle.figure.canvas.mpl_connect('key_press_event', self.onKeyPressed)
        self.lclick_x = []            
        self.lclick_y = []            

    def onKeyPressed(self, event):
        if event.key == 'escape':
            self.lclick_x = []
            self.lclick_y = []
            print('  Clear recent selection')
        elif event.key == 'up':
            self.lclick_y[-1] = self.lclick_y[-1] - 5
        elif event.key == 'down':
            self.lclick_y[-1] = self.lclick_y[-1] + 5
        elif event.key == 'right':
            self.lclick_x[-1] = self.lclick_x[-1] + 5
        elif event.key == 'left':
            self.lclick_x[-1] = self.lclick_x[-1] - 5
        self.drawLines()

    def onMouseClicked(self, event):
#        print('click', event)
        if event.inaxes!=self.Rectangle.axes: 
            return
        if event.button == 1:
            self.lclick_x.append(event.xdata)
            self.lclick_y.append(event.ydata)
        elif event.button == 3:
            self.rclick_x = event.xdata
            self.rclick_y = event.ydata
            # remove the last selection
            if len(self.lclick_x) > 0:
                self.lclick_x = self.lclick_x[:-1]
                self.lclick_y = self.lclick_y[:-1]
            elif len(self.RectList) > 0:
                self.RectList = self.RectList[:-1]

        if self.AspectRatio != None and len(self.lclick_x) == 2:
            Rect = self.getRectCornersFrom2Points(self.lclick_x, self.lclick_y)
            self.RectList.append(Rect)
            self.lclick_x = []            
            self.lclick_y = []            
        elif len(self.lclick_x) == 4:
            Rect = [[x,y] for x,y in zip(self.lclick_x, self.lclick_y)]
            Rect = self.correctPointOrder(Rect)
            self.RectList.append(Rect)
            self.lclick_x = []            
            self.lclick_y = []            
        self.drawLines()
        
    def findCorner(self, Corner, CornerType = 'topleft', WindowSize = 100, Threshold = 50):
        x, y = Corner
        HWindowSize = int(WindowSize/2)
        window = self.Image[y-HWindowSize:y+HWindowSize+1, x-HWindowSize:x+HWindowSize+1,:].astype(np.float)
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
        
    def drawLines(self):
        xs, ys = [], []
        for Rect in self.RectList:
            tl, bl, br, tr = Rect
            xs = xs + [tl[0], bl[0], br[0], tr[0], tl[0], np.nan]
            ys = ys + [tl[1], bl[1], br[1], tr[1], tl[1], np.nan]
        if len(self.lclick_x) > 1:
            xs = xs + [x for x in self.lclick_x]
            ys = ys + [y for y in self.lclick_y]
        self.Rectangle.set_data(xs, ys)
        self.Rectangle.figure.canvas.draw()
        
    def getRectCornersFrom2Points(self, lclick_x, lclick_y):
        Lenght = np.sqrt((lclick_x[0] - lclick_x[1])**2 + \
                         (lclick_y[0] - lclick_y[1])**2)
        Height = Lenght/np.sqrt(1+self.AspectRatio**2)
        Width = Height*self.AspectRatio
        Centre = np.asarray([lclick_x[0] + lclick_x[1], lclick_y[0] + lclick_y[1]])/2.0
        Angle = np.arctan2(Height, Width) - \
                np.arctan2(lclick_y[1] - lclick_y[0], lclick_x[1] - lclick_x[0])
        InitRect = self.createRectangle(Centre, Width, Height, Angle)
        CornerTypes = ['topleft', 'bottomleft', 'bottomright', 'topright']
        Rect = []
        for Corner, Type in zip(InitRect, CornerTypes):
            Corner = self.findCorner(Corner, Type)
            Rect.append(Corner)
        return Rect

    def correctPointOrder(self, Rect, tolerance = 40):
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

    def createRectangle(self, Centre, Width, Height, Angle):
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

def selectColorCard(Img, AspectRatio):                    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('left click at top-left corner and right click at bottom-right corner to to select pot area')
    ax.imshow(Img)
    Rectangle, = ax.plot([0], [0])  # empty line/Rectangle
    Rectangles = RectangleBuilder(Rectangle, AspectRatio, Img)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    plt.show()
    return Rectangles.RectList
    
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

def main(argv):
    HelpString = 'selectPots.py -i <image file> ' + \
                    '-p <pot config file> '+ \
                    '-o <output file>\n' + \
                 'Example:\n' + \
                 "$ ./estimateColorCorrection.py -i /home/chuong/Data/GC03L-temp/corrected/IMG_6425.JPG -c /home/chuong/Data/GC03L-temp/corrected/TrayConfig.yml\n" +\
                 "$ ./estimateColorCorrection.py -f /home/chuong/Data/GC03L-temp/corrected/ -p IMG*JPG -c /home/chuong/Data/GC03L-temp/corrected/TrayConfig.yml -o /home/chuong/Data/GC03L-temp/rectified/"
    try:
        opts, args = getopt.getopt(argv,"hi:f:p:c:g:a:o:",\
            ["ifile=","ifolder=","ipattern=","colorcard=","gamafile=","aspectratio=","ofolder="])
    except getopt.GetoptError:
        print(HelpString)
        sys.exit(2)
    if len(opts) == 0:
        print(HelpString)
        sys.exit()

    ImageFile = ''
    ImageFolder = ''
    ImageFilePattern = '*jpg'
    OutputFolder = ''
    TrayImgWidth = None
    TrayImgHeight = None
    AspectRatio = 300.0/200.0 # None 
    ColorCardFile = 'CameraTrax_24ColorCard_2x3in.png'
    GamaFile = 'ColorGama.yml'
    for opt, arg in opts:
        if opt == '-h':
            print(HelpString)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            ImageFile = arg
        elif opt in ("-f", "--ifolder"):
            ImageFolder = arg
        elif opt in ("-p", "--ipattern"):
            ImageFilePattern = arg
        elif opt in ("-c", "--colorcard"):
            ColorCardFile = arg
        elif opt in ("-g", "--gamafile"):
            GamaFile = arg
        elif opt in ("-a", "--aspectratio"):
            AspectRatio = float(arg)
        elif opt in ("-o", "--ofolder"):
            OutputFolder = arg

    P24ColorCard = cv2.imread(ColorCardFile)[:,:,::-1]
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
    
    Image = cv2.imread(ImageFile)[:,:,::-1]
    RectList = selectColorCard(Image, AspectRatio)
    for Rect in RectList:
        print('Rect = \n', Rect)
        Centre, Width, Height, Angle = getRectangleParamters(Rect)
        MapX, MapY = createMap(Centre, Width, Height, Angle)
        RectifiedColorCard = cv2.remap(Image, MapX, MapY, cv2.INTER_CUBIC)
        
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
#            if Captured_R < 254 and Captured_G < 254 and Captured_B < 254:
#                # only accepts unsaturated colors
#                Captured_Colors.append(np.asarray([Captured_R, Captured_G, Captured_B], dtype = np.float))
        print('Captured_Colors = \n', Captured_Colors)

        # initial values
        ColorMatrix = np.eye(3)
        ColorConstant = np.zeros([3,1])
        ColorGamma = np.ones([3,1])
        print('ColorMatrix = \n', ColorMatrix)
        print('ColorConstant = \n', ColorConstant)
        print('ColorGamma = \n', ColorGamma)
        Arg = np.zeros([9 + 3 + 3])
        Arg[:9] = ColorMatrix.reshape([9])
        Arg[9:12] = ColorConstant.reshape([3])
        Arg[12:15] = ColorGamma.reshape([3])
        
        ArgRefined, _ = optimize.leastsq(getColorMatchingErrorVectorised, Arg, args=(Colors, Captured_Colors), maxfev=10000)
        
        ColorMatrix = ArgRefined[:9].reshape([3,3])
        ColorConstant = ArgRefined[9:12].reshape([3,1])
        ColorGamma = ArgRefined[12:15]
        print('ColorMatrix = \n', ColorMatrix)
        print('ColorConstant = \n', ColorConstant)
        print('ColorGamma = \n', ColorGamma)
        
        ColorCardCorrected = correctColorVectorised(RectifiedColorCard, ColorMatrix, ColorConstant, ColorGamma)
        ImageCorrected = correctColorVectorised(Image.astype(np.float), ColorMatrix, ColorConstant, ColorGamma)
        
        plt.figure()
        plt.imshow(RectifiedColorCard/255)
        plt.title('Captured CameraTrax 24-color card')
        plt.figure()
        plt.imshow(ColorCardCorrected/255)
        plt.title('Corrected CameraTrax 24-color card')
        
        plt.figure()
        plt.imshow(Image)
        plt.title('Captured Chamber Image')
        plt.figure()
        plt.imshow(ImageCorrected/255)
        plt.title('Corrected Chamber Image')
        
        cv2.imwrite('CapturedImage.jpg', np.uint8(Image[:,:,2::-1]))
        cv2.imwrite('CorrectedImage.jpg', np.uint8(ImageCorrected[:,:,2::-1]))
        
    plt.figure()
    plt.imshow(P24ColorCard/255)
    plt.title('Original CameraTrax 24-color card')
    plt.show()

        
if __name__ == "__main__":
   main(sys.argv[1:])


