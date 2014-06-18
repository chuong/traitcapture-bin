"""
Created on Thu Dec 12 08:38:21 2013
 
@author: Sukhbinder Singh
 
Simple QTpy and MatplotLib example with Zoom/Pan
 
Built on the example provided at
How to embed matplotib in pyqt - for Dummies
 
http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
 
"""
import sys
from PyQt4 import QtGui, QtCore
 
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
 
import cv2
import numpy as np
import utils
import cv2yml
 
class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
 
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
         
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()
 
        # Just some button 
        self.colorcardRadioButton = QtGui.QRadioButton('Select color car&d')
        self.colorcardRadioButton.setChecked(False)
        self.colorcardRadioButton.clicked.connect(self.selectWhat)
        
        self.trayRadioButton = QtGui.QRadioButton('Select &tray')
        self.trayRadioButton.setChecked(False)
        self.trayRadioButton.clicked.connect(self.selectWhat)

        self.potRadioButton = QtGui.QRadioButton('Select &pot')
        self.potRadioButton.setChecked(False)
        self.potRadioButton.clicked.connect(self.selectWhat)

        self.loadImageButton = QtGui.QPushButton('&Load image')
        self.loadImageButton.clicked.connect(self.loadImage)
 
        self.rotateImageButton = QtGui.QPushButton('&Rotate 90-deg')
        self.rotateImageButton.clicked.connect(self.rotateImage90Degrees)
 
        self.loadCamCalibButton = QtGui.QPushButton('Load &cam. param.')
        self.loadCamCalibButton.clicked.connect(self.loadCamCalib)
 
        self.saveGeometriesButton = QtGui.QPushButton('&Save selected geometries')
        self.saveGeometriesButton.clicked.connect(self.saveSelectedGeometries)
 
        self.saveTraysButton = QtGui.QPushButton('&Save selected tray images')
        self.saveTraysButton.clicked.connect(self.saveSelectedTrayImages)
 
        self.saveColorcadButton = QtGui.QPushButton('&Save sel. col. card images')
        self.saveColorcadButton.clicked.connect(self.saveSelectedColorcardImages)
 
        self.zoomButton = QtGui.QPushButton('&Zoom')
        self.zoomButton.setCheckable(True)
        self.zoomButton.clicked.connect(self.zoom)
         
        self.panButton = QtGui.QPushButton('&Pan')
        self.panButton.setCheckable(True)
        self.panButton.clicked.connect(self.pan)
         
        self.homeButton = QtGui.QPushButton('&Home')
        self.homeButton.clicked.connect(self.home)
        
        self.status = QtGui.QTextEdit('')
        self.status.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.mousePosition = QtGui.QLabel('')
 
        # set the layout
        layout = QtGui.QHBoxLayout()
        rightWidget = QtGui.QWidget()
        buttonlayout = QtGui.QVBoxLayout(rightWidget)
        buttonlayout.addWidget(self.loadImageButton)
        buttonlayout.addWidget(self.rotateImageButton)
        buttonlayout.addWidget(self.loadCamCalibButton)
        buttonlayout.addWidget(self.colorcardRadioButton)
        buttonlayout.addWidget(self.trayRadioButton)
        buttonlayout.addWidget(self.potRadioButton)
        buttonlayout.addWidget(self.zoomButton)
        buttonlayout.addWidget(self.panButton)
        buttonlayout.addWidget(self.homeButton)
        buttonlayout.addWidget(self.saveGeometriesButton)
        buttonlayout.addWidget(self.saveColorcadButton)
        buttonlayout.addWidget(self.saveTraysButton)
        buttonlayout.addWidget(self.status)
        buttonlayout.addWidget(self.mousePosition)
        rightWidget.setMaximumWidth(200)
        leftLayout = QtGui.QVBoxLayout()
        leftLayout.addWidget(self.toolbar)
        leftLayout.addWidget(self.canvas)

        layout.addWidget(rightWidget)
        layout.addLayout(leftLayout)
        self.setLayout(layout)
 
        self.group = QtGui.QButtonGroup()
        self.group.addButton(self.colorcardRadioButton)
        self.group.addButton(self.trayRadioButton) 
        self.group.addButton(self.potRadioButton) 
        
        self.panMode = False
        self.zoomMode = False
        
        self.ax = None
        self.plotRect = None
        self.plotImg = None
        self.image = None
        self.UndistMapX = None
        self.UndistMapY = None
        self.trayAspectRatio = 0.835
        self.colorcardAspectRatio = 1.5
        self.potAspectRatio = 1.0
        self.leftClicks = []
        
#        # change cursor shape
#        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor ))
        
        # Ouput parameters
        self.colorcardList = []
        self.trayList = []
        self.rotationAngle = 0
        self.isDistortionCorrected = False

    def selectWhat(self):
        if self.trayRadioButton.isChecked():
            self.status.append('Start selecting tray.')
        elif self.colorcardRadioButton.isChecked():
            self.status.append('Start selecting color bar.')
        else:
            self.status.append('Start selecting pot.')
        
    def home(self):
        self.toolbar.home()
    def zoom(self):
        self.toolbar.zoom()
        if not self.zoomMode:
            self.zoomMode = True
            self.panMode = False
            self.panButton.setChecked(False)
        else:
            self.zoomMode = False
    def pan(self):
        self.toolbar.pan()
        if not self.panMode:
            self.panMode = True
            self.zoomMode = False
            self.zoomButton.setChecked(False)
        else:
            self.panMode = False
         
    def loadImage(self):
        ''' load and show an image'''
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open image', '/home/chuong/Data/phenocam/a_data/TimeStreams/Borevitz/BVZ0033/BVZ0033-GC02L~fullres-orig/2014/2014_05/2014_05_23/2014_05_23_08')
        self.status.append('Loading image...')
        app.processEvents()
        self.image = cv2.imread(str(fname))[:,:,::-1]
        self.status.append('Loaded image from ' + str(fname))
        
        # reset all outputs
        self.colorcardList = []
        self.trayList = []
        self.potList = []
        self.rotationAngle = 0
        self.isDistortionCorrected = False

        # Undistort image if mapping available
        if not self.isDistortionCorrected and self.UndistMapX != None and self.UndistMapY != None:
            self.image = cv2.remap(self.image, self.UndistMapX, self.UndistMapY, cv2.INTER_CUBIC)
            self.isDistortionCorrected = True

        if self.image != None:
            if self.ax == None:
                self.ax = self.figure.add_subplot(111)
                self.ax.figure.canvas.mpl_connect('button_press_event', self.onMouseClicked)
                self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMoves)
                self.ax.figure.canvas.mpl_connect('figure_enter_event', self.changeCursor)
            self.ax.hold(False)
            if self.plotImg == None:
                self.plotImg = self.ax.imshow(self.image)
            else:
                self.plotImg.set_data(self.image)
            self.figure.tight_layout()
            self.canvas.draw()
        
    def changeCursor(self, event):
#        cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor ))
        
    def updateFigure(self):
        xs, ys = [], []
        for Rect in self.colorcardList:
            tl, bl, br, tr = Rect
            xs = xs + [tl[0], bl[0], br[0], tr[0], tl[0], np.nan]
            ys = ys + [tl[1], bl[1], br[1], tr[1], tl[1], np.nan]
        for Rect in self.trayList:
            tl, bl, br, tr = Rect
            xs = xs + [tl[0], bl[0], br[0], tr[0], tl[0], np.nan]
            ys = ys + [tl[1], bl[1], br[1], tr[1], tl[1], np.nan]
        for Rect in self.potList:
            tl, bl, br, tr = Rect
            xs = xs + [tl[0], bl[0], br[0], tr[0], tl[0], np.nan]
            ys = ys + [tl[1], bl[1], br[1], tr[1], tl[1], np.nan]
        for x,y in self.leftClicks:
            xs = xs + [x]
            ys = ys + [y]
#        if self.crosshair != None:
#            xs = xs + [np.nan, 0, self.image.shape[1], np.nan, self.crosshair[0], self.crosshair[0], np.nan]
#            ys = ys + [np.nan, self.crosshair[1], self.crosshair[1], np.nan, 0, self.image.shape[0], np.nan]
        if len(xs) > 0 and len(ys) > 0:
            if self.plotRect == None:
                self.ax.hold(True)
                self.plotRect, = self.ax.plot(xs, ys, 'b')
                self.ax.hold(False)
                self.ax.set_xlim([0,self.image.shape[1]])
                self.ax.set_ylim([0,self.image.shape[0]])
                self.ax.invert_yaxis()
            else:
                self.plotRect.set_data(xs, ys)
        self.canvas.draw()
           
        app.processEvents()

         
    def loadCamCalib(self):
        ''' load camera calibration image and show an image'''
        CalibFile = QtGui.QFileDialog.getOpenFileName(self, 'Open image', '/home/chuong/Workspace/traitcapture-bin/unwarp_rectify/data')
        ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs = utils.readCalibration(CalibFile)
        self.status.append('Loaded camera parameters from ' + CalibFile) 
        print('CameraMatrix =', CameraMatrix)
        print('DistCoefs =', DistCoefs)
        self.UndistMapX, self.UndistMapY = cv2.initUndistortRectifyMap(CameraMatrix, DistCoefs, \
            None, CameraMatrix, ImageSize, cv2.CV_32FC1)    

        if self.image != None:
            self.image = cv2.remap(self.image, self.UndistMapX, self.UndistMapY, cv2.INTER_CUBIC)
            self.isDistortionCorrected = True
            self.status.append('Corrected image distortion.') 
            if self.plotImg == None:
                self.plotImg = self.ax.imshow(self.image)
            else:
                self.plotImg.set_data(self.image)
            self.canvas.draw()
        
    def saveSelectedGeometries(self):
        ''' save selected geometries'''
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save selected geometries', '/home/chuong/Workspace/traitcapture-bin/unwarp_rectify/data')
        colorcardList2 = []
        for colorcard in self.colorcardList:
            colorcardList2 = colorcardList2 + colorcard
        trayList2 = []
        for tray in self.trayList:
            trayList2 = trayList2 + tray
        potList2 = []
        for pot in self.potList:
            potList2 = potList2 + pot
        dicdata = {'colorcardself.crosshair = NoneList':np.asarray(colorcardList2), \
                   'trayList':np.asarray(trayList2), \
                   'potList':np.asarray(potList2), \
                   'rotationAngle':self.rotationAngle, \
                   'distortionCorrected':int(self.isDistortionCorrected)}
        cv2yml.dic2yml(fname, dicdata)
        self.status.append('Saved selected geometries to ' + fname)
        
    def saveSelectedTrayImages(self):
        ''' save selected trays'''
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save selected tray images', '/home/chuong/Workspace/traitcapture-bin/unwarp_rectify/data')
        medianWidth, medianHeight = utils.getMedianRectSize(self.trayList)
        rectifiedTrayImages = utils.rectifyRectImages(self.image, self.trayList, MedianSize = [medianWidth, medianHeight])
        for i,rectifiedImage in enumerate(rectifiedTrayImages):
            cv2.imwrite(str(fname) %i, rectifiedImage)

    def saveSelectedColorcardImages(self):
        ''' save selected trays'''
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save selected color card images', '/home/chuong/Workspace/traitcapture-bin/unwarp_rectify/data')
        medianWidth, medianHeight = utils.getMedianRectSize(self.colorcardList)
        rectifiedColorcardImages = utils.rectifyRectImages(self.image, self.colorcardList, MedianSize = [medianWidth, medianHeight])
        for i,rectifiedImage in enumerate(rectifiedColorcardImages):
            cv2.imwrite(str(fname) %i, rectifiedImage)

    def rotateImage90Degrees(self):
        if self.image == None:
            self.status.append('No image to rotate.')
            return
        self.rotationAngle = self.rotationAngle + 90
        if self.rotationAngle >= 360:
            self.rotationAngle = self.rotationAngle - 360
        self.image = np.rot90(self.image) #.astype(uint8)
        self.status.append('Rot. angle = %d deg' %self.rotationAngle) 
        if self.plotImg == None:
            self.plotImg = self.ax.imshow(self.image)
        else:
            self.plotImg.set_data(self.image)
        self.canvas.draw()

    def onMouseClicked(self, event):
        if self.panMode or self.zoomMode:
            return
        print('click', event.button, event.xdata, event.ydata)
        
        if event.button == 1 and event.xdata != None and event.ydata != None:
            self.leftClicks.append([event.xdata, event.ydata])
            print('self.leftClicks =', self.leftClicks)
            Rect = []
            AspectRatio = None
            if self.trayRadioButton.isChecked():
                AspectRatio = self.trayAspectRatio
            elif self.colorcardRadioButton.isChecked():
                AspectRatio = self.colorcardAspectRatio
            elif self.potRadioButton.isChecked():
                AspectRatio = self.potAspectRatio
                
            if len(self.leftClicks) == 2 and AspectRatio != None:
                if self.potRadioButton.isChecked():
                    Rect = utils.getRectCornersFrom2Points(self.image, self.leftClicks, AspectRatio, Rounded = True)
                else:
                    Rect = utils.getRectCornersFrom2Points(self.image, self.leftClicks, AspectRatio)
                self.leftClicks = []            
            elif len(self.leftClicks) == 4:
                Rect = [[x,y] for x,y in self.leftClicks]
                Rect = utils.correctPointOrder(Rect)
                self.leftClicks = []            
    
            if len(Rect) > 0:
                if self.trayRadioButton.isChecked():
                    self.trayList.append(Rect)
                    self.status.append('Added tray selection.')
                elif self.colorcardRadioButton.isChecked():
                    self.colorcardList.append(Rect)
                    self.status.append('Added color card selection.')
                else:
                    self.potList.append(Rect)
                    self.status.append('Added pot selection.')
            self.updateFigure()
        elif event.button == 3:
            # remove the last selection
            if len(self.leftClicks) > 0:
                self.leftClicks = self.leftClicks[:-1]
                self.status.append('Removed a previous click')
            else:
                if self.trayRadioButton.isChecked() and len(self.trayList) > 0:
                    self.trayList = self.trayList[:-1]
                    self.status.append('Removed a previous tray selection')
                elif self.colorcardRadioButton.isChecked() and len(self.colorcardList) > 0:
                    self.colorcardList = self.colorcardList[:-1]
                    self.status.append('Removed a previous color card selection.')
                elif self.potRadioButton.isChecked() and len(self.potList) > 0:
                    self.potList = self.potList[:-1]
                    self.status.append('Removed a previous pot selection')
            self.updateFigure()
        else:
            print('Ignored click')

    def onMouseMoves(self, event):
        if event.inaxes == self.ax:
            self.mousePosition.setText('x=%d, y=%d' %(event.xdata, event.ydata))
#            self.crosshair = [event.xdata, event.ydata]
        else:
            self.mousePosition.setText('')
#            self.crosshair = None
#        self.updateFigure()
        
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
    
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtGui.QMessageBox.question(self, 'Message', 
                         quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
    
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
 
    main = Window()
    main.setWindowTitle('Select Color Card, Trays, and Pot')
    main.show()
 
    sys.exit(app.exec_())