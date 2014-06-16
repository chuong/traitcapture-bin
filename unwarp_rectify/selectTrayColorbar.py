"""
Created on Thu Dec 12 08:38:21 2013
 
@author: Sukhbinder Singh
 
Simple QTpy and MatplotLib example with Zoom/Pan
 
Built on the example provided at
How to embed matplotib in pyqt - for Dummies
 
http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
 
"""
import sys
from PyQt4 import QtGui
 
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt
 
import cv2
import numpy as np
import utils
 
class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
 
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
 
         
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()
 
        # Just some button 
        self.colorcardRadioButton = QtGui.QRadioButton('Select color card')
        self.colorcardRadioButton.setChecked(False)
        
        self.trayRadioButton = QtGui.QRadioButton('Select tray')
        self.trayRadioButton.setChecked(False)

        self.loadImageButton = QtGui.QPushButton('Load image')
        self.loadImageButton.clicked.connect(self.loadImage)
 
        self.rotateImageButton = QtGui.QPushButton('Rotate 90-deg')
        self.rotateImageButton.clicked.connect(self.rotateImage90Degrees)
 
        self.loadCamCalibButton = QtGui.QPushButton('Load cam. param.')
        self.loadCamCalibButton.clicked.connect(self.loadCamCalib)
 
        self.saveButton = QtGui.QPushButton('Save selections')
        self.saveButton.clicked.connect(self.save)
 
        self.zoomButton = QtGui.QPushButton('Zoom')
        self.zoomButton.setCheckable(True)
        self.zoomButton.clicked.connect(self.zoom)
         
        self.panButton = QtGui.QPushButton('Pan')
        self.panButton.setCheckable(True)
        self.panButton.clicked.connect(self.pan)
         
        self.homeButton = QtGui.QPushButton('Home')
        self.homeButton.clicked.connect(self.home)
        
        self.status = QtGui.QTextEdit('')
 
        # set the layout
        layout = QtGui.QHBoxLayout()
        rightWidget = QtGui.QWidget()
        buttonlayout = QtGui.QVBoxLayout(rightWidget)
        buttonlayout.addWidget(self.loadImageButton)
        buttonlayout.addWidget(self.rotateImageButton)
        buttonlayout.addWidget(self.loadCamCalibButton)
        buttonlayout.addWidget(self.colorcardRadioButton)
        buttonlayout.addWidget(self.trayRadioButton)
        buttonlayout.addWidget(self.zoomButton)
        buttonlayout.addWidget(self.panButton)
        buttonlayout.addWidget(self.homeButton)
        buttonlayout.addWidget(self.saveButton)
        buttonlayout.addWidget(self.status)
#        buttonlayout.addStretch(1)
        rightWidget.setFixedWidth(150)
        leftLayout = QtGui.QVBoxLayout()
        leftLayout.addWidget(self.toolbar)
        leftLayout.addWidget(self.canvas)

        layout.addWidget(rightWidget)
        layout.addLayout(leftLayout)
        self.setLayout(layout)
 
        self.group = QtGui.QButtonGroup()
        self.group.addButton(self.colorcardRadioButton)
        self.group.addButton(self.trayRadioButton) 
        
        self.panMode = False
        self.zoomMode = False
        
        self.ax = None
        self.plotRect = None
        self.image = None
        self.isDistortionCorrected = False
        self.UndistMapX = None
        self.UndistMapY = None
        self.trayAspectRatio = 0.835
        self.colorcardAspectRatio = 1.5
        self.leftClicks = []
        self.colorcardList = []
        self.trayList = []
        
        
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
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open image', '/home/chuong/Data/phenocam/a_data/TimeStreams/Borevitz/BVZ0033/BVZ0033-GC02L~fullres-orig/2014/2014_05/2014_05_27/2014_05_27_11/')
        self.status.setText('Loading image...')
        app.processEvents()
        self.image = cv2.imread(str(fname))[:,:,::-1]

        # Undistort image if mapping available
        if not self.isDistortionCorrected and self.UndistMapX != None and self.UndistMapY != None:
            self.image = cv2.remap(self.image, self.UndistMapX, self.UndistMapY, cv2.INTER_CUBIC)
            self.isDistortionCorrected = True

        self.status.setText('Loading done.')
        if self.image != None:
            if self.ax == None:
                self.ax = self.figure.add_subplot(111)
                self.ax.figure.canvas.mpl_connect('button_press_event', self.onMouseClicked)
            self.ax.hold(False)
            self.ax.imshow(self.image)
            self.figure.tight_layout()
            self.canvas.draw()
        
    def updateFigure(self):
        self.status.setText('Update figure...') 
        xs, ys = [], []
        for Rect in self.trayList:
            tl, bl, br, tr = Rect
            xs = xs + [tl[0], bl[0], br[0], tr[0], tl[0], np.nan]
            ys = ys + [tl[1], bl[1], br[1], tr[1], tl[1], np.nan]
        for Rect in self.colorcardList:
            tl, bl, br, tr = Rect
            xs = xs + [tl[0], bl[0], br[0], tr[0], tl[0], np.nan]
            ys = ys + [tl[1], bl[1], br[1], tr[1], tl[1], np.nan]
        for x,y in self.leftClicks:
            xs = xs + [x]
            ys = ys + [y]
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
           
#        self.status.setText('Plot done.') 
        app.processEvents()

         
    def loadCamCalib(self):
        ''' load camera calibration image and show an image'''
        CalibFile = QtGui.QFileDialog.getOpenFileName(self, 'Open image', '/home/chuong/Workspace/traitcapture-bin/unwarp_rectify/data')
        ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs = utils.readCalibration(CalibFile)
        print('CameraMatrix =', CameraMatrix)
        print('DistCoefs =', DistCoefs)
        self.UndistMapX, self.UndistMapY = cv2.initUndistortRectifyMap(CameraMatrix, DistCoefs, \
            None, CameraMatrix, ImageSize, cv2.CV_32FC1)    

        if self.image != None:
            self.image = cv2.remap(self.image, self.UndistMapX, self.UndistMapY, cv2.INTER_CUBIC)
            self.isDistortionCorrected = True
            self.ax.imshow(self.image)
            self.canvas.draw()
        
    def save(self):
        ''' save selection'''
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save selection', '/home/chuong/Data/phenocam/a_data/TimeStreams/Borevitz/BVZ0033/BVZ0033-GC02L~fullres-orig/2014/2014_05/2014_05_27/2014_05_27_11/')
        
    def rotateImage90Degrees(self):
        self.image = np.rot90(self.image) #.astype(uint8)
        self.ax.imshow(self.image)
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
                
            if len(self.leftClicks) == 2 and AspectRatio != None:
                Rect = utils.getRectCornersFrom2Points(self.image, self.leftClicks, AspectRatio)
                self.leftClicks = []            
            elif len(self.leftClicks) == 4:
                Rect = [[x,y] for x,y in self.leftClicks]
                Rect = utils.correctPointOrder(Rect)
                self.leftClicks = []            
    
            if len(Rect) > 0:
                if self.trayRadioButton.isChecked():
                    self.trayList.append(Rect)
                else:
                    self.colorcardList.append(Rect)
            self.updateFigure()
        elif event.button == 3:
            # remove the last selection
            if len(self.leftClicks) > 0:
                self.leftClicks = self.leftClicks[:-1]
            else:
                if self.trayRadioButton.isChecked() and len(self.trayList) > 0:
                    self.trayList = self.trayList[:-1]
                elif self.colorcardRadioButton.isChecked() and len(self.colorcardList) > 0:
                    self.colorcardList = self.colorcardList[:-1]
            self.updateFigure()
        else:
            print('Ignored click')
               
                        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
 
    main = Window()
    main.setWindowTitle('Select Color Card and Trays')
    main.show()
 
    sys.exit(app.exec_())