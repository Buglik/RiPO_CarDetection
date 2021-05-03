# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(0, 135, 800, 465))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("starting page.png"))
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 125, 800, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.detectionCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.detectionCheckBox.setGeometry(QtCore.QRect(700, 20, 85, 25))
        self.detectionCheckBox.setObjectName("detectionCheckBox")
        self.startCameraButton = QtWidgets.QPushButton(self.centralwidget)
        self.startCameraButton.setGeometry(QtCore.QRect(20, 20, 100, 25))
        self.startCameraButton.setObjectName("startCameraButton")
        self.startPhotoButton = QtWidgets.QPushButton(self.centralwidget)
        self.startPhotoButton.setGeometry(QtCore.QRect(20, 100, 100, 25))
        self.startPhotoButton.setObjectName("startPhotoButton")
        self.startVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.startVideoButton.setGeometry(QtCore.QRect(20, 60, 100, 25))
        self.startVideoButton.setDefault(False)
        self.startVideoButton.setObjectName("startVideoButton")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.startPhotoButton.clicked.connect(MainWindow.startPhotoButton_Clicked)
        self.startVideoButton.clicked.connect(MainWindow.startVideoButton_Clicked)
        self.startCameraButton.clicked.connect(MainWindow.startCameraButton_Clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RiPO J.Pawleniak 248897"))
        self.detectionCheckBox.setText(_translate("MainWindow", "Detection"))
        self.startCameraButton.setText(_translate("MainWindow", "Start camera"))
        self.startPhotoButton.setText(_translate("MainWindow", "Choose photo"))
        self.startVideoButton.setText(_translate("MainWindow", "Choose video"))