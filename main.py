import sys
import cv2 as cv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from MainWindow import Ui_MainWindow


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = cv.VideoCapture(0)
        self.is_camera_on = False
        self.is_video_on = False

        # Timer: 30ms capture a frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.framePlayer)
        self.timer.setInterval(30)

        self.faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


    def startCameraButton_Clicked(self):
        if self.is_video_on:
            self.startVideoButton_Clicked()


        self.is_camera_on = ~self.is_camera_on
        if self.is_camera_on:
            self.camera = cv.VideoCapture(1)
            self.startCameraButton.setText("Stop")
            self.timer.start()
        else:
            self.image.setPixmap(QtGui.QPixmap("starting page.png"))
            self.startCameraButton.setText("Start cam")
            self.timer.stop()
            self.camera.release()

    def startPhotoButton_Clicked(self):
        if self.is_camera_on:
            self.startCameraButton_Clicked()
        elif self.is_video_on:
            self.startVideoButton_Clicked()
            # Open the file selection dialog
        filename,  _ = QFileDialog.getOpenFileName(self, 'Open picture')
        if filename:
            img = cv.imread(str(filename))

            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(imgGray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            rows, cols, channels = img.shape
            bytesPerLine = channels * cols
            QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.image.setPixmap(QPixmap.fromImage(QImg))

    def startVideoButton_Clicked(self):
        if self.is_camera_on:
            self.startCameraButton_Clicked()

        self.is_video_on = ~self.is_video_on
        if self.is_video_on:
            # Open the file selection
            filename, _ = QFileDialog.getOpenFileName(self, 'Open video')
            print(filename)
            # filename = 'test.mp4'
            if filename:
                self.camera = cv.VideoCapture(filename)
                self.startVideoButton.setText("Stop")
                self.timer.start()
        else:
            self.timer.stop()
            self.camera.release()
            self.image.setPixmap(QtGui.QPixmap("starting page.png"))
            self.startVideoButton.setText("Choose video")

    def framePlayer(self):

        ret, frame = self.camera.read()

        if ret:

            imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(imgGray,1.1,4)

            for(x,y,w,h) in faces:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


            img_rows, img_cols, channels = frame.shape
            bytesPerLine = channels * img_cols

            cv.cvtColor(frame, cv.COLOR_BGR2RGB, frame)
            QImg = QImage(frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
            self.image.setPixmap(QPixmap.fromImage(QImg))
        else:
            if self.is_video_on:
                self.startVideoButton_Clicked()
            elif self.is_camera_on:
                self.startCameraButton_Clicked()





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())