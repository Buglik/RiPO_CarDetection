import sys
import cv2 as cv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from MainWindow import Ui_MainWindow

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np


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

        # self.faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        print("Loading models ...")

        self.configs = config_util.get_configs_from_pipeline_file('Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config')
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)

        self.category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')

        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join('Tensorflow/workspace/models/my_ssd_mobnet', 'ckpt-6')).expect_partial()


    def startCameraButton_Clicked(self):
        if self.is_video_on:
            self.startVideoButton_Clicked()


        self.is_camera_on = ~self.is_camera_on
        if self.is_camera_on:
            self.camera = cv.VideoCapture(0)
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

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect(self, frame):
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.5,
            agnostic_mode=False)
        return image_np_with_detections

    def framePlayer(self):

        ret, frame = self.camera.read()

        if ret:

            # print(frame.dtype)
            frame = self.detect(frame)
            # print(frame.dtype)
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