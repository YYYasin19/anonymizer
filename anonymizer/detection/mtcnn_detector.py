import mtcnn
import PIL.Image as Image
import numpy as np

class MTCNNDetector():


    def __init__(self, kind='face'):
        self.detector = mtcnn.MTCNN(weights_file="anonymizer/anonymizer/detection/mtcnn_weights.npy")

    def detect(self, image, detection_threshold):
        np_image = np.array(image)
        boxes = self.detector.detect_faces(np_image)

        # transform boxes
        return boxes