from cv2.cv2 import imread, CascadeClassifier

from anonymizer.anonymizer.utils import Box


class OpenCVDetector():

    def __init__(self, kind='face',*args, **kwargs):
        self.kind = kind
        self.classifier = CascadeClassifier('anonymizer/anonymizer/detection/haarcascade_frontalface_default.xml')
        return

    def detect(self, image, *args, **kwargs):
        # detect boxes
        boxes = self.classifier.detectMultiScale(image)

        return self._convert_boxes(boxes)


    def _convert_boxes(self, boxes):
        box_objects = []
        for box in boxes:
            x,y,x_len, y_len = box
            box_objects.append(Box(
                x_min=x, x_max=x+x_len,
                y_min=y, y_max=y+y_len,
                kind=self.kind, score=1.0
            ))

        return box_objects