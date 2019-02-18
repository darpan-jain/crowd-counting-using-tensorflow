import numpy as np
import tensorflow as tf
import cv2
import time
import glob
from progressbar import *

widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
pbar = ProgressBar(widgets=widgets, maxval=10000000)

class People_Counter:

    def __init__(self, path):
        self.path = path
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0') # Defining tensors for the graph
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0') # Each box denotes part of image with a person detected 
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0') # Score represents the confidence for the detected person
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded}) # Using the model for detection

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = './data/utils/my_model.pb'
    peop_counter = People_Counter(path=model_path)
    threshold = 0.4
    no=1
    for n in pbar(glob.glob("./data/images/test/*.jpg")):
        count=0
        img = cv2.imread(n)
        img = cv2.resize(img, (640, 480))

        boxes, scores, classes, num = peop_counter.detect(img)

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                count+=1
        cv2.putText(img,'Count = '+str(count),(10,400),cv2.FONT_HERSHEY_SIMPLEX, 1.25,(255,255,0),2,cv2.LINE_AA)
        cv2.imwrite("./results/result%04i.jpg" %no, img)
        no+=1
print("\n\t\t\tSuccessfully saved all results!\n")
