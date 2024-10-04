import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
from absl import app, flags, logging
from absl.flags import FLAGS
import yolov4.core.utils as utils
from yolov4.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time

class YoloModel():
    def __init__(self):
        flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
        flags.DEFINE_string('weights', 'yolov4/Pleural_Line_TensorFlow',
                            'path to weights file')
        flags.DEFINE_integer('size', 416, 'resize images to')
        flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
        flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
        flags.DEFINE_string('image', '/shared_data/YOLO_Updated_PL_Model_Results/Sliding/image_677741729740_clean/frame0.png', 'path to input image')
        flags.DEFINE_string('output', 'result.png', 'path to output image')
        flags.DEFINE_float('iou', 0.45, 'iou threshold')
        flags.DEFINE_float('score', 0.25, 'score threshold')

        FLAGS(['detect.py'])

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.input_size = FLAGS.size
        image_path = FLAGS.image

        start = time.time()
        print('loading model\n')
        self.saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        end = time.time()
        elapsed_time = end - start
        print('model loaded\n')
        print('Took %.2f seconds to load model\n' % (elapsed_time))

    def get_bounding_boxes(self, original_image):

        start = time.time()
#         print('image resizing\n')
        image_data = cv2.resize(original_image, (self.input_size, self.input_size))
        image_data = image_data / 255.
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        
#         print('running as tensorflow\n')

        #print('loading model\n')
#         print(FLAGS.weights)
#         saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        #print('model loaded\n')

        if FLAGS.framework == 'tflite':
#         print('running as tflite\n')
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = self.saved_model_loaded.signatures['serving_default']
    #         print('batch data\n')
            batch_data = tf.constant(images_data)
    #         print('computing bounding box data\n')
            yolo_start_time = time.time()
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
#                 print("VALUE", value)
            yolo_end_time = time.time()
            yolo_elapsed_time = yolo_end_time - yolo_start_time
#         print('Took %.2f seconds to run yolo\n' % (yolo_elapsed_time))

#         print('non max suppression\n')
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.5,
            score_threshold=0.25
        )

#         print('formatting bounding box data\n')
        boxes = boxes.numpy()

        # remove bounding boxes with zero area
        boxes = boxes.tolist()
        boxes = boxes[0]
        boxes_list = []
        for box in boxes:
            sum = 0
            for value in box:
                sum += value
            if sum > 0:
                boxes_list.append(box)
        boxes_list = [boxes_list]
        boxes = np.array(boxes_list)

        end = time.time()
        elapsed_time = end - start
#         print('Took %.2f seconds to run whole bounding box function\n' % (elapsed_time))
        return boxes
