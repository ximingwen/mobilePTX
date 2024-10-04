from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
import yolov4.core.utils as utils
from yolov4.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time
import torch
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

class YoloModel():
    def __init__(self, dataset):
        flags.DEFINE_string('framework', 'tflite', '(tf, tflite, trt')
        if dataset == 'pnb':
            flags.DEFINE_string('weights', 'yolov4/Pleural_Line_TensorFlow/pnb_prelim_yolo/yolov5-416.tflite',
                                'path to weights file')
        elif dataset == 'onsd':
            flags.DEFINE_string('weights', 'yolov4/Pleural_Line_TensorFlow/onsd_prelim_yolo/yolov5-416.tflite',
                                'path to weights file')
        else:
            flags.DEFINE_string('weights', 'yolov4/yolov4-416.tflite',
                                'path to weights file')
        flags.DEFINE_integer('size', 416, 'resize images to')
        flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
        flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
        flags.DEFINE_string('image', '/shared_data/YOLO_Updated_PL_Model_Results/Sliding/image_677741729740_clean/frame0.png', 'path to input image')
        flags.DEFINE_string('output', 'result.png', 'path to output image')
        flags.DEFINE_float('iou', 0.45, 'iou threshold')
        flags.DEFINE_float('score', 0.25, 'score threshold')

        FLAGS(['detect.py'])
    
        flags.score = 0.1

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.input_size = FLAGS.size
        image_path = FLAGS.image

#         start = time.time()
#         print('loading model\n')
#         self.saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
#         end = time.time()
#         elapsed_time = end - start
#         print('model loaded\n')
#         print('Took %.2f seconds to load model\n' % (elapsed_time))

    def get_bounding_boxes_v5(self, original_image):

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
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))][0]
            
            output = non_max_suppression(torch.tensor(pred))[0]
            boxes = output[:, :4].numpy()
            classes = output[:, 5].numpy()
            scores = output[:, 4].numpy()
            
            return boxes, classes, scores
            
#             if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
#                 boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([self.input_size, self.input_size]))
#             else:
                #boxes, pred_conf = filter_boxes(boxes, scores, score_threshold=0.25, input_shape=tf.constant([self.input_size, self.input_size]))
#         else:
#             infer = self.saved_model_loaded.signatures['serving_default']
#     #         print('batch data\n')
#             batch_data = tf.constant(images_data)
#     #         print('computing bounding box data\n')
#             yolo_start_time = time.time()
#             pred_bbox = infer(batch_data)
#             for key, value in pred_bbox.items():
#                 boxes = value[:, :, 0:4]
#                 pred_conf = value[:, :, 4:]
# #                 print("VALUE", value)
#             yolo_end_time = time.time()
#             yolo_elapsed_time = yolo_end_time - yolo_start_time
# #         print('Took %.2f seconds to run yolo\n' % (yolo_elapsed_time))
# #         print(boxes)

# #         print('non max suppression\n')
#         boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#             scores=tf.reshape(
#                 pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#             max_output_size_per_class=50,
#             max_total_size=50,
#             iou_threshold=0.5,
#             score_threshold=0.25
#         )

# #         print('formatting bounding box data\n')
#         boxes = boxes.numpy()

#         # remove bounding boxes with zero area
#         boxes = boxes.tolist()
#         boxes = boxes[0]
#         classes = classes[0]
#         scores = scores[0]
#         boxes_list = []
#         class_list = []
#         score_list = []
#         for box, class_idx, score in zip(boxes, classes, scores):
#             sum = 0
#             for value in box:
#                 sum += value
#             if sum > 0:
#                 boxes_list.append(box)
#                 class_list.append(class_idx)
#                 score_list.append(score)
#         boxes_list = [boxes_list]
#         class_list = [class_list]
#         score_list = [score_list]
#         boxes = np.array(boxes_list)
#         classes = np.array(class_list)
#         scores = np.array(score_list)

        end = time.time()
        elapsed_time = end - start
#         print('Took %.2f seconds to run whole bounding box function\n' % (elapsed_time))
        return None

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
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([self.input_size, self.input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([self.input_size, self.input_size]))
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
#         print(boxes)

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
        classes = classes[0]
        scores = scores[0]
        boxes_list = []
        class_list = []
        score_list = []
        for box, class_idx, score in zip(boxes, classes, scores):
            sum = 0
            for value in box:
                sum += value
            if sum > 0:
                boxes_list.append(box)
                class_list.append(class_idx)
                score_list.append(score)
        boxes_list = [boxes_list]
        class_list = [class_list]
        score_list = [score_list]
        boxes = np.array(boxes_list)
        classes = np.array(class_list)
        scores = np.array(score_list)

        end = time.time()
        elapsed_time = end - start
#         print('Took %.2f seconds to run whole bounding box function\n' % (elapsed_time))
        return boxes, classes, scores
