import tensorflow as tf
import numpy as np
import cv2


def non_max_suppression(inputs, model_size, 
                            max_output_size, max_output_size_per_class, 
                            iou_threshold, confidence_treshold):

    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox /= model_size[0]

    scores = confs * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_treshold
    )

    return boxes, scores, classes, valid_detections


def resize_image(inputs, model_size):
    inputs = tf.image.resize(inputs, model_size)
    return inputs


def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def output_boxes(inputs, model_size, max_output_size, 
                        max_output_size_per_class, 
                        iou_threshold, confidence_threshold):

    center_x, center_y, width, height, confidence, classes = (
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    )

    top_left_x = center_x - width/2.
    top_left_y = center_y - height/2.
    bottom_right_x = center_x + width/2.
    bottom_right_y = center_y + height/2.

    inputs = tf.concat(
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis=-1
    )

    boxes_dicts = non_max_suppression(
        inputs, model_size, max_output_size, 
        max_output_size_per_class, iou_threshold, 
        confidence_threshold
    )

    return boxes_dicts


def draw_outputs(img, boxes, objectness, classes, 
                    nums, class_names):

    boxes, objectness, classes, nums = (
        boxes[0], objectness[0], classes[0], nums[0]
        )
    boxes = np.array(boxes)
    
    for i in range(nums):
        x1y1 = tuple(
            (boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
        x2y2 = tuple(
            (boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))

        img = cv2.rectangle(img, (x1y1), (x2y2), (255, 0, 0), 5)

        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    return img

"""
Returns all the plates from the image as separate pictures, for later detection
Iterates through all the detections, and looks for plate detection
Crops that part of image
Parameters
----------
img: image where we want to detect plates
classes: all the classes in the net as numbers
class_names: names of the aforementioned classes
nums: number of detections 
"""
def get_plates(img, boxes, classes, nums, class_names):
    boxes, classes, nums = boxes[0], classes[0], nums[0]
    boxes = np.array(boxes)
    plate_images = []
    plate_coordinates = []

    for i in range(nums):
        class_name = class_names[int(classes[i])]
        if class_name != 'license-plate':
            continue

        x1y1 = tuple(
            (boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32)
        )
        x2y2 = tuple(
            (boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32)
        )

        x1, y1 = x1y1
        x2, y2 = x2y2

        plate_images.append(img[y1:y2, x1:x2])
        plate_coordinates.append(x1y1)

    return plate_images, plate_coordinates

