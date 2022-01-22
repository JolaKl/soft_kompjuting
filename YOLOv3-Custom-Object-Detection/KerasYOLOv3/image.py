import tensorflow as tf
from keras.models import load_model

from KerasYOLOv3 import imgutils
from KerasYOLOv3.utils import get_plates, load_class_names, output_boxes, draw_outputs, resize_image
from KerasYOLOv3.contourutils import select_roi
import cv2
import numpy as np
from KerasYOLOv3.yolo3 import YOLOv3Net

model_size = (416, 416,3)
num_classes = 2
class_name = '../classes.names'
max_output_size = 40
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5
cfgfile = '../yolov3_testing.cfg'
weightfile = 'weights/yolov3_weights.tf'
img_path = "../test_images/car66.jpg"  # car5.jpg  car43.jpeg

def get_predicted_label(prediction):
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    prediction_index = int(np.argmax(np.round(prediction), axis=1))
    return classes[prediction_index]

def process_image_file(img_path, model, debug=False, remove_noise=False):
    image = imgutils.load_image(img_path)
    process_image(image, model, debug, remove_noise)


def process_image(image, model, debug=False, remove_noise=False):
    image = np.array(image)
    image = tf.expand_dims(image, axis=0)
    resized_frame = resize_image(image, (model_size[0], model_size[1]))
    pred = model.predict(resized_frame)

    boxes, scores, classes, nums = output_boxes(
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )

    class_names = load_class_names(class_name)

    image = np.squeeze(image)
    plates, plates_coords = get_plates(image.copy(), boxes, classes, nums, class_names)
    image = draw_outputs(image, boxes, scores, classes, nums, class_names)

    for plate_img, plate_coord in zip(plates, plates_coords):
        plate_img = imgutils.resize_image(plate_img, 200, 60)
        img_gs = imgutils.get_binary(plate_img)
        img_noise_removed = imgutils.get_binary_remove_noise(plate_img)
        input_img = img_noise_removed if remove_noise else img_gs

        if debug: imgutils.display_image(img_gs)
        if debug: imgutils.display_image(img_noise_removed)

        edges = cv2.Canny(input_img, 70, 400)

        if debug: imgutils.display_image(edges)

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_copy = plate_img.copy()
        selected_regions, region_images = select_roi(img_copy, contours, intersect=True)

        ocr_model = load_model('char_model')
        print('License plate prediction: ', end='')
        plate_text = ''
        for char_image in region_images:
            char_image = imgutils.resize_image(char_image, 30, 50)
            char_image = imgutils.image_bin_to_clr(imgutils.invert(imgutils.get_binary_remove_noise(char_image)))
            # imgutils.display_image(image)
            img_array = np.array([char_image])
            prediction = ocr_model.predict(img_array)
            predicted_char = get_predicted_label(prediction)
            print(predicted_char, end='')
            plate_text += predicted_char
        print()
        if debug:
            cv2.putText(selected_regions, plate_text, (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            imgutils.display_image(selected_regions)
        cv2.putText(image, plate_text, plate_coord, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    imgutils.display_image(image, color=True)
    

        
def process_video(video_path, model, remove_noise=False):  
    class_names = load_class_names(class_name)
    ocr_model = load_model('char_model')


    cv2.namedWindow('License detection')
    cap = cv2.VideoCapture(video_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)

            boxes, scores, classes, nums = output_boxes(
                pred, model_size,
                max_output_size=max_output_size,
                max_output_size_per_class=max_output_size_per_class,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold
            )

            frame = np.squeeze(frame)
            plates, plates_coords = get_plates(frame.copy(), boxes, classes, nums, class_names)
            annotated_frame = draw_outputs(frame, boxes, scores, classes, nums, class_names)

            for plate_img, plate_coord in zip(plates, plates_coords):
                plate_img = imgutils.resize_image(plate_img, 200, 60)
                img_gs = imgutils.get_binary(plate_img)
                img_noise_removed = imgutils.get_binary_remove_noise(plate_img)
                input_img = img_noise_removed if remove_noise else img_gs

                edges = cv2.Canny(input_img, 70, 400)

                contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                img_copy = plate_img.copy()
                selected_regions, region_images = select_roi(img_copy, contours, intersect=True)

                print('License plate prediction: ', end='')
                plate_text = ''
                for char_image in region_images:
                    char_image = imgutils.resize_image(char_image, 30, 50)
                    char_image = imgutils.image_bin_to_clr(imgutils.invert(imgutils.get_binary_remove_noise(char_image)))
                    img_array = np.array([char_image])
                    prediction = ocr_model.predict(img_array)
                    predicted_char = get_predicted_label(prediction)
                    print(predicted_char, end='')
                    plate_text += predicted_char
                print()
                cv2.putText(annotated_frame, plate_text, plate_coord, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

            cv2.imshow('License detection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')





def main():
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    model.load_weights(weightfile)

    video = True

    if video:
        process_video('../test_images/car73.mp4', model, True)
    else:
        process_image_file('../test_images/car69.png', model, True)

        for i in range(7):
            path = f'../test_images/car6{i}.jpg'
            process_image_file(path, model)

        # cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)
        # imgutils.display_image(img_copy)
        

if __name__ == '__main__':
    main()