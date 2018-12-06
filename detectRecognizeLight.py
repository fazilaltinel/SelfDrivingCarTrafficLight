import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os


def recognize_color(im):
    '''
    Given a traffic light image, returns class of the image (red, yellow, green)

    Args:
        im: traffic light image (detected bbox of traffic light)

    Returns:
        predicted_label: output label of the light color
                        ([1, 0, 0]: red, [0, 1, 0]: yellow, [0, 0, 1]: green)
    '''

    # Standardize input and convert it from RGB to HSV
    standard_im = cv2.resize(im, (32, 32))
    hsv = cv2.cvtColor(standard_im, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Define the mask & its boundaries, based on value channel
    # The limits of the masks are deduced by trial & error
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([255, 255, 40])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    masked_image=np.copy(hsv)
    masked_image[mask_hsv != 0] = [0, 0, 0]

    # Grayscale is used to enhance the quality of brightness
    grayimage = cv2.cvtColor(standard_im, cv2.COLOR_RGB2GRAY)

    # Using image slicing, crop the image into 3 horizontal parts
    upper_part_v = masked_image[3:12, 5:27, 2]
    mid_part_v = masked_image[12:21, 5:27, 2]
    lower_part_v = masked_image[21:30, 5:27, 2]

    upper_part_g = grayimage[3:12, 5:27]
    mid_part_g = grayimage[12:21, 5:27]
    lower_part_g = grayimage[21:30, 5:27]

    # Average brightness of all the pixels on value channel & grayscale for the masked image
    area = 9.0*22
    avg_brightness_upper = (np.sum(upper_part_g[:,:]) + np.sum(upper_part_v[:,:]))/area
    avg_brightness_mid =  (np.sum(mid_part_g[:, :]) + np.sum(mid_part_v[:, :]))/area
    avg_brightness_lower =  (np.sum(lower_part_g[:, :]) + np.sum(lower_part_v[:, :]))/area

    # Feature vector
    feature = []
    feature = [avg_brightness_upper, avg_brightness_mid, avg_brightness_lower]

    # Convert feature vector to output label
    predicted_label = [0, 0, 0]
    predicted_label[np.argmax(feature)] = 1
    return predicted_label


def draw_boxes(image_fed, best_boxes_roi, best_boxes_classes, best_boxes_scores, frame_w, frame_h, num_pred, video_writer):
    '''
    Draws boxes for detected objects

    Args:
        image_fed: image batch in which object detection performed
        best_boxes_roi: roi of detected objects in images
        best_boxes_classes: classes of detected objects in images
        best_boxes_scores: scores of detected objects in images
        frame_w: output frame width
        frame_h: output frame height
        num_pred: number of predictions
        video_writer: video writer object to write drawn frame to video
    '''

    for i in range(best_boxes_roi.shape[0]):
        im = np.reshape(image_fed[i], (frame_w, frame_h, 3))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        for j in range(num_pred):

            # Choose traffic lights from detected objects (label number of traffic light category in COCO dataset is 10)
            if best_boxes_scores[i][j] > 0.35 and best_boxes_classes[i][j] == 10.:
                x = best_boxes_roi[i][j][1]
                y = best_boxes_roi[i][j][0]
                x_max = best_boxes_roi[i][j][3]
                y_max = best_boxes_roi[i][j][2]

                # Classify color of the traffic light detected
                color = recognize_color(im[int(x):int(x_max), int(y):int(y_max), :])

                # Use different colors according to classified color of traffic light
                if color == [1, 0, 0]:
                    cv2.rectangle(im, (x,y), (x_max,y_max), (255, 0, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im, 'Red', (x,y), font, 1e-3*frame_h, (0, 0, 255), 2)
                elif color == [0, 1, 0]:
                    cv2.rectangle(im, (x,y), (x_max,y_max), (255, 0, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im, 'Yellow', (x,y), font, 1e-3*frame_h, (0, 255, 255), 2)
                elif color == [0, 0, 1]:
                    cv2.rectangle(im, (x,y), (x_max,y_max), (255, 0, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im, 'Green', (x,y), font, 1e-3*frame_h, (0, 255, 0), 2)

        video_writer.write(im)


def main():
    im_size = 512
    pb_dir = './model/frozen_inference_graph.pb'  # Pretrained model path
    img_dir = './object-dataset'  # Dataset folder path
    vid_out = './out'  # Output video path

    # Load pretrained model
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile(pb_dir, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')

            img = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')
            sess = tf.Session(graph=graph)

    vid_out = vid_out + '/outputVideo.mp4'
    num_iter = 67
    batch_size = 32
    num_pred = 30
    frame_w = 512
    frame_h = 512
    video_writer = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'MP4V'), 5.0, (frame_w, frame_h))

    # Read image name list from the directory and sort it (os.listdir generates output in random order)
    images = sorted(os.listdir(img_dir))
    k = 2
    for i in range((len(images)-2)//batch_size):
        image_bat = []
        for j in range(batch_size):
            # Read images from dataset directory
            image = cv2.imread(img_dir + '/' + images[k])
            # Resize images and make them batch
            image = cv2.resize(image, (im_size, im_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_bat.append(image)
            image_batch = np.asarray(image_bat)
            k = k + 1

        # Give image batch to the model as input
        feed_dict = {img:image_batch}
        y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run([detection_boxes,
                                                                           detection_scores,
                                                                           num_detections,
                                                                           detection_classes], feed_dict=feed_dict)

        # Process detection outputs
        best_boxes_roi = []
        best_boxes_scores = []
        best_boxes_classes = []
        for i in range(y_p_boxes.shape[0]):
            temp = y_p_boxes[i, :num_pred] * frame_w
            best_boxes_roi.append(temp)
            best_boxes_scores.append(y_p_scores[i, :num_pred])
            best_boxes_classes.append(y_p_classes[i, :num_pred])
        best_boxes_roi = np.asarray(best_boxes_roi)
        best_boxes_scores = np.asarray(best_boxes_scores)
        best_boxes_classes = np.asarray(best_boxes_classes)

        # Draw boxes for detected objects
        draw_boxes(image_batch, best_boxes_roi, best_boxes_classes, best_boxes_scores, frame_w, frame_h, num_pred, video_writer)

    video_writer.release()


if __name__ == "__main__":
    main()
