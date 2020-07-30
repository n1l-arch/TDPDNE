import math
import pickle
from pathlib import Path
from random import randint
import json
from multiprocessing import Pool, Queue
import traceback

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

import cv_tools
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config

def multi_worker(queue):
    while 1:
        try:
            align_data = queue.get()
            if isinstance(align_data, str) and align_data == 'kill process':
                print('Killing listener.')
                break
            try:
                aligned = align(**align_data['mask_info'])
                padded = pad_image(aligned)
                cv2.imwrite(str(align_data['save_name']), padded)
            except:
                pass
        except:
            # prints exceptions from another process
            traceback.print_exc()
            pass

def vis(image_path, r):
    img = load_img(image_path)
    img = img_to_array(img)
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                ['BG', 'pp'], r['scores'],
                                title="Predictions")


def detect(model, image_path):
    image = load_img(image_path)
    image = img_to_array(image)

    result = model.detect([image])

    return result[0]


def align(image, mask_rcnn_res):
    top_left, bottom_right, mask = get_best_bbox(mask_rcnn_res)
    mask_img = (mask*255).astype(np.uint8)
    centre, e1, _ = cv_tools.compute_PCA(mask_img)

    # make a binary image to represent the bounding box so we can rotate later
    bb_mask = make_bb_mask(image, top_left, bottom_right)

    tilt_angle = get_tilt(centre, e1)

    rotated = rotate_image(image, -tilt_angle, image_centre=centre)
    rotated_mask = rotate_image(bb_mask, -tilt_angle, image_centre=centre)

    top_left, bottom_right = get_largest_bbox(rotated_mask)

    cropped = cv_tools.crop(rotated, top_left, bottom_right)

    return cropped


def pad_image(image, side_length=512):
    padded_image = cv_tools.pad_to_square(image)
    resize_dim = (side_length, side_length)

    if padded_image.shape[0] < side_length:
        interpolation = cv2.INTER_CUBIC
        padded_image = cv2.resize(padded_image, resize_dim,
                                  interpolation=interpolation)
    elif padded_image.shape[0] > side_length:
        interpolation = cv2.INTER_AREA
        padded_image = cv2.resize(padded_image, resize_dim,
                                  interpolation=interpolation)

    return padded_image


def get_largest_bbox(mask):
    foreground_rows, foreground_cols = np.where(mask == 1)

    top_left = min(foreground_cols), min(foreground_rows)
    bottom_right = max(foreground_cols), max(foreground_rows)

    mask = cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask, top_left, bottom_right, (255, 0, 0), 2)
    # cv_tools.show(cv_tools.resize(mask, 0.3))

    return top_left, bottom_right


def rotate_image(image, angle, image_centre=None):
    if image_centre is None:
        image_centre = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_centre, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def make_bb_mask(image, top_left, bottom_right):
    x1, y1 = top_left[0], top_left[1]
    x2, y2 = bottom_right[0], bottom_right[1]
    height, width = image.shape[:2]

    bb_mask = np.zeros((height, width))

    bb_mask[y1:y2, x1:x2] = 1

    return bb_mask


def angle_trunc(a):
    while a < 0.0:
        a += math.pi * 2
    return a


def get_tilt(p1, p2, tilt_from='vertical', degrees=True):

    if not tilt_from in ['vertical', 'horizontal']:
        raise ValueError('Tilt must be calculated from vertical or horizontal')

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    deltaY = y2 - y1
    deltaX = x2 - x1

    rads = angle_trunc(math.atan2(deltaY, deltaX))
    if abs(rads) > math.pi:
        rads = -(2*math.pi - rads)
    if tilt_from == 'vertical':
        rads = math.pi/2 - rads
    if degrees:
        return math.degrees(rads)
    return rads


def get_best_bbox(result):
    best_index = np.argmax(result['scores'])
    y1, x1, y2, x2 = result['rois'][best_index]
    mask = result['masks'][:, :, best_index]

    top_left = (x1, y1)
    bottom_right = (x2, y2)

    return top_left, bottom_right, mask


class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"

    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background)
    # kangaroo + BG
    NUM_CLASSES = 1+1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # setting Max ground truth instances
    MAX_GT_INSTANCES = 10


if __name__ == '__main__':

    # LOAD MASK R-CNN
    config = myMaskRCNNConfig()
    # Loading the model in the inference mode
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
    model_path = Path('mask_rcnn_model.h5')
    if not model_path.exists():
        download_file_from_google_drive('1gJDCxx1cExbdCxhtPIFfBxa6F4obzRHY', model_path)
    # loading the trained weights o the custom dataset
    model.load_weights(model_path, by_name=True)

    save_folder = Path('../aligned_images')
    if not save_folder.exists():
        save_folder.mkdir()

    images = list(Path('../images').iterdir())

    # just a text file that lists the already done images in case we
    # want to stop and continue later
    progress_file = Path('done.txt')
    if not progress_file.exists():
        progress_file.mkdir()

    with open('done.txt', 'r') as f:
        done = f.read().split('\n')

    # alignment and mask r-cnn happen in parallel
    # alignment uses cpu, mask r-cnn uses the gpu
    # this way we cut the time needed by a factor of
    # at least 4
    
    # alignment queue
    queue = Queue()
    Pool = Pool(4, multi_worker, (queue,))

    for i in tqdm(images):
        skip = False
        save_name = save_folder/i.name
        if save_name.exists() or str(i.name) in done:
            skip = True
        try:
            image = cv2.imread(str(i))
            # image not available
            if image.shape == (81, 161, 3):
                skip = True
        except:
            print('Error in:', i.name)
            skip = True

        if not skip:
            result = detect(model, i)
            # if mask r-cnn detects a pp, put into
            # the alignment queue for processing
            if len(result['rois']):
                queue.put({
                            'mask_info': {'image': image,
                                        'mask_rcnn_res': result},
                            'save_name': save_name
                        })

        with open('done.txt', 'a') as f:
            f.write(str(i.name) + '\n')
    queue.put('kill process')