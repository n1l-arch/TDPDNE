import os
import random

import cv2
import numpy as np

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def show(image: np.ndarray) -> None:
    """
    Displays the image in an opencv window.

    Args:
        image: 2D np.array
    """
    cv2.imshow('img', image)
    cv2.waitKey(0)


def resize(image: np.ndarray, resize_factor=None, resize_to=None) -> np.ndarray:
    if resize_factor is None and resize_to is None:
        raise ValueError('Must resize by a factor or to a size.')
    elif resize_factor is not None and resize_to is not None:
        raise ValueError('Must choose only one of resizing by a factor or to a size.')

    if resize_factor:
        if 0 < resize_factor < 1:
            interpolation = cv2.INTER_AREA
        elif resize_factor > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            raise ValueError('Resize factor must be above 0, and not 1.')

        resized = cv2.resize(image, None, fx=resize_factor,
                             fy=resize_factor, interpolation=interpolation)

    elif resize_to:
        if image.shape[0] < resize_to:
            interpolation = cv2.INTER_CUBIC

        elif image.shape[0] > resize_to:
            interpolation = cv2.INTER_AREA
        
        elif image.shape[0] == resize_to:
            # resizing to the same shape
            interpolation = cv2.INTER_AREA

        resized = cv2.resize(image, (resize_to, resize_to),
                             interpolation=interpolation)

    return resized


def crop(image: np.ndarray, top_left: Tuple[int], bottom_right: Tuple[int]) -> np.ndarray:
    height, width = image.shape[:2]

    x1, y1 = top_left[0], top_left[1]
    x2, y2 = bottom_right[0], bottom_right[1]

    return image[y1:y2, x1:x2]


def compute_PCA(image: np.ndarray, display=False) -> tuple:
    mat = np.argwhere(image != 0)
    mat[:, [0, 1]] = mat[:, [1, 0]]
    mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

    # mean (e. g. the geometrical center)
    # and eigenvectors (e. g. directions of principal components)
    m, e = cv2.PCACompute(mat, mean=np.array([]))

    # now to draw: let's scale our primary axis by 100,
    # and the secondary by 50
    centre = tuple(m[0])
    endpoint1 = tuple(m[0] + e[0]*100)
    endpoint2 = tuple(m[0] + e[1]*50)

    if display:
        img_show = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        cv2.circle(img_show, centre, 5, (255, 255, 0))
        cv2.line(img_show, centre, endpoint1, (255, 255, 0))
        cv2.line(img_show, centre, endpoint2, (255, 255, 0))

        show(img_show)

    return centre, endpoint1, endpoint2


def threshold(image: np.ndarray, show_thresholded: bool = False, show_morphed: bool = False) -> np.ndarray:
    """
    Main thresholding function.

    Args:
        image: 2D np.array
        show_thresholded: Will display the thresholded image if True
        show_morphed: Will display the morphologically transformed image if True

    Returns:
        morphed: Thresholded and morphologically transformed image
    """
    # Adaptive thresholding to account for lighting
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -2)
    if show_thresholded:
        show(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.erode(thresh, kernel, iterations=1)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=10)

    if show_morphed:
        show(morphed)

    return morphed


def find_contours(image: np.ndarray) -> List[np.ndarray]:
    """
    Finds the contours in a given image.

    Args:
        image: 2D numpy array

    Returns:
        contours: list of contours found with cv2.findContours()
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    return contours


def draw_contours(image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """
    Draws the contours in a given image using cv2.drawContours.
    input image must be RGB so the lines can be seen
    """

    clr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(clr_img, contours, -1, (0, 255, 0), 3)

    return clr_img


def draw_lines(image: np.ndarray, lines: List[np.ndarray]) -> np.ndarray:
    """
    Draws the lines found in an image from cv2.HoughLines()
    input image must be RGB so the lines can be seen

    Args:
        image: 2D numpy array
        lines: lines returned from cv2.HoughLines()

    Returns:
        image: 2D numpy array
    """

    for line in lines:
        # visit https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
        # for explanation on each line parameter
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    return image


def hough_lines(image: np.ndarray) -> (np.ndarray, List[np.ndarray]):
    """
    Uses the hough transform to find lines in an image.
    Converts to a coloured image first, so the lines can be seen

    Args:
        image: 2D numpy array

    Returns:
        clr_img: the coloured image (which can be used to draw the lines)
        lines: the lines that were returned from cv2.HoughLines()
    """
    clr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(image, 50, 150, apertureSize=3)  # Canny edge detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)  # Hough line detection

    return clr_img, lines


def filter_lines(lines: List[np.ndarray], angle_thresh: list = [(0, 45), (135, 180)]) -> List[np.ndarray]:
    """
    Filters a list of lines found with cv2.HoughLines()
    Can filter based on the angles you want to exclude.

    OpenCV makes zero degrees a vertical line, and increases
    going clockwise up to 180. Since these lines are of
    infinite length, this covers a full circle.

    Args:
        lines: list of lines found with cv2.HoughLines()
        angle_thresh: list of angle thresholds, each indicated with a tuple or list.
                      For example, if you want from 0-30 degrees and 150-180,
                      set angle thresh to [(0,30),(150,180)].

    Returns:
        filtered_lines: list of lines that are filtered.
    """

    filtered_lines = []
    for line in lines:
        # theta in radians
        theta = line[0][1]
        # convert to degrees
        degrees = theta*(180/math.pi)
        for thresh in angle_thresh:
            if thresh[0] <= degrees <= thresh[1]:
                filtered_lines.append(line)
    return filtered_lines


def segment_by_angle_kmeans(lines: np.ndarray, k: int = 2, **kwargs) -> List[List[np.ndarray]]:
    """Groups lines based on angle with k-means.
    Used to make sure we don't check intersections between two
    almost parallel lines. Uses k-means clustering to make
    two groups with different angles.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def segmented_intersections(lines: np.ndarray) -> List[List[int]]:
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def intersection(line1: np.ndarray, line2: np.ndarray) -> List[int]:
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    # make sure A is invertable
    if np.linalg.cond(A) < 1/sys.float_info.epsilon:
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
    else:
        x0 = y0 = None
    return [x0, y0]


def filter_intersects(intersects: List[List[int]], pad_thickness: int, height: int, width: int) -> List[List[int]]:
    """
    Filters the intersections found with intersection()

    Args:
        intersects: list of intersections found with intersection()
        pad_thickness: the padding that was set
        height: height of the image
        width: width of the image

    Returns:
        filtered_intersects: list of filtered intersections 
    """
    filtered_intersects = []
    for x, y in intersects:
        if x is not None and y is not None:
            # the intersection can be between the edge of
            # the image (including the pad thickness)
            # and the intersection can be between a third of
            # the way down the picture, to above the picture (including the pad thickness)
            if -pad_thickness < x < width+pad_thickness and -pad_thickness < y < height*0.2:
                filtered_intersects.append([x, y])

    return filtered_intersects


def draw_intersections(image: np.ndarray, intersects: list, pad_thickness: int) -> np.ndarray:
    """
    Draws the intersections between the lines as little circles.

    Args:
        image: 2D numpy array
        intersects: list of intersections
        pad_thickness: the padding that was set

    Returns:
        image: image with the intersections drawn on it
    """
    for x, y in intersects:
        cv2.circle(image, (x+pad_thickness, y+pad_thickness),
                   2, (0, 255, 0), -1)

    return image


def pad_to_square(img, pad_value=0):
    rows, columns = img.shape[:2]
    dimensions = len(img.shape)

    if rows < columns:
        # image is wider than it is tall
        # to make it square,, you have to add height
        pad_height = columns - rows
        pad_width = 0
    elif rows > columns:
        # image is taller than it is wide
        # to make it square, you have to add width
        pad_height = 0
        pad_width = rows - columns
    else:
        # image is already square
        pad_height = pad_width = 0

    # evenly distribute the padding between the two sides
    top_pad = bottom_pad = pad_height // 2
    left_pad = right_pad = pad_width // 2

    # if the padding needed is odd, add 1 row/column
    if (pad_width % 2) != 0:
        left_pad += 1
    elif (pad_height % 2) != 0:
        top_pad += 1

    # if we don't want the background to be 0
    if pad_value != 0:
        # finding all edge values of the image
        # concatenates values found in the top, right, bottom, left edges respectively
        edges = [img[0, :-1], img[:-1, -1], img[-1, ::-1], img[-2:0:-1, 0]]
        edges = np.concatenate(edges)
        # pad with the median of the edges (which should be the background)
        pad_value = int(np.median(edges))

    if dimensions == 2:
        padded = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant',
                        constant_values=pad_value)
    elif dimensions == 3:
        padded = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), 'constant',
                        constant_values=pad_value)

    return padded


def find_bbox(contours):
    xmin = 999
    ymin = 999
    xmax = 0
    ymax = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x+w)
        ymax = max(ymax, y+h)

    return xmin, ymin, xmax, ymax


def crop_resize(img, resize_size=64, foreground='black', show_bbox=False):
    """
    Finds ROI, crops it out, then resizes to the desired square image.
    """
    if foreground == 'black':
        threshold_type = cv2.THRESH_BINARY_INV
    elif foreground == 'white':
        threshold_type = cv2.THRESH_BINARY
    else:
        raise ValueError('foreground must be either black or white')

    # threshold and find contours for bounding box
    _, thresh = cv2.threshold(
        img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    xmin, ymin, xmax, ymax = find_bbox(contours)

    if show_bbox:
        colour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(colour_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        cv2.imshow('bounding box', colour_img)

    # crop
    roi = img[ymin:ymax, xmin:xmax]

    # add padding to keep aspect ratio
    roi = pad_to_square(roi)

    return cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
