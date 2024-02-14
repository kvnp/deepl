import cv2
from matplotlib.pyplot import draw
import numpy as np
import os
import shutil
import csv

crop_x = 256
crop_y = 256


xy_pairs = []
def get_random_xy_pairs(xmax, ymax):
    pair = (np.random.randint(0, xmax), np.random.randint(0, ymax))
    if pair not in xy_pairs:
        xy_pairs.append(pair)
        return pair
    else:
        return get_random_xy_pairs(xmax, ymax)


def random_cutout(amount):
    orig = cv2.imread("./dataset/complete/complete_original.png")
    draw = cv2.imread("./dataset/complete/complete_drawing.png")
    
    max_x = orig.shape[1] - crop_x
    max_y = orig.shape[0] - crop_y

    original_crops = []
    drawing_crops = []

    for _ in range(amount):
        x, y = get_random_xy_pairs(max_x, max_y)
        original_crops.append(orig[y: y + crop_y, x: x + crop_x])
        drawing_crops.append(draw[y: y + crop_y, x: x + crop_x])

    return original_crops, drawing_crops

def create_dataset(amount):
    original_crops, drawing_crops = random_cutout(amount)
    for i in range(amount):
        cv2.imwrite("./dataset/original/" + str(i) + ".png", original_crops[i])
        cv2.imwrite("./dataset/drawing/" + str(i) + ".png", drawing_crops[i])