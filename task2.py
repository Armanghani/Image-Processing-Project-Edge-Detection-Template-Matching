"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import math

import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="data/proj1-task2-png.png",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="data/a.pgm",
        choices=["./data/a.pgm", "./data/b.pgm", "./data/c.pgm"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def size_of_image(kernel):
    '''
    Gives the sizew of the image/kernel! 

    '''
    rows_ = len(kernel)
    col_ = len(kernel[0])

    return rows_,col_

def mean_(image):
    '''
    Calculates the mean of the image or kernel wherever applied!

    '''
    img_sum = 0
    for img_r in image:
        for img_c in img_r:
            img_sum = img_sum + img_c
    img_mean = img_sum/(len(image)*len(image[0]))

    image = img_mean
    return image

def edge_detection(image):
    '''
    Finds the Gradient direction using edge detection from the Task1.py file

    '''
    image_edge_x = detect_edges(image,sobel_x,True)
    image_edge_y = detect_edges(image,sobel_y,True)

    image_edges = copy.deepcopy(image_edge_x)

    for ii, row in enumerate(image_edge_y):
        for jj, col in enumerate(row):
            image_edges[ii][jj] = math.atan2(image_edge_y[ii][jj],image_edge_x[ii][jj])

    return image_edges   

  


def detect(img, template):
    """Detect a given character using normalized cross-correlation (NCC)."""


    temp_h = len(template)    
    temp_w = len(template[0])   


    template_mean = mean_(template)

    template_norm = 0
    template_zero_mean = []

    for i in range(temp_h):
        row = []
        for j in range(temp_w):
            val = template[i][j] - template_mean
            row.append(val)
            template_norm += val * val
        template_zero_mean.append(row)

    template_norm = template_norm ** 0.5


    coordinates = []


    img_h = len(img)
    img_w = len(img[0])

    for i in range(img_h - temp_h):
        for j in range(img_w - temp_w):

            patch = utils.crop(
                img,
                i, i + temp_h,
                j, j + temp_w
            )

            patch_mean = mean_(patch)

            patch_norm = 0
            patch_zero_mean = []

            for x in range(temp_h):
                row = []
                for y in range(temp_w):
                    val = patch[x][y] - patch_mean
                    row.append(val)
                    patch_norm += val * val
                patch_zero_mean.append(row)

            patch_norm = patch_norm ** 0.5

            if patch_norm == 0:
                continue

            numerator = 0
            for x in range(temp_h):
                for y in range(temp_w):
                    numerator += (
                        template_zero_mean[x][y] *
                        patch_zero_mean[x][y]
                    )

            ncc = numerator / (template_norm * patch_norm)
            if ncc > 0.78:
                coordinates.append((i, j))

    return coordinates



def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)
    print(size_of_image(img))
    print(size_of_image(template))

    coordinates = detect(img,template)
    print(coordinates)
    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()

