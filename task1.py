import argparse
import copy
import os

import cv2
import numpy as np

import utils

# Prewitt operator
prewitt_x = [[1, 0, -1],[1,0,-1],[1,0,-1]] 
prewitt_y = [[1] * 3, [0] * 3, [-1] * 3]

# Sobel operator
sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="data/proj1-task1.jpg",
        help="path to the image used for edge detection")
    parser.add_argument(
        "--kernel", type=str, default="Prewitt",
        choices=["prewitt", "sobel", "Prewitt", "Sobel"],
        help="type of edge detector used for edge detection")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def read_image(img_path, show=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if show:
        show_image(img)
    img = [[int(pixel) for pixel in row] for row in img]
    # print("Trying to read:", img_path)
    # print("Result:", img)

    return img


def show_image(img, delay=1000):
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def write_image(img, img_saving_path):
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            img = (255 * img).astype(np.uint8)
    cv2.imwrite(img_saving_path, img)


def pixel_conv_sum(image):
    s = 0
    for row in image:
        for val in row:
            s += val
    return s


def convolve2d(img, kernel):
    k_h = len(kernel)
    k_w = len(kernel[0])
    pad_h = k_h // 2
    pad_w = k_w // 2

    img_pad = utils.zero_pad(img, pad_h, pad_w)
    img_conv = copy.deepcopy(img)

    for i in range(len(img)):
        for j in range(len(img[0])):
            patch = utils.crop(img_pad, i, i + k_h, j, j + k_w)
            prod = utils.elementwise_mul(patch, kernel)
            img_conv[i][j] = pixel_conv_sum(prod)
    return img_conv


def normalize(img):
    min_val = min([min(row) for row in img])
    max_val = max([max(row) for row in img])
    if max_val == min_val:
        return [[0 for _ in row] for row in img]
    norm_img = copy.deepcopy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            norm_img[i][j] = (img[i][j] - min_val) / (max_val - min_val)
    return norm_img


def detect_edges(img, kernel, norm=True):
    img_edges = convolve2d(img, kernel)
    if norm:
        img_edges = normalize(img_edges)
    return img_edges


def edge_magnitude(edge_x, edge_y):
    edge_mag = copy.deepcopy(edge_x)
    max_val = 0
    for i in range(len(edge_x)):
        for j in range(len(edge_x[0])):
            val = (edge_x[i][j] ** 2 + edge_y[i][j] ** 2) ** 0.5
            edge_mag[i][j] = val
            if val > max_val:
                max_val = val
    for i in range(len(edge_mag)):
        for j in range(len(edge_mag[0])):
            edge_mag[i][j] /= max_val
    return edge_mag


# ----------- NEW PART -----------

def make_combined_image(original, edge_x, edge_y, edge_mag, save_path):
    """

    """
    if isinstance(original, list): original = np.asarray(original, dtype=np.uint8)
    if isinstance(edge_x, list): edge_x = (255*np.asarray(edge_x)).astype(np.uint8)
    if isinstance(edge_y, list): edge_y = (255*np.asarray(edge_y)).astype(np.uint8)
    if isinstance(edge_mag, list): edge_mag = (255*np.asarray(edge_mag)).astype(np.uint8)


    h, w = original.shape
    edge_x = cv2.resize(edge_x, (w, h))
    edge_y = cv2.resize(edge_y, (w, h))
    edge_mag = cv2.resize(edge_mag, (w, h))

    combined = cv2.hconcat([original, edge_x, edge_y, edge_mag])
    cv2.imwrite(save_path, combined)

# --------------------------------


def main():
    args = parse_args()

    img = read_image(args.img_path, True)

    if args.kernel in ["prewitt", "Prewitt"]:
        kernel_x = prewitt_x
        kernel_y = prewitt_y
        op_name = "prewitt"
    elif args.kernel in ["sobel", "Sobel"]:
        kernel_x = sobel_x
        kernel_y = sobel_y
        op_name = "sobel"
    else:
        raise ValueError("Kernel type not recognized.")

    if not os.path.exists(args.rs_directory):
        os.makedirs(args.rs_directory)

    # X edges
    img_edge_x = detect_edges(img, kernel_x, False)
    img_edge_x = normalize(img_edge_x)
    img_edge_x_np = (255 * np.asarray(img_edge_x)).astype(np.uint8)
    write_image(img_edge_x_np,
        os.path.join(args.rs_directory, f"{op_name}_edge_x.jpg")
    )

    # Y edges
    img_edge_y = detect_edges(img, kernel_y, False)
    img_edge_y = normalize(img_edge_y)
    img_edge_y_np = (255 * np.asarray(img_edge_y)).astype(np.uint8)
    write_image(img_edge_y_np,
        os.path.join(args.rs_directory, f"{op_name}_edge_y.jpg")
    )

    # Magnitude
    img_edges = edge_magnitude(img_edge_x, img_edge_y)
    img_edges = normalize(img_edges)
    img_edges_np = (255 * np.asarray(img_edges)).astype(np.uint8)
    write_image(img_edges_np,
        os.path.join(args.rs_directory, f"{op_name}_edge_mag.jpg")
    )


    # -------- combined output --------
    original_np = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    make_combined_image(
        original_np,
        img_edge_x_np,
        img_edge_y_np,
        img_edges_np,
        os.path.join(args.rs_directory, f"{op_name}_combined.jpg")
    )



if __name__ == "__main__":
    main()
