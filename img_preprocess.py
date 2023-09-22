import cv2
import random
import numpy as np
from option import opt


def img_normal(img_inputs):
    outputs = np.float32(img_inputs / 255.)
    return outputs


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def img_trans(img_inputs, idx):
    if idx == 2:
        outputs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in img_inputs]
    elif idx == 3:
        outputs = [cv2.rotate(img, cv2.ROTATE_180) for img in img_inputs]
    elif idx == 4:
        outputs = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in img_inputs]
    else:
        outputs = [cv2.flip(img, idx) for img in img_inputs]

    return outputs


def modcrop(img, h_start, w_start, crop_size):
    outputs = img[h_start:h_start + crop_size, w_start:w_start + crop_size, :]
    return outputs


def modcrop_size(img_inputs, scale):
    img = np.copy(img_inputs)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r = H % scale
        W_r = W % scale
        img = img[:H - H_r, :W - W_r, :]

    return img


def bgr2ycbcr(img_inputs, only_y=True):
    # img_inputs: BGR channels
    '''
    only_y: only return Y channel
    img_Input:
        [0, 255]
    '''
    # convert
    if only_y:
        img_trans = np.dot(img_inputs, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        img_trans = np.matmul(img_inputs, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    img_outputs = np.float32(img_trans)
    return img_outputs


if __name__ == '__main__':
    f1 = './000.png'
    img1 = imread(f1)
    img2 = bgr2ycbcr(img1)
    print(img1)
