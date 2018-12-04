import cv2
import glob
import os
import shutil
import sys
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


# image slice parameter
hight_rate = 0.92
width_rate = 0
lower, upper, position, del_ratio = 150, 255, hight_rate, 0.05


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


# 判断图片底部是否有白色污染
def compute_white_ratio(gray, lower, position, hight, width):
    img_mask = gray[int(hight*position):hight, int(width*0):width]
    hight1, width1 = img_mask.shape
    total_area = hight1 * width1
    tmp = img_mask[img_mask > lower]
    white_area = len(tmp)
    white_ratio = white_area / total_area
    return white_ratio


def Gaussian_Blur(img, image_name, boxes, scale):
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    hight, width = img.shape[0:2]
    for box in boxes:
        minX = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        if box[8] >= 0.8 and minX > int(0.4 * width):
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            img_mask = img[min_y:max_y, min_x:max_x]
            blur = cv2.GaussianBlur(img_mask, (99, 99), 0)
            img[min_y:max_y, min_x:max_x] = blur

    return img


def inpaint_ctpn(sess, net, image_name):
    other_name = image_name.split('\\')[-1]
    tmp = other_name.split('.')
    base_name = tmp[0] + '_blur.' + tmp[1]

    img = cv2.imread(image_name)
    hight, width = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = compute_white_ratio(gray, lower, position, hight, width)
    img_section = img[int(hight*hight_rate):hight, int(width*width_rate):width]

    if white_ratio <= del_ratio:
        print('\033[34m use inpaint deal with image \033[0m')
        thresh = cv2.inRange(img_section, np.array([lower, lower, lower]), np.array([upper, upper, upper]))
        kernel = np.ones((3, 3), np.uint8)
        hi_mask = cv2.dilate(thresh, kernel, iterations=1)
        specular = cv2.inpaint(img_section, hi_mask, 5, flags=cv2.INPAINT_TELEA)
        img[int(hight * position):hight, int(width * 0):int(width)] = specular
    else:
        timer = Timer()
        timer.tic()
        img_sec_new, scale = resize_im(img_section, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(sess, net,  img_sec_new)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img_sec_new.shape[:2])

        # draw_boxes(img, image_name, boxes, scale)
        img_blur = Gaussian_Blur(img_sec_new, image_name, boxes, scale)
        try:
            img[int(hight * hight_rate):hight, int(width * width_rate):width] = img_blur
        except:
            img = img
        timer.toc()
        print(('\033[34mDetection took {:.3f}s for '
               '{:d} object proposals\033[0m').format(timer.total_time, boxes.shape[0]))

    cv2.imwrite(os.path.join("../data/results", base_name), img)

    # timer.toc()
    # print(('\033[34m pre-process took time {:.5f}s \033[0m'.format(timer.total_time)))


if __name__ == '__main__':
    timer = Timer()
    timer.tic()
    if os.path.exists("../data/results/"):
        shutil.rmtree("../data/results/")
    os.makedirs("../data/results/")

    cfg_from_file('../ctpn/text.yml')

    # init session
    # config = tf.ConfigProto(allow_soft_placement=True)  # 允许动态分配CPU内存
    config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=4,
                                device_count={'CPU': 4})
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    # print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    timer.toc()
    print(('\033[34mLoading model speed time {:.5f}s \033[0m'.format(timer.total_time)))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))

        inpaint_ctpn(sess, net, im_name)
