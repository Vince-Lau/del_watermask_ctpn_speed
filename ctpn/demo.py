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
hight_rate = 0.90
width_rate = 0
# scale = 1


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    hight, width = img.shape[0:2]
    base_name = image_name.split('\\')[-1]
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            minY = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            if minY > 0 * (hight/scale) and box[8] >= 0.8:
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                # if box[8] >= 0.9:
                #     color = (0, 255, 0)
                # elif box[8] >= 0.8:
                #     color = (255, 0, 0)
                color = (0, 255, 0)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
                cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

                min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
                min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
                max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
                max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

                line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
                f.write(line)
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)


def Gaussian_Blur(img, image_name, boxes, scale):
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    hight, width = img.shape[0:2]
    for box in boxes:
        minY = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
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


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()
    other_name = image_name.split('\\')[-1]
    tmp = other_name.split('.')
    base_name = tmp[0] + '_blur.' + tmp[1]

    img = cv2.imread(image_name)
    hight, width = img.shape[0:2]
    img_section = img[int(hight*hight_rate):hight, int(width*width_rate):width]

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

    cv2.imwrite(os.path.join("data/results", base_name), img)
    timer.toc()
    print(('\033[34mDetection took {:.3f}s for '
           '{:d} object proposals\033[0m').format(timer.total_time, boxes.shape[0]))
    # timer.toc()
    # print(('\033[34m pre-process took time {:.5f}s \033[0m'.format(timer.total_time)))


if __name__ == '__main__':
    timer = Timer()
    timer.tic()
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('../ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
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
        ctpn(sess, net, im_name)
