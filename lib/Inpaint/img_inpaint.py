# -*- coding:utf-8 -*-
# @Time    : 2018/12/4 11:07
# @Author  : yuanjing liu
# @Email   : lauyuanjing@163.com
# @File    : img_inpaint.py
# @Software: PyCharm

import cv2
import os
import numpy as np

# path = './others/test2.jpg'
# path = './oringe/15410457011016.jpg'
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 判断图片底部是否有白色污染
def compute_white_ratio(gray, lower, position, hight, width):
    img_mask = gray[int(hight*position):hight, int(width*0):width]
    hight1, width1 = img_mask.shape
    total_area = hight1 * width1
    tmp = img_mask[img_mask > lower]
    white_area = len(tmp)
    white_ratio = white_area / total_area
    return white_ratio


def del_watermask_ratio(img, image_name, lower, upper, position, del_ratio):
    other_name = image_name.split('\\')[-1]
    tmp = other_name.split('.')
    base_name = tmp[0] + '_blur.' + tmp[1]

    hight, width = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = compute_white_ratio(gray, lower, position, hight, width)
    if white_ratio <= del_ratio:
        img_mask = img[int(hight * position):hight, int(width * 0):width]
        thresh = cv2.inRange(img_mask, np.array([lower, lower, lower]), np.array([upper, upper, upper]))
        kernel = np.ones((3, 3), np.uint8)
        hi_mask = cv2.dilate(thresh, kernel, iterations=1)
        specular = cv2.inpaint(img_mask, hi_mask, 5, flags=cv2.INPAINT_TELEA)
        img_copy = img.copy()
        img_copy[int(hight * position):hight, int(width * 0):int(width)] = specular
    else:
        img_copy = img.copy()
    return img_copy


def test_img_nowatermask_ratio(foldername):
    lower, upper, position, del_ratio = 180, 255, 0.92, 0.3
    zheng, wu = 0, 0
    for r, dirs, files in os.walk(foldername):
        # Get all the images
        for file in files:
            pth = os.sep.join([r, file])
            img = cv2.imread(pth)
            img0 = cv2.imread(pth)
            img_clear, J = del_watermask_ratio(img, lower, upper, position, del_ratio)
            new = file.split('.')
            if J == True:
                cv2.imwrite('./result/zheng' + '/' + file, img0)
                cv2.imwrite('./result/zheng' + '/' + new[0] + '_1.' + new[1], img_clear)
                zheng += 1
            else:
                cv2.imwrite('./result/wu' + '/' + file, img0)
                cv2.imwrite('./result/wu' + '/' + new[0] + '_1.' + new[1], img_clear)
                wu += 1
    print('判断<没有>白色污染<并进行>去水印处理的个数：%.f' % zheng)
    print('判断<有>白色污染<没有进行>去水印处理的个数：%.f' % wu)

    return zheng, wu


if __name__ == '__main__':
    position = 0.92
    lower, upper = 180, 255
    zheng, wu = test_img_nowatermask_ratio('./test4')  # 根据以上测试结果，再次去水印，验证准确率
