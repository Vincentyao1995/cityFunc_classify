# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import os
import cv2
import numpy as np
import operator
from tqdm import tqdm

class Region:
    def __init__(self, x, y, w, h, index):
        self.id = index
        self.x1 = x
        self.x2 = x + w
        self.y1 = y
        self.y2 = y + h
        self.area = w * h


def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound.x1
    cy1 = candidateBound.y1
    cx2 = candidateBound.x2
    cy2 = candidateBound.y2

    gx1 = groundTruthBound.x1
    gy1 = groundTruthBound.y1
    gx2 = groundTruthBound.x2
    gy2 = groundTruthBound.y2

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (carea + garea - area)

    return iou


def main():
    # loading astronaut image
    # img = skimage.data.astronaut()
    small_img_list = []
    imgs_sub_folder_root_path = r"./dataset/haidian_streetblock/sub_block_afterSS"
    imgs_folder_path = r"./dataset/haidian_streetblock/hd_clip_jpg"
    if not os.path.exists(imgs_folder_path):
        print('street block images does not exists, please check the filepath!')
    if not os.path.exists(imgs_sub_folder_root_path):
        os.mkdir(imgs_sub_folder_root_path)
    for img in tqdm(os.listdir(imgs_folder_path)):
        img_name = img.split(".")[0]
        print('\n%s is generating sub-street block using selective search!\n' % img_name)
        img_path = os.path.join(imgs_folder_path, img)
        img_ss_folder_path = os.path.join(imgs_sub_folder_root_path, img_name)
        if not os.path.exists(img_ss_folder_path):
            os.mkdir(img_ss_folder_path)
        # ------ read the image ------
        img_content = cv2.imread(img_path)
        # ------ remove the small images according the width and height ------
        img_w = img_content.shape[0]
        img_h = img_content.shape[1]
        if img_w < 350 and img_h < 350:
            small_img_list.append(img)
            continue

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img_content, scale=500, sigma=0.8, min_size=2)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            # if r['size'] < 2000:
            #     continue
            # distorted rects
            x, y, w, h = r['rect']
            if w < 150 or h < 150:
                continue
            if w / h > 2 or h / w > 2:
                continue
            candidates.add(r['rect'])

        # print(candidates)
        # draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(img_content)
        i = 0
        regions = []
        for x, y, w, h in candidates:
            # print(x, y, w, h)
            r = Region(y, x, h, w, i)
            regions.append(r)
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            sub_img = img_content[y:y + h, x:x + w, :]
            sub_path = os.path.join(img_ss_folder_path, 'sub_' + str(i) + '.jpg')
            # sub_path = 'sub_' + str(i) + '.jpg'
            # print(sub_path)
            cv2.imwrite(sub_path, sub_img)
            i += 1
        # plt.show()

        n = len(regions)
        iou_matrix = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i + 1, n):
                iou = calculateIoU(regions[i], regions[j])
                iou_matrix[i][j] = iou
        np.savetxt(os.path.join(img_ss_folder_path, "iou_matrix.txt"), iou_matrix)
        # 按照area进行排序
        cmpfun = operator.attrgetter('area')
        regions.sort(key=cmpfun)
        ids = []
        for r in regions:
            ids.append(r.id)
        np.savetxt(os.path.join(img_ss_folder_path, "area_asc_ids.txt"), np.array(ids))

    np.savetxt("small_imgs", np.array(small_img_list))


if __name__ == "__main__":
    main()
