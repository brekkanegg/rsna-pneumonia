import pandas as pd
import numpy as np
# draw

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import random
import tqdm
from tqdm import tqdm

from PIL import Image

import argparse
import os


import re

s = 'asdf=5;iwantthis123jasd'
result = re.search('asdf=5;(.*)123jasd', s)

def str2bool(v):
    return v.lower() in ('true')

# control here
parser = argparse.ArgumentParser()

parser.add_argument('--csv', type=str, help='target csv')  # v5_T0.3
parser.add_argument('--csv_dir', type=str, default='ensemble2', help='pred csv dir')  # v5_T0.3
parser.add_argument('--save_only_bbox', '--sob', type=str2bool, default=True)
parser.add_argument('--mode', '--m', type=str, default='test')


pargs = parser.parse_args()


import pickle

# DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_train_images'
# DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_test_images'

# seeds = ['42', '7_51_epoch', '77', '34', '17']

#  'deeper_head-5_he_concat-False_lr-0.003_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-115_shuffle_every-5_train_size-2400_13_0.91_test.csv',

# seeds = ['deeper_head-5_he_concat-False_lr-0.003_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-115_shuffle_every-5_train_size-2400_13_0.91_test.csv']

# result = 'temp/iou_th-0.12_mode-test_overfit-True_overlap_th-0.8_prob_th-0.9_seeds-205_212_217_221_th-0.95_Cls(DensePT_v5_T0.35)_testP2.csv'
# save_dir = 'overfit_0.949_v5_0.35_test'

#seed-205_212_217_221_jam-hem_jam2_hardjam_iou_th-0.12_prob_th-0.949_ov_th-0.8_csv_th-0.9_mode-test_PT_v5_T0.3_testP2.csv
result = os.path.join(pargs.csv_dir, pargs.csv)

save_dir = []
save_dir.append(re.search('jam-(.*)_iou_th', result).group(1))  # hem_jam2_hardjam
save_dir.append(re.search('prob_th-(.*)_ov_th', result).group(1))  # 0.949
save_dir.append(re.search('_T(.*)_testP2', result).group(1))  # 0.35, ..
save_dir.append(re.search('mode-(.*)_PT', result).group(1))  # train, test

save_dir = '_'.join(save_dir) + '_only_bbox-' + str(pargs.save_only_bbox).lower()
print(save_dir)
#
# result = 'ensemble2/' + 'iou_th-0.12_jam-hem_jam2_mode-test_overlap_th-0.8_prob_th-0.949_seeds-205_212_217_221_th-0.9_PT_v5_T0.35_testP2.csv'
# save_dir = 'hem_jam2_0.949_v5_0.35_test'
#
# result = 'ensemble2/' + 'iou_th-0.12_jam-hem_jam2_mode-test_overlap_th-0.8_prob_th-0.949_seeds-205_212_217_221_th-0.9_PT_v5_T0.3_testP2.csv'
# save_dir = 'hem_jam2_0.949_v5_0.3_test'


pred = pd.read_csv(result)
# mode = save_dir[-4:]

if pargs.mode == 'test':
    DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_test_images'
else:
    DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_train_images'


print(save_dir)

if not os.path.exists('images3/'+save_dir):
    os.makedirs('images3/'+save_dir)


# ensemble = pd.read_csv('ensemble/' + save_dir)


patientIds = pred['patientId']

def add_bbox(pred, ax, pid, c='red', ensemble=False):
    # print(c)
    pred_bbox_string = pred[pred['patientId'] == pid]['PredictionString'].iloc[0]
    if pred_bbox_string is not np.nan:

        if ensemble:
            pred_bbox_string = pred_bbox_string.split(' ')[1:] ######### fixme: ""
        else:
            pred_bbox_string = pred_bbox_string.split(' ')[1:] #########
        num_pred_bbox_string = int(len(pred_bbox_string) / 5)

        pred_bboxes = []
        for pi in range(num_pred_bbox_string):
            pb = pred_bbox_string[5 * pi:5 * (pi + 1)]
            pred_bboxes.append(pb)

        # Create a Rectangle patch
        for pb in pred_bboxes:
            # print(pb)
            prob, x, y, w, h = float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3]), float(pb[4])

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=c, facecolor='none')
            text = plt.text(x, y - 6.0, str(prob), color=c, fontsize='xx-large')

            ax.add_patch(rect)

        if len(pred_bboxes) > 0:
            return True

def add_gt_bbox(ax, pid, c='red'):
    # print(c)
    gt = pd.read_csv(DATA_DIR + 'stage_2_train_labels.csv')

    gt_bboxes = gt[gt['patientId'] == pid].iloc[:, 1:-1]
    for i in range(len(gt_bboxes)):
        prob, x, y, w, h = 1.0, *gt_bboxes.iloc[i, :]
        if x is not np.nan:
            x, y, w, h = float(x), float(y), float(w), float(h)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=c, facecolor='none')
            text = plt.text(x, y - 6.0, str(prob), color=c, fontsize='xx-large')

            ax.add_patch(rect)



for pid in tqdm(patientIds):
    # print(pid)
    # fp = '/home/minki/ramdisk/rsna/data/kaggle/stage_1_test_images_woong/' + pid + '.png'
    fp = DATA_DIR + '/' + pid + '.png'

    png = Image.open(fp)
    image = np.array(png)

    fig, ax = plt.subplots(1, figsize=(8, 8))

    # Display the image
    ax.imshow(image, cmap='gray')

    colors = ['yellow', 'gray', 'blue', 'purple', 'brown']
    # for i, p in enumerate(preds):
    have_bbox = add_bbox(pred, ax, pid, 'yellow')

    if pargs.mode == 'val':
        add_gt_bbox(ax, pid, 'green')

    if pargs.save_only_bbox:
        if have_bbox:
            fig.savefig('images3/' + save_dir + '/' + pid + '.png')
    else:
        fig.savefig('images3/' + save_dir + '/' + pid + '.png')

#     plt.show()
#     break

