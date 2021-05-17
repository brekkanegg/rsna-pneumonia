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


# test_png_dir = os.path.join(DATA_DIR, 'Test_P2/images')





# seeds = ['42', '7_51_epoch', '77', '34', '17']

# seeds = ['42'
#          '77'
#         ]
# seeds_text = '42_77'



# 0.195
# seeds = ['deeper_head-2_he_concat-False_lr-0.003_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-116_shuffle_every-5_train_size-2400_18_0.95_test.csv',
#          'deeper_head-5_he_concat-False_lr-0.003_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-115_shuffle_every-5_train_size-2400_13_0.91_test.csv',
#         ]
# seeds_text = 'deep-2_data-7_ep-19_th-0.95_deep-5_data-7_ep-14-th-0.91'


# seeds = [
#     'deeper_head-5_he_concat-False_lr-0.001_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-125_shuffle_every-5_train_size-3600_29_0.94_test.csv',
#     'deeper_head-5_he_concat-False_lr-0.001_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-126_shuffle_every-5_train_size-3600_27_0.94_test.csv'
#     ]
# seeds_text = 'data-7_ep-29_-th-0.94_data-8_ep-27_th-0.94'

# 42, 77
# seeds = [
#     'seed_42_test2_52_0.9_test.csv',
#     'seed_77_test2_58_0.9_test.csv'
# ]
# seeds_text = 'seeds_42_0.9_77_0.9_test2'9

# seeds = [
#     'seed_115_test2_15_0.9_test.csv',
#     'seed_116_test2_20_0.9_test.csv'
# ]
# seeds_text = '115_0.9_116_0.9_test2'

# seeds = [
# 'seed-205_lr-0.003_tsize-3600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.9_pval_map_0.3316_recall_0.7809.csv',
# # 'seed-206_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_48_0.9_val_map_0.3144_recall_0.7963.csv',
# 'seed-212_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_19_0.9_pval_map_0.3295_recall_0.7795.csv',
# # 'seed-213_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_19_0.9_val_map_0.3176_recall_0.7992.csv',
# 'seed-217_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_22_0.9_pval_map_0.3274_recall_0.7704.csv',
# 'seed-221_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.9_pval_map_0.3332_recall_0.7739.csv',
# ]
#
#
# seeds = [
# 'seed-205_lr-0.003_tsize-3600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.95_test.csv',
# # 'seed-206_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_48_0.95_test.csv',
# 'seed-212_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.95_test.csv',
# 'seed-217_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_22_0.95_test.csv',
# 'seed-221_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_21_0.95_test.csv',
# # 'seed-213_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.95_test.csv',
# # 'seed-214_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_23_0.95_test.csv',
# ]
#
# #
# seeds = [
#     'seed_205_overfit_6_0.95_test.csv',
#     'seed_212_overfit_6_0.95_test.csv',
#     'seed_217_overfit_6_0.95_test.csv',
#     'seed_221_overfit_6_0.95_test.csv'
# ]
#
# seeds = [
#     'seed_205_hem_jam2_6_0.9_test.csv',
#     'seed_212_hem_jam2_6_0.9_test.csv',
#     'seed_217_hem_jam2_6_0.9_test.csv',
#     'seed_221_hem_jam2_6_0.9_test.csv'
# ]


seeds = [
    'seed_205_hem_jam2_hardjam_5_0.9_test.csv',
    'seed_212_hem_jam2_hardjam_5_0.9_test.csv',
    'seed_217_hem_jam2_hardjam_5_0.9_test.csv',
    'seed_221_hem_jam2_hardjam_5_0.9_test.csv'
]


# seed_205_overfit_qjam_6_0.9_test.csv
# seed_212_overfit_qjam_6_0.9_test.csv
# seed_217_overfit_qjam_6_0.9_test.csv
# seed_221_overfit_qjam_6_0.9_test.csv




#
#
# seeds = [
#     'seed-205_lr-0.003_tsize-3600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.9_pval_map_0.3964_recall_0.7426.csv',
#     'seed-212_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_20_0.9_pval_map_0.4157_recall_0.7765.csv',
#     'seed-217_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_22_0.9_pval_map_0.4220_recall_0.7680.csv',
#
#     'seed-221_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7_21_0.9_pval_map_0.4217_recall_0.7758.csv'
# ]
def str2bool(v):
    return v.lower() in ('true')

# control here
parser = argparse.ArgumentParser()

parser.add_argument('--csv', type=str, nargs='+', help='csvs to ensemble')

parser.add_argument('--iou_th', type=float, default=0.12, help='iou threshold')
parser.add_argument('--prob_th', type=float, default=0.959, help='prob threshold')
parser.add_argument('--ov_th', type=float, default=0.8, help='overlap threshold')

parser.add_argument('--mode', type=str, default='test', help='val or test')
parser.add_argument('--csv_th', type=float, default=0.9, help='each threshold')

parser.add_argument('--jam', type=str, default='overfit_qjam', help='which jam')


parser.add_argument('--draw', type=str2bool, default='false', help='draw pred bboxes')
parser.add_argument('--draw_all', type=str2bool, default='false', help='draw all pred bboxes')


pargs = parser.parse_args()


seeds_text = [s[5:8] for s in pargs.csv]
seeds_text = '_'.join(seeds_text)

results = ['result2/{}'.format(s) for s in pargs.csv]

if pargs.mode == 'test':
    # DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_train_images'
    DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_test_images'
else:
    DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle/stage_2_train_images'



import collections
save_dict = collections.OrderedDict()
save_dict['seed'] = seeds_text
save_dict['jam'] = pargs.jam
save_dict['iou_th'] = pargs.iou_th
save_dict['prob_th'] = pargs.prob_th
save_dict['ov_th'] = pargs.ov_th
save_dict['csv_th'] = pargs.csv_th
save_dict['mode'] = pargs.mode


save_dir = ['{}-{}'.format(key, save_dict[key]) for key in save_dict.keys()]
save_dir = '_'.join(save_dir)
print(save_dir)


# Merge

merge_result = pd.DataFrame()

pids = pd.read_csv(results[0]).sort_values(by=['patientId']).reset_index()['patientId']
merge_result['patientId'] = pids

for i, s in enumerate(seeds):
    temp = pd.read_csv(results[i]).sort_values(by=['patientId'])
    temp = temp.reset_index()
    merge_result['PredictionString_' + str(s)] = temp['PredictionString']

merge_result.head()


# In[6]:


def getCoords(box):  # x1,y1, w,h
    x1 = float(box[0])
    x2 = float(box[0]) + float(box[2])
    y1 = float(box[1])
    y2 = float(box[1]) + float(box[3])
    return x1, x2, y1, y2

def compute_overlaps(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    i_left = max(x11, x21)
    i_top = max(y11, y21)
    i_right = min(x12, x22)
    i_bottom = min(y12, y22)

    if i_right < i_left or i_bottom < i_top:
        return 0.0

    intersect_area = (i_right - i_left) * (i_bottom - i_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    overlap1 = intersect_area / box1_area
    overlap2 = intersect_area / box2_area

    return max(overlap1, overlap2)

def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    i_left = max(x11, x21)
    i_top = max(y11, y21)
    i_right = min(x12, x22)
    i_bottom = min(y12, y22)

    if i_right < i_left or i_bottom < i_top:
        return 0.0

    intersect_area = (i_right - i_left) * (i_bottom - i_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

def parse_detection(s):
    dets = []
    if not pd.isnull(s):
    # if s is not np.nan:
        s = s.split(' ')[1:]
        num_dets = int(len(s) / 5)
        for i in range(num_dets):
            temp_det = s[5 * i: 5 * (i + 1)]
            temp_det = [float(bb) for bb in temp_det]
            dets.append(temp_det)

    return dets

def unparse_detection(s):
    if len(s) == 0:
        return np.nan
    else:
        s = [str(item) for sublist in s for item in sublist]
        out = ' ' + ' '.join(s)
        return out


# two model case
def get_ensemble_dets(s1, s2, mode='union', iou_th=pargs.iou_th,
                      prob_th=pargs.prob_th, size_th=10000, overlap_th=pargs.ov_th):
    used = []
    out = s1 + s2
    for ss1 in s1:
        if ss1 in used:
            continue
        for ss2 in s2:
            if ss2 in used:
                continue

            iou = computeIOU(ss1[1:], ss2[1:])
            ov = compute_overlaps(ss1[1:], ss2[1:])

            if iou > iou_th:

                merged = [(a1 + a2) / 2 for a1, a2 in zip(ss1, ss2)]

                try:
                    out.remove(ss1)
                    out.remove(ss2)
                except ValueError:
                    pass

                out.append(merged)
                used.append(ss1)
                used.append(ss2)

            elif ov > overlap_th:
                idx = float(ss1[0]) < float(ss2[0])
                merged = [ss1, ss2][idx]

                try:
                    out.remove(ss1)
                    out.remove(ss2)
                except ValueError:
                    pass

                out.append(merged)
                used.append(ss1)
                used.append(ss2)

    out = [o for o in out if o[0] > prob_th]
    return out


# In[13]:


# two case
def ensemble_two(x, s1, s2):
    s1 = parse_detection(x['PredictionString_{}'.format(s1)])
    s2 = parse_detection(x['PredictionString_{}'.format(s2)])
    edets = get_ensemble_dets(s1, s2)
    edets = unparse_detection(edets)
    return edets


# In[14]:


# multiple case
def ensemble(x, slist):
    s0 = parse_detection(x['PredictionString_{}'.format(slist[0])])
    s1 = parse_detection(x['PredictionString_{}'.format(slist[1])])
    edets = get_ensemble_dets(s0, s1)
    #     print(edets)
    if len(slist) > 2:
        for _s in slist[2:]:
            _s = parse_detection(x['PredictionString_{}'.format(_s)])
            edets = get_ensemble_dets(edets, _s)

    edets = unparse_detection(edets)

    return edets

from mrcnn.utils import iou

def check_boxes(boxes):
    if pd.isnull(boxes):
        return boxes

    boxes = parse_detection(boxes)
    boxes_left = np.copy(boxes)
    median_boxes = []
    while boxes_left.shape[0] > 0:
        maximum_index = np.argmax(boxes_left[:, 0])
        maximum_box = np.copy(boxes_left[maximum_index])
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
        if boxes_left.shape[0] == 0:
            median_boxes.append(maximum_box)
            break
        similarities = iou(boxes_left[:, 1:], np.expand_dims(maximum_box[1:], 0))
        grouped = boxes_left[similarities >= 0.2]
        boxes_left = boxes_left[similarities < 0.2]
        grouped = np.concatenate([maximum_box[np.newaxis, ...], grouped], axis=0)
        median_box = np.median(grouped, axis=0)
        median_boxes.append(median_box)

    decoded = unparse_detection(np.array(median_boxes))


    return decoded
# In[16]:



merge_result_2 = merge_result.copy()
merge_result_2['PredictionString'] = merge_result.apply(lambda x: ensemble(x, seeds), axis=1)


merge_result_2['PredictionString'] = merge_result_2.apply(lambda x: check_boxes(x['PredictionString']), axis=1)


print(merge_result_2.head())

# In[16]:


# seed_42_77 = merge_result[['patientId', 'PredictionString']]
# seed_42_77.to_csv('ensemble2seed_42_77.csv', index=False)


# In[17]:


# fn = fn + '_0.2_0.95_10000'
print(save_dir)
ensemble_csv = merge_result_2[['patientId', 'PredictionString']]
ensemble_csv.to_csv('ensemble2/{}.csv'.format(save_dir), index=False)
print('Ensemble file saved!')

# In[18]:


# draw image


# In[19]:


# DATA_DIR = '/notebooks/minki/rsna/data/kaggle'


# In[18]:


if not os.path.exists('images2/' + save_dir):
    os.makedirs('images2/' + save_dir)

# In[20]:


# gt = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
ensemble = pd.read_csv('ensemble2/' + save_dir + '.csv')

preds = [pd.read_csv(r) for r in results]

# pred2 = pd.read_csv('result/' + 'seed_42_test_0.9' + '.csv')
# pred3 = pd.read_csv('result/' + 'seed_77_test_0.9' + '.csv')

# val = pd.read_csv('test2_(1)_Cls(PT_v5_T0.15)_FPR_T0.4_Result.csv')


# In[21]:


# ensemble['PredictionString'][1].split(' ')


# In[22]:

def add_gt_bbox(ax, pid, c='red'):
    # print(c)
    gt = pd.read_csv('/home/minki/ramdisk/rsna/data/kaggle/stage_2_train_labels.csv')

    gt_bboxes = gt[gt['patientId'] == pid].iloc[:, 1:-1]
    for i in range(len(gt_bboxes)):
        prob, x, y, w, h = 1.0, *gt_bboxes.iloc[i, :]
        if x is not np.nan:
            x, y, w, h = float(x), float(y), float(w), float(h)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=c, facecolor='none')
            text = plt.text(x, y - 6.0, str(prob), color=c, fontsize='xx-large')

            ax.add_patch(rect)


def add_bbox(pred, ax, pid, c='red', ensemble=False):

    pred_bbox_string = pred[pred['patientId'] == pid]['PredictionString'].iloc[0]
    if pred_bbox_string is not np.nan:

        if ensemble:
            pred_bbox_string = pred_bbox_string.split(' ')[1:]
        else:
            pred_bbox_string = pred_bbox_string.split(' ')[1:]
        num_pred_bbox_string = int(len(pred_bbox_string) / 5)

        pred_bboxes = []
        for pi in range(num_pred_bbox_string):
            pb = pred_bbox_string[5 * pi:5 * (pi + 1)]
            pred_bboxes.append(pb)

        # Create a Rectangle patch
        for pb in pred_bboxes:

            prob, x, y, w, h = float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3]), float(pb[4])

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=c, facecolor='none')
            text = plt.text(x, y - 6.0, str(prob), color=c, fontsize='xx-large')

            ax.add_patch(rect)


# In[24]:

if pargs.draw:
    for pid in tqdm(ensemble['patientId'][:]):
        # fp = '/home/minki/rsna/data/kaggle/stage_1_test_images/' + pid + '.png'
        fp = os.path.join(DATA_DIR, pid+'.png')

        png = Image.open(fp)
        image = np.array(png)
        #     ds = pydicom.read_file(fp)
        #     image = ds.pixel_array

        fig, ax = plt.subplots(1, figsize=(8, 8))

        # Display the image
        ax.imshow(image, cmap='gray')

        colors = ['orange', 'yellow', 'blue', 'purple', 'brown', 'gray']
        if pargs.draw_all:
            for i, p in enumerate(preds):
                add_bbox(p, ax, pid, colors[i])

        if pargs.mode == 'val':
            add_gt_bbox(ax, pid, 'green')
        add_bbox(ensemble, ax, pid, 'red', True)

        fig.savefig('images2/' + save_dir + '/' + pid + '.png')


