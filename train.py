# fixme:
"""
retrain
remove datasaving history

"""


#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
# import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob

from sklearn import preprocessing
from PIL import Image
from scipy import stats

import argparse
import pickle
import warnings
from helper import *


def str2bool(v):
    return v.lower() in ('true')

# control here
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='7', help='gpu id')
parser.add_argument('--train_size', '--ts', type=int, default=1600)
parser.add_argument('--val_size', '--vs', type=int, default=400)

parser.add_argument('--batch_size', '--bs', type=int, default=8)

parser.add_argument('--seed', '--s', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

# becareful with bn, deeper_head, he
parser.add_argument('--option', type=str, default='standard', help='standard, bn, he, shuffle_5, deeper_head, ignoreseg, woong, he')

parser.add_argument('--bn', '--batch_norm', type=str2bool, default=True)
parser.add_argument('--shuffle_epoch', '--se', type=int, default=5)
parser.add_argument('--deeper_head', '--deep', type=int, default=5)
parser.add_argument('--he_concat', '--he', type=str2bool, default=False)
parser.add_argument('--ignore_seg', '--igseg', type=str2bool, default=True)
parser.add_argument('--cls_reg_weight_ratio', '--crwr', type=float, default=2.0)


parser.add_argument('--head_init', '--hinit', type=str, default='glorot_uniform')
parser.add_argument('--rpn_roi_iou_pos_th', '--ioupth', type=float, default=0.6)
parser.add_argument('--rpn_roi_iou_neg_th', '--iounth', type=float, default=0.4)
parser.add_argument('--anc_iou_pos_th', '--aioupth', type=float, default=0.7)
parser.add_argument('--anc_iou_neg_th', '--aiounth', type=float, default=0.3)

parser.add_argument('--woong_data', '--wd', type=str2bool, default=False)
parser.add_argument('--use_all_data', '--uad', type=str2bool, default=False)
parser.add_argument('--use_clean_data', '--ucd', type=str2bool, default=False)



parser.add_argument('--max_epoch', '--mep', type=int, default=50)
parser.add_argument('--load_rpn_head', '--lrpn', type=str2bool, default=True)
parser.add_argument('--rpn_anchor_ratios', '--rpnar', type=int, default=3)


parser.add_argument('--ucd_n', '--ucdn', type=str, default='7')


parser.add_argument('--valset', '--val', type=int, default=1)



# parser.add_argument('--use_42_pretrain', '--42pre', type=str2bool, default=False)


parser.add_argument('--is_train', type=str2bool, default=True)

parser.add_argument('--submit', type=str2bool, default=False, help='test set')
parser.add_argument('--pval', type=str2bool, default=False, help='pval')

parser.add_argument('--submit_confth', '--sct' ,type=float, default=0.95, help='conf th')
parser.add_argument('--inference_epoch', '--iep', type=int, default=0)
parser.add_argument('--do_just', type=int, default=0)
parser.add_argument('--check_phase1', type=str2bool, default=False)


parser.add_argument('--nms', type=str, default='nms', help='or NMW')




parser.add_argument('--image_size', '--ims', type=int, default=512)







pargs = parser.parse_args()

import collections
save_dict = collections.OrderedDict()
save_dict['seed'] = pargs.seed
save_dict['lr'] = pargs.lr
save_dict['tsize'] = pargs.train_size
save_dict['anchors'] =   pargs.rpn_anchor_ratios
save_dict['loadfpn'] = pargs.load_rpn_head
save_dict['deep'] = pargs.deeper_head
save_dict['regcls'] = pargs.cls_reg_weight_ratio
save_dict['roiiou'] = str(pargs.rpn_roi_iou_pos_th) + '_' + str(pargs.rpn_roi_iou_neg_th)
save_dict['clean'] = str(pargs.use_clean_data) + '_' + str(pargs.ucd_n)

save_dir = ['{}-{}'.format(key, save_dict[key]) for key in save_dict.keys()]
save_dir = '_'.join(save_dir)

save_dir = save_dir.replace('True', 'true').replace('False', 'false')
print(save_dir)


# In[9]:

# old version  seed 115, 116
# save_dict = {'train_size': pargs.train_size,  # 1600
#              'seed': pargs.seed,  # do not change this
#              'lr': pargs.lr,
#              'shuffle_every': pargs.shuffle_epoch,  # 'add_HE_channel', 'shuffle_every_5', 'deeper_head'
#              'he_concat': pargs.he_concat,
#              'deeper_head': pargs.deeper_head,  #### 없을 수도
#              'reg_cls_weight': pargs.cls_reg_weight_ratio,
#              'pos_iou': pargs.rpn_roi_iou_pos_th,
#              'neg_iou': pargs.rpn_roi_iou_neg_th,
#              }
#
# save_dir = ['{}-{}'.format(key, save_dict[key]) for key in sorted(save_dict.keys())]
# save_dir = save_dir.replace('True', 'true').replace('False', 'false')
#
# print(save_dir)





# In[10]:


DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle'
ROOT_DIR = '/home/minki/rsna/Mask_RCNN'

# In[11]:


# train_png_dir = os.path.join(DATA_DIR, 'stage_1_train_images_HE_woong')
# def get_png_fps(png_dir):
#     png_fps = glob.glob(png_dir + '/' + '*.png')
#     return list(set(png_fps))
# t_fps = get_png_fps(train_png_dir)
# mm = []
# for i in tqdm(t_fps):
#     m = np.mean(np.array(Image.open(i)))
#     mm.append(m)
# print(np.mean(mm))


# In[12]:


# change here

os.environ["CUDA_VISIBLE_DEVICES"] = pargs.gpu

# In[13]:


# Import Mask RCNN
# sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# In[14]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# In[15]:


### test phase 1
train_png_dir = os.path.join(DATA_DIR, 'stage_2_train_images')
# test_png_dir = os.path.join(DATA_DIR, 'stage_1_test_images')

## test phase 2
test_png_dir = os.path.join(DATA_DIR, 'stage_2_test_images')

# In[16]:


# !wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/premask_rcnn_coco.h5

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"


# In[17]:


def get_png_fps(png_dir):
    png_fps = glob.glob(png_dir + '/' + '*.png')
    return list(set(png_fps))


def parse_dataset(dicom_dir, anns):
    image_fps = get_png_fps(dicom_dir)

    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.png')
        try:
            image_annotations[fp].append(row)
        except KeyError:
            pass
    return image_fps, image_annotations


# In[18]:


# The following parameters have been selected to reduce running time for demonstration purposes
# These are not optimal

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = save_dir

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = pargs.batch_size

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = pargs.image_size
    IMAGE_MAX_DIM = pargs.image_size

    # if IMAGE_MAX_DIM == 1024:
    #     IMAGE_RESIZE_MODE = "resample"


    BACKBONE = 'resnet101'
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (32, 64, 96, 128, 256)

    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

    # fixme:
    STEPS_PER_EPOCH = int(pargs.train_size / pargs.batch_size)
    VALIDATION_STEPS = int(pargs.val_size / pargs.batch_size)

    LEARNING_RATE = pargs.lr
    TRAIN_BN = pargs.bn  # False

    # if args['option'] == 'use_ztrans':
    #     MEAN_PIXEL = np.array([0.0, 0.0, 0.0])

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }


    ######### Modification ############################################################################

    if pargs.rpn_anchor_ratios == 4:
        RPN_ANCHOR_RATIOS = [0.33, 0.5, 1, 2]

    DEEPER_HEAD = pargs.deeper_head  #2
    SHUFFLE_EPOCH = pargs.shuffle_epoch #5


    HEAD_INIT = pargs.head_init #glorot

    if pargs.ignore_seg:
        LOSS_WEIGHTS['mrcnn_mask_loss'] = 0


    LOSS_WEIGHTS["mrcnn_class_loss"] *= pargs.cls_reg_weight_ratio  #1
    print(LOSS_WEIGHTS)

    NMS_OPTION = pargs.nms  # nms
    if NMS_OPTION != 'nms':  # fixme
        DETECTION_MAX_INSTANCES = 10
        DETECTION_MIN_CONFIDENCE = 0.1  ## match target distribution
        DETECTION_NMS_THRESHOLD = 0.9

        DETECTION_MAX_INSTANCES_LATER = 3
        DETECTION_MIN_CONFIDENCE_LATER = 0.78  ## match target distribution
        DETECTION_NMS_THRESHOLD_LATER = 0.01


    RPN_ROI_IOU_POS_THRESHOLD = pargs.rpn_roi_iou_pos_th # 0.5
    RPN_ROI_IOU_NEG_THRESHOLD = pargs.rpn_roi_iou_neg_th # 0.5
    ANC_IOU_POS_THRESHOLD = pargs.anc_iou_pos_th  # 0.5
    ANC_IOU_NEG_THRESHOLD = pargs.anc_iou_neg_th  # 0.5

    ###################################################################################################


config = DetectorConfig()
config.display()


# In[19]:


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        png = Image.open(fp)
        image = np.array(png)

        # if 'use_ztrans' in args['option']:
        #     image = stats.zscore(image, axis=None)

        if pargs.he_concat:
            fp2 = fp.replace("images", "images_HE")
            png2 = Image.open(fp2)
            image2 = np.array(png2)
            image = np.stack((image, image, image2), -1)

            # If grayscale. Convert to RGB for consistency.
        elif len(image.shape) != 3 or image.shape[2] != 3:

            image = np.stack((image,) * 3, -1)


        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:

            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1

        return mask.astype(np.bool), class_ids.astype(np.int32)


# In[20]:


# training dataset
# anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_2_train_labels.csv'))






# print(anns.head())

# In[21]:


image_fps, image_annotations = parse_dataset(train_png_dir, anns=anns)

# In[ ]:


# In[22]:


# image_fps = sorted(image_fps)
#
# png = Image.open(image_fps[0])
# image = np.array(png)
# print(image_fps[0])
# /home/minki/rsna/data/kaggle/stage_1_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.png


# In[23]:


# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

# In[24]:


# image_fps_list = list(image_fps)
# np.random.seed(42)  # 42
# np.random.shuffle(sorted(image_fps_list))


select_train = ['train_v1_normal.npy',
                'train_v1_abnormal.npy',
                'train_v1_pnuemo.npy']

select_val1 = ['valset_v1_normal.npy',
               'valset_v1_abnormal.npy',
               'valset_v1_pnuemo.npy']
select_val2 = ['valset_v2_normal.npy',
               'valset_v2_abnormal.npy',
               'valset_v2_pnuemo.npy']
select_val3 = ['valset_v3_normal.npy',
               'valset_v3_abnormal.npy',
               'valset_v3_pnuemo.npy']


if pargs.valset == 1:
    select_val = select_val1
    select_train.extend(select_val2)
    test_val = select_val3
elif pargs.valset == 2:
    select_val = select_val2
    select_train.extend(select_val3)
    test_val = select_val1
elif pargs.valset == 3:
    select_val = select_val3
    select_train.extend(select_val1)
    test_val = select_val2




# select_train = ['valset_v{}_normal.npy'.format(pargs.valset),
#                 'valset_v{}_abnormal.npy'.format(pargs.valset),
#                 'valset_v{}_pneumo.npy'.format(pargs.valset)]

train_dirs = []
for npy in select_train:
    temp = list(np.load(DATA_DIR + '/valset_P2/' + npy))
    train_dirs.extend(temp)
image_fps_train = [DATA_DIR + '/stage_2_train_images/' + i + '.png' for i in train_dirs]




val_dirs = []
for npy in select_val:
    temp = list(np.load(DATA_DIR + '/valset_P2/' + npy))
    val_dirs.extend(temp)
image_fps_val = [DATA_DIR + '/stage_2_train_images/' + i + '.png' for i in val_dirs]

print('train_images: ', len(train_dirs), 'val_images: ', len(val_dirs))


val_test_dirs = []
for npy in test_val:
    temp = list(np.load(DATA_DIR + '/valset_P2/' + npy))
    val_test_dirs.extend(temp)
image_fps_val_test = [DATA_DIR + '/stage_2_train_images/' + i + '.png' for i in val_test_dirs]





temp_val_dirs = []
temp_vals = select_val1 + select_val2 + select_val3
for npy in temp_vals:
    temp = list(np.load(DATA_DIR + '/valset_P2/' + npy))
    temp_val_dirs.extend(temp)
temp_fps_val = [DATA_DIR + '/stage_2_train_images/' + i + '.png' for i in temp_val_dirs]

print('tot_val_images: ', len(temp_val_dirs))



# print(image_fps_list[0])
# /home/minki/rsna/data/kaggle/stage_1_train_images/b1ebd57e-6037-42a0-9e31-612968c4ff92.png


# In[25]:


# if 'use_89' in args['option']:
#     image_fps_list = [i for i in image_fps_list if (i[61] == '8') or (i[61] == '9')]
# print(len(image_fps_list))

# In[38]:


# val_image_ids = np.load('npy/seed_42_val.npy')
# image_fps_val = [os.path.join(DATA_DIR, 'stage_1_train_images/{}.png'.format(id)) for id in val_image_ids]
#
# image_fps_train = list(set(image_fps_list) - set(image_fps_val))
#
# np.random.seed(pargs.seed)  # 42
# np.random.shuffle(sorted(image_fps_train))
#
# print(len(image_fps_train), len(image_fps_val))
#
# print(image_fps_val[0])
# /home/minki/rsna/data/kaggle/stage_1_train_images/ff0b66d5-ef14-45c4-8bd5-3282f45c163c.png

# x = anns[anns['Target'] == 1]['patientId']
# x = np.unique(x)
# np.save('npy2/train_pneumonia_ids.npy', x)

#
# if pargs.woong_data:
#     woong_data = get_png_fps(os.path.join(DATA_DIR, 'Drawing_GoodBBox'))
#     woong_ids = [w.split('_GoodBBox/')[1][:-4] for w in woong_data]
#     woong_images = ['/home/minki/ramdisk/rsna/data/kaggle/stage_1_train_images/{}.png'.format(id) for id in
#                     woong_ids]
#
#     pneumonia_train_ids = list(np.load('npy2/train_pneumonia_ids.npy'))
#     pneumonia_train_images = ['/home/minki/ramdisk/rsna/data/kaggle/stage_1_train_images/{}.png'.
#                                   format(id) for id in pneumonia_train_ids]
#
#     image_fps_list = list(set(image_fps_list) - set(pneumonia_train_images)) + woong_images
#     image_fps_list = sorted(image_fps_list)
#
#     np.random.seed(42)  # 42
#     np.random.shuffle(image_fps_list)
#
#
#     val_size = 1500
#     image_fps_val = image_fps_list[:val_size]
#     image_fps_train = image_fps_list[val_size:]
#     print(image_fps_train[0])
#     print(image_fps_val[0])
    #/home/minki/ramdisk/rsna/data/kaggle/stage_1_train_images/ddf67528-4268-4315-95c6-6f1560ef536f.png
    #/home/minki/ramdisk/rsna/data/kaggle/stage_1_train_images/c9c40af5-bf20-4889-9eb1-9ad3e9ec3413.png

# else:
#     val_size = 1500
#     image_fps_val = image_fps_list[:val_size]
#     image_fps_train = image_fps_list[val_size:]
#     print(image_fps_train[0])
#     print(image_fps_val[0])


# pneumo_val_image_ids = np.load('npy/list_sameChest14_pneumo.npy')[:, 0]
# image_fps_val = [os.path.join(DATA_DIR, 'stage_1_train_images/{}.png'.format(id)) for id in pneumo_val_image_ids]
# image_fps_train = sorted(list(set(image_fps_train) - set(image_fps_val)))
# image_fps_val = sorted(list(set(image_fps_val + image_fps_val)))



########################

#
# if pargs.use_clean_data:
#     image_fps_train_ids = np.load(DATA_DIR+ '/clean/train_clean_v{}_pnuemo.npy'.format(pargs.ucd_n))
#     image_fps_val_ids = np.load(DATA_DIR+ '/clean/valset_clean_v{}_pnuemo.npy'.format(pargs.ucd_n))
#
#     image_fps_train = set([os.path.join(DATA_DIR, 'stage_1_train_images/{}.png'.format(id)) for id in
#                        image_fps_train_ids])
#     image_fps_train = sorted(list(image_fps_train))
#     image_fps_val = set([os.path.join(DATA_DIR, 'stage_1_train_images/{}.png'.format(id)) for id in
#                        image_fps_val_ids])
#     image_fps_val = sorted(list(image_fps_val))
#
#     config.VALIDATION_STEPS = int(len(image_fps_val) / pargs.batch_size)


config.VALIDATION_STEPS = int(len(image_fps_val) / pargs.batch_size)

#####################
if pargs.use_all_data:
    config.STEPS_PER_EPOCH = int(len(image_fps_train) / pargs.batch_size)


np.random.seed(pargs.seed)  # 42
np.random.shuffle(sorted(image_fps_train))

print('data distribution')
print(len(image_fps_train), len(image_fps_val))


# image_fps_val = glob.glob('/home/minki/ramdisk/rsna/data/kaggle/stage_1_test_images/*.png')
# pneumonia_val_images = []
# for img in tqdm(image_fps_val):
# #     print(img.split('/stage_1_test_images/')[1][:-4])
#     if anns[anns['patientId'] == img.split('/stage_2_test_images/')[1][:-4]].iloc[0]['Target'] == 1:
#         pneumonia_val_images.append(img)
#
# print('pneumonia images in valset: ', len(pneumonia_val_images))
#







# In[39]:


# val_size = 1500
# image_fps_val = image_fps_list[:val_size]
# image_fps_train = image_fps_list[val_size:]

# print(len(image_fps_train), len(image_fps_val))


# In[43]:


# image_fps_val[0].split('/stage_1_train_images/')[1][:-4]

# In[44]:



pneumonia_val_images = []
for img in tqdm(image_fps_val):
    if anns[anns['patientId'] == img.split('/stage_2_train_images/')[1][:-4]].iloc[0]['Target'] == 1:
        pneumonia_val_images.append(img)

print('pneumonia images in valset: ', len(pneumonia_val_images))
# pneumonia_val_images[:5]

# 619


# In[45]:


print(save_dir)

# In[46]:


# image_fps_train[0][54:62]

# In[26]:


# image_fps_train_pids = [d[61:-4] for d in image_fps_train]
# train_array = np.array(image_fps_train_pids)
# np.save('npy/{}_train'.format(save_dir), train_array)

# image_fps_val_pids = [d[61:-4] for d in image_fps_val]
# val_array = np.array(image_fps_val_pids)
# np.save('npy/{}_val'.format(save_dir), val_array)


# In[ ]:


# In[47]:


# prepare the training dataset
# image_fps_trains = [image_fps_train[pargs.train_size*i:pargs.train_size*(i+1)]
#                     for i in range(int(len(image_fps_train)/pargs.train_size))]
# dataset_train_0 = DetectorDataset(image_fps_trains[0], image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_train_0.prepare()
# dataset_train_1 = DetectorDataset(image_fps_trains[1], image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_train_1.prepare()
# dataset_train_2 = DetectorDataset(image_fps_trains[2], image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_train_2.prepare()
# dataset_train_3 = DetectorDataset(image_fps_trains[3], image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_train_3.prepare()




dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# In[48]:


# Show annotation(s) for a DICOM image
# test_fp = np.random.choice(image_fps_train)
# print(image_annotations[test_fp])

# In[49]:


# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

# In[50]:


# Load and display random sample and their bounding boxes

# class_ids = [0]
# while class_ids[0] == 0:  ## look for a mask
#     image_id = random.choice(dataset_train.image_ids)
#     image_fp = dataset_train.image_reference(image_id)
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#
# print(image.shape)
#
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# masked = np.zeros(image.shape[:2])
# for i in range(mask.shape[2]):
#     masked += image[:, :, 0] * mask[:, :, i]
# plt.imshow(masked, cmap='gray')
# plt.axis('off')
#
# print(image_fp)
# print(class_ids)

# In[51]:


# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.Fliplr(0.1),
    iaa.OneOf([  ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.01, 0.025)),
    ]),
    iaa.OneOf([  ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([  ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# test on the same image as above
# imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
# plt.figure(figsize=(30, 12))
# _ = plt.imshow(imggrid[:, :, 0], cmap='gray')

# In[ ]:


# In[37]:




# In[32]:


# save_dir

# In[ ]:


# In[52]:



# In[53]:




# In[ ]:


# In[ ]:


# In[ ]:


# In[31]:


if pargs.is_train:
    warnings.filterwarnings("ignore")

    if pargs.deeper_head > 100:  # training only deeper head
        pass
        # model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR + '/ckpt')
        #
        # print('Using seed 42 pretrained weights')
        # weights_42 = 'ckpt/seed_42/mask_rcnn_seed_42_0052.h5'
        #
        # model.load_weights(weights_42, by_name=True)
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE * 2, epochs=2, layers='heads', augmentation=None)  ## no need to augment yet
        # history = model.keras_model.history.history
        #
        # # In[34]:
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE, epochs=6, layers='heads', augmentation=augmentation)
        # new_history = model.keras_model.history.history
        # for k in new_history: history[k] = history[k] + new_history[k]
        # # In[35]:
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 5, epochs=16, layers='heads', augmentation=augmentation)
        # new_history = model.keras_model.history.history
        # for k in new_history: history[k] = history[k] + new_history[k]
        #
        # # Store data (serialize)
        # with open('history2/{}.pickle'.format(save_dir), 'wb') as f:
        #     pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10, epochs=40, layers='heads', augmentation=augmentation)
        # new_history = model.keras_model.history.history
        # for k in new_history: history[k] = history[k] + new_history[k]
        #
        # # Store data (serialize)
        # with open('history2/{}.pickle'.format(save_dir), 'wb') as f:
        #     pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

        ##### temporary code to retrain:
        # print('working')
        # model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR + '/ckpt')
        # latest_epoch = 15
        #
        # model_path = '/home/minki/rsna/Mask_RCNN/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir,
        #                                                                               latest_epoch + 1)
        # model_path = model_path.replace('False', 'false')
        # model_path = model_path.replace('True', 'true')
        # model.load_weights(model_path, by_name=True)
        # model.epoch = latest_epoch + 1
        #
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10, epochs=pargs.max_epoch, layers='all', augmentation=augmentation)
        # history = model.keras_model.history.history
        #
        # # Store data (serialize)
        # with open('history2/{}.pickle'.format(save_dir), 'wb') as f:
        #     pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # epochs = range(1, len(next(iter(history.values()))) + 1)
        # print(pd.DataFrame(history, index=epochs))
        #
        # fix_val_loss = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
        # best_epoch = np.argmin(fix_val_loss)
        #
        # # best_epoch = np.argmin(history["val_loss"])
        # print("Best Epoch:", best_epoch + 1)  # +15


    else: # default
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR + '/ckpt')

        # Exclude the last layers because they require a matching
        # number of classes
        print('Using coco pretrained weights')
        if pargs.load_rpn_head:
            model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask",
            ])


        # import h5py
        # f = h5py.File(COCO_WEIGHTS_PATH, 'r')
        # print(list(f.keys()))

        else:
            model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
                'mrcnn_bbox',
                'mrcnn_bbox_fc',
                'mrcnn_bbox_loss',
                'mrcnn_class',
                'mrcnn_class_bn1',
                'mrcnn_class_bn2',
                'mrcnn_class_conv1',
                'mrcnn_class_conv2',
                'mrcnn_class_logits',
                'mrcnn_class_loss',
                'mrcnn_mask',
                'mrcnn_mask_bn1',
                'mrcnn_mask_bn2',
                'mrcnn_mask_bn3',
                'mrcnn_mask_bn4',
                'mrcnn_mask_conv1',
                'mrcnn_mask_conv2',
                'mrcnn_mask_conv3',
                'mrcnn_mask_conv4',
                'mrcnn_mask_deconv',
                'mrcnn_mask_loss',
                'rpn_bbox',
                'rpn_bbox_loss',
                'rpn_class',
                'rpn_class_logits',
                'rpn_class_loss',
                'rpn_model',
            ])

        # In[33]:



        print(vars(pargs))
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE * 2, epochs=4, layers='heads', augmentation=None)  ## no need to augment yet

        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE, epochs=pargs.max_epoch, layers='all', augmentation=augmentation)
        #
        # print(vars(pargs))
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 5, epochs=16, layers='all', augmentation=augmentation)

        print(vars(pargs))
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE, epochs=pargs.max_epoch, layers='all', augmentation=augmentation)
        print(vars(pargs))

        # fixme:
        try:
            print('\n\n', save_dir.replace('False', 'false'))
            # history = pd.read_csv('/home/minki/rsna/Mask_RCNN/ckpt/'+save_dir.replace('False', 'false') + '/training_log.csv')
            history = pd.read_csv(
                ROOT_DIR + '/ckpt/' + save_dir.replace('False', 'false') + '/training_log.csv')
            print('history: ', history)

        except FileNotFoundError:
            with open('history2/{}.pickle'.format(save_dir), 'rb') as f:
                history = pickle.load(f)
            epochs = range(16, len(next(iter(history.values()))) + 16)
            print(pd.DataFrame(history, index=epochs))


        fix_val_loss = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
        best_epoch = np.argmin(fix_val_loss)

        # best_epoch = np.argmin(history["val_loss"])
        print(vars(pargs))
        print("Best Epoch:", best_epoch + 1)

        # ##### temporary code to retrain:
        # print('working')
        # model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR + '/ckpt')
        # latest_epoch = 15
        #
        # model_path = '/home/minki/rsna/Mask_RCNN/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir,
        #                                                                               latest_epoch + 1)
        # model_path = model_path.replace('False', 'false')
        # model_path = model_path.replace('True', 'true')
        # model.load_weights(model_path, by_name=True)
        # model.epoch = latest_epoch + 1
        #
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 5, epochs=pargs.max_epoch, layers='all', augmentation=augmentation)
        # history = model.keras_model.history.histor[[y
        #
        # # Store data (serialize)
        # with open('history2/{}.pickle'.format(save_dir), 'wb') as f:
        #     pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

elif pargs.do_just > 0:
    if pargs.do_just == 42:
        best_epoch = 52
    elif pargs.do_just == 7:
        best_epoch = 58
    elif pargs.do_just == 115:
        best_epoch = 14
    elif pargs.do_just == 116:
        best_epoch = 19
else:
    # best_epoch = 15
    # model_path = 'ckpt/deeper_head-2_he_concat-false_lr-0.001_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-100_shuffle_every-5_train_size-2400/mask_rcnn_deeper_head-2_he_concat-false_lr-0.001_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-100_shuffle_every-5_train_size-2400_0016.h5'

    # # fixme
    if pargs.inference_epoch == 0:

        try:
            # history = pd.read_csv('/home/minki/rsna/Mask_RCNN/ckpt/'+save_dir.replace('False', 'false') + '/training_log.csv')
            print('\n\n', save_dir)
            history = pd.read_csv(
                ROOT_DIR + '/ckpt/' + save_dir + '/training_log.csv')
            print('history: ', history)

        except FileNotFoundError:

            with open('history2/{}.pickle'.format(save_dir), 'rb') as f:
                history = pickle.load(f)
            epochs = range(16, len(next(iter(history.values()))) + 16)
            print(pd.DataFrame(history, index=epochs))


        fix_val_loss = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
        best_epoch = np.argmin(fix_val_loss) + 1

        # best_epoch = np.argmin(history["val_loss"])
        print(vars(pargs))
        print("Best Epoch:", best_epoch)
    else:
        best_epoch = pargs.inference_epoch
        print("Best Epoch:", best_epoch)
# In[47]:




# In[ ]:


# # inference

# In[42]:
#'/home/minki/rsna/Mask_RCNN/ckpt/deeper_head-2_he_concat-False_lr-0.001_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-100_shuffle_every-5_train_size-2400/mask_rcnn_deeper_head-2_he_concat-False_lr-0.001_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-100_shuffle_every-5_train_size-2400_0016.h5'

if best_epoch >= 10:
    # model_path = '/home/minki/rsna/Mask_RCNN/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir, best_epoch + 1)
    model_path = ROOT_DIR + '/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir, best_epoch)
else:
    # model_path = '/home/minki/rsna/Mask_RCNN/ckpt/{}/mask_rcnn_{}_000{}.h5'.format(save_dir, save_dir, best_epoch + 1)
    model_path = ROOT_DIR + '/ckpt/{}/mask_rcnn_{}_000{}.h5'.format(save_dir, save_dir, best_epoch)

# model_path = model_path.replace('False', 'false')
# model_path = model_path.replace('True', 'true')
# print('inference: ', model_path)


if pargs.do_just == 42:
    print('Use seed 42 model! 52epoch, 0.5858')
    model_path = '/home/minki/rsna/Mask_RCNN/ckpt/{}/mask_rcnn_{}_00{}.h5'.format('seed_42', 'seed_42', best_epoch)
    save_dir = 'seed_42_test2'
elif pargs.do_just == 77:
    print('Use seed 77 model! 58epoch 0.5775')
    model_path = '/home/minki/rsna/Mask_RCNN/ckpt/{}/mask_rcnn_{}_00{}.h5'.format('seed_77', 'seed_77', best_epoch)
    save_dir = 'seed_77_test2'
elif pargs.do_just == 115:
    print('Use seed 115 model!')
    save_dir = 'deeper_head-5_he_concat-false_lr-0.003_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-115_shuffle_every-5_train_size-2400'
    model_path = ROOT_DIR + '/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir, best_epoch)
    save_dir = 'seed_115_test2'
elif pargs.do_just == 116:
    print('Use seed 116 model!')
    save_dir = 'deeper_head-2_he_concat-false_lr-0.003_neg_iou-0.4_pos_iou-0.6_reg_cls_weight-2.0_seed-116_shuffle_every-5_train_size-2400'
    model_path = ROOT_DIR + '/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir, best_epoch)
    save_dir = 'seed_116_test2'



# In[43]:


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    TRAIN_BN = False

inference_config = InferenceConfig()

# Recreate the model in inference mode
model_i = modellib.MaskRCNN(mode='inference',
                            config=inference_config,
                            model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model_i.load_weights(model_path, by_name=True)


# In[ ]:


# set color for class
# def get_colors_for_class_ids(class_ids):
#     colors = []
#     for class_id in class_ids:
#         if class_id == 1:
#             colors.append((.941, .204, .204))
#     return colors


# In[ ]:


# Show few example of ground truth vs. predictions on the validation dataset
# dataset = dataset_val
# fig = plt.figure(figsize=(10, 30))
#
# for i in range(6):
#
#     image_id = random.choice(dataset.image_ids)
#
#     original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
#                                                                                        image_id, use_mini_mask=False)
#
#     if len(gt_bbox) > 0:
#         print(original_image.shape)
#         plt.subplot(6, 2, 2 * i + 1)
#         visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                                     dataset.class_names,
#                                     colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
#
#         plt.subplot(6, 2, 2 * i + 2)
#         results = model_i.detect([original_image])  # , verbose=1)
#         r = results[0]
#         visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                                     dataset.class_names, r['scores'],
#                                     colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])

# In[ ]:


# Get filenames of test dataset DICOM images
test_image_fps = get_png_fps(test_png_dir)


# In[ ]:


# Fix

# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='/Mask_RCNN/submission.csv', min_conf=0.97):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    # resize_factor = ORIG_SIZE

    result_df = pd.DataFrame(columns=['patientId', 'PredictionString'])

    idx = -1
    for image_id in tqdm(image_fps):
        idx += 1
        png = Image.open(image_id)
        image = np.array(png)
        if pargs.he_concat:
            fp2 = image_id.replace("image", "images_HE")
            png2 = Image.open(fp2)
            image2 = np.array(png2)
            image = np.stack((image, image, image2), -1)

            # If grayscale. Convert to RGB for consistency2.
        elif len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)

        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        results = model_i.detect([image])
        r = results[0]

        # todo: insert ensemble
        # config.DETECTION_NMS_THRESHOLD
        # max_output_size=config.DETECTION_MAX_INSTANCES,

        if len(r['rois']) > 1:
            if pargs.nms != 'nms':
                from mrcnn.utils import iou as calc_iou
                from mrcnn.utils import minmax_iou

                r['scores'] = np.expand_dims(r['scores'], 1)
                r['class_ids'] = np.expand_dims(r['class_ids'], 1)
                predictions = np.concatenate((r['scores'], r['class_ids'], r['rois']), 1)

                if pargs.nms == 'nms_2':
                    boxes_left = np.copy(predictions)
                    maxima = []
                    while boxes_left.shape[0] > 0:
                        maximum_index = np.argmax(boxes_left[:, 0])
                        maximum_box = np.copy(boxes_left[maximum_index])
                        maxima.append(maximum_box)
                        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
                        if boxes_left.shape[0] == 0: break
                        similarities = calc_iou(boxes_left[:, 1:], maximum_box[1:], coords="minmax")
                        boxes_left = boxes_left[similarities <= config.DETECTION_NMS_THRESHOLD_LATER]

                    decoded = np.array(maxima)

                elif pargs.nms == 'median':
                    boxes_left = np.copy(predictions)
                    median_boxes = []
                    while boxes_left.shape[0] > 0:
                        maximum_index = np.argmax(boxes_left[:, 0])
                        maximum_box = np.copy(boxes_left[maximum_index])
                        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
                        if boxes_left.shape[0] == 0:
                            median_boxes.append(maximum_box)
                            break
                        similarities = minmax_iou(boxes_left[:, 2:], np.expand_dims(maximum_box[2:], 0))
                        grouped = boxes_left[similarities >= config.DETECTION_NMS_THRESHOLD_LATER]
                        boxes_left = boxes_left[similarities < config.DETECTION_NMS_THRESHOLD_LATER]
                        grouped = np.concatenate([maximum_box[np.newaxis, ...], grouped], axis = 0)
                        median_box = np.median(grouped, axis=0)
                        median_boxes.append(median_box)

                    decoded = np.array(median_boxes)

                elif pargs.nms == 'wmean':
                    boxes_left = np.copy(predictions)
                    wmean_boxes = []
                    while boxes_left.shape[0] > 0:
                        maximum_index = np.argmax(boxes_left[:, 0])
                        maximum_box = np.copy(boxes_left[maximum_index])
                        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
                        if boxes_left.shape[0] == 0:
                            wmean_boxes.append(maximum_box)
                            break
                        similarities = minmax_iou(boxes_left[:, 2:], np.expand_dims(maximum_box[2:], 0))
                        grouped = boxes_left[similarities >= config.DETECTION_NMS_THRESHOLD_LATER]
                        boxes_left = boxes_left[similarities < config.DETECTION_NMS_THRESHOLD_LATER]
                        grouped = np.concatenate([maximum_box[np.newaxis, ...], grouped], axis = 0)
                        wmean_box = np.average(grouped, weights=grouped[:, 0], axis=0)
                        wmean_boxes.append(wmean_box)

                    decoded = np.array(wmean_boxes)
                    decoded[:, 0] = np.sqrt(decoded[:, 0])
                    # fixme get square root of wmean_boxes

                if decoded.shape[0] > config.DETECTION_MAX_INSTANCES_LATER:
                    decoded = decoded[:config.DETECTION_MAX_INSTANCES_LATER, :]


                r['scores'] = decoded[:, 0]
                r['class_ids'] = decoded[:, 1]
                r['rois'] = decoded[:, 2:]

        ###

        out_str = ""
        #         out_str += patient_id
        #         out_str += ","
        result_df.loc[idx, 'patientId'] = patient_id

        assert (len(r['rois']) == len(r['class_ids']) == len(r['scores']))
        if len(r['rois']) == 0:
            pass
        else:
            num_instances = len(r['rois'])

            for i in range(num_instances):
                if r['scores'][i] > min_conf:
                    out_str += ' '
                    out_str += str(round(r['scores'][i], 2))
                    out_str += ' '

                    # x1, y1, width, height
                    x1 = r['rois'][i][1]
                    y1 = r['rois'][i][0]
                    width = r['rois'][i][3] - x1
                    height = r['rois'][i][2] - y1
                    bboxes_str = "{} {} {} {}".format(x1 * resize_factor, y1 * resize_factor, width * resize_factor,
                                                      height * resize_factor)
                    out_str += bboxes_str

        result_df.loc[idx, 'PredictionString'] = out_str
    #         if idx == 10: break

    #     result_df.to_csv(submission_fp, index=False)
    return result_df




# In[ ]:

#
# # show a few test image detection example
# def _visualize():
#     image_id = random.choice(test_image_fps)
#     ds = pydicom.read_file(image_id)
#
#     # original image
#     image = ds.pixel_array
#
#     # assume square image
#     resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
#
#     # If grayscale. Convert to RGB for consistency.
#     if len(image.shape) != 3 or image.shape[2] != 3:
#         image = np.stack((image,) * 3, -1)
#     resized_image, window, scale, padding, crop = utils.resize_image(
#         image,
#         min_dim=config.IMAGE_MIN_DIM,
#         min_scale=config.IMAGE_MIN_SCALE,
#         max_dim=config.IMAGE_MAX_DIM,
#         mode=config.IMAGE_RESIZE_MODE)
#
#     patient_id = os.path.splitext(os.path.basename(image_id))[0]
#     print(patient_id)
#
#     results = model_i.detect([resized_image])
#     r = results[0]
#     for bbox in r['rois']:
#         print(bbox)
#         x1 = int(bbox[1] * resize_factor)
#         y1 = int(bbox[0] * resize_factor)
#         x2 = int(bbox[3] * resize_factor)
#         y2 = int(bbox[2] * resize_factor)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)
#         width = x2 - x1
#         height = y2 - y1
#         print("x {} y {} h {} w {}".format(x1, y1, width, height))
#     plt.figure()
#     plt.imshow(image, cmap=plt.cm.gist_gray)
#
#
# _visualize()
# _visualize()
# _visualize()
# _visualize()
#





print('Prediction start')
print(vars(pargs))

if pargs.submit:
    submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_test.csv'.
                                 format(best_epoch+1, pargs.submit_confth))
    result_df = predict(test_image_fps, filepath=submission_fp, min_conf=pargs.submit_confth)
    result_df.to_csv(submission_fp, index=False)

elif pargs.pval:
    submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_pval.csv'.
                                 format(best_epoch, 0.9))
    result_df = predict(pneumonia_val_images, filepath=submission_fp, min_conf=0.9)
    result_df.to_csv(submission_fp, index=False)

else:
    submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_val.csv'.
                                 format(best_epoch, 0.9))
    # result_df = predict(pneumonia_val_images, filepath=submission_fp, min_conf=0.9)
    # result_df = predict(image_fps_val_test, filepath=submission_fp, min_conf=0.9)
    # result_df.to_csv(submission_fp, index=False)



    # 23451(train+val2) 1780(val1) 1780(val3)
    # result_df = predict(image_fps_val, filepath=submission_fp, min_conf=0.9)


    submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_val_fpr.csv'.
                                 format(best_epoch, 0.9))
    # result_df = predict(image_fps_val, filepath=submission_fp, min_conf=0.9)
    # result_df = predict(image_fps_train + image_fps_val + image_fps_val_test, filepath=submission_fp, min_conf=0.9)
    result_df = predict(temp_fps_val, filepath=submission_fp, min_conf=0.5)

    result_df.to_csv(submission_fp, index=False)
    print(submission_fp)

# result_df.to_csv(submission_fp, index=False)
# print(submission_fp)

# In[ ]:


# output = pd.read_csv(submission_fp)
# output.head(60)


# In[ ]:



# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:






# In[ ]:

if not pargs.submit:
    print('Start evaluation')


    def minmax_iou(boxes1, boxes2):
        intersection = \
            np.maximum(
                0, np.minimum(boxes1[:, 1], boxes2[:, 1]) \
                   - np.maximum(boxes1[:, 0], boxes2[:, 0])
            ) \
            * np.maximum(
                0, np.minimum(boxes1[:, 3], boxes2[:, 3]) \
                   - np.maximum(boxes1[:, 2], boxes2[:, 2])
            )
        union = (boxes1[:, 1] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 2]) \
                + (boxes2[:, 1] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 2]) \
                - intersection

        return intersection / union


    # def simple_evaluator(
    #         result_csv_path,
    #         gt_csv_path,
    #         iou_thresholds=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
    #         prob_thresholds=[x / 100 for x in range(50, 100)],
    #         return_redefined_df=False
    # ):
    #     gt_df = pd.read_csv(gt_csv_path)
    #     result_df = pd.read_csv(result_csv_path)  # , index_col=0
    #     result_puid = result_df['patientId'].drop_duplicates()
    #     result_df = result_df.dropna()
    #     _gt_num = 0
    #     result_redefined = []
    #
    #     fn_list = []
    #     for each_result_puid in tqdm(result_puid[10:20]):
    #         _this_tp_num = 0
    #         _this_fp_num = 0
    #         _this_dp_num = 0
    #         sub_result_df = result_df[result_df['patientId'] == each_result_puid]
    #         sub_gt_df = gt_df[gt_df['patientId'] == each_result_puid]
    #         sub_gt_scoreboard = np.zeros(len(sub_gt_df))
    #         if len(sub_result_df) != 0:
    #             result_string = sub_result_df['PredictionString'].get_values()[0]
    #             result_string = result_string.split(' ')  # [1:]
    #             _num_of_result = int(len(result_string) / 5)
    #             if len(sub_gt_df) == 1 and sub_gt_df['Target'].get_values() == 0:
    #                 _this_gt_num = 0
    #                 _max_iou = 0
    #                 for i in range(_num_of_result):
    #                     prob = float(result_string[5 * i])
    #                     xmin = float(result_string[5 * i + 1])
    #                     ymin = float(result_string[5 * i + 2])
    #                     xmax = xmin + float(result_string[5 * i + 3])
    #                     ymax = ymin + float(result_string[5 * i + 4])
    #                     result = 'FP1'
    #                     result_redefined.append(
    #                         [each_result_puid,
    #                          prob, xmin, xmax, ymin, ymax,
    #                          result, _max_iou]
    #                     )
    #
    #
    #             else:
    #                 _this_gt_num = len(sub_gt_df)
    #                 gt_bboxes = sub_gt_df.get_values()[:, 1:-1]
    #                 gt_bboxes[:, 2] = gt_bboxes[:, 2] + gt_bboxes[:, 0]
    #                 gt_bboxes[:, 3] = gt_bboxes[:, 3] + gt_bboxes[:, 1]
    #                 gt_bboxes[:, [2, 1]] = gt_bboxes[:, [1, 2]]
    #                 for i in range(_num_of_result):
    #                     prob = float(result_string[5 * i])
    #                     xmin = float(result_string[5 * i + 1])
    #                     ymin = float(result_string[5 * i + 2])
    #                     xmax = xmin + float(result_string[5 * i + 3])
    #                     ymax = ymin + float(result_string[5 * i + 4])
    #                     bbox = np.array([[xmin, xmax, ymin, ymax]])
    #                     _max_iou = np.max(minmax_iou(bbox, gt_bboxes))
    #                     _max_iou_arg = np.argmax(minmax_iou(bbox, gt_bboxes))
    #                     if _max_iou > iou_thresholds[0]:
    #                         if sub_gt_scoreboard[_max_iou_arg] == 0:
    #                             sub_gt_scoreboard[_max_iou_arg] = 1
    #                             _this_tp_num += 1
    #                             result = 'TP'
    #
    #                         else:
    #                             _this_dp_num += 1
    #                             result = 'dp_TP'
    #
    #
    #                     else:
    #                         _this_fp_num += 1
    #                         result = 'FP2'
    #                         # if _max_iou == 0:
    #                         #     result = 'FP1'
    #
    #                     result_redefined.append(
    #                         [each_result_puid,
    #                          prob, xmin, xmax, ymin, ymax,
    #                          result, _max_iou]
    #                     )
    #
    #             # print(each_result_puid)
    #             # print(sub_gt_scoreboard)
    #             if sum(sub_gt_scoreboard) < len(sub_gt_scoreboard):
    #                 fn_list.append(each_result_puid)
    #
    #             # print(gt_bboxes)
    #
    #
    #
    #         else:
    #             _max_iou = 0
    #             _num_of_result = 0
    #             if len(sub_gt_df) == 1 and sub_gt_df['Target'].get_values() == 0:
    #                 _this_gt_num = 0
    #                 _max_iou = 0
    #
    #             else:
    #                 _this_gt_num = len(sub_gt_df)
    #
    #         _gt_num += _this_gt_num
    #         """
    #         print('{} - GT / DETECTED / ASSIGNED / DUPLICATED / FP : {} {} {} {} {}'.format(
    #             each_result_puid,
    #             _this_gt_num, _num_of_result,
    #             _this_tp_num, _this_dp_num, _this_fp_num)
    #         )
    #         """
    #
    #     # print(fn_list)
    #
    #     result_redefined_df = pd.DataFrame(
    #         result_redefined,
    #         columns=['puid', 'prob',
    #                  'xmin', 'xmax', 'ymin', 'ymax',
    #                  'result', 'max_iou']
    #     )
    #     recall_04 = []
    #     mAPs = []
    #
    #     for each_prob_threshold in prob_thresholds:
    #         precisions = []
    #         print('=' * 48, '\nprob_threshold : {:.3f}\n{}'.format(each_prob_threshold, '-' * 48))
    #         sub_df = result_redefined_df[result_redefined_df['prob'] > each_prob_threshold]
    #         for each_iou_threshold in iou_thresholds:
    #             sub_df_tp = sub_df[sub_df['max_iou'] > each_iou_threshold]
    #             sub_df_tp = sub_df_tp[sub_df_tp['result'] == 'TP']
    #             sub_df_fp1 = sub_df[sub_df['result'] == 'FP1']
    #             sub_df_fp2 = sub_df[sub_df['result'] == 'FP2']
    #             precision = len(sub_df_tp) / (_gt_num + len(sub_df_fp1) + len(sub_df_fp2))
    #             precisions.append(precision)
    #             print('iou_threshold : {:.2f} - AP : {:.4f}'.format(each_iou_threshold, precision))
    #             print('recall: ', len(sub_df_tp) / _gt_num)
    #             if each_iou_threshold == 0.40:
    #                 recall_04.append('{:.4f}'.format(len(sub_df_tp) / _gt_num))
    #
    #         mAP = np.mean(precisions)
    #         mAPs.append('{:.4f}'.format(mAP))
    #         print('\nmAP : {:.4f}\n'.format(mAP))
    #
    #     return precisions, mAPs, recall_04, _gt_num, result_redefined_df, fn_list


    mAP, mAPs, recall_04, gt_num, result_df, fn_list = simple_evaluator_temp(
        gt_csv_path='/home/minki/ramdisk/rsna/data/kaggle/stage_2_train_labels.csv',
        result_csv_path=submission_fp
    )

    print(gt_num,
          len(result_df[result_df['result'] == 'TP']),
          len(fn_list),  # len(result_df[result_df['result'] == 'FN']),
          len(result_df[result_df['result'] == 'FP1']),
          len(result_df[result_df['result'] == 'FP2']))



    print('max mAP: ', max(mAPs), 'max recall: ', max(recall_04))
    print('mAPs: ', mAPs[-20:])
    print('recalls: ', recall_04[-20:])
    os.rename(submission_fp, submission_fp[:-4] + '_map_{}_recall_{}.csv'.format(max(mAPs), max(recall_04)))
    submission_fp = submission_fp[:-4] + '_map_{}_recall_{}.csv'.format(max(mAPs), max(recall_04))
    print(submission_fp)

    result_df_path = submission_fp.replace('result2/', 'resultdf/')[:-4] + '.csv'
    result_df.to_csv(result_df_path, index=False)


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[54]:



# In[ ]:




