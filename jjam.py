"""

Final Submission Step 1:

hard example mining

seeds: 205, 212, 217, 221
seed-205_lr-0.003_tsize-3600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7
seed-212_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7
seed-217_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7
seed-221_lr-0.001_tsize-1600_anchors-3_loadfpn-true_deep-5_regcls-2.0_roiiou-0.6_0.4_clean-false_7

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



parser.add_argument('--max_epoch', '--mep', type=int, default=5)
parser.add_argument('--load_rpn_head', '--lrpn', type=str2bool, default=True)
parser.add_argument('--rpn_anchor_ratios', '--rpnar', type=int, default=3)


parser.add_argument('--ucd_n', '--ucdn', type=str, default='7')


parser.add_argument('--valset', '--val', type=int, default=1)



# parser.add_argument('--use_42_pretrain', '--42pre', type=str2bool, default=False)


parser.add_argument('--is_train', type=str2bool, default=True)

parser.add_argument('--submit', type=str2bool, default=True, help='test set')
parser.add_argument('--pval', type=str2bool, default=False, help='pval')

parser.add_argument('--submit_confth', '--sct' ,type=float, default=0.9, help='conf th')
parser.add_argument('--inference_epoch', '--iep', type=int, default=0)
parser.add_argument('--do_just', type=int, default=0)
parser.add_argument('--check_phase1', type=str2bool, default=False)


parser.add_argument('--nms', type=str, default='nms', help='or NMW')

parser.add_argument('--weight_decay', '--wdc', type=float, default=1e-4)


parser.add_argument('--superjam', '--sj', type=str2bool, default=True)
parser.add_argument('--qjam', '--qj', type=str2bool, default=False)


parser.add_argument('--image_size', '--ims', type=int, default=512)


parser.add_argument('--from_hem', type=str2bool, default=False)




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




DATA_DIR = '/home/minki/ramdisk/rsna/data/kaggle'
ROOT_DIR = '/home/minki/rsna/Mask_RCNN'



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


# config.VALIDATION_STEPS = int(len(image_fps_val) / pargs.batch_size)
config.VALIDATION_STEPS = 1 ## validation set have no more meaning

#####################
if pargs.use_all_data:
    config.STEPS_PER_EPOCH = int(len(image_fps_train) / pargs.batch_size)






# pneumonia_val_images = []
# for img in tqdm(image_fps_val):
#     if anns[anns['patientId'] == img.split('/stage_2_train_images/')[1][:-4]].iloc[0]['Target'] == 1:
#         pneumonia_val_images.append(img)
#
# print('pneumonia images in valset: ', len(pneumonia_val_images))
# pneumonia_val_images[:5]

# 619


# In[45]:


print(save_dir)

image_fps_jam = image_fps_val + image_fps_val_test
np.random.seed(pargs.seed)  # 42
np.random.shuffle(sorted(image_fps_jam))

dataset_jam = DetectorDataset(image_fps_jam, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_jam.prepare()

print('data distribution(=val1+val3 data)')
print(len(image_fps_jam))


#
#
# dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_train.prepare()
#
#
#
# # prepare the validation dataset
# dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_val.prepare()
#
#
# dataset_val_test = DetectorDataset(image_fps_val_test, image_annotations, ORIG_SIZE, ORIG_SIZE)
# dataset_val_test.prepare()


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


# In[ ]:


# In[31]:


if pargs.is_train:
    warnings.filterwarnings ("ignore")

    # orig_save_dir = save_dir
    # load model
    if pargs.from_hem:
        save_dir = 'seed_{}_hem'.format(pargs.seed)


    history = pd.read_csv(
        ROOT_DIR + '/ckpt/' + save_dir + '/training_log.csv')
    history['fixed_val_loss'] = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
    print('history: ', history)
    # fix_val_loss = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
    best_epoch = np.argmin(history['fixed_val_loss']) + 1

    # check config, configs should be same as the beginning
    model_weight = ROOT_DIR + '/ckpt/{}/mask_rcnn_{}_00{}.h5'.format(save_dir, save_dir, best_epoch)

    if pargs.from_hem:
        save_dir = 'seed_{}_hem_jjam'.format(pargs.seed)
        if pargs.superjam:
            save_dir = save_dir.replace('_jjam', '_superjjam')
    else:
        save_dir = 'seed_{}_jjam'.format(pargs.seed)
        if pargs.superjam:
            save_dir = save_dir.replace('_jjam', '_superjjam')

    config.NAME = save_dir

    if pargs.superjam:
        config.DEEPER_HEAD += 2

    model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR + '/ckpt')

    print(save_dir)


    # todo:
    model.load_weights(model_weight, by_name=True)

    if pargs.superjam:

        weights_conv_add4 = model.keras_model.get_layer('mrcnn_class_conv_add4').get_weights()
        weights_bn_add4 = model.keras_model.get_layer('mrcnn_class_bn_add4').get_weights()
        weights_conv_add5 = model.keras_model.get_layer('mrcnn_class_conv_add5').get_weights()
        weights_bn_add5 = model.keras_model.get_layer('mrcnn_class_bn_add5').get_weights()

        model.keras_model.get_layer('mrcnn_class_conv_add6').set_weights(weights_conv_add4)
        model.keras_model.get_layer('mrcnn_class_bn_add6').set_weights(weights_bn_add4)
        model.keras_model.get_layer('mrcnn_class_conv_add7').set_weights(weights_conv_add5)
        model.keras_model.get_layer('mrcnn_class_bn_add7').set_weights(weights_bn_add5)

    # np.random.seed(42)
    # np.random.shuffle(sorted(image_fps_val_test))


    model.config.STEPS_PER_EPOCH = int(len(image_fps_jam) / pargs.batch_size)
    config.VALIDATION_STEPS = 1  ## validation set have no more meaning

    ## retrain focus on hard examples
    # print('Hard example mining cycle 1 start!')

    model.train(dataset_jam, dataset_jam,
                learning_rate=1e-4, epochs=pargs.max_epoch, layers='all', augmentation=augmentation)

    # note: loss is no more a guide line, check the final map and recall
















    # fixme:

    print('\n\n', save_dir)
    # history = pd.read_csv('/home/minki/rsna/Mask_RCNN/ckpt/'+save_dir.replace('False', 'false') + '/training_log.csv')
    history = pd.read_csv(
        ROOT_DIR + '/ckpt/' + save_dir + '/training_log.csv')
    history['fixed_val_loss'] = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
    print('history: ', history)


    # fix_val_loss = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
    best_epoch = np.argmin(history['fixed_val_loss']) + 1

    # best_epoch = np.argmin(history["val_loss"])
    print(vars(pargs))
    print("Best Epoch:", best_epoch)


else:

    if pargs.from_hem:
        save_dir = 'seed_{}_hem_jjam'.format(pargs.seed)
        if pargs.superjam:
            save_dir = save_dir.replace('_jjam', '_superjjam')
    else:
        save_dir = 'seed_{}_jjam'.format(pargs.seed)
        if pargs.superjam:
            save_dir = save_dir.replace('_jjam', '_superjjam')

    if pargs.inference_epoch == 0:

        print('\n\n', save_dir)
        history = pd.read_csv(
            ROOT_DIR + '/ckpt/' + save_dir + '/training_log.csv')
        history['fixed_val_loss'] = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
        print('history: ', history)


        # fix_val_loss = [vl - rbl for (vl, rbl) in zip(history["val_loss"], history['val_rpn_bbox_loss'])]
        best_epoch = np.argmin(history['fixed_val_loss']) + 1

        # best_epoch = np.argmin(history["val_loss"])
        print(vars(pargs))
        print("Best Epoch:", best_epoch)
    else:
        best_epoch = pargs.inference_epoch
        print("Best Epoch:", best_epoch)
# In[47]:



# no meaning for best epoch
best_epoch = 5



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
# print('inference: ', model_path[)




# In[43]:


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    TRAIN_BN = False
    if pargs.superjam:
        DEEPER_HEAD = pargs.deeper_head + 2


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


        #### NMS
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

        ####

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

    return result_df










print('Prediction start')
print(vars(pargs))

if pargs.submit:
    submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_test.csv'.
                                 format(best_epoch+1, pargs.submit_confth))
    result_df = predict(test_image_fps, filepath=submission_fp, min_conf=pargs.submit_confth) # 0.9
    result_df.to_csv(submission_fp, index=False)
    print(submission_fp)


# elif pargs.pval:
#     submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_pval.csv'.
#                                  format(best_epoch, 0.9))
#     result_df = predict(pneumonia_val_images, filepath=submission_fp, min_conf=0.9)
#     result_df.to_csv(submission_fp, index=False)
#     print(submission_fp)

else:
    print('used test val to jam the model, no more validation score')
    # submission_fp = os.path.join(ROOT_DIR, 'result2', save_dir + '_{}_{}_val.csv'.
    #                              format(best_epoch, 0.9))
    # # result_df = predict(pneumonia_val_images, filepath=submission_fp, min_conf=0.9)
    #
    # result_df = predict(image_fps_val_test, filepath=submission_fp, min_conf=0.9)
    # # result_df = predict(image_fps_train+image_fps_val+image_fps_val_test, filepath=submission_fp+'/_fpr', min_conf=pargs.submit_confth)



# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:






# In[ ]:

if not pargs.submit:
    print('Start evaluation')




    mAP, mAPs, recall_04, gt_num, result_df, fn_list = simple_evaluator_temp(
        gt_csv_path=DATA_DIR + '/stage_2_train_labels.csv',
        result_csv_path=submission_fp
    )

    print(gt_num,
          len(result_df[result_df['result'] == 'TP']),
          len(fn_list),
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




