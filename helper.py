import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
import os
from mrcnn import utils

def predict_temp(image_fps, model_i, config, min_conf=0.97):
    # assume square image
    resize_factor = 1024 / config.IMAGE_SHAPE[0]
    # resize_factor = ORIG_SIZE

    result_df = pd.DataFrame(columns=['patientId', 'PredictionString'])

    idx = -1
    for image_id in tqdm(image_fps):
        idx += 1
        png = Image.open(image_id)
        image = np.array(png)
        # if pargs.he_concat:
        #     fp2 = image_id.replace("image", "images_HE")
        #     png2 = Image.open(fp2)
        #     image2 = np.array(png2)
        #     image = np.stack((image, image, image2), -1)

            # If grayscale. Convert to RGB for consistency2.
        if len(image.shape) != 3 or image.shape[2] != 3:
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


def simple_evaluator_temp(
        result_csv_path,
        gt_csv_path,
        iou_thresholds=(0.30, 0.40, 0.50, 0.60),
        prob_thresholds=[x / 100 for x in range(50, 100)],
        return_redefined_df=False
):
    gt_df = pd.read_csv(gt_csv_path)
    result_df = pd.read_csv(result_csv_path)  # , index_col=0
    result_puid = result_df['patientId'].drop_duplicates()
    result_df = result_df.dropna()
    _gt_num = 0
    result_redefined = []

    fn_list = []
    for each_result_puid in tqdm(result_puid):
        _this_tp_num = 0
        _this_fp_num = 0
        _this_dp_num = 0
        sub_result_df = result_df[result_df['patientId'] == each_result_puid]
        sub_gt_df = gt_df[gt_df['patientId'] == each_result_puid]
        sub_gt_scoreboard = np.zeros(len(sub_gt_df))
        if len(sub_result_df) != 0:
            result_string = sub_result_df['PredictionString'].get_values()[0]
            result_string = result_string.split(' ') [1:]
            _num_of_result = int(len(result_string) / 5)
            if len(sub_gt_df) == 1 and sub_gt_df['Target'].get_values() == 0:
                _this_gt_num = 0
                _max_iou = 0
                for i in range(_num_of_result):
                    prob = float(result_string[5 * i])
                    xmin = float(result_string[5 * i + 1])
                    ymin = float(result_string[5 * i + 2])
                    xmax = xmin + float(result_string[5 * i + 3])
                    ymax = ymin + float(result_string[5 * i + 4])
                    result = 'FP1'  # 없는데 잡음
                    result_redefined.append(
                        [each_result_puid,
                         prob, xmin, xmax, ymin, ymax,
                         result, _max_iou]
                    )


            else:
                _this_gt_num = len(sub_gt_df)
                gt_bboxes = sub_gt_df.get_values()[:, 1:-1]
                gt_bboxes[:, 2] = gt_bboxes[:, 2] + gt_bboxes[:, 0]
                gt_bboxes[:, 3] = gt_bboxes[:, 3] + gt_bboxes[:, 1]
                gt_bboxes[:, [2, 1]] = gt_bboxes[:, [1, 2]]
                for i in range(_num_of_result):
                    prob = float(result_string[5 * i])
                    xmin = float(result_string[5 * i + 1])
                    ymin = float(result_string[5 * i + 2])
                    xmax = xmin + float(result_string[5 * i + 3])
                    ymax = ymin + float(result_string[5 * i + 4])
                    bbox = np.array([[xmin, xmax, ymin, ymax]])
                    try:
                        _max_iou = np.max(minmax_iou(bbox, gt_bboxes))
                    except:
                        import pdb; pdb.set_trace()
                    _max_iou_arg = np.argmax(minmax_iou(bbox, gt_bboxes))
                    if _max_iou > iou_thresholds[0]:
                        if sub_gt_scoreboard[_max_iou_arg] == 0:
                            sub_gt_scoreboard[_max_iou_arg] = 1
                            _this_tp_num += 1
                            result = 'TP'

                        else:
                            _this_dp_num += 1
                            result = 'dp_TP'


                    else:
                        _this_fp_num += 1
                        result = 'FP2'  # iou 부족
                        # if _max_iou == 0:
                        #     result = 'FP1'

                    result_redefined.append(
                        [each_result_puid,
                         prob, xmin, xmax, ymin, ymax,
                         result, _max_iou]
                    )

                if sum(sub_gt_scoreboard) < len(sub_gt_scoreboard):
                    fn_list.append(each_result_puid)  # 'FN'





        else:
            _max_iou = 0
            _num_of_result = 0
            if len(sub_gt_df) == 1 and sub_gt_df['Target'].get_values() == 0:
                _this_gt_num = 0
                _max_iou = 0

            else:
                _this_gt_num = len(sub_gt_df)

        _gt_num += _this_gt_num
        """
        print('{} - GT / DETECTED / ASSIGNED / DUPLICATED / FP : {} {} {} {} {}'.format(
            each_result_puid, 
            _this_gt_num, _num_of_result, 
            _this_tp_num, _this_dp_num, _this_fp_num)
        )
        """

    # print(fn_list)

    result_redefined_df = pd.DataFrame(
        result_redefined,
        columns=['puid', 'prob',
                 'xmin', 'xmax', 'ymin', 'ymax',
                 'result', 'max_iou']
    )
    recall_04 = []
    mAPs = []

    for each_prob_threshold in prob_thresholds:
        precisions = []
        # print('=' * 48, '\nprob_threshold : {:.3f}\n{}'.format(each_prob_threshold, '-' * 48))
        sub_df = result_redefined_df[result_redefined_df['prob'] > each_prob_threshold]
        for each_iou_threshold in iou_thresholds:
            sub_df_tp = sub_df[sub_df['max_iou'] > each_iou_threshold]
            sub_df_tp = sub_df_tp[sub_df_tp['result'] == 'TP']
            sub_df_fp1 = sub_df[sub_df['result'] == 'FP1']
            sub_df_fp2 = sub_df[sub_df['result'] == 'FP2']
            # precision = len(sub_df_tp) / (_gt_num + len(sub_df_fp1) + len(sub_df_fp2))
            precision = len(sub_df_tp) / (len(sub_df_tp) + len(sub_df_fp1) + len(sub_df_fp2))
            precisions.append(precision)
            # print('iou_threshold : {:.2f} - AP : {:.4f}'.format(each_iou_threshold, precision))
            # print('recall: ', len(sub_df_tp) / _gt_num)
            if each_iou_threshold == 0.40:
                recall_04.append('{:.4f}'.format(len(sub_df_tp) / _gt_num))

        mAP = np.mean(precisions)
        mAPs.append('{:.4f}'.format(mAP))
        # print('\nmAP : {:.4f}\n'.format(mAP))

    return precisions, mAPs, recall_04, _gt_num, result_redefined_df, fn_list


# def simple_evaluator_temp(
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
#     for each_result_puid in tqdm(result_puid):
#         _this_tp_num = 0
#         _this_fp_num = 0
#         _this_dp_num = 0
#         sub_result_df = result_df[result_df['patientId'] == each_result_puid]
#         sub_gt_df = gt_df[gt_df['patientId'] == each_result_puid]
#         sub_gt_scoreboard = np.zeros(len(sub_gt_df))
#         if len(sub_result_df) != 0:
#             result_string = sub_result_df['PredictionString'].get_values()[0]
#             result_string = result_string.split(' ')[1:]
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
#                     elif _max_iou == 0:
#                         result = 'FN'
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
#         # print('=' * 48, '\nprob_threshold : {:.3f}\n{}'.format(each_prob_threshold, '-' * 48))
#         sub_df = result_redefined_df[result_redefined_df['prob'] > each_prob_threshold]
#         for each_iou_threshold in iou_thresholds:
#             sub_df_tp = sub_df[sub_df['max_iou'] > each_iou_threshold]
#             sub_df_tp = sub_df_tp[sub_df_tp['result'] == 'TP']
#             sub_df_fp1 = sub_df[sub_df['result'] == 'FP1']
#             sub_df_fp2 = sub_df[sub_df['result'] == 'FP2']
#             precision = len(sub_df_tp) / (_gt_num + len(sub_df_fp1) + len(sub_df_fp2))
#             precisions.append(precision)
#             # print('iou_threshold : {:.2f} - AP : {:.4f}'.format(each_iou_threshold, precision))
#             # print('recall: ', len(sub_df_tp) / _gt_num)
#             if each_iou_threshold == 0.40:
#                 recall_04.append('{:.4f}'.format(len(sub_df_tp) / _gt_num))
#
#         mAP = np.mean(precisions)
#         mAPs.append('{:.4f}'.format(mAP))
#         # print('\nmAP : {:.4f}\n'.format(mAP))
#
#     return precisions, mAPs, recall_04, _gt_num, result_redefined_df