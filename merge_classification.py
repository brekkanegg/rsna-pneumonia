import csv
import argparse
import os

def str2bool(v):
    return v.lower() in ('true')

# control here
parser = argparse.ArgumentParser()

parser.add_argument('--cls_csv', '--cls', type=str, default='classification_TestDense_Result_PT_v5_T0.3_testP2.csv', help='classification csv version and threshold')  # v5_T0.3
parser.add_argument('--pred_csv', '--pred', type=str, help='pred csv')  # v5_T0.3
parser.add_argument('--pred_dir', type=str, default='ensemble2', help='pred csv dir')  # v5_T0.3



pargs = parser.parse_args()



## classification delete in CSV detector result
delimiter = ','



# label_csv_cls_path = './classification_TestDense_Result_PT_v4_T0.15_test2.csv'
# label_csv_cls_path = './classification_TestDense_Result_PT_v4_T0.25.csv'
# label_csv_cls_path = './classification_TestDense_Result_PT_v5_T0.35_testP2.csv'
# label_csv_cls_path = 'classification_TestDense_Result_PT_v5_T0.3_testP2.csv'
label_csv_cls_path = pargs.cls_csv


add_name = label_csv_cls_path.split('Result')[1]


list_ClsLabel = []
with open(label_csv_cls_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=delimiter)  # , delimiter=' ', quotechar='|'
    for row in spamreader:
        if len(row) <= 0:
            print('pass :', len(row))
            continue
        list_ClsLabel.append(row)

import pandas as pd

# filename = '/ramdisk/Pneumonia/nih_for_test1_images.csv'
# df = pd.read_csv(filename)


# iou_th-0.12_overlap_th-0.8_prob_th-0.95_seeds-42_77

# label_csv_bbox_path = '/data3/iorism/ChestPA/RSNA_Pneumonia/result/iou_th-0.12_overlap_th-0.8_prob_th-0.95_seeds-42_77.csv'
# label_csv_bbox_path = './ensemble2/iou_th-0.12_overlap_th-0.8_prob_th-0.96_seeds-115_0.9_116_0.9_test2.csv'
# label_csv_bbox_path = './result/anchors-3_clean-false_7_deep-5_loadfpn-true_lr-0.001_regcls-2.0_roiiou-0.6_0.4_seed-211_tsize-2400_41_0.9_val_map_0.1271_recall_0.5268.csv'
# label_csv_bbox_path = './ensemble2/iou_th-0.12_overlap_th-0.8_prob_th-0.96_seeds-205_212_217_221_test2.csv'
# label_csv_bbox_path = './result2/seed_231_overfit_6_0.95_test.csv'
# label_csv_bbox_path = './ensemble2/iou_th-0.12_overlap_th-0.8_prob_th-0.935_seeds-205_212_217_221_False.csv'  # update on test phase 1
#


# label_csv_bbox_path = './ensemble2/iou_th-0.12_jam-hem_jam2_mode-test_overlap_th-0.8_prob_th-0.949_seeds-205_212_217_221_th-0.9.csv'  # update on test phase 1
label_csv_bbox_path = os.path.join(pargs.pred_dir, pargs.pred_csv)

list_bboxLabel = 'patientId,PredictionString'
with open(label_csv_bbox_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=delimiter)  # , delimiter=' ', quotechar='|'

    for row in spamreader:
        if row[0] == 'patientId':
            continue
        bboxs = row[1].split(' ')
        # print(len(bboxs), row)
        nIter = int(len(bboxs) / 5)
        if nIter < 1:
            list_bboxLabel += '\n' + row[0] + ','
        else:
            bIsFind = False

            '''
            tmp_idx = df['patientId'] == row[0]
            found = df.loc[tmp_idx]
            #imageIndex = list(found['imageIndex'])
            #imageIndex = imageIndex[0].replace('.png','')
            class2 = list(found['class2'])

            bIsFind = True
            if class2[0].find('Pneumonia')>=0 or class2[0].find('pneumonia')>=0:
                print ('incet:',class2[0] )
                list_bboxLabel += '\n' + row[0] + ',' + str(row[1]) 
            else:
                if class2[0].find('Infiltration')>=0 or class2[0].find('No Finding')>=0:
            '''
            for patient in list_ClsLabel:
                if patient[0].find(row[0]) >= 0:
                    # print(patient)
                    bIsFind = True
                    # print('find :',patient)
                    if patient[1] == 'False':  # False, FALSE
                        # print ('del model:',class2[0] )
                        list_bboxLabel += '\n' + row[0] + ','
                    else:
                        # print ('incet model:',class2[0] )
                        list_bboxLabel += '\n' + row[0] + ',' + str(row[1])
                    break

                # else:
                #    print ('del:',class2[0] )
                #    list_bboxLabel += '\n' + row[0] + ','

            if bIsFind == False:
                pass
                # print('error No Findings', row[0])

with open(label_csv_bbox_path.replace('.csv', add_name),
          'w') as f:  # PT_v5_2_T0.1 / PT_v3_T0.15 / PT_v5_2_T0.35
    f.write("%s" % list_bboxLabel)