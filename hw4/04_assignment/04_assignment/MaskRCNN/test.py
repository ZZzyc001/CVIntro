from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data

# the outputs includes: 'boxes', 'labels', 'masks', 'scores'


def cal_iou_box(box1, box2):
    mask1 = torch.zeros((128, 128), dtype=torch.bool)
    mask2 = torch.zeros((128, 128), dtype=torch.bool)
    mask1[int(box1[1]):int(box1[3]), int(box1[0]):int(box1[2])] = True
    mask2[int(box2[1]):int(box2[3]), int(box2[0]):int(box2[2])] = True
    intersection = torch.sum(mask1 & mask2)
    union = torch.sum(mask1 | mask2)
    return intersection.float() / union


def cal_iou_mask(mask1, mask2):
    intersection = (mask1 + mask2 - 1).clamp(0, 1).sum()
    union = (mask1 + mask2).clamp(0, 1).sum()
    return intersection.float() / union


def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):
    prec = np.zeros((3, 1001))
    rec = np.zeros((3, 1001))

    threshold = np.linspace(0., 1., 101)

    ## put all the output element together and sort them by their scores
    out = []
    for i in range(len(output_list)):
        pr_mask = output_list[i]['masks']
        pr_labels = output_list[i]['labels']
        n_pr = pr_mask.shape[0]
        gt_mask = gt_labels_list[i]['masks']
        gt_labels = gt_labels_list[i]['labels']

        for j in range(n_pr):
            if pr_labels[j] == gt_labels:
                out.append({
                    "mask": pr_mask[j],
                    "label": pr_labels[j],
                    "score": output_list[i]['scores'][j],
                    "gt_mask": gt_mask[0],
                    "gt_label": gt_labels
                })

    out.sort(key=lambda x: x['score'], reverse=True)

    for t in range(len(out)):
        TP = np.zeros(3)
        FP = np.zeros(3)
        FN = np.zeros(3)
        for p in out:
            if p['score'] > out[t]['score']:
                if cal_iou_mask(p['mask'], p['gt_mask']) > iou_threshold:
                    TP[p['label'] - 1] += 1
                else:
                    FP[p['label'] - 1] += 1
            else:
                if cal_iou_mask(p['mask'], p['gt_mask']) <= iou_threshold:
                    FN[p['label'] - 1] += 1

        prec[:, t] = np.where(TP + FP > 0, TP / (TP + FP), 1.)
        rec[:, t] = np.where(TP > 0, TP / (TP + FN), 0)

    # print('prec', prec)
    # print('rec', rec)

    AP = np.zeros(3)

    for t in range(len(out) - 1):
        AP += (rec[:, t + 1] - rec[:, t]) * prec[:, t + 1]

    AP /= rec[:, len(out) - 1] - rec[:, 0]
    return AP.mean()


def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    prec = np.zeros((3, 1001))
    rec = np.zeros((3, 1001))

    threshold = np.linspace(0., 1., 101)

    ## put all the output element together and sort them by their scores
    out = []
    for i in range(len(output_list)):
        pr_mask = output_list[i]['boxes']
        pr_labels = output_list[i]['labels']
        n_pr = pr_mask.shape[0]
        gt_mask = gt_labels_list[i]['boxes']
        gt_labels = gt_labels_list[i]['labels']

        for j in range(n_pr):
            if pr_labels[j] == gt_labels:
                out.append({
                    "box": pr_mask[j],
                    "label": pr_labels[j],
                    "score": output_list[i]['scores'][j],
                    "gt_box": gt_mask[0],
                    "gt_label": gt_labels
                })

    out.sort(key=lambda x: x['score'])

    for t in range(len(out)):
        TP = np.zeros(3)
        FP = np.zeros(3)
        FN = np.zeros(3)
        for p in out:
            if p['score'] > out[t]['score']:
                if cal_iou_box(p['box'], p['gt_box']) > iou_threshold:
                    TP[p['label'] - 1] += 1
                else:
                    FP[p['label'] - 1] += 1
            else:
                if cal_iou_box(p['box'], p['gt_box']) <= iou_threshold:
                    FN[p['label'] - 1] += 1
        prec[:, t] = np.where(TP + FP > 0, TP / (TP + FP), 1.)
        rec[:, t] = np.where(TP > 0, TP / (TP + FN), 0)

    AP = np.zeros(3)

    for t in range(len(out) - 1):
        AP += (rec[:, t + 1] - rec[:, t]) * prec[:, t + 1]

    AP /= rec[:, len(out) - 1] - rec[:, 0]

    return AP.mean()


dataset_test = SingleShapeDataset(100)

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               collate_fn=utils.collate_fn)

num_classes = 4

# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cuda')

# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(
    torch.load('../exps/weight/maskrcnn_19.pth', map_location='cuda'))

model.eval()
path = "../results/MaskRCNN/"

# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(100):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        plot_save_output(path + str(i) + "_result.png", imgs, output[0])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)

print(mAP_detection)
print(mAP_segmentation)

np.savetxt(path + "mAP.txt", np.asarray([mAP_detection, mAP_segmentation]))
