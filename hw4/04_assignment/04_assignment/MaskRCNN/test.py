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
    AP = np.zeros(len(output_list))

    TP = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)

    for i in range(len(output_list)):
        pr_mask = output_list[i]['masks']
        pr_labels = output_list[i]['labels']
        n_pr = pr_mask.shape[0]

        if n_pr == 0:
            continue

        gt_mask = gt_labels_list[i]['masks']
        gt_labels = gt_labels_list[i]['labels']

        for j in range(n_pr):
            if pr_labels[j] == gt_labels:
                if cal_iou_mask(pr_mask[j], gt_mask[0]) > iou_threshold:
                    TP[gt_labels - 1] += 1
                else:
                    FP[gt_labels - 1] += 1
            else:
                if cal_iou_mask(pr_mask[j], gt_mask[0]) > iou_threshold:
                    FN[gt_labels - 1] += 1

    AP = TP * TP / (TP + FP) / (TP + FN)

    return AP.mean()


def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    AP = np.zeros(len(output_list))

    TP = np.zeros(3)
    FP = np.zeros(3)
    FN = np.zeros(3)

    for i in range(len(output_list)):
        pr_box = output_list[i]['boxes']
        pr_labels = output_list[i]['labels']
        n_pr = pr_box.shape[0]

        if n_pr == 0:
            continue

        gt_box = gt_labels_list[i]['boxes']
        gt_labels = gt_labels_list[i]['labels']

        for j in range(n_pr):
            if pr_labels[j] == gt_labels:
                if cal_iou_box(pr_box[j], gt_box[0]) > iou_threshold:
                    TP[gt_labels - 1] += 1
                else:
                    FP[gt_labels - 1] += 1
            else:
                if cal_iou_box(pr_box[j], gt_box[0]) > iou_threshold:
                    FN[gt_labels - 1] += 1

    AP = TP * TP / (TP + FP) / (TP + FN)

    return AP.mean()


dataset_test = SingleShapeDataset(10)

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
    for i in range(10):
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
