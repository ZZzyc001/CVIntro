import os
from matplotlib import image
import numpy as np
import torch
import torch.utils.data
import random
import cv2
import math
from utils import plot_save_dataset


class SingleShapeDataset(torch.utils.data.Dataset):

    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h // 4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        if shape_id == 1:
            cv2.rectangle(mask, (x - s, y - s), (x + s, y + s), 1, -1)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[
                (x, y - s),
                (x - s / math.sin(math.radians(60)), y + s),
                (x + s / math.sin(math.radians(60)), y + s),
            ]],
                              dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray([random.randint(0, 255)
                                  for _ in range(3)])[None, None, :]

        obj_ids = np.zeros((n_class))

        shape_code = random.randint(1, 3)
        self._draw_shape(img, masks[0, :], shape_code)
        obj_ids[0] = shape_code

        boxes = np.zeros((n_class, 4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0, :] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


class MySingleShapeDataset(torch.utils.data.Dataset):

    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h // 4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        if shape_id == 1:
            cv2.rectangle(mask, (x - s, y - s), (x + s, y + s), 1, -1)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[
                (x, y - s),
                (x - s / math.sin(math.radians(60)), y + s),
                (x + s / math.sin(math.radians(60)), y + s),
            ]],
                              dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, shape_code):
        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray([random.randint(0, 255)
                                  for _ in range(3)])[None, None, :]

        self._draw_shape(img, masks[0, :], shape_code)

        boxes = np.zeros((n_class, 4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0, :] = np.asarray([xmin, ymin, xmax, ymax])

        target = {}
        target["boxes"] = boxes
        target["masks"] = masks

        return img, target

    def __len__(self):
        return self.size


# ----------TODO------------
# Implement ShapeDataset.
# Refer to `SingleShapeDataset` for the shape parameters
# ----------TODO------------


class ShapeDataset(torch.utils.data.Dataset):

    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)
        self.nms_thres = 0.7
        self.single_shape = MySingleShapeDataset(self.size)

    def __getitem__(self, idx):
        np.random.seed(idx)
        n_shapes = random.randint(1, 3)

        image_id = torch.tensor([idx])
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        boxes = np.zeros((n_shapes, 4), dtype=np.int32)
        masks = np.zeros((n_shapes, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray([random.randint(0, 255)
                                  for _ in range(3)])[None, None, :]

        shape_code = np.random.randint(1, 4, (n_shapes))

        def check_overlap(masks, mask):
            if len(masks) == 0:
                return False
            for m in masks:
                tot = m.sum()
                outer = np.where(m > mask, m, np.zeros_like(m)).sum()
                if outer / tot < self.nms_thres:
                    return True

                tot = mask.sum()
                outer = np.where(mask > m, mask, np.zeros_like(mask)).sum()
                if outer / tot < self.nms_thres:
                    return True
            return False

        t_mask = np.zeros((self.h, self.w))

        for i in range(n_shapes):
            while True:
                t_img, t_target = self.single_shape[shape_code[i]]
                if not check_overlap(masks[:i], t_target["masks"][0]):
                    break

            boxes[i, :] = t_target["boxes"][0]
            masks[i, :] = t_target["masks"][0]
            img = np.where(
                np.expand_dims(masks[i] - t_mask, -1) >= 0, t_img, img)
            t_mask = np.where(masks[i] > t_mask, masks[i], t_mask)

            ## put object i on the other mask
            for j in range(i):
                masks[j] = np.where(masks[j] > masks[i], masks[j],
                                    np.zeros_like(masks[j]))
                pos = np.where(masks[j])
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                boxes[j, :] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(shape_code, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


if __name__ == '__main__':
    # dataset = SingleShapeDataset(10)
    dataset = ShapeDataset(10)
    path = "results/"

    for i in range(10):
        imgs, labels = dataset[i]
        print(labels)
        plot_save_dataset(path + str(i) + "_data.png", imgs, labels)
