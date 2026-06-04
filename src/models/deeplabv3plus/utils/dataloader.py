import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class SeverityControlledLesionCopyPaste:
    def __init__(
        self,
        annotation_lines,
        dataset_path,
        num_classes,
        prob=0.0,
        max_components=3,
        class_weights=None,
        min_area=8,
        max_area_ratio=0.12,
    ):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.prob = float(prob)
        self.max_components = int(max_components)
        self.min_area = int(min_area)
        self.max_area_ratio = float(max_area_ratio)
        self.class_weights = class_weights or {2: 1.0, 3: 2.0, 4: 2.0, 5: 3.0}
        self.bank = []
        self._build_bank(annotation_lines)

    def _build_bank(self, annotation_lines):
        image_dir = os.path.join(self.dataset_path, "VOC2007/JPEGImages")
        mask_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass")
        for line in annotation_lines:
            name = line.split()[0]
            image_path = os.path.join(image_dir, name + ".jpg")
            mask_path = os.path.join(mask_dir, name + ".png")
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                continue
            image = np.array(cvtColor(Image.open(image_path)), np.uint8)
            mask = np.array(Image.open(mask_path), np.uint8)
            leaf_area = max(int(np.isin(mask, [1, 2, 3, 4, 5]).sum()), 1)
            lesion_area = int(np.isin(mask, [2, 3, 4, 5]).sum())
            severity = lesion_area / leaf_area
            h, w = mask.shape[:2]
            max_area = max(int(h * w * self.max_area_ratio), self.min_area)
            for class_id in range(2, min(self.num_classes, 6)):
                component_mask = (mask == class_id).astype(np.uint8)
                if component_mask.sum() < self.min_area:
                    continue
                count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
                for idx in range(1, count):
                    x, y, bw, bh, area = stats[idx]
                    if area < self.min_area or area > max_area:
                        continue
                    comp = labels[y : y + bh, x : x + bw] == idx
                    if comp.sum() < self.min_area:
                        continue
                    self.bank.append(
                        {
                            "class_id": class_id,
                            "image": image[y : y + bh, x : x + bw].copy(),
                            "mask": comp.astype(np.uint8),
                            "area": int(area),
                            "severity": float(severity),
                        }
                    )

    def _pick_component(self):
        if not self.bank:
            return None
        weights = np.array([self.class_weights.get(item["class_id"], 1.0) for item in self.bank], dtype=np.float64)
        weights = weights / weights.sum()
        return self.bank[int(np.random.choice(len(self.bank), p=weights))]

    @staticmethod
    def _resize_component(component, scale):
        patch = component["image"]
        mask = component["mask"]
        h, w = mask.shape[:2]
        nh = max(int(h * scale), 2)
        nw = max(int(w * scale), 2)
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        return patch, mask

    @staticmethod
    def _severity_bin(mask):
        leaf_area = max(int(np.isin(mask, [1, 2, 3, 4, 5]).sum()), 1)
        lesion_area = int(np.isin(mask, [2, 3, 4, 5]).sum())
        ratio = lesion_area / leaf_area
        if ratio < 0.03:
            return "low"
        if ratio < 0.08:
            return "medium"
        return "high"

    def apply(self, image, mask):
        if not self.bank or np.random.rand() >= self.prob:
            return image, mask

        image = np.array(image, np.uint8).copy()
        mask = np.array(mask, np.uint8).copy()
        leaf_positions = np.argwhere(np.isin(mask, [1, 2, 3, 4, 5]))
        if len(leaf_positions) == 0:
            return image, mask

        severity_bin = self._severity_bin(mask)
        if severity_bin == "low":
            n_components = np.random.randint(1, self.max_components + 1)
            scale_range = (0.75, 1.35)
        elif severity_bin == "medium":
            n_components = np.random.randint(1, max(2, self.max_components))
            scale_range = (0.65, 1.15)
        else:
            n_components = 1
            scale_range = (0.55, 0.95)

        h, w = mask.shape[:2]
        for _ in range(n_components):
            component = self._pick_component()
            if component is None:
                break
            patch, comp_mask = self._resize_component(component, np.random.uniform(*scale_range))
            ph, pw = comp_mask.shape[:2]
            if ph >= h or pw >= w:
                continue
            cy, cx = leaf_positions[np.random.randint(0, len(leaf_positions))]
            x1 = int(np.clip(cx - pw // 2, 0, w - pw))
            y1 = int(np.clip(cy - ph // 2, 0, h - ph))
            target_leaf = np.isin(mask[y1 : y1 + ph, x1 : x1 + pw], [1, 2, 3, 4, 5])
            paste_mask = comp_mask & target_leaf
            if paste_mask.sum() < self.min_area:
                continue
            roi = image[y1 : y1 + ph, x1 : x1 + pw]
            roi[paste_mask] = patch[paste_mask]
            image[y1 : y1 + ph, x1 : x1 + pw] = roi
            target = mask[y1 : y1 + ph, x1 : x1 + pw]
            target[paste_mask] = component["class_id"]
            mask[y1 : y1 + ph, x1 : x1 + pw] = target
        return image, mask


class DeeplabDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
        sclp=False,
        sclp_prob=0.0,
        sclp_max_components=3,
        sclp_class_weights=None,
    ):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.sclp = None
        if self.train and sclp and sclp_prob > 0:
            self.sclp = SeverityControlledLesionCopyPaste(
                annotation_lines,
                dataset_path,
                num_classes,
                prob=sclp_prob,
                max_components=sclp_max_components,
                class_weights=sclp_class_weights,
            )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
        if self.sclp is not None:
            jpg, png = self.sclp.apply(jpg, png)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
