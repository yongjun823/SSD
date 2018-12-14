import os
import time

import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        print('loading annotations into memory...')
        tic = time.time()
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        with open(image_sets_file) as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.anns = {}
        for image_id in self.ids:
            size, boxes, labels, difficult = self._get_annotation(image_id)
            self.anns[image_id] = {
                'boxes': np.array(boxes, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64),
                'difficult': np.array(difficult, dtype=np.bool),
                'info': size
            }
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann = self.anns[image_id]
        boxes, labels, is_difficult = ann['boxes'], ann['labels'], ann['difficult']
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_annotation(self, index):
        image_id = self.ids[index]
        ann = self.anns[image_id]
        return image_id, ann

    def get_img_info(self, index):
        image_id = self.ids[index]
        ann = self.anns[image_id]
        return ann['info']

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        root = ET.parse(annotation_file)
        size = {
            'width': int(root.find('size').find('width').text),
            'height': int(root.find('size').find('height').text)
        }
        objects = root.findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(VOCDataset.class_names.index(class_name))
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str))

        return size, boxes, labels, is_difficult

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
