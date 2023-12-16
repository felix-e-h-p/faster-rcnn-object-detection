
import os
import torch
import cv2
import numpy as np
import glob as glob

import albumentations as A

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from albumentations.pytorch import ToTensorV2

class OnshoreWindTurbinesDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_paths = glob.glob(os.path.join(self.dir_path, '*.png'))
        self.all_images = sorted([image_path.split(os.path.sep)[-1] for image_path in self.image_paths])

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            sample = self.transforms(image=image_resized, bboxes=target['boxes'], labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)



class OnshoreWindTurbinesDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, valid_dir, width, height, classes, batch_size, num_workers):
        super().__init__()
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.width = width
        self.height = height
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_transform(self):
        return A.Compose([
            A.Flip(0.5),
            A.RandomRotate90(0.5),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
            ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

    def valid_transform(self):
        return A.Compose([
            ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

    def test_transform(self):
        return A.Compose([
            ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

    def collate_fn(self, batch):
        images, targets = zip(*batch)
        return images, targets

    def setup(self, stage=None):

        self.train_dataset = OnshoreWindTurbinesDataset(self.train_dir, self.width, self.height, self.classes,
                                                        transforms=self.train_transform())

        self.valid_dataset = OnshoreWindTurbinesDataset(self.valid_dir, self.width, self.height, self.classes,
                                                        transforms=self.valid_transform())

    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def valid_loader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
