
# data_transform.py
# This module provides classes and functions for transforming and loading datasets for onshore wind turbines detection.

import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et

from train_config import CLASSES, WIDTH, HEIGHT, TRAIN_DIR, VALID_DIR, BATCH_SIZE, NUM_WORKERS

from torch.utils.data import Dataset, DataLoader

from utils import collate_fn, train_transform, valid_transform


class OnshoreWindTurbines(Dataset):
    """
    Custom PyTorch Dataset class for onshore wind turbines detection.
    """

    def __init__(self, dir_path, width, height, classes, transforms=None):
        """
        Initializes the dataset.

        Parameters:
        - dir_path (str): Path to the directory containing images and XML annotations.
        - width (int): Width of the resized images.
        - height (int): Height of the resized images.
        - classes (list): List of class names.
        - transforms (callable): Image and annotation transformations (e.g., data augmentation).
        """        
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_paths = glob.glob(self.dir_path + '/*.png')
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        """
        Retrieves and processes an image and its annotations.

        Parameters:
        - idx (int): Index of the image in the dataset.

        Returns:
        - tuple: Resized image and target annotations.
        """
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # Iterate through xml files in custom directory

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Designate bounding coordinate labels to a class - dict format

        for member in root.findall('object'):

            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        # Convert bounding box coordinates to tensors

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Specify coordinate class labels as target features

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.all_images)


def create_train_dataset():
    """
    Creates and returns the training dataset.

    Returns:
    - OnshoreWindTurbines: Training dataset.
    """
    train_dataset = OnshoreWindTurbines(TRAIN_DIR, WIDTH, HEIGHT, CLASSES, train_transform())
    return train_dataset


def create_validation_dataset():
    """
    Creates and returns the validation dataset.

    Returns:
    - OnshoreWindTurbines: Validation dataset.
    """
    valid_dataset = OnshoreWindTurbines(VALID_DIR, WIDTH, HEIGHT, CLASSES, valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers):
    """
    Creates and returns the training data loader.

    Parameters:
    - train_dataset (OnshoreWindTurbines): Training dataset.
    - num_workers (int): Number of parallel data loading processes.

    Returns:
    - DataLoader: Training data loader.
    """    
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)
    return train_loader


def create_validation_loader(valid_dataset, num_workers):
    """
    Creates and returns the validation data loader.

    Parameters:
    - valid_dataset (OnshoreWindTurbines): Validation dataset.
    - num_workers (int): Number of parallel data loading processes.

    Returns:
    - DataLoader: Validation data loader.
    """    
    valid_loader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)
    return valid_loader
