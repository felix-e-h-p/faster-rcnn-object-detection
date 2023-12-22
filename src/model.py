
# model.py
# This module contains the definition of the Faster R-CNN object detection model.

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes):
    """
    Creates and returns a Faster R-CNN object detection model.

    Parameters:
        num_classes (int): Number of classes for object detection.

    Returns:
        FasterRCNN: The configured Faster R-CNN model.
    """
    # Use MobileNetV2 as the backbone
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280

    # Configure anchor generator for the region proposal network (RPN)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Configure region of interest (RoI) pooler for feature extraction
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # Create the Faster R-CNN model
    model = FasterRCNN(backbone,
                       num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    
    return model
