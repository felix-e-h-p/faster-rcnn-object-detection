
import torch
import pytorch_lightning as pl

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torchvision.ops

from train_config import LEARNING_RATE, GAMMA, NUM_CLASSES

class LightningFasterRCNN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = self.create_model()

    def forward(self, x):
        return self.model(x)

    def create_model(self):
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(backbone,
                           NUM_CLASSES,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
        return model

    def configure_optimisers(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=GAMMA)
        return [optimiser], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets)
        loss_dict = output['losses']
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        return {'train_loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets)
        loss_dict = output['losses']
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        return {'val_loss': avg_loss}
