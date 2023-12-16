
import torch
import time

from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl

from train_config import DEVICE, NUM_EPOCHS, NUM_CLASSES, OUT_DIR, NUM_WORKERS, LEARNING_RATE, GAMMA
from train_config import CLASSES, WIDTH, HEIGHT, TRAIN_DIR, VALID_DIR, BATCH_SIZE, NUM_WORKERS

from model_pl import LightningFasterRCNN

#from data_transform_pl import create_train_dataset, create_validation_dataset, create_train_loader, create_validation_loader
from data_transform_pl import OnshoreWindTurbinesDataModule


def calculate_mAP(data_loader, model):

    metric_test = MeanAveragePrecision()
    preds_single = []
    targets_single = []

    for batch_idx, (images, targets) in enumerate(data_loader, 1):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        targets_single.extend(targets)
        model.eval()

        with torch.no_grad():
            pred = model(images)
        preds_single.extend(pred)

    metric_test.update(preds_single, targets_single)
    test_map = metric_test.compute()
    print(f"Validation Mean Average Precision: {test_map['map']:.3f}")

    return test_map['map']


if __name__ == '__main__':

    torch.cuda.is_available()
    torch.cuda.empty_cache()

    data_module = OnshoreWindTurbinesDataModule(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        width=WIDTH,
        height=HEIGHT,
        classes=CLASSES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    data_module.setup()

    model = LightningFasterRCNN()

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu'
    )

    trainer.fit(model, data_module.train_loader(), data_module.valid_loader())

    calculate_mAP(data_module.valid_loader(), model.model)
