
# train.py
# This script handles the training and validation of a Faster R-CNN object detection model.

import torch
import time

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from train_config import DEVICE, NUM_EPOCHS, NUM_CLASSES, OUT_DIR, NUM_WORKERS, LEARNING_RATE, GAMMA
from model import create_model
from utils import Averager, SaveBestModel, save_model, save_loss_plot
from data_transform import create_train_dataset, create_validation_dataset, create_train_loader, create_validation_loader


def train(train_data_loader, model):
    """
    Train the Faster R-CNN model on the training dataset.

    Parameters:
        train_data_loader (DataLoader): DataLoader for the training dataset.
        model (FasterRCNN): Faster R-CNN model.
        optimiser (torch.optim.Optimizer): Optimizer for model training.
        train_loss_hist (Averager): Averager object to track training loss.
        train_itr (int): Training iteration number.

    Returns:
        list: List of training losses for each iteration.
    """
    print('Training')
    global train_itr
    global train_loss_list

    model.train()

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    train_loss_list = []

    for i, data in enumerate(prog_bar):
        optimiser.zero_grad()

        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimiser.step()
        train_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list


def validate(valid_data_loader, model):
    """
    Validate the Faster R-CNN model on the validation dataset.

    Parameters:
        valid_data_loader (DataLoader): DataLoader for the validation dataset.
        model (FasterRCNN): Faster R-CNN model.
        val_loss_hist (Averager): Averager object to track validation loss.
        val_itr (int): Validation iteration number.

    Returns:
        list: List of validation losses for each iteration.
    """
    print('Validating')
    global val_itr
    global val_loss_list

    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    val_loss_list = []

    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list


def calculate_mAP(data_loader, model):
    """
    Calculate Mean Average Precision (mAP) on the validation dataset.

    Parameters:
        data_loader (DataLoader): DataLoader for the validation dataset.
        model (FasterRCNN): Faster R-CNN model.

    Returns:
        float: Validation Mean Average Precision (mAP).
    """
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

    train_dataset = create_train_dataset()
    valid_dataset = create_validation_dataset()

    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_validation_loader(valid_dataset, NUM_WORKERS)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Optimiser call, calibration and specify base parameters

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    optimiser = torch.optim.Adam(params,
                                 lr=LEARNING_RATE)

    scheduler = StepLR(optimiser, step_size=30, gamma=GAMMA)

    # Averager class initiated for Training and Validation loss

    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    train_loss_list = []
    val_loss_list = []

    model_name = 'model'
    save_best_model = SaveBestModel()

    # Training and Validation loop per epoch

    for epoch in range(NUM_EPOCHS):

        scheduler.step()

        print(f"\nEpoch {epoch+1} of {NUM_EPOCHS}")
        train_loss_hist.reset()
        val_loss_hist.reset()
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)

        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")

        mAP = calculate_mAP(valid_loader, model)

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        save_best_model(val_loss_hist.value, epoch, model, optimiser)
        save_model(epoch, model, optimiser)

        if train_loss is not None and val_loss is not None:
            print("Train Loss:", train_loss)
            print("Validation Loss:", val_loss)
            save_loss_plot(OUT_DIR, train_loss, val_loss)
        else:
            print("Skipping save_loss_plot due to None values in train_loss or val_loss")
 
        time.sleep(5)