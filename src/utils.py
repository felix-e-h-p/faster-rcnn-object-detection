
import albumentations as A
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2

from train_config import CLASSES, DEVICE, OUT_DIR


class Averager:

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimiser):

        if current_valid_loss < self.best_valid_loss:

            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            # Must specify absoloute path for saving training and validation results

            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict()},
                        os.path.join(OUT_DIR, 'best_model.pth'))


def collate_fn(batch):
    return tuple(zip(*batch))


def train_transform():
    return A.Compose([A.Flip(0.5),
                      A.RandomRotate90(0.5),
                      A.MedianBlur(blur_limit=3, p=0.1),
                      A.Blur(blur_limit=3, p=0.1),
                      ToTensorV2(p=1.0)],
                      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def valid_transform():
    return A.Compose([ToTensorV2(p=1.0)],
                      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def test_transform():
    return A.Compose([ToTensorV2(p=1.0)],
                      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def show_transformed_image(train_loader):
    if len(train_loader) > 0:
        for i in range(1):

            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()

            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]],
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            cv2_imshow(sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def save_model(epoch, model, optimiser):

    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict()},
                os.path.join(OUT_DIR, 'last_model.pth'))


def save_loss_plot(output_dir, train_loss, val_loss):

    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(output_dir + '/train_loss.png')
    figure_2.savefig(output_dir + '/valid_loss.png')
    print('Saving completed')
    plt.close('all')


def delete_previous_output(folder):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            