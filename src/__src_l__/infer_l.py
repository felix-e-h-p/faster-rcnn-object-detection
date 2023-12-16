import cv2
import os
import torch
import numpy as np
from glob import glob
import time

from train_config import DEVICE, DETECTION_THRESHOLD, NUM_CLASSES, INFER_DIR, OUT_DIR
from model import LightningFasterRCNN

if __name__ == '__main__':

    colours = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))
    
    model = LightningFasterRCNN.load_from_checkpoint(os.path.join(OUT_DIR, 'best_model.pth'))
    model = model.model.to(DEVICE).eval()

    test_images = glob(os.path.join(INFER_DIR, '*.png'))

    for i in range(len(test_images)):

        image_name = os.path.basename(test_images[i]).split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(DEVICE)
        image = torch.unsqueeze(image, 0)
        start_time = time.time()

        with torch.no_grad():
            outputs = model(image)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # Direct correspondence to pre-stated bounding boxes
        if len(outputs[0]['boxes']) != 0:

            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
            draw_boxes = boxes.copy()
            pred_classes = [model.classes[i] for i in outputs[0]['labels'].cpu().numpy()]

            pred_scores = outputs[0]['scores'].detach().cpu().numpy()
            pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
            pred_boxes = pred_boxes[pred_scores >= DETECTION_THRESHOLD].astype(np.int32)
            pred_labels = outputs[0]['labels'][:len(pred_boxes)]

            print(f"Image: {image_name}")
            print("Predicted Boxes:")

            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                colour = colours[model.classes.index(class_name)]
                print(f"Class: {class_name}, Box: {box}")

                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            colour, 2)

                cv2.putText(orig_image, class_name,
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour,
                            2, lineType=cv2.LINE_AA)

                # Crop and extract sub-image of target corresponding to predicted box coordinates
                section = orig_image[box[1]:box[3], box[0]:box[2]]
                section_filename = os.path.join(INFER_DIR, f"{image_name}_box{j}.png")
                cv2.imwrite(section_filename, section)

            cv2.imshow('Image', orig_image)
            cv2.waitKey(1)
            inf_dir = INFER_DIR
            cv2.imwrite(os.path.join(inf_dir, f"{image_name}.png"), orig_image)

        print(f"Image {i+1} done...")
        print('-' * 50)

    print('Testing stage complete')
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    cv2.destroyAllWindows()
