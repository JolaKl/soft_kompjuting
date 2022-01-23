from torch._C import dtype
import config

import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

# TODO: Import iou and non max surpression from utils.py
# TODO: Make that utils.py

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, S=[13, 26, 52], C=2, transform=None) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # bboxes = np.loadtxt(fname=label_path, delimiter=' ', ndmin=2) # [class, x, y, w, h] je oblik podataka u skupu
        # bboxes = np.roll(bboxes, 4, axis=1).tolist() # [x, y, w, h, class] je specifikacijom definisan oblik, nisam siguran da li je ovo neophodno ali eto
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 jer [p_object_present, x, y, w, h, class]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            # Mreza ima 3 prediction scale nivoa, svaki ima po 5 anchor-a
            # prolazimo kroz anchore i gledamo rezultate
            for anchor_index in anchor_indices:
                scale_index = anchor_index // self.num_anchors_per_scale
                anchor_on_scale = anchor_index % self.num_anchors_per_scale
                S = self.S[scale_index]
                # Trebaju nam podaci o celiji slike, jer je u YOLO slika
                # podeljena na poddelove (celije)
                cell_i, cell_j = int(S*y), int(S*x) # x = 0.5, S=13 --> int(6.5) = 6 (sesta celija)
                anchor_taken = targets[scale_index][anchor_on_scale, cell_i, cell_j, 0]

                if not anchor_taken and not has_anchor[scale_index]:
                    targets[scale_index][anchor_on_scale, cell_i, cell_j, 0] = 1
                    cell_x, cell_y = S*x - cell_j, S*y - cell_i
                    cell_width, cell_height = (
                        width * S,
                        height * S,
                    )
                    box_coordinates = torch.tensor(
                        [cell_x, cell_y, cell_width, cell_height]
                    ) 
                    targets[scale_index][anchor_on_scale, cell_i, cell_j, 1:5] = box_coordinates
                    targets[scale_index][anchor_on_scale, cell_i, cell_j, 5] = int(class_label)
                    has_anchor[scale_index] = True
                    

                elif not anchor_taken and iou_anchors[anchor_index] > self.ignore_iou_thresh:
                    targets[scale_index][anchor_on_scale, cell_i, cell_j, 0] = -1

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    # dataset = YOLODataset(
    #     "COCO/train.csv",
    #     "COCO/images/images/",
    #     "COCO/labels/labels_new/",
    #     S=[13, 26, 52],
    #     anchors=anchors,
    #     transform=transform,
    # )

    dataset = YOLODataset(
        "../data/unified_dataset/train.csv",
        "../data/unified_dataset/images/",
        "../data/unified_dataset/labels/",
        anchors=anchors,
        S=[13, 26, 52],
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()

