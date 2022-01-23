import torch
import os
import pandas as pd
from PIL import Image
import cv2


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=2, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [ 
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace('\n', '').split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)  # .resize((448, 448))
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            cell_i, cell_j = int(self.S * y), int(self.S * x)
            cell_x, cell_y = self.S * x - cell_j, self.S * y - cell_i
            cell_width, cell_height = (
                width * self.S,
                height * self.S
            )

            # if there's no obj in cell(i, j)
            if label_matrix[cell_i, cell_j, self.C] == 0:
                label_matrix[cell_i, cell_j, self.C] = 1
                box_coordinates = torch.tensor(
                    [cell_x, cell_y, cell_width, cell_height]
                )
                label_matrix[cell_i, cell_j, self.C+1:self.C+5] = box_coordinates
                label_matrix[cell_i, cell_j, class_label] = 1

        return image, label_matrix
