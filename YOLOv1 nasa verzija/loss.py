import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=2):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S  # Kernel size
        self.B = B  # No. of bounding boxes
        self.C = C  # No. of classes
        self. lambda_no_obj = 0.5
        self. lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(
            predictions[..., self.C+1:self.C+5],
            target[..., self.C+1:self.C+5]
            )  # [0..C-1] - verovatnoce klasa [C] verovatnoca prisutnosti [C+1, C+5] - koordinate b-box
        iou_b2 = intersection_over_union(
            predictions[..., self.C+6:self.C+10],
            target[..., self.C+1:self.C+5]
            )

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)  # IObj_i

        # COORDINATES LOSS
        # Explanation: 
        # best_box - index of the most accurate bounding box
        # box_predictions je deo loss-a koji nam govori verovatnocu da se u boudning box-u nalazi objekat
        # box_predictions uzima vrednost najboljeg bounding box-a
        # kako imaju 2 bbox-a onda best_box * predictions[second_box_index] daje 0 ako je prvi bbox najbolj, (jer je best_box = 0)
        # (1 - best_box) * predictions[first_box_index] daje 0 ako je drugi bbox najbolji (jer je best_box = 1)
        # Ovo se racuna samo ako box postoji u celiji (exists_box = 1)
        box_predictions = exists_box * (
            best_box * predictions[..., self.C+6, self.C+10]
            + (1 - best_box) * predictions[..., self.C+1, self.C+5]
        )

        box_targets = exists_box * target[..., self.C+1, self.C+5]

        # Biramo w i h bounding box-a da bi ga korenovali, po specifikaciji za loss iz originalnog rada
        eps = 1e-6  # treba jer ako je vrednost predikcije 0, onda ce optimizator puci jer ne moze izvod 0
        box_predictions[..., 2:4] = (
            torch.sign(box_predictions[..., 2:4])  # da bi sve bilo pozitivno jer kad se predikcije
            * torch.sqrt(torch.abs(box_predictions[..., 2:4] + eps))  # inicajluzuju mogu biti negativne 
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # OBJECT LOSS
        # Explanation: 
        # pred_box - biramo bbox koji je odgovoran za sadrzavanje objekta
        # proveravamo da li ima objekat tu, i ako ima racunamo loss
        pred_box = (
            best_box * predictions[..., self.C+5:self.C+6]
            + (1 - best_box) * predictions[..., self.C:self.C+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        # NO OBJECT LOSS
        # Za prvi bbox
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # Za drugi bbox
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # CLASS LOSS
        #(N, S, S, C) -> (N*S*S, C)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss,
            + object_loss
            + self.lambda_no_obj * no_object_loss
            + class_loss
        )

        return loss


