from numpy.core.fromnumeric import mean
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import YoloDataset
from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YOLOLoss

seed = 123
torch.manual_seed(seed)

# Hyperparams
LEARNING_RATE = 2E-5
DEVICE = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'
IMG_DIR = 'YOUR/PATH/TO/IMAGES/HERE'  # 'unified_dataset/images'
LABEL_DIR = 'YOUR/PATH/TO/LABELS/HERE'  # 'unified_dataset/labels'

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())

    print(f'Mean loss was {sum(mean_loss)/len(mean_loss)}')


def main():
    print('Creating model...')
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    print('Finished.')

    print('Initializing optimizer...')
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY 
    )
    print('Finished')

    print('Initializing loss function...')
    loss_fn = YOLOLoss()
    print('Finished')

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    print('Initializing train dataset...')
    train_dataset = YoloDataset(
        'unified_dataset/train.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )
    print('Finished')

    print('Initializing test dataset...')
    test_dataset = YoloDataset(
        'unified_dataset/test.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )
    print('Finished')

    print('Creating train data loader...')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )
    print('Finished')


    print('Creating test data loader...')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )
    print('Finished')

    print('Starting trainging...')
    for epoch in range(EPOCHS):
        print(f'EPOCH NO. {epoch}')
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint'
        )

        print(f'Train mAP: {mean_avg_prec}')

        if mean_avg_prec > 0.9:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            break

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == '__main__':
    main()
    