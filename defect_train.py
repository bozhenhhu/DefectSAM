# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import cv2
import json

# set seeds
seed = 42
# np.random.seed(seed)
torch.manual_seed(seed)
# random.seed(seed)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

# os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class DefectDataset(Dataset):
    def __init__(self, data_root='defect/test_set.txt', label_dir='defect/data/labels_coco/labels',
                 height=1024, width=1024, heating_num=50, batchsize=8, bbox_shift=10):
        file = open(data_root, 'r')
        self.data_path = [line.strip() for line in file]
        file.close()

        self.label_dir = label_dir
        self.height = height
        self.width = width
        self.heating_num = heating_num
        self.batchsize = batchsize

        self.bbox_shift = bbox_shift
        print(f"number of samples: {len(self.data_path)}")

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        file_path = self.data_path[index]
        basename = file_path.split('/')[-1].split('.')[0]
        label_path = os.path.join(self.label_dir,'{}_label.png'.format(basename))

        label_img = cv2.imread(label_path, 0)
        scale_x = self.width / label_img.shape[1]
        scale_y = self.height / label_img.shape[0]
        label_img = cv2.resize(label_img, (self.width, self.height))
        label_img[label_img < 120] = 120
        label_img[label_img != 120 ] = 0 #0,black,1 white
        label_img[label_img != 0] = 255 #label convert
        label_img = label_img / 255.

        data_struct = sio.loadmat(file_path)
        data = data_struct['data']
        t_len = data.shape[2]
        sub = data[:, :, -1]
        data = data[:, :, self.heating_num:min(t_len, self.heating_num+160)]
        data = data - np.tile(sub[:, :, np.newaxis], (1, 1, data.shape[2]))

        random_indices = np.random.choice(data.shape[2], size=self.batchsize, replace=False)
        data = data[:, :, random_indices]
        data = cv2.resize(data, (self.width, self.height))
        data = np.transpose(data, (2, 0, 1))
        data = data / 255.

        labels = np.tile(label_img[np.newaxis, :, :], (data.shape[0], 1, 1))
        data = np.tile(data[:, np.newaxis, :, :], (1, 3, 1, 1))

        label_json = os.path.join(self.label_dir,'{}_label.json'.format(basename))
        bboxes = []
        with open(label_json, 'r') as fp:
            label_coord = json.load(fp)
            num_classes = len(label_coord['shapes'])
            masked_image = np.zeros((labels.shape[0], num_classes, labels.shape[1], labels.shape[2]))
            for i in range(num_classes):
                shapes = label_coord['shapes'][i]
                points = shapes['points']
                x_min, y_min = points[0][0], points[0][1]
                x_max, y_max = points[1][0], points[1][1]
                x_min = int(x_min * scale_x)
                x_max = int(x_max * scale_x)
                y_min = int(y_min * scale_y)
                y_max = int(y_max * scale_y)
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(self.width, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(self.height, y_max + random.randint(0, self.bbox_shift))
                bboxes.append([x_min, y_min, x_max, y_max])
                masked_image[:, i, y_min:y_max, x_min:x_max] = labels[:, y_min:y_max, x_min:x_max]

        bboxes = np.array(bboxes)
        bboxes = np.tile(bboxes[np.newaxis, :, :], (data.shape[0], 1, 1))

        return (
            torch.tensor(masked_image).float(),
            torch.tensor(data).float(),
            torch.tensor(bboxes).float(),
            basename,
        )

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="defect/data",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="DefectSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="weights/sam_vit_b_01ec64.pth")
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training")
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("-width", type=int, default=1024)
parser.add_argument("-height", type=int, default=1024)
parser.add_argument("-heating_num", type=int, default=50)
parser.add_argument("-sample_rate", type=int, default=4)

parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },)

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)


class DefectSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        predicted_masks = torch.zeros((image.shape[0], boxes.shape[1], image.shape[2], image.shape[3]), device=image.device)
        # do not compute gradients for prompt encoder
        for i in range(boxes.shape[1]):
            box = boxes[:, i, :]
            with torch.no_grad():
                box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            predicted_masks[:, i, :, :] = ori_res_masks.squeeze(1)
        
    def inference(self,):
        pass
        


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, os.path.join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    Defect_model = DefectSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    Defect_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in Defect_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in Defect_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(Defect_model.image_encoder.parameters()) + list(
        Defect_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    num_epochs = args.num_epochs
    iter_num = 0
    train_losses = []
    val_losses = []
    best_loss = 1e10
    train_dataset = DefectDataset(data_root='defect/train_set.txt', height=args.height, width=args.width, 
                                  heating_num=args.heating_num, batchsize=args.sample_rate)
    val_dataset = DefectDataset(data_root='defect/val_set.txt', height=args.height, width=args.width, 
                                heating_num=args.heating_num, batchsize=args.sample_rate)
    # test_dataset = DefectDataset(data_root='defect/test_set.txt', height=args.height, width=args.width, 
    #                              heating_num=args.heating_num, batchsize=args.sample_rate)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            Defect_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_epoch_loss = 0
        for step, (labels, data, bboxes, names_temp) in enumerate(train_dataloader):
            data = torch.flatten(data, start_dim=0, end_dim=1)
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
            optimizer.zero_grad()
            boxes_np = bboxes.detach().cpu().numpy()
            labels, data = labels.to(device), data.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = Defect_model(data, boxes_np)
                    loss = seg_loss(medsam_pred, labels) + ce_loss(
                        medsam_pred, labels.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = Defect_model(data, boxes_np)
                seg_loss_ = seg_loss(medsam_pred, labels)
                ce_loss_ = ce_loss(medsam_pred, labels.float())
                loss = seg_loss_ + ce_loss_
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f'Epoch: {epoch}, Step: {step}, seg_loss: {seg_loss_.item()}, ce_loss_: {ce_loss_.item()}')

            train_epoch_loss += loss.item()
            iter_num += 1

        train_epoch_loss /= step
        train_losses.append(train_epoch_loss)
        if args.use_wandb:
            wandb.log({"train_epoch_loss": train_epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train_epoch_loss: {train_epoch_loss}'
        )

        val_epoch_loss = 0
        for step, (labels, data, bboxes, names_temp) in enumerate(val_dataloader):
            data = torch.flatten(data, start_dim=0, end_dim=1)
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
            boxes_np = bboxes.detach().cpu().numpy()
            labels, data = labels.to(device), data.to(device)
            with torch.no_grad():
                medsam_pred = Defect_model(data, boxes_np)
                seg_loss_ = seg_loss(medsam_pred, labels)
                ce_loss_ = ce_loss(medsam_pred, labels.float())
                loss = seg_loss_ + ce_loss_

            print(f'Epoch: {epoch}, Step: {step}, val_seg_loss: {seg_loss_.item()}, val_ce_loss_: {ce_loss_.item()}')

            val_epoch_loss += loss.item()
            iter_num += 1
        val_epoch_loss /= step
        val_losses.append(val_epoch_loss)
        if args.use_wandb:
            wandb.log({"val_epoch_loss": val_epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, val_epoch_loss: {val_epoch_loss}'
        )
        
        ## save the best model
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            checkpoint = {
                "model": Defect_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "defectsam_model_best.pth"))
        losses = {'train_losses': train_losses, 'val_losses': val_losses}
        with open(os.path.join(model_save_path, "losses.json"), 'w') as f:
            json.dump(losses, f)
        # %% plot loss
        # plt.plot(losses)
        # plt.title("Dice + Cross Entropy Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        # plt.close()


if __name__ == "__main__":
    main()
