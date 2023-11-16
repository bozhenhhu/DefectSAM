# -*- coding: utf-8 -*-
"""
test the image encoder and mask decoder
freeze prompt image encoder
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
def calculate_iou(y_hat, y):
    intersection = np.logical_and(y_hat, y)
    union = np.logical_or(y_hat, y)
    iou = np.sum(intersection) / np.sum(union)
    return iou

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
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

@torch.no_grad()
def defectsam_inference(medsam_model, img_embed, boxes, H, W):

    predicted_masks = np.zeros((img_embed.shape[0], boxes.shape[1], H, W))
    for i in range(boxes.shape[1]):
        box = boxes[:, i, :]
        box_torch = torch.as_tensor(box, dtype=torch.float, device=img_embed.device)
        if len(box.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        predicted_masks[:, i, :, :] = medsam_seg
    return predicted_masks

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
        # for param in self.prompt_encoder.parameters():
        #     param.requires_grad = False

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
        
    def inference(self, img_embed, boxes, H, W):
        predicted_masks = np.zeros((img_embed.shape[0], boxes.shape[1], H, W))
        for i in range(boxes.shape[1]):
            box = boxes[:, i, :]
            box_torch = torch.as_tensor(box, dtype=torch.float, device=img_embed.device)
            if len(box.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, _ = self.mask_decoder(
                image_embeddings=img_embed,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )

            low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

            low_res_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
            medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
            predicted_masks[:, i, :, :] = medsam_seg
        return predicted_masks
        


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
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="defect/output",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=list,
    default=[10,200,100,250], #[95, 255, 190, 350]
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="weights/defect_vit_b.pth",
    help="path to the trained model",
)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-width", type=int, default=1024)
parser.add_argument("-height", type=int, default=1024)
parser.add_argument("-heating_num", type=int, default=50)
parser.add_argument("-sample_rate", type=int, default=4)

args = parser.parse_args()


device = torch.device(args.device)

medsam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()


test_dataset = DefectDataset(data_root='defect/test_set.txt', height=args.height, width=args.width, 
                                heating_num=args.heating_num, batchsize=args.sample_rate)

print("Number of training samples: ", len(test_dataset))
    
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

IOU_plane = []
IOU_R = []
R_type = ['036g', '029g', '035g', '012g']
for step, (labels, data, bboxes, names_temp) in enumerate(test_dataloader):
    data = torch.flatten(data, start_dim=0, end_dim=1)
    labels = torch.flatten(labels, start_dim=0, end_dim=1)
    bboxes = torch.flatten(bboxes, start_dim=0, end_dim=1)
    boxes_np = bboxes.detach().cpu().numpy()
    labels, data = labels.to(device), data.to(device)
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(data)
    defectsam_seg = defectsam_inference(medsam_model, image_embedding, boxes_np, args.height, args.width)
    # print('medsam', medsam_seg.shape)
    # print('data', data.shape)
    # print('label', labels.shape)
    labels = labels.cpu().numpy()
    data = data.cpu().numpy()
    for i in range(data.shape[0]):
        label_img = labels[0]
        label_img = label_img.astype(bool)
        pre_img = defectsam_seg[i]
        pre_img = pre_img.astype(bool)
        y = label_img[0]
        y_hat = pre_img[0]
        for j in range(label_img.shape[0]):
            y = np.logical_or(y, label_img[j])
        for j in range(pre_img.shape[0]):
            y_hat = np.logical_or(y_hat, pre_img[j])
        IOU = calculate_iou(y_hat, y)
        if names_temp[0] in R_type:
            IOU_R.append(IOU)
        else:
            IOU_plane.append(IOU)
print('avg plane IOU', sum(IOU_plane)/len(IOU_plane))
print('avg R IOU', sum(IOU_R)/len(IOU_R))
    # for i in range(data.shape[0]):
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111)
    #     img = data[i].cpu().numpy()
    #     img = img.transpose(1, 2, 0)
    #     # img = data_np[0].astype(float)
    #     ax.imshow(img)

    #     print(len(medsam_seg))
    #     for mask in medsam_seg[i]:
    #         show_mask(mask, ax, random_color=False)
        
    #     # for box in boxes_np[0]:
    #     #     show_box(box, ax)
    #     ax.axis("off")

    #     # plt.show()
    #     plt.tight_layout()
    #     plt.savefig('defect/output/MedSAM/{}_{}.png'.format(names_temp[0], i))
    # plt.shxianow()
    # break