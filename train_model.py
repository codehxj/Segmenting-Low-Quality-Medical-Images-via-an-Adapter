import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
from matplotlib import pyplot as plt
from torch.cuda import amp
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from model import CRIS
from PIL import Image
from typing import List, Union
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from loguru import logger as Logger
from dataset_med import MyDataset
from utils_cris import JointTransform2D, ImageToImage2D, ImageToImage2D_, Image2D
from test_dataset_cris import MyDataset

parser = argparse.ArgumentParser(description='CRIS')
parser.add_argument('--train_dataset', default='D:/Data/Code_Dataset/Glas/Train_Folder', type=str)
parser.add_argument('--val_dataset', default='D:/Data/Code_Dataset/Glas/Val_Folder', type=str)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--direc', default="D:/Data/CRIS_Med", type=str,help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='batch size (default: 1)')

args = parser.parse_args()
direc = args.direc
# imgsize = 128
# imgchant = 3

tf_train = JointTransform2D(crop=None, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
# train_dataset = ImageToImage2D(args.train_dataset, tf_train)
# val_dataset = ImageToImage2D_(args.val_dataset, tf_val)
train_dataset = ImageToImage2D(args.train_dataset)
val_dataset = ImageToImage2D(args.val_dataset)
predict_dataset = Image2D(args.val_dataset)
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)
#***********************************************************************

# 参数：
batch_size = 2


# 获取数据集：
train_path = r'D:\work_project\CRIS.pytorch-master14\test_func\datasets\Covid19_text\Train_Folder'
val_path = r'D:\work_project\CRIS.pytorch-master14\test_func\datasets\Covid19_text\Val_Folder'
# train_path = r'D:\work_project\CRIS.pytorch-master15\test_func\datasets\Glas\Train_Folder'
# val_path = r'D:\work_project\CRIS.pytorch-master15\test_func\datasets\Glas\Val_Folder'
train_dataset = MyDataset(train_path,'Train')               # 训练集
val_dataset = MyDataset(val_path,'Val')                     # 验证集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=True)

batch_num = (len(train_data_loader))/batch_size             # batch个数



# train函数：
def train(train_data_loader,model,j):
    model.train()
    total_loss_epoch = 0
    for i, data in enumerate(train_data_loader):
        images, labels, text, text_med = data  # images:[1,3,1000,1000]Tensor  labels:[1,1,1000,1000]
        # imgs_SAM = imgs_SAM.to(device=device)
        images = images.to(device=device)
        text = text.to(device=device)
        labels = labels.to(device=device)
        # imgs_SAM = np.array(imgs_SAM)  # (2,416,416,3)

        # forward
        # with amp.autocast():                # 有gpu可用，只有cpu不支持
        with amp.autocast():
            pred, target, loss = model(text_med, images, text, labels)
        total_loss_epoch = total_loss_epoch + loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # GPU加速
        # 防止梯度爆炸，设置一个阈值，参数中是0.
        # if args.max_norm:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # 为了看清楚打印的整体测试集上的损失loss，打印每次训练的损失只有是100的整数倍才打印
        # if total_train_step%100==0:
        # print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
    return total_loss_epoch / batch_num
    # print("Epoch:{},avg_Loss:{}".format(j + 1, (total_loss_epoch / batch_num)))



# val函数：
def validate(val_data_loader,model):
    iou_list = []
    model.eval()
    for i, data in enumerate(val_data_loader):
        images, masks, text, text_med = data
        images = images.to(device=device)
        text = text.to(device=device)
        masks = masks.to(device=device)
        preds = model(text_med,images, text)                 # 104      preds:[2,1,104,104]   batch_size=2
        preds = torch.sigmoid(preds)                # 0-1
        # preds和images尺寸不一致，将preds插值到images大小
        if preds.shape[-2:] != images.shape[-2:]:   # images:  [2,3,416,416]
            preds = F.interpolate(preds,
                                  size=images.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True)
        preds = np.array(preds > 0.35)  # 大于0.35的都置为True,其余的都置为False    [2,1,416,416]
        masks = np.array(masks)
        # preds可视化：
        for p in range(preds.shape[0]):  # 遍历每张图片
            # 将预测结果转换为0-255的灰度图
            preds_img = (preds[p, 0] * 255).astype(np.uint8)
            # 将标签数据转换为0-255的灰度图
            labels_img = (masks[p, 0] * 255).astype(np.uint8)
            # 将背景标记为黑色，目标标记为白色
            preds_img[preds_img == 0] = 0
            preds_img[preds_img == 255] = 255
            labels_img[labels_img == 0] = 0
            labels_img[labels_img == 255] = 255
            # 可视化分割结果图和标签图
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(preds_img, cmap='gray')
            axs[0].axis('off')
            axs[0].set_title('preds')
            axs[1].imshow(labels_img, cmap='gray')
            axs[1].axis('off')
            axs[1].set_title('label')
            # 保存分割结果图和标签图到以轮数命名的文件夹中
            plt.savefig(f'output_data/epoch_{j}/batch_{i}_img_{p}.png')
            plt.clf()               # 清除当前图形,
            plt.close('all')        # 关闭所有 防止内存不够
        # 交集：
        inter = np.logical_and(preds, masks)
        union = np.logical_or(preds, masks)
        iou = np.sum(inter) / (np.sum(union) + 1e-6)
        # print("iou", iou)
        iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = iou_list.mean()

    return iou_list



# 参数：
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
# # 获取数据集：
# train_path = 'D:/Data/Code_Dataset/Glas/Train_Folder'
# val_path = 'D:/Data/Code_Dataset/Glas/Val_Folder'
# save_path = 'D:/Data/CRIS_Med'
# # train_path = r'D:\work_project\CRIS.pytorch-master15\test_func\datasets\Glas\Train_Folder'
# # val_path = r'D:\work_project\CRIS.pytorch-master15\test_func\datasets\Glas\Val_Folder'
# train_dataset = MyDataset(train_path,'Train')               # 训练集
# val_dataset = MyDataset(val_path,'Val')                     # 验证集
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# val_data_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=True)

# batch_num = (len(trainloader))/batch_size             # batch个数

#构造模型：
model = CRIS()
model.to(device=device)
# checkpoint = torch.load('best_model_CRIS.pth')
# model_state_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
# model_state_dict.update(pretrained_dict)
# model.load_state_dict(model_state_dict)

# 构造优化器：
# build optimizer & lr scheduler
param_list = list(model.parameters())
optimizer = torch.optim.Adam(param_list,lr=0.0001,weight_decay=0.)
# 学习率调度器
scheduler = MultiStepLR(optimizer,milestones=[35],gamma=0.1)
scaler = amp.GradScaler()                 # GPU加速的，CPU不支持会报错
# scaler = GradScaler()                       # CPU下适用

#==================== 训练过程： ====================

epoch = 100
best_IoU = 0.0

# logger = Logger(['epoch', 'train_loss', 'val_iou'])
for j in range(epoch):
    # 保存val中分割结果的路径
    if not os.path.exists('output_data'):
        os.makedirs('output_data')
        # 创建以轮数命名的文件夹
    if not os.path.exists(f'output_data/epoch_{j}'):
        os.makedirs(f'output_data/epoch_{j}')

    # ====================== train 训练 ==========================================
    # avg_train_loss = train(train_data_loader, model, j)
    model.train()
    total_loss_epoch = 0
    for batch_idx, data in enumerate(trainloader):
        images, labels, _,text = data  # images:[1,3,1000,1000]Tensor  labels:[1,1,1000,1000]
        # imgs_SAM = imgs_SAM.to(device=device)

        images = images.to(device=device)
        text = text.to(device=device)
        labels = labels.to(device=device)

        # ========================forward====================================
        # with amp.autocast():                # 有gpu可用，只有cpu不支持
        with amp.autocast():
            pred, target, loss = model(images, text, labels)
        total_loss_epoch = total_loss_epoch + loss.item()

        # =======================backward================================
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # GPU加速
        # 防止梯度爆炸，设置一个阈值，参数中是0.
        # if args.max_norm:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # 为了看清楚打印的整体测试集上的损失loss，打印每次训练的损失只有是100的整数倍才打印
        # if total_train_step%100==0:
        # print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
    avg_train_loss =  total_loss_epoch / batch_num  #每一轮的loss值


    # ====================== evaluation 评估指标==================================
    # iou_out = validate(val_data_loader, model)

    iou_list = []
    model.eval()
    for batch_idx, data in enumerate(valloader):
        images, masks, rest, text = data
        if isinstance(rest[0], str):
            image_filename = rest[0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        images = images.to(device=device)
        text = text.to(device=device)
        masks = masks.to(device=device)
        preds = model(images, text)  # 104      preds:[1,1,104,104]   batch_size=2

        preds = torch.sigmoid(preds)  # 0-1
        # preds和images尺寸不一致，将preds插值到images大小
        if preds.shape[-2:] != images.shape[-2:]:  # images:  [2,3,416,416]
            preds = F.interpolate(preds,
                                  size=images.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True)
        preds = preds.cpu()
        preds = np.array(preds > 0.35)  # 大于0.35的都置为True,其余的都置为False    [2,1,416,416]
        masks = np.array(masks.cpu())

        # preds可视化：
        for p in range(preds.shape[0]):  # 遍历每张图片
            # 将预测结果转换为0-255的灰度图
            preds_img = (preds[p, 0] * 255).astype(np.uint8)
            # 将标签数据转换为0-255的灰度图
            labels_img = (masks[p, 0] * 255).astype(np.uint8)
            # 将背景标记为黑色，目标标记为白色
            preds_img[preds_img == 0] = 0
            preds_img[preds_img == 255] = 255
            labels_img[labels_img == 0] = 0
            labels_img[labels_img == 255] = 255
            # 可视化分割结果图和标签图
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(preds_img, cmap='gray')
            axs[0].axis('off')
            axs[0].set_title('preds')
            axs[1].imshow(labels_img, cmap='gray')
            axs[1].axis('off')
            axs[1].set_title('label')
            save_fulldir = args.direc + "/train_val_output" + "/{}/".format(epoch)
            if not os.path.isdir(save_fulldir):
                os.makedirs(save_fulldir)
            # cv2.imwrite(save_fulldir + image_filename, yHaT_[0, 1, :, :])
            save_p = os.path.join(save_fulldir,'epoch_{j}/batch_{i}_img_{p}.png')
            # 保存分割结果图和标签图到以轮数命名的文件夹中
            plt.savefig(save_p)
            plt.clf()  # 清除当前图形,
            plt.close('all')  # 关闭所有 防止内存不够
        # 交集：
        inter = np.logical_and(preds, masks)
        union = np.logical_or(preds, masks)
        iou = np.sum(inter) / (np.sum(union) + 1e-6)
        # print("iou", iou)
        iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = iou_list.mean()
    val_iou_out =  iou_list         # 验证集的iou指标

    print("Epoch:{},train_Loss:{},avg_iou:{}".format(j + 1, avg_train_loss,val_iou_out))

    # update lr
    scheduler.step()
    torch.cuda.empty_cache()

    # logger.add_entry(j, train_loss=avg_epoch_loss, val_loss=iou_out)
    # logger.print_last_entry()

    # 保存最好的模型
    if val_iou_out > best_IoU:
        best_IoU = val_iou_out
        torch.save({'epoch': j,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_iou': best_IoU},
                   'best_model_CRIS.pth')



