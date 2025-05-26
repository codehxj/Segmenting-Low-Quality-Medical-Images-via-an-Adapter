import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from MedCLIP__.medclip import MedCLIPProcessor, MedCLIPModel,MedCLIPVisionModel
from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder

from SAM_.segment_anything import sam_model_registry, SamPredictor
from SAM_.segment_anything.utils.transforms import ResizeLongestSide
from collections import defaultdict

class SRBranch(nn.Module):
    def __init__(self, n_features, up_scale, bn=True, act='relu'):
        super(SRBranch, self).__init__()
        m = []
        if (up_scale & (up_scale - 1)) == 0:  # Is scale = 2^n?
            for i in range(int(math.log(up_scale, 2))):
                m.append(nn.ConvTranspose2d(n_features, n_features // 2,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=True))
                m.append(nn.Conv2d(n_features // 2, n_features // 2, kernel_size=1))
                if bn:
                    m.append(nn.BatchNorm2d(n_features // 2))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                n_features = n_features // 2
        else:
            raise NotImplementedError
        self.upsample = nn.Sequential(*m)

        self.classfier = nn.Conv2d(n_features, 3, kernel_size=1)
    def forward(self, x):
        feature = self.upsample(x)
        sr = self.classfier(feature)

        return feature, sr

class CRIS(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load("pretrain/RN50.pt").eval()
        self.backbone = build_model(clip_model.state_dict(), 17).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=[512,1024,1024], out_channels=[256,512,1024])
        # Decoder
        self.decoder = TransformerDecoder(num_layers=3,
                                          d_model=512,
                                          nhead=8,
                                          dim_ffn=2048,
                                          dropout=0.1,
                                          return_intermediate=False)
        # Projector
        self.proj = Projector(1024, 512 // 2, 3)

        self.processor = MedCLIPProcessor()
        self.model_med = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        self.state_dict1 = torch.load("pytorch_model.bin")
        self.model_med.load_state_dict(self.state_dict1)
        # 变换vis的三个维度到512
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)

        # SAM之后和neck之后的concate 再经过一个conv操作：
        self.sam_conv_layer = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, stride=1, padding=1)

        # 定义变换层，用于将CRIS的resnet的三层输出的，不同维度的图像特征 变换到[b,512]的维度，
        # 旨在和medclip的编码器的输出维度对齐，实现从[b,channel,h,w] ---> [b,512]

        #定义超分层：
        self.SRBranch = SRBranch(512,2*16)
        #定义超分的损失：
        self.SR_loss = torch.nn.MSELoss()


        # self.onlyseg = self.onlyseg()
        # self.onlySR = self.onlySR()
        # self.seseg = self.SRSeg()

    def reshape_(self, v):
        # 定义自适应平均池化层
        adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 对x进行自适应平均池化
        v1_ = adaptive_avgpool(v)  # 进行平均池化操作
        # v1_ = torch.squeeze(v1_)  # 压缩最后两个维度
        v1_ = v1_.view(v1_.size(0),-1)  # 压缩最后两个维度
        return v1_

        # MSE损失函数

    def mse_loss(self, x, y):
        loss = torch.mean(torch.pow(x - y, 2))
        return loss
        # 余弦相似度损失函数：

    def cos_loss(self, x, y):
        similarity = F.cosine_similarity(x, y, dim=1)
        loss = F.mse_loss(similarity, torch.ones(2))
        return loss

    def clip_loss(self, img, text, logit_scale):  # 输入是图像和文本的编码，以及温度参数
        logit_scale = nn.Parameter(torch.log(torch.tensor(1 / logit_scale)))
        logit_scale.data = torch.clamp(logit_scale.data, 0, 4.6025)  # 温度参数限制范围
        logit_scale = logit_scale.exp()
        logits_per_img = logit_scale * img @ text.t()  # 计算相似度矩阵，每行是图像，每列是文本，图像对文本的相似度
        logits_per_text = logits_per_img.t()  # 相似度矩阵转置得到文本对图像的相似度矩阵

        batch_size = img.shape[0]
        # labels = torch.arange(batch_size).cpu().long()  # 生成类别标签
        labels = torch.arange(batch_size).cuda()        # 生成类别标签
        loss_img = F.cross_entropy(logits_per_img, labels)  # 做交叉熵损失，对应的图像文本对 使其相似度最大 不匹配的最小
        loss_text = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_img + loss_text) / 2
        return total_loss

    # 添加SAM_encoder模块，该函数输入是一个图像 格式：(416,416,3)的array
    def SAM_encoder(self, image):
        # 加载模型
        device = 'cuda'
        sam = sam_model_registry["vit_b"](checkpoint="SAM_/sam_vit_b_01ec64.pth")
        sam.to(device=device)
        predictor = SamPredictor(sam)
        transformed_data = defaultdict(dict)  # 如果访问一个不存在的键，则会返回指定的默认值（默认为 None）而不是抛出KeyError异常

        transform = ResizeLongestSide(sam.image_encoder.img_size)
        input_image = transform.apply_image(image)  # array (1024,1024,3)
        input_image_torch = torch.as_tensor(input_image, device=device)  # tensor [1024,1024,3]
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]  # tensor[1,3,1024,1024]
        input_image = sam.preprocess(
            transformed_image)  # SAM的数据预处理，transformed_image--归一化操作-->input_image [1,3,1024,1024]
        original_image_size = image.shape[:2]  # 获取原始图像的大小(224,224)  [:2]切片前两个大小
        input_size = tuple(transformed_image.shape[-2:])  # 切片切出 [1024,1024]
        transformed_data[0]['image'] = input_image  # [1,3,1024,1024]
        transformed_data[0]['input_size'] = input_size  # (1024,1024)
        transformed_data[0]['original_image_size'] = original_image_size  # (224,224)

        input_image = transformed_data[0]['image'].to(device)  # [1, 3, 1024, 1024]

        # print("1")
        with torch.no_grad():
            img_embedding = sam.image_encoder(input_image)
            # print(img_embedding.shape)  # torch.Size([1, 256, 64, 64])
        return img_embedding

    # 该函数输入是一个list 格式：(2,416,416,3)的array
    def SAM_encoder_process(self, image):
        # 生成一个tensors[]
        tensors = []
        for m in range(len(image)):
            emb = self.SAM_encoder(image[m])
            tensors.append(emb)
            # print(emb.shape)
        sam_emb = torch.cat(tensors, dim=0)
        # print(sam_emb.shape)  # [2,256,64,64]
        return sam_emb

    def onlyseg(self, imgs_SAM, text_med, lr_img, img, word, mask):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(lr_img)

        encoder_vis = vis  # vis 赋值给encoder_vis，对encoder_vis进行Medclip操作
        # ************************** img和text输入mecclip ***************************************
        inputs = self.processor(
            text=list(text_med),
            images=lr_img,
            return_tensors="pt",
            padding=True
        )
        med_out = self.model_med(**inputs)
        med_out_img = med_out['img_embeds']  # medclip的img编码
        med_out_text = med_out['text_embeds']  # medclip的text编码
        # print("med_out_text:",med_out_text.shape)5t

        # vis的三个输出进行维度变换
        v1, v2, v3 = encoder_vis
        v1_ = self.reshape_(v1)  # v1的维度就是512，直接进行变换
        # # v1[2,512,52,52]   v2[2,1024,26,26]    v3[2,1024,13,13]
        # 对v2 v3进行卷积操作,维度变换到512
        v2 = self.conv(v2)  # [b,512,26,26]
        v3 = self.conv(v3)  # [b,512,13,13]
        v2_ = self.reshape_(v2)  # [b,512]
        v3_ = self.reshape_(v3)  # [b,512]              此处的v1_  v2_  v3_就是维度变换到512之后的结果
        # print("vvvvvvv",v1_.shape,v2_.shape,v3_.shape)

        # # 采用MSE损失函数来 计算medclip的输出和vis的三个向量之间的相似度
        # loss1 = self.mse_loss(med_out_text,v1_)
        # loss2 = self.mse_loss(med_out_text,v2_)
        # loss3 = self.mse_loss(med_out_text,v3_)
        # loss_ = (loss1 + loss2 +loss3)/3
        # # print("loss___:",loss_)

        # # 采用余弦相似度来计算 medclip的输出和vis的三个向量之间的相似度
        # loss1 = self.cos_loss(med_out_img,v1_)
        # loss2 = self.cos_loss(med_out_img,v2_)
        # loss3 = self.cos_loss(med_out_img,v3_)
        # loss_ = (loss1 + loss2 + loss3) / 3

        # 用clip训练用的损失函数，交叉熵损失：
        loss1 = self.clip_loss(v1_, med_out_text, 0.07)
        loss2 = self.clip_loss(v2_, med_out_text, 0.07)
        loss3 = self.clip_loss(v3_, med_out_text, 0.07)
        loss_ = (loss1 + loss2 + loss3) / 3

        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)

        # *******************增加SAM的encoder模块**************************************：
        # 中间concate上SAM_encoder的输出：
        sam_emb = self.SAM_encoder_process(imgs_SAM)  # 输出维度是[b,256,64,64]
        # resize到(26，26)大小
        sam_emb = F.interpolate(sam_emb, size=(26, 26), mode='bilinear', align_corners=True)
        # 进行concate操作：
        sam_concate = torch.concat([fq, sam_emb], dim=1)  # concate之后[b,768,26,26]
        fq = self.sam_conv_layer(sam_concate)  # 经过一道卷积之后 [b,512,26,26]
        #***************************************************************
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            loss = (loss + loss_) / 2
            return pred.detach(), mask, loss
        else:
            return pred.detach()


    def onlySR(self, imgs_SAM, text_med, lr_img, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(lr_img)

        encoder_vis = vis  # vis 赋值给encoder_vis，对encoder_vis进行Medclip操作
        # ************************** img和text输入mecclip ***************************************
        inputs = self.processor(
            text=list(text_med),
            images=lr_img,
            return_tensors="pt",
            padding=True
        )
        med_out = self.model_med(**inputs)
        med_out_img = med_out['img_embeds']  # medclip的img编码
        med_out_text = med_out['text_embeds']  # medclip的text编码
        # print("med_out_text:",med_out_text.shape)5t

        # vis的三个输出进行维度变换
        v1, v2, v3 = encoder_vis
        v1_ = self.reshape_(v1)  # v1的维度就是512，直接进行变换
        # # v1[2,512,52,52]   v2[2,1024,26,26]    v3[2,1024,13,13]
        # 对v2 v3进行卷积操作,维度变换到512
        v2 = self.conv(v2)  # [b,512,26,26]
        v3 = self.conv(v3)  # [b,512,13,13]
        v2_ = self.reshape_(v2)  # [b,512]
        v3_ = self.reshape_(v3)  # [b,512]              此处的v1_  v2_  v3_就是维度变换到512之后的结果
        # print("vvvvvvv",v1_.shape,v2_.shape,v3_.shape)

        # # 采用MSE损失函数来 计算medclip的输出和vis的三个向量之间的相似度
        # loss1 = self.mse_loss(med_out_text,v1_)
        # loss2 = self.mse_loss(med_out_text,v2_)
        # loss3 = self.mse_loss(med_out_text,v3_)
        # loss_ = (loss1 + loss2 +loss3)/3
        # # print("loss___:",loss_)

        # # 采用余弦相似度来计算 medclip的输出和vis的三个向量之间的相似度
        # loss1 = self.cos_loss(med_out_img,v1_)
        # loss2 = self.cos_loss(med_out_img,v2_)
        # loss3 = self.cos_loss(med_out_img,v3_)
        # loss_ = (loss1 + loss2 + loss3) / 3

        # 用clip训练用的损失函数，交叉熵损失：
        loss1 = self.clip_loss(v1_, med_out_text, 0.07)
        loss2 = self.clip_loss(v2_, med_out_text, 0.07)
        loss3 = self.clip_loss(v3_, med_out_text, 0.07)
        loss_ = (loss1 + loss2 + loss3) / 3

        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)

        # *******************增加SAM的encoder模块**************************************：
        # 中间concate上SAM_encoder的输出：
        sam_emb = self.SAM_encoder_process(imgs_SAM)  # 输出维度是[b,256,64,64]
        # resize到(26，26)大小
        sam_emb = F.interpolate(sam_emb, size=(14, 14), mode='bilinear', align_corners=True)
        # 进行concate操作：
        sam_concate = torch.concat([fq, sam_emb], dim=1)  # concate之后[b,768,26,26]
        fq = self.sam_conv_layer(sam_concate)  # 经过一道卷积之后 [b,512,26,26]
        # ***************************************************************
        b, c, h, w = fq.size()

        # 将fq输入到超分模块中进行训练。
        feature_sr, sr_pred = self.SRBranch(fq)
        if self.training:
            # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img.float())
            sr_loss = self.SR_loss(sr_pred,img.float())
            loss = (sr_loss + loss_)/2
            return sr_pred.detach(), img, loss
        else:
            return sr_pred.detach()



    def SRSeg(self, imgs_SAM, text_med, lr_img, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(lr_img)

        encoder_vis = vis  # vis 赋值给encoder_vis，对encoder_vis进行Medclip操作
        # ************************** img和text输入mecclip ***************************************
        inputs = self.processor(
            text=list(text_med),
            images=lr_img,
            return_tensors="pt",
            padding=True
        )
        med_out = self.model_med(**inputs)
        med_out_img = med_out['img_embeds']  # medclip的img编码
        med_out_text = med_out['text_embeds']  # medclip的text编码
        # print("med_out_text:",med_out_text.shape)5t

        # vis的三个输出进行维度变换
        v1, v2, v3 = encoder_vis
        v1_ = self.reshape_(v1)  # v1的维度就是512，直接进行变换
        # # v1[2,512,52,52]   v2[2,1024,26,26]    v3[2,1024,13,13]
        # 对v2 v3进行卷积操作,维度变换到512
        v2 = self.conv(v2)  # [b,512,26,26]
        v3 = self.conv(v3)  # [b,512,13,13]
        v2_ = self.reshape_(v2)  # [b,512]
        v3_ = self.reshape_(v3)  # [b,512]              此处的v1_  v2_  v3_就是维度变换到512之后的结果
        # print("vvvvvvv",v1_.shape,v2_.shape,v3_.shape)

        # # 采用MSE损失函数来 计算medclip的输出和vis的三个向量之间的相似度
        # loss1 = self.mse_loss(med_out_text,v1_)
        # loss2 = self.mse_loss(med_out_text,v2_)
        # loss3 = self.mse_loss(med_out_text,v3_)
        # loss_ = (loss1 + loss2 +loss3)/3
        # # print("loss___:",loss_)

        # # 采用余弦相似度来计算 medclip的输出和vis的三个向量之间的相似度
        # loss1 = self.cos_loss(med_out_img,v1_)
        # loss2 = self.cos_loss(med_out_img,v2_)
        # loss3 = self.cos_loss(med_out_img,v3_)
        # loss_ = (loss1 + loss2 + loss3) / 3

        # 用clip训练用的损失函数，交叉熵损失：
        loss1 = self.clip_loss(v1_, med_out_text, 0.07)
        loss2 = self.clip_loss(v2_, med_out_text, 0.07)
        loss3 = self.clip_loss(v3_, med_out_text, 0.07)
        loss_ = (loss1 + loss2 + loss3) / 3

        word, state = self.backbone.encode_text(word)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)

        # *******************增加SAM的encoder模块**************************************：
        # 中间concate上SAM_encoder的输出：
        sam_emb = self.SAM_encoder_process(imgs_SAM)  # 输出维度是[b,256,64,64]
        # resize到(26，26)大小
        sam_emb = F.interpolate(sam_emb, size=(26, 26), mode='bilinear', align_corners=True)
        # 进行concate操作：
        sam_concate = torch.concat([fq, sam_emb], dim=1)  # concate之后[b,768,26,26]
        fq = self.sam_conv_layer(sam_concate)  # 经过一道卷积之后 [b,512,26,26]
        # ***************************************************************


        fq_copy = fq

        # fq先进入分割分支：
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)


        #fq_copy进入超分分支：
        # 将fq输入到超分模块中进行训练。
        feature_sr, sr_pred = self.SRBranch(fq)
        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            seg_loss = F.binary_cross_entropy_with_logits(pred, mask)
            sr_loss = self.SR_loss(sr_pred, img)
            loss = (sr_loss + loss_ + seg_loss) / 3
            return pred.detach(), mask, loss
        else:
            return pred.detach()

    def get_model(self,imgs_SAM, text_med, lr_img, img, word, mask,epoch):
        if(epoch<=2):
            return self.onlySR(imgs_SAM,text_med, lr_img, img, word, mask)
        else:
            if(epoch<=4):
                return self.onlyseg(imgs_SAM, text_med, lr_img, img, word, mask)
            else:
                return self.SRSeg(imgs_SAM, text_med, lr_img, img, word, mask)


    def forward(self, imgs_SAM, text_med, lr_img, img, word, epoch, mask):
        return self.get_model(imgs_SAM, text_med, lr_img, img, word, mask, epoch)

