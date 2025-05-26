import torch
import torch.nn as nn
import torch.nn.functional as F

from MedCLIP__.medclip import MedCLIPProcessor, MedCLIPModel,MedCLIPVisionModel
from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder


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



    def forward(self, text_med, img, word, mask=None):
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
        vis = self.backbone.encode_image(img)

        encoder_vis = vis  # vis 赋值给encoder_vis，对encoder_vis进行Medclip操作
        # ************************** img和text输入mecclip ***************************************
        inputs = self.processor(
            text=list(text_med),
            images=img,
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
