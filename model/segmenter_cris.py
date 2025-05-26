import torch
import torch.nn as nn
import torch.nn.functional as F

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



    def forward(self, img, word, mask=None):
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
            return pred.detach(), mask, loss
        else:
            return pred.detach()
